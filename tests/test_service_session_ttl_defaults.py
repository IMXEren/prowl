"""Idle TTL defaults for isolated sessions.

A new isolated session gets a configured idle TTL when its caller omits one, while a
shared session stays unlimited unless a TTL is stated. These are pure tests: the
registry is driven by a hand-advanced monotonic clock and the service by a spec'd
backend mock, so no browser process and no network are involved.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from prowl.service import sessions as sessions_module
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchResult
from prowl.service.sessions import ISOLATED_MODE, SessionInfo, SessionRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


class _CleanupRecorder:
    """Record every cleanup the registry drives."""

    def __init__(self) -> None:
        self.calls: list[SessionInfo] = []

    async def __call__(self, info: SessionInfo) -> None:
        self.calls.append(info)


class _FakeClock:
    """A monotonic clock the registry reads while a test advances idle time by hand."""

    __slots__ = ("_now",)

    def __init__(self, now: float) -> None:
        self._now = now

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


async def _wait_until(predicate: Callable[[], bool], *, attempts: int = 400) -> None:
    """Yield until *predicate* holds, failing rather than hanging when it never does."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    msg = "the expected state was not reached in time"
    raise AssertionError(msg)


async def _hold_lease(
    registry: SessionRegistry, session_id: str, entered: asyncio.Event, release: asyncio.Event
) -> None:
    async with registry.lease(session_id):
        entered.set()
        await release.wait()


def _service(**overrides: Any) -> tuple[Service, Any]:
    backend = Mock(spec=Backend)
    backend.fetch.return_value = FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={},
        response="",
        cookies=[],
        user_agent=None,
    )
    return Service(ServiceConfig(**overrides), backend), backend


class SessionTtlDefaultConfigTests(TestCase):
    """``ServiceConfig.session_ttl_minutes`` defaults to one hour and parses its env var."""

    def test_the_default_isolated_ttl_is_one_hour(self) -> None:
        self.assertEqual(ServiceConfig().session_ttl_minutes, 60)

    def test_the_env_var_sets_the_default(self) -> None:
        with patch.dict(os.environ, {"PROWL_SESSION_TTL_MINUTES": "90"}):
            self.assertEqual(ServiceConfig.from_env().session_ttl_minutes, 90)

    def test_an_absent_or_empty_env_var_keeps_the_default(self) -> None:
        for value in ["", "   "]:
            with patch.dict(os.environ, {"PROWL_SESSION_TTL_MINUTES": value}):
                self.assertEqual(ServiceConfig.from_env().session_ttl_minutes, 60)

    def test_non_positive_and_non_numeric_env_values_are_refused(self) -> None:
        for invalid in ["0", "-3", "many", "1.5"]:
            with (
                patch.dict(os.environ, {"PROWL_SESSION_TTL_MINUTES": invalid}),
                self.assertRaisesRegex(ValueError, "PROWL_SESSION_TTL_MINUTES"),
            ):
                ServiceConfig.from_env()


class RegistryIsolatedTtlDefaultTests(IsolatedAsyncioTestCase):
    """The registry default applies to new isolated sessions only, and never to an explicit value."""

    async def test_a_new_isolated_session_takes_the_default_and_shared_stays_unlimited(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=60)

        await registry.create("iso", mode=ISOLATED_MODE)
        isolated = registry._entries["iso"]
        self.assertEqual(isolated.ttl_minutes, 60)
        self.assertIsNotNone(isolated.expires_at)

        await registry.create("shared")
        shared = registry._entries["shared"]
        self.assertIsNone(shared.ttl_minutes)
        self.assertIsNone(shared.expires_at)

    async def test_ensure_of_a_new_isolated_session_takes_the_default(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=15)
        info = await registry.ensure("iso", mode=ISOLATED_MODE)
        self.assertEqual(info.ttl_minutes, 15)
        self.assertIsNotNone(registry._entries["iso"].expires_at)

    async def test_an_explicit_ttl_overrides_the_default(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=60)
        await registry.create("iso", 5, mode=ISOLATED_MODE)
        self.assertEqual(registry._entries["iso"].ttl_minutes, 5)

    async def test_an_explicit_none_creates_a_new_isolated_session_without_expiry(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=60)
        await registry.create("iso", None, mode=ISOLATED_MODE)
        self.assertIsNone(registry._entries["iso"].ttl_minutes)
        self.assertIsNone(registry._entries["iso"].expires_at)

    async def test_an_omitted_ensure_refresh_preserves_a_custom_ttl(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=60)
        await registry.create("iso", 5, mode=ISOLATED_MODE)
        info = await registry.ensure("iso", mode=ISOLATED_MODE)
        self.assertEqual(info.ttl_minutes, 5)

    async def test_an_explicit_none_ensure_clears_the_default(self) -> None:
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=60)
        await registry.create("iso", mode=ISOLATED_MODE)
        info = await registry.ensure("iso", None, mode=ISOLATED_MODE)
        self.assertIsNone(info.ttl_minutes)
        self.assertIsNone(registry._entries["iso"].expires_at)

    async def test_no_configured_default_keeps_direct_registry_callers_unlimited(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("iso", mode=ISOLATED_MODE)
        self.assertIsNone(registry._entries["iso"].ttl_minutes)

    def test_a_non_positive_configured_default_is_refused(self) -> None:
        for invalid in [0, -5]:
            with self.assertRaisesRegex(ValueError, "default_isolated_ttl_minutes"):
                SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=invalid)


class RegistryIsolatedTtlIdleDeadlineTests(IsolatedAsyncioTestCase):
    """Idle expiry, active and queued lease immunity, and the final-release rearm."""

    async def test_an_active_lease_is_immune_and_the_final_release_rearms_the_deadline(self) -> None:
        clock = _FakeClock(1000.0)
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=10)
        with patch.object(sessions_module, "time", clock):
            await registry.create("iso", mode=ISOLATED_MODE)
            entry = registry._entries["iso"]
            self.assertEqual(entry.expires_at, 1600.0)

            entered = asyncio.Event()
            release = asyncio.Event()
            holder = asyncio.create_task(_hold_lease(registry, "iso", entered, release))
            await _wait_until(entered.is_set)

            clock.advance(700.0)
            self.assertEqual(await registry.purge_expired(), [])

            release.set()
            await holder
            assert entry.expires_at is not None
            self.assertGreater(entry.expires_at, clock.monotonic())

            clock.advance(599.0)
            self.assertEqual(await registry.purge_expired(), [])
            clock.advance(2.0)
            self.assertEqual(await registry.purge_expired(), ["iso"])

    async def test_a_queued_lease_also_blocks_expiry(self) -> None:
        clock = _FakeClock(1000.0)
        registry = SessionRegistry(max_sessions=4, default_isolated_ttl_minutes=10)
        with patch.object(sessions_module, "time", clock):
            await registry.create("iso", mode=ISOLATED_MODE)
            held = asyncio.Event()
            release = asyncio.Event()
            first = asyncio.create_task(_hold_lease(registry, "iso", held, release))
            await _wait_until(held.is_set)
            queued = asyncio.create_task(_hold_lease(registry, "iso", asyncio.Event(), asyncio.Event()))
            await _wait_until(lambda: registry._entries["iso"].active == 2)

            clock.advance(700.0)
            self.assertEqual(await registry.purge_expired(), [])

            release.set()
            await first
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            self.assertIn("iso", await registry.list_sessions())

    async def test_expiry_runs_cleanup_once_and_fences_the_id(self) -> None:
        clock = _FakeClock(1000.0)
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder, default_isolated_ttl_minutes=10)
        with patch.object(sessions_module, "time", clock):
            await registry.create("iso", mode=ISOLATED_MODE)
            clock.advance(601.0)
            self.assertEqual(await registry.purge_expired(), ["iso"])
            self.assertEqual(len(recorder.calls), 1)
            self.assertEqual(recorder.calls[0].ttl_minutes, 10)
            self.assertEqual(await registry.list_sessions(), [])

            self.assertEqual(await registry.purge_expired(), [])
            self.assertEqual(len(recorder.calls), 1)


class ServiceIsolatedTtlDefaultTests(IsolatedAsyncioTestCase):
    """The service wires the configured default and keeps an omitted wire TTL unspecified."""

    async def test_sessions_create_uses_the_isolated_default(self) -> None:
        service, _backend = _service(session_ttl_minutes=45)
        self.assertEqual(service.sessions.default_isolated_ttl_minutes, 45)

        _status, body = await service.handle({"cmd": "sessions.create", "session": "iso", "sessionMode": "isolated"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(service.sessions._entries["iso"].ttl_minutes, 45)

    async def test_sessions_create_leaves_a_shared_session_unlimited(self) -> None:
        service, _backend = _service(session_ttl_minutes=45)
        await service.handle({"cmd": "sessions.create", "session": "shared"})
        self.assertIsNone(service.sessions._entries["shared"].ttl_minutes)
        self.assertIsNone(service.sessions._entries["shared"].expires_at)

    async def test_a_request_auto_create_uses_the_isolated_default(self) -> None:
        service, _backend = _service(session_ttl_minutes=45)
        _status, body = await service.handle(
            {"cmd": "request.get", "url": "https://example.com/", "session": "iso", "sessionMode": "isolated"}
        )
        self.assertEqual(body["status"], "ok")
        self.assertEqual(service.sessions._entries["iso"].ttl_minutes, 45)

    async def test_an_explicit_wire_ttl_overrides_the_default(self) -> None:
        service, _backend = _service(session_ttl_minutes=45)
        await service.handle(
            {"cmd": "sessions.create", "session": "iso", "sessionMode": "isolated", "session_ttl_minutes": 5}
        )
        self.assertEqual(service.sessions._entries["iso"].ttl_minutes, 5)

    async def test_an_omitted_wire_ttl_preserves_a_custom_ttl_on_refresh(self) -> None:
        service, _backend = _service(session_ttl_minutes=45)
        await service.handle(
            {"cmd": "sessions.create", "session": "iso", "sessionMode": "isolated", "session_ttl_minutes": 5}
        )
        await service.handle({"cmd": "request.get", "url": "https://example.com/", "session": "iso"})
        self.assertEqual(service.sessions._entries["iso"].ttl_minutes, 5)
