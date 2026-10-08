"""Cleanup provenance tests for the logical session registry.

The registry records why a session's cleanup ran: an explicit destroy, automatic
retirement (TTL expiry or admission eviction), or registry shutdown. The origin is
immutable across retries, so a failed automatic cleanup stays automatic even when a
later destroy retries it. Every case is deterministic and uses no browser.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from prowl.service.errors import SessionError
from prowl.service.sessions import ISOLATED_MODE, SessionInfo, SessionRegistry

if TYPE_CHECKING:
    from prowl.service.sessions import _Entry


class _CleanupError(Exception):
    """A backend cleanup failure used to exercise the tombstone path."""


class _CleanupRecorder:
    """Record cleanup calls, optionally gating or failing the first ones."""

    def __init__(self, *, failures: int = 0) -> None:
        self.calls: list[SessionInfo] = []
        self.started = asyncio.Event()
        self.gate: asyncio.Event | None = None
        self._failures = failures

    async def __call__(self, info: SessionInfo) -> None:
        self.calls.append(info)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if len(self.calls) <= self._failures:
            msg = "cleanup failed"
            raise _CleanupError(msg)


async def _start_shutdown(registry: SessionRegistry) -> asyncio.Task[None]:
    waiting = asyncio.Event()
    original = registry._await_close

    async def await_close(entry: _Entry) -> None:
        waiting.set()
        await original(entry)

    with patch.object(registry, "_await_close", side_effect=await_close):
        task = asyncio.create_task(registry.aclose())
        await asyncio.wait_for(waiting.wait(), timeout=2.0)
    return task


async def _hold_lease(
    registry: SessionRegistry, session_id: str, entered: asyncio.Event, release: asyncio.Event
) -> None:
    async with registry.lease(session_id):
        entered.set()
        await release.wait()


class SessionCloseReasonTests(IsolatedAsyncioTestCase):
    """Each cleanup path stamps the origin of its attempt on the callback metadata."""

    async def test_explicit_destroy_reports_the_destroy_reason(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None, mode=ISOLATED_MODE, egress="a")
        await registry.destroy("s")
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].close_reason, "destroy")
        self.assertFalse(recorder.calls[0].evicted)

    async def test_ttl_expiry_reports_the_evicted_reason(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5, egress="a")
        registry._entries["s"].expires_at = time.monotonic() - 1
        self.assertEqual(await registry.purge_expired(), ["s"])
        self.assertEqual(recorder.calls[0].close_reason, "evicted")
        self.assertTrue(recorder.calls[0].evicted)

    async def test_admission_eviction_reports_the_evicted_reason(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, isolated_per_egress_limit=1, cleanup=recorder)
        await registry.ensure("old", 5, mode=ISOLATED_MODE)
        await registry.ensure("new", 5, mode=ISOLATED_MODE)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].id, "old")
        self.assertEqual(recorder.calls[0].close_reason, "evicted")
        self.assertTrue(recorder.calls[0].evicted)

    async def test_live_shutdown_reports_the_shutdown_reason(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await asyncio.wait_for(entered.wait(), timeout=2.0)

        closing = await _start_shutdown(registry)
        self.assertEqual(len(recorder.calls), 0)
        release.set()
        await holder
        await asyncio.wait_for(closing, timeout=2.0)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].close_reason, "shutdown")
        self.assertFalse(recorder.calls[0].evicted)

    async def test_shutdown_leaves_an_in_flight_destroy_reason_untouched(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await asyncio.wait_for(recorder.started.wait(), timeout=2.0)

        closing = await _start_shutdown(registry)
        self.assertFalse(closing.done())
        recorder.gate.set()
        await destroy
        await asyncio.wait_for(closing, timeout=2.0)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].close_reason, "destroy")
        self.assertFalse(recorder.calls[0].evicted)


class SessionCloseReasonRetryTests(IsolatedAsyncioTestCase):
    """A retried cleanup keeps the immutable origin of the first attempt."""

    async def test_a_retried_eviction_stays_evicted_even_through_destroy(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        registry._entries["s"].expires_at = time.monotonic() - 1
        with self.assertRaises(_CleanupError):
            await registry.purge_expired()

        await registry.destroy("s")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(recorder.calls[0].close_reason, "evicted")
        self.assertEqual(recorder.calls[1].close_reason, "evicted")
        self.assertTrue(recorder.calls[1].evicted)

    async def test_a_retried_shutdown_stays_shutdown(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        with self.assertRaises(_CleanupError):
            await registry.aclose()

        await registry.retry_cleanup("s")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(recorder.calls[0].close_reason, "shutdown")
        self.assertEqual(recorder.calls[1].close_reason, "shutdown")
        self.assertFalse(recorder.calls[1].evicted)


class SessionCloseReasonFenceTests(IsolatedAsyncioTestCase):
    """A reason the caller never asks for is still recorded where the origin requires it."""

    async def test_a_recreated_generation_starts_from_the_default_reason(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        registry._entries["s"].expires_at = time.monotonic() - 1
        await registry.purge_expired()
        info = await registry.ensure("s", None)
        self.assertEqual(info.close_reason, "destroy")
        self.assertFalse(info.evicted)

    async def test_a_repeated_destroy_retries_with_the_destroy_reason(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        with self.assertRaises(_CleanupError):
            await registry.destroy("s")
        await registry.destroy("s")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(recorder.calls[1].close_reason, "destroy")
        self.assertFalse(recorder.calls[1].evicted)

    async def test_shutdown_fences_the_reason_for_a_rejected_admission(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        closing = asyncio.create_task(registry.aclose())
        await asyncio.wait_for(recorder.started.wait(), timeout=2.0)
        with self.assertRaises(SessionError):
            await registry.ensure("new", None)
        recorder.gate.set()
        await asyncio.wait_for(closing, timeout=2.0)
        self.assertEqual(recorder.calls[0].close_reason, "shutdown")
        self.assertFalse(recorder.calls[0].evicted)
