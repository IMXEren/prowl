"""Resource-bound tests for the logical session registry.

These cover the Phase 3 admission policy: when an isolated per-egress budget is
configured, a full registry evicts the oldest idle session through the existing
cleanup fence instead of failing, never an active or queued one, and only once its
callback completes. Without a configured budget the legacy no-eviction rejection
stands. Every case is deterministic and uses no browser.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase

from prowl.service.errors import SessionLimitError, SessionNotFoundError
from prowl.service.sessions import ISOLATED_MODE, SessionInfo, SessionRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


class _CleanupError(Exception):
    """A backend cleanup failure used to exercise the retained-tombstone path."""


class _CleanupRecorder:
    """Record cleanup calls, optionally gating and failing the first ones."""

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


class SessionIdleEvictionTests(IsolatedAsyncioTestCase):
    """The idle LRU picks the oldest session and tracks lease activity."""

    async def test_the_oldest_idle_session_is_evicted_and_a_lease_reorders_it(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=2, cleanup=recorder, isolated_per_egress_limit=2)
        await registry.create("a")
        await registry.create("b")
        registry._entries["a"].last_used = -2.0
        registry._entries["b"].last_used = -1.0

        # A completed lease refreshes "a", so "b" is now the oldest idle session.
        async with registry.lease("a"):
            pass
        self.assertGreaterEqual(registry._entries["a"].last_used, 0.0)

        self.assertEqual(await registry.create("c"), "c")
        self.assertEqual([info.id for info in recorder.calls], ["b"])
        self.assertEqual(await registry.list_sessions(), ["a", "c"])


class SessionEvictionImmunityTests(IsolatedAsyncioTestCase):
    """Active and queued leases are never eviction candidates."""

    async def test_active_and_queued_leases_are_not_evicted(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a")

        entered = asyncio.Event()
        release = asyncio.Event()
        queued_entered = asyncio.Event()
        queued_release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "a", entered, release))
        await entered.wait()
        queued = asyncio.create_task(_hold_lease(registry, "a", queued_entered, queued_release))
        await _wait_until(lambda: registry._entries["a"].active == 2)

        with self.assertRaises(SessionLimitError):
            await registry.create("b")
        self.assertEqual(recorder.calls, [])

        release.set()
        await queued_entered.wait()
        queued_release.set()
        await asyncio.gather(holder, queued)

        # With the leases drained "a" is idle, so admission evicts it instead of failing.
        self.assertEqual(await registry.create("b"), "b")
        self.assertEqual([info.id for info in recorder.calls], ["a"])


class SessionEgressBudgetTests(IsolatedAsyncioTestCase):
    """The isolated budget is scoped to one effective egress."""

    async def test_the_isolated_budget_evicts_only_the_matching_egress(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=10, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a", mode=ISOLATED_MODE, egress="x")
        await registry.create("b", mode=ISOLATED_MODE, egress="y")

        # "x" is at its budget, so the new "x" session evicts the only one and leaves "y".
        self.assertEqual(await registry.create("c", mode=ISOLATED_MODE, egress="x"), "c")
        self.assertEqual([info.id for info in recorder.calls], ["a"])
        self.assertEqual(await registry.list_sessions(), ["b", "c"])

        self.assertEqual(await registry.create("d", mode=ISOLATED_MODE, egress="y"), "d")
        self.assertEqual([info.id for info in recorder.calls], ["a", "b"])


class SessionEvictionFenceTests(IsolatedAsyncioTestCase):
    """Eviction holds capacity until cleanup completes and honours the fence."""

    async def test_capacity_is_held_until_the_evicted_cleanup_completes(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a")
        recorder.gate = asyncio.Event()

        creating = asyncio.create_task(registry.create("b"))
        await _wait_until(recorder.started.is_set)
        self.assertFalse(creating.done())
        with self.assertRaises(SessionLimitError):
            await registry.create("c")

        recorder.gate.set()
        self.assertEqual(await creating, "b")
        self.assertEqual([info.id for info in recorder.calls], ["a"])

    async def test_cancelling_an_admission_keeps_the_cleanup_fenced(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a")
        recorder.gate = asyncio.Event()

        creating = asyncio.create_task(registry.create("b"))
        await _wait_until(recorder.started.is_set)
        creating.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await creating

        self.assertIn("a", registry._closing)
        with self.assertRaises(SessionNotFoundError):
            async with registry.lease("a"):
                pass

        recorder.gate.set()
        await _wait_until(lambda: "a" not in registry._closing)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(await registry.create("b"), "b")


class SessionCleanupFailureRetentionTests(IsolatedAsyncioTestCase):
    """A failed eviction surfaces and retains its capacity reservation."""

    async def test_a_failed_eviction_retains_capacity_until_a_retry(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a")

        with self.assertRaises(_CleanupError):
            await registry.create("b")
        with self.assertRaises(SessionLimitError):
            await registry.create("c")

        await registry.retry_cleanup("a")
        self.assertEqual(await registry.create("c"), "c")
        self.assertEqual([info.id for info in recorder.calls], ["a", "a"])


class SessionLegacyPolicyTests(IsolatedAsyncioTestCase):
    """Without a budget the registry rejects; a zero budget disables isolated sessions."""

    async def test_legacy_rejection_and_zero_isolated_budget(self) -> None:
        recorder = _CleanupRecorder()
        legacy = SessionRegistry(max_sessions=1, cleanup=recorder)
        await legacy.create("a")
        with self.assertRaises(SessionLimitError):
            await legacy.create("b")
        self.assertEqual(recorder.calls, [])
        self.assertEqual(await legacy.list_sessions(), ["a"])

        disabled = SessionRegistry(max_sessions=4, isolated_per_egress_limit=0)
        await disabled.create("shared")
        with self.assertRaises(SessionLimitError):
            await disabled.create("i-default", mode=ISOLATED_MODE)
        with self.assertRaises(SessionLimitError):
            await disabled.create("i-x", mode=ISOLATED_MODE, egress="x")
        self.assertEqual(await disabled.create("shared-two"), "shared-two")


class CombinedPressureTests(IsolatedAsyncioTestCase):
    async def test_combined_pressure_evicts_only_the_matching_isolated_session(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=2, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("shared")
        await registry.create("isolated", mode=ISOLATED_MODE, egress="east")
        await registry.create("replacement", mode=ISOLATED_MODE, egress="east")
        self.assertEqual([info.id for info in recorder.calls], ["isolated"])
        self.assertEqual(await registry.list_sessions(), ["replacement", "shared"])
        await registry.aclose()

    async def test_zero_isolated_budget_never_evicts_a_shared_session(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=0)
        await registry.create("shared")
        with self.assertRaises(SessionLimitError):
            await registry.create("isolated", mode=ISOLATED_MODE)
        self.assertEqual(recorder.calls, [])
        self.assertEqual(await registry.list_sessions(), ["shared"])
        await registry.aclose()

    async def test_active_matching_session_does_not_evict_unrelated_idle_sessions(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=2, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("shared")
        await registry.create("isolated", mode=ISOLATED_MODE, egress="east")
        async with registry.lease("isolated"):
            with self.assertRaises(SessionLimitError):
                await registry.create("replacement", mode=ISOLATED_MODE, egress="east")
        self.assertEqual(recorder.calls, [])
        self.assertEqual(await registry.list_sessions(), ["isolated", "shared"])
        await registry.aclose()
