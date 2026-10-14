"""Lifecycle tests for the logical session registry.

These cover the Phase 1 session contract: mode and egress binding, TTL
preservation, lease metadata from the admitted generation, and the destroy,
expiry, cancellation, and shutdown fencing that keeps an id unavailable until
its backend cleanup finishes. Every case is deterministic and uses no browser.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase

from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME
from prowl.service.errors import SessionError, SessionLimitError, SessionNotFoundError
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE, SessionInfo, SessionRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


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


class SessionMetadataTests(IsolatedAsyncioTestCase):
    """Mode, egress binding, and TTL preservation across ensure calls."""

    async def test_shared_is_the_default_mode(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        info = await registry.ensure("s", None)
        self.assertEqual(info.mode, SHARED_MODE)
        self.assertIsNone(info.egress)
        self.assertIsNone(info.ttl_minutes)

    async def test_isolated_mode_is_opt_in_and_sticky(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        info = await registry.ensure("s", 5, mode=ISOLATED_MODE)
        self.assertEqual(info.mode, ISOLATED_MODE)
        self.assertEqual(info.egress, DEFAULT_EGRESS_NAME)
        again = await registry.ensure("s", mode=ISOLATED_MODE)
        self.assertEqual(again.mode, ISOLATED_MODE)
        with self.assertRaises(SessionError):
            await registry.ensure("s", mode=SHARED_MODE)

    async def test_egress_binds_once_and_rejects_a_change(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None, egress="a")
        with self.assertRaises(SessionError):
            await registry.ensure("s", None, egress="b")
        info = await registry.ensure("s", None, egress="a")
        self.assertEqual(info.egress, "a")

    async def test_a_shared_session_stays_unbound_until_first_use(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        self.assertIsNone((await registry.ensure("s", None)).egress)
        self.assertEqual((await registry.ensure("s", None, egress="a")).egress, "a")
        self.assertEqual((await registry.ensure("s", None)).egress, "a")

    async def test_omitted_ttl_preserves_the_configured_ttl(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("s", 5)
        before = registry._entries["s"].expires_at
        await asyncio.sleep(0.01)
        info = await registry.ensure("s")
        self.assertEqual(info.ttl_minutes, 5)
        after = registry._entries["s"].expires_at
        assert after is not None
        assert before is not None
        self.assertGreater(after, before)

    async def test_explicit_none_ttl_clears_the_configured_ttl(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("s", 5)
        info = await registry.ensure("s", None)
        self.assertIsNone(info.ttl_minutes)
        self.assertIsNone(registry._entries["s"].expires_at)

    async def test_create_generates_an_id_and_list_stays_a_list_of_ids(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        self.assertEqual(await registry.create("a", None), "a")
        generated = await registry.create(None, None)
        self.assertIsInstance(generated, str)
        self.assertEqual(await registry.list_sessions(), sorted(["a", generated]))


class SessionLeaseMetadataTests(IsolatedAsyncioTestCase):
    """The lease yields the admitted generation's immutable metadata."""

    async def test_lease_yields_the_admitted_generations_metadata(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", 5, mode=ISOLATED_MODE, egress="a")
        async with registry.lease("s") as info:
            self.assertEqual(info, SessionInfo(id="s", mode=ISOLATED_MODE, egress="a", ttl_minutes=5))

    async def test_lease_of_none_yields_none(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        async with registry.lease(None) as info:
            self.assertIsNone(info)

    async def test_metadata_is_immutable(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        info = await registry.ensure("s", None)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            info.mode = ISOLATED_MODE  # type: ignore[misc]

    async def test_a_lease_during_destroy_fails(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await _wait_until(entered.is_set)
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(lambda: registry._entries["s"].destroying)
        with self.assertRaises(SessionNotFoundError):
            async with registry.lease("s"):
                pass
        release.set()
        await holder
        await destroy

    async def test_lease_of_an_unknown_session_fails(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        with self.assertRaises(SessionNotFoundError):
            async with registry.lease("missing"):
                pass


class SessionCleanupLifecycleTests(IsolatedAsyncioTestCase):
    """Cleanup runs once and fences the id until it finishes."""

    async def test_destroy_cleanup_fences_recreation_until_it_finishes(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None, mode=ISOLATED_MODE, egress="a")
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)

        self.assertFalse(destroy.done())
        self.assertEqual(await registry.list_sessions(), [])
        with self.assertRaises(SessionNotFoundError):
            async with registry.lease("s"):
                pass
        recreate = asyncio.create_task(registry.ensure("s", None))
        await asyncio.sleep(0.05)
        self.assertFalse(recreate.done())

        recorder.gate.set()
        await destroy
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0], SessionInfo(id="s", mode=ISOLATED_MODE, egress="a", ttl_minutes=None))
        info = await recreate
        self.assertEqual(info.mode, SHARED_MODE)
        self.assertIsNone(info.egress)

    async def test_duplicate_ensure_does_not_refresh_while_closing(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)
        pending = asyncio.create_task(registry.ensure("s", 30))
        await asyncio.sleep(0.05)
        self.assertFalse(pending.done())
        recorder.gate.set()
        await destroy
        info = await pending
        self.assertEqual(info.ttl_minutes, 30)

    async def test_cleanup_failure_retains_a_tombstone_and_raises(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        with self.assertRaises(_CleanupError):
            await registry.destroy("s")
        self.assertEqual(await registry.list_sessions(), [])
        with self.assertRaises(SessionError):
            await registry.ensure("s", None)
        with self.assertRaises(SessionError):
            async with registry.lease("s"):
                pass

    async def test_retry_cleanup_clears_a_failed_tombstone(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None, egress="a")
        with self.assertRaises(_CleanupError):
            await registry.destroy("s")
        await registry.retry_cleanup("s")
        self.assertEqual(len(recorder.calls), 2)
        info = await registry.ensure("s", None, egress="b")
        self.assertEqual(info.egress, "b")

    async def test_retry_cleanup_does_not_duplicate_an_active_cleanup(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)
        with self.assertRaises(SessionError):
            await registry.retry_cleanup("s")
        recorder.gate.set()
        await destroy
        self.assertEqual(len(recorder.calls), 1)

    async def test_cancelling_a_destroy_waiter_does_not_abandon_cleanup(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)
        destroy.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await destroy

        with self.assertRaises(SessionNotFoundError):
            async with registry.lease("s"):
                pass
        recorder.gate.set()
        await _wait_until(lambda: "s" not in registry._closing)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual((await registry.ensure("s", None)).id, "s")

    async def test_max_sessions_counts_a_closing_session(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=1, cleanup=recorder)
        await registry.ensure("a", None)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("a"))
        await _wait_until(recorder.started.is_set)
        with self.assertRaises(SessionLimitError):
            await registry.create("b", None)
        recorder.gate.set()
        await destroy
        self.assertEqual(await registry.create("b", None), "b")

    async def test_destroy_of_an_unknown_session_is_safe(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        with self.assertRaises(SessionNotFoundError):
            await registry.destroy("missing")


class SessionExpiryTests(IsolatedAsyncioTestCase):
    """Idle TTL expiry runs cleanup once and never touches active leases."""

    async def test_purge_expired_runs_cleanup_and_hides_the_id(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5, egress="a")
        registry._entries["s"].expires_at = time.monotonic() - 1
        self.assertEqual(await registry.purge_expired(), ["s"])
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.calls[0].egress, "a")
        self.assertEqual(await registry.list_sessions(), [])
        self.assertIsNone((await registry.ensure("s", None)).ttl_minutes)

    async def test_expiry_does_not_touch_an_active_lease(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await _wait_until(entered.is_set)
        registry._entries["s"].expires_at = time.monotonic() - 1
        self.assertEqual(await registry.purge_expired(), [])
        self.assertEqual(len(recorder.calls), 0)
        release.set()
        await holder
        # The completed lease re-armed the idle deadline, so a use counts as activity.
        deadline = registry._entries["s"].expires_at
        assert deadline is not None
        self.assertGreater(deadline, time.monotonic())
        registry._entries["s"].expires_at = time.monotonic() - 1
        self.assertEqual(await registry.purge_expired(), ["s"])
        self.assertEqual(len(recorder.calls), 1)

    async def test_concurrent_destroy_and_expiry_close_once(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await _wait_until(entered.is_set)
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(lambda: registry._entries["s"].destroying)
        registry._entries["s"].expires_at = time.monotonic() - 1
        await registry.purge_expired()
        release.set()
        await holder
        await destroy
        self.assertEqual(len(recorder.calls), 1)


class SessionCancellationTests(IsolatedAsyncioTestCase):
    """Cancelled leases release their reference at every stage."""

    async def test_cancelled_before_admission_leaves_no_reference(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        task = asyncio.create_task(_hold_lease(registry, "s", asyncio.Event(), asyncio.Event()))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(registry._entries["s"].active, 0)
        async with registry.lease("s") as info:
            assert info is not None
            self.assertEqual(info.id, "s")

    async def test_cancelled_after_admission_waits_for_the_per_session_lock(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        first = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await _wait_until(entered.is_set)

        queued = asyncio.create_task(_hold_lease(registry, "s", asyncio.Event(), asyncio.Event()))
        await _wait_until(lambda: registry._entries["s"].active == 2)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertEqual(registry._entries["s"].active, 1)

        destroy = asyncio.create_task(registry.destroy("s"))
        await asyncio.sleep(0.02)
        self.assertFalse(destroy.done())
        release.set()
        await first
        await destroy
        self.assertEqual(await registry.list_sessions(), [])

    async def test_cancelled_inside_a_lease_releases_the_reference(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)

        async def body() -> None:
            async with registry.lease("s"):
                await asyncio.sleep(10)

        task = asyncio.create_task(body())
        await _wait_until(lambda: registry._entries["s"].active == 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(registry._entries["s"].active, 0)


class SessionShutdownTests(IsolatedAsyncioTestCase):
    """Shutdown drains leases before it runs cleanup."""

    async def test_aclose_drains_an_active_lease_before_cleanup(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await _wait_until(entered.is_set)

        closing = asyncio.create_task(registry.aclose())
        await asyncio.sleep(0.05)
        self.assertEqual(len(recorder.calls), 0)
        release.set()
        await holder
        await asyncio.wait_for(closing, timeout=2.0)
        self.assertEqual(len(recorder.calls), 1)

    async def test_drain_waits_for_every_active_lease(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("a", None)
        await registry.ensure("b", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        first = asyncio.create_task(_hold_lease(registry, "a", entered, release))
        second = asyncio.create_task(_hold_lease(registry, "b", asyncio.Event(), release))
        await _wait_until(entered.is_set)

        drained = asyncio.create_task(registry.drain())
        await asyncio.sleep(0.02)
        self.assertFalse(drained.done())
        release.set()
        await asyncio.gather(first, second)
        await asyncio.wait_for(drained, timeout=2.0)


class SessionLeaseValidationTests(IsolatedAsyncioTestCase):
    """The lease validates and binds mode and egress against the generation it admits."""

    async def test_lease_rejects_a_mode_that_disagrees_with_the_generation(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("s", None)
        with self.assertRaises(SessionError):
            async with registry.lease("s", mode=ISOLATED_MODE):
                pass

    async def test_lease_binds_an_egress_once_and_rejects_a_change(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        async with registry.lease("s", egress="a") as info:
            assert info is not None
            self.assertEqual(info.egress, "a")
        async with registry.lease("s") as info:
            assert info is not None
            self.assertEqual(info.egress, "a")
        with self.assertRaises(SessionError):
            async with registry.lease("s", egress="b"):
                pass

    async def test_lease_with_none_binds_the_default_egress(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        async with registry.lease("s", egress=None) as info:
            assert info is not None
            self.assertEqual(info.egress, DEFAULT_EGRESS_NAME)
        self.assertEqual((await registry.ensure("s")).egress, DEFAULT_EGRESS_NAME)

    async def test_an_omitted_egress_leaves_an_unbound_session_unbound(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        async with registry.lease("s") as info:
            assert info is not None
            self.assertIsNone(info.egress)

    async def test_completing_a_lease_re_arms_the_idle_deadline(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", 5)
        async with registry.lease("s"):
            registry._entries["s"].expires_at = time.monotonic() - 1
        deadline = registry._entries["s"].expires_at
        assert deadline is not None
        self.assertGreater(deadline, time.monotonic())
        self.assertEqual(await registry.list_sessions(), ["s"])

    async def test_a_recreated_generation_rejects_the_previous_mode(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.create("s", None, mode=ISOLATED_MODE)
        await registry.ensure("s", mode=ISOLATED_MODE)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)
        recreate = asyncio.create_task(registry.ensure("s", mode=SHARED_MODE))
        await asyncio.sleep(0.05)
        self.assertFalse(recreate.done())
        recorder.gate.set()
        await destroy
        info = await recreate
        self.assertEqual(info.mode, SHARED_MODE)
        with self.assertRaises(SessionError):
            async with registry.lease("s", mode=ISOLATED_MODE):
                pass
        async with registry.lease("s", mode=SHARED_MODE) as leased:
            assert leased is not None
            self.assertEqual(leased.mode, SHARED_MODE)


class SessionCleanupVisibilityTests(IsolatedAsyncioTestCase):
    """A cleanup failure is visible to the sweeper, shutdown, and a repeated destroy."""

    async def test_a_failed_expiry_is_visible_to_purge_expired(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", 5)
        registry._entries["s"].expires_at = time.monotonic() - 1
        with self.assertRaises(_CleanupError):
            await registry.purge_expired()
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(await registry.list_sessions(), [])

    async def test_a_failed_shutdown_surfaces_after_every_cleanup_drains(self) -> None:
        recorder = _CleanupRecorder(failures=2)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("a", None)
        await registry.ensure("b", None)
        with self.assertRaises(_CleanupError):
            await registry.aclose()
        self.assertEqual(len(recorder.calls), 2)

    async def test_shutdown_refuses_new_work(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        closing = asyncio.create_task(registry.aclose())
        await _wait_until(recorder.started.is_set)
        with self.assertRaises(SessionError):
            await registry.ensure("new", None)
        with self.assertRaises(SessionError):
            await registry.create("other", None)
        with self.assertRaises(SessionError):
            async with registry.lease("s"):
                pass
        recorder.gate.set()
        await asyncio.wait_for(closing, timeout=2.0)

    async def test_a_repeated_destroy_retries_a_failed_cleanup(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        with self.assertRaises(_CleanupError):
            await registry.destroy("s")
        await registry.destroy("s")
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual(await registry.list_sessions(), [])

    async def test_a_cancelled_cleanup_leaves_a_tombstone_without_a_busy_loop(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.ensure("s", None)
        recorder.gate = asyncio.Event()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(recorder.started.is_set)
        close_task = registry._closing["s"].close_task
        assert close_task is not None
        close_task.cancel()
        with self.assertRaises(SessionError):
            await destroy
        # The id is a tombstone, not an already-set event a caller would spin on.
        with self.assertRaises(SessionError):
            await registry.ensure("s", None)
        with self.assertRaises(SessionError):
            async with registry.lease("s"):
                pass
        recorder.gate.set()
        await asyncio.wait_for(registry.destroy("s"), timeout=2.0)
        self.assertEqual(len(recorder.calls), 2)
        self.assertEqual((await registry.ensure("s", None)).id, "s")


class CleanupRetryLeaseSafetyTests(IsolatedAsyncioTestCase):
    async def test_cancelled_cleanup_retry_still_waits_for_active_leases(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.create("s", mode=ISOLATED_MODE)
        entered = asyncio.Event()
        release = asyncio.Event()
        holder = asyncio.create_task(_hold_lease(registry, "s", entered, release))
        await entered.wait()
        destroy = asyncio.create_task(registry.destroy("s"))
        await _wait_until(lambda: registry._entries["s"].close_task is not None)
        failed_cleanup = registry._entries["s"].close_task
        assert failed_cleanup is not None
        await asyncio.sleep(0)
        failed_cleanup.cancel()
        with self.assertRaises(SessionError):
            await destroy
        retry = asyncio.create_task(registry.destroy("s"))
        await _wait_until(lambda: registry._entries["s"].close_task is not failed_cleanup)
        try:
            await asyncio.sleep(0)
            self.assertEqual(recorder.calls, [])
            self.assertFalse(retry.done())
        finally:
            release.set()
            await holder
            await retry
        self.assertEqual(len(recorder.calls), 1)
