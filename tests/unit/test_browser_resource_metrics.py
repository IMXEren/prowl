"""Native resource counters for one browser identity's contexts, restarts and groups.

These are native-spec contract tests: they pin what each counter means at the seam that
registers, closes or evicts a resource, that only recovery after a successful rollback counts
as a restart, and that the facade snapshot is a fixed integer mapping. Aggregating these
values into the service metrics registry, and supplying ``evicted=True`` from automatic
cleanup, are separate slices.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, Mock

from playwright.async_api import BrowserContext as PWBrowserCtx

from prowl.browser.browser import Browser
from prowl.browser.config import BrowserConfig
from prowl.browser.driver.contexts import BrowserContextManager
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.exceptions import BrowserContextError, BrowserStartError
from prowl.browser.lifecycle.startup import BrowserLifecycle

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _FakeContext:
    """Playwright context double whose native close can be made to fail."""

    def __init__(self, *, close_failures: int = 0) -> None:
        self._closed = False
        self._remaining_failures = close_failures
        self.close_calls = 0

    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self.close_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            msg = "context close failed"
            raise RuntimeError(msg)
        self._closed = True


def _context(close_failures: int = 0) -> PWBrowserCtx:
    return Mock(spec=PWBrowserCtx, wraps=_FakeContext(close_failures=close_failures))


def _creator(context: PWBrowserCtx) -> Callable[[], Awaitable[PWBrowserCtx]]:
    """Return a factory coroutine that yields *context*."""

    async def create() -> PWBrowserCtx:
        return context

    return create


class ContextCounterTests(IsolatedAsyncioTestCase):
    """Registration, reuse and eviction decide the manager's cumulative counters."""

    async def test_shared_registration_counts_once_and_a_rebind_of_the_same_context_does_not(self) -> None:
        """A registered shared context counts once; rebinding the same native context is free."""
        manager = BrowserContextManager()
        first, second = _context(), _context()

        manager.bind_shared(first)
        manager.bind_shared(first)
        self.assertEqual(manager.context_created_total, 1)
        self.assertEqual(manager.count, 1)

        manager.bind_shared(second)

        self.assertEqual(manager.context_created_total, 2)
        self.assertEqual(manager.count, 1)

    async def test_isolated_creation_counts_once_while_reuse_does_not(self) -> None:
        """Each isolated session registration counts once; reuse returns the registered handle."""
        manager = BrowserContextManager()

        first = await manager.get_or_create("session-a", _creator(_context()))
        again = await manager.get_or_create("session-a", _creator(_context()))
        await manager.get_or_create("session-b", _creator(_context()))

        self.assertIs(first, again)
        self.assertEqual(manager.context_created_total, 2)
        self.assertEqual(manager.count, 2)

    async def test_factory_failure_cap_refusal_and_cancellation_never_count(self) -> None:
        """Only a successful new registration counts: no failure, refusal or cancellation does."""
        manager = BrowserContextManager()
        failing = BrowserContextManager()
        capped = BrowserContextManager(1)

        async def fail() -> PWBrowserCtx:
            msg = "factory failed"
            raise RuntimeError(msg)

        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            await failing.get_or_create("session-a", fail)

        await capped.get_or_create("session-a", _creator(_context()))
        with self.assertRaisesRegex(BrowserContextError, "the limit"):
            await capped.get_or_create("session-b", _creator(_context()))

        started, finish = asyncio.Event(), asyncio.Event()

        async def slow() -> PWBrowserCtx:
            started.set()
            await finish.wait()
            return _context()

        task = asyncio.create_task(manager.get_or_create("session-c", slow))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(failing.context_created_total, 0)
        self.assertEqual(capped.context_created_total, 1)
        self.assertEqual(manager.context_created_total, 0)
        self.assertEqual(manager.count, 0)

    async def test_automatic_eviction_counts_once_after_a_failed_close_is_retried(self) -> None:
        """A failed close keeps the handle and counts nothing; the retry evicts exactly once."""
        manager = BrowserContextManager()
        await manager.get_or_create("session-a", _creator(_context(close_failures=1)))

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            await manager.close_isolated("session-a", evicted=True)

        self.assertEqual(manager.context_evicted_total, 0)
        self.assertIsNotNone(manager.isolated("session-a"))

        self.assertTrue(await manager.close_isolated("session-a", evicted=True))

        self.assertEqual(manager.context_evicted_total, 1)
        self.assertFalse(await manager.close_isolated("session-a", evicted=True))
        self.assertEqual(manager.context_evicted_total, 1)

    async def test_explicit_destroy_and_unknown_close_are_not_evictions(self) -> None:
        """An explicit close, a bulk destroy and an unknown id never count as evictions."""
        manager = BrowserContextManager()
        await manager.get_or_create("session-a", _creator(_context()))
        await manager.get_or_create("session-b", _creator(_context()))

        await manager.close_isolated("session-a")
        await manager.close_isolated("unknown-session", evicted=True)
        await manager.close_all_isolated()

        self.assertEqual(manager.context_evicted_total, 0)
        self.assertEqual(manager.count, 0)
        self.assertEqual(manager.context_created_total, 2)

    async def test_reset_preserves_cumulative_counters_and_clears_the_gauge(self) -> None:
        """A generation reset forgets handles and keeps the identity's cumulative counters."""
        manager = BrowserContextManager()
        manager.bind_shared(_context())
        await manager.get_or_create("session-a", _creator(_context()))
        await manager.close_isolated("session-a", evicted=True)

        manager.reset()

        self.assertEqual(manager.count, 0)
        self.assertEqual(manager.context_created_total, 2)
        self.assertEqual(manager.context_evicted_total, 1)


class BrowserResourceMetricsFacadeTests(IsolatedAsyncioTestCase):
    """The facade snapshot contract and the ``evicted`` flag it passes to the runtime."""

    def setUp(self) -> None:
        self.runtime = BrowserRuntimeState(max_groups=3)
        self.original_runtime = Browser._runtime
        self.original_lifecycle = Browser._lifecycle
        Browser._runtime = self.runtime
        Browser._lifecycle = BrowserLifecycle(self.runtime)

    def tearDown(self) -> None:
        Browser._runtime = self.original_runtime
        Browser._lifecycle = self.original_lifecycle

    async def test_snapshot_has_fixed_integer_keys_and_facade_close_counts_only_evictions(self) -> None:
        """The snapshot is a fixed integer mapping, and only the flagged facade close evicts."""
        self.runtime.contexts.bind_shared(_context())
        await self.runtime.contexts.get_or_create("session-a", _creator(_context()))
        self.runtime.active_groups.add(MagicMock())

        metrics = Browser.resource_metrics()

        self.assertEqual(
            metrics,
            {
                "context_count": 2,
                "context_created_total": 2,
                "context_evicted_total": 0,
                "browser_restart_total": 0,
                "tabgroups_active": 1,
            },
        )
        self.assertTrue(all(isinstance(value, int) for value in metrics.values()))

        await Browser.close_context("session-a")
        await self.runtime.contexts.get_or_create("session-b", _creator(_context()))
        await Browser.close_context("session-b", evicted=True)
        self.runtime.active_groups.clear()

        closed = Browser.resource_metrics()

        self.assertEqual(closed["context_count"], 1)
        self.assertEqual(closed["context_created_total"], 3)
        self.assertEqual(closed["context_evicted_total"], 1)
        self.assertEqual(closed["tabgroups_active"], 0)


class _RetireDriver:
    """Driver double for the recovery seam the lifecycle retires through."""

    def __init__(self, *, attached: bool = False, dead: bool = True, rollback_fails: bool = False) -> None:
        self.main_ctx: PWBrowserCtx | None = _context() if dead else None
        self.main_ctx_owned = not attached
        self.rollback_fails = rollback_fails
        self.rollback_calls = 0

    async def rollback_start(self) -> None:
        """Retire the dead generation, or fail the retirement when configured."""
        self.rollback_calls += 1
        if self.rollback_fails:
            msg = "rollback failed"
            raise RuntimeError(msg)
        self.main_ctx = None


class BrowserRestartCounterTests(IsolatedAsyncioTestCase):
    """Recovery of a dead owned generation is the only restart the lifecycle counts."""

    def _lifecycle(self, driver: _RetireDriver) -> BrowserLifecycle:
        native = Mock(spec=BrowserRuntimeState)
        native.main_ctx = driver.main_ctx
        native.main_ctx_owned = driver.main_ctx_owned
        native.shared_pd = None
        native.cdp_browser = None
        native.cdp_playwright = None

        async def rollback() -> None:
            await driver.rollback_start()
            native.main_ctx = driver.main_ctx

        native.rollback_start = AsyncMock(side_effect=rollback)
        return BrowserLifecycle(native)

    async def test_owned_recovery_counts_once_and_other_outcomes_do_not(self) -> None:
        """A successful owned rollback counts one restart; a clean start, a rejected remote
        reconnect and a failed retirement count nothing, and the counter survives config."""
        recovered = _RetireDriver()
        lifecycle = self._lifecycle(recovered)

        await lifecycle._retire_stale_generation()

        self.assertEqual(lifecycle.browser_restart_total, 1)
        self.assertEqual(recovered.rollback_calls, 1)

        clean = _RetireDriver(dead=False)
        clean_lifecycle = self._lifecycle(clean)
        await clean_lifecycle._retire_stale_generation()
        self.assertEqual(clean.rollback_calls, 0)
        self.assertEqual(clean_lifecycle.browser_restart_total, 0)

        attached = _RetireDriver(attached=True)
        attached_lifecycle = self._lifecycle(attached)
        with self.assertRaisesRegex(BrowserStartError, "Reconnect explicitly"):
            await attached_lifecycle._retire_stale_generation()
        self.assertEqual(attached.rollback_calls, 0)
        self.assertEqual(attached_lifecycle.browser_restart_total, 0)

        failing = _RetireDriver(rollback_fails=True)
        failing_lifecycle = self._lifecycle(failing)
        with self.assertRaisesRegex(RuntimeError, "rollback failed"):
            await failing_lifecycle._retire_stale_generation()
        self.assertEqual(failing.rollback_calls, 1)
        self.assertEqual(failing_lifecycle.browser_restart_total, 0)

        lifecycle.apply_config(BrowserConfig(), is_running=lambda: False)
        self.assertEqual(lifecycle.browser_restart_total, 1)
