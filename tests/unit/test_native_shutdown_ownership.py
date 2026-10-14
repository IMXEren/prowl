"""Exercise native failure retention without leaving the patched browser facade."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext as PWContext
from playwright.async_api import Playwright
from pydoll.browser.chromium import Chrome

from prowl.browser.browser import Browser, BrowserShutdownState
from prowl.browser.config import BrowserConfig
from prowl.browser.driver import BrowserRuntimeState
from prowl.browser.exceptions import BrowserShutdownError, BrowserStartError
from prowl.browser.lifecycle import BrowserLifecycle
from prowl.browser.proxy.egress import EgressError, EgressPool

if TYPE_CHECKING:
    from collections.abc import Iterator


def _runtime() -> BrowserRuntimeState:
    runtime = BrowserRuntimeState(max_groups=1)
    context = MagicMock(spec=PWContext)
    context.pages = []
    context.is_closed.return_value = False
    context.close = AsyncMock()
    runtime.main_ctx = context
    runtime.main_ctx_owned = True
    runtime.shared_pd = MagicMock(spec=Chrome)
    runtime.shared_pd.close = AsyncMock()
    runtime.cdp_browser = MagicMock(spec=PWBrowser)
    runtime.cdp_browser.close = AsyncMock()
    runtime.cdp_playwright = MagicMock(spec=Playwright)
    runtime.cdp_playwright.stop = AsyncMock()
    runtime.contexts.bind_shared(context)
    return runtime


@contextmanager
def _facade(runtime: BrowserRuntimeState) -> Iterator[BrowserLifecycle]:
    lifecycle = BrowserLifecycle(runtime)
    with patch.object(Browser, "_runtime", runtime), patch.object(Browser, "_lifecycle", lifecycle):
        yield lifecycle


class NativeShutdownTests(IsolatedAsyncioTestCase):
    async def test_failure_retains_main_and_caches_same_error(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        close = AsyncMock(side_effect=RuntimeError("main close failed"))
        assert context is not None
        context.close = close
        with _facade(runtime) as lifecycle:
            with self.assertRaises(BrowserShutdownError) as first:
                await Browser.shutdown()
            with self.assertRaises(BrowserShutdownError) as second:
                await Browser.shutdown()
            self.assertIs(first.exception, second.exception)
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.FAILED)
            self.assertIs(runtime.main_ctx, context)
            self.assertTrue(runtime.main_ctx_owned)
            self.assertEqual(runtime.contexts.count, 1)
            close.assert_awaited_once()
            close.side_effect = None
            await Browser.retry_shutdown()

    async def test_explicit_retry_releases_retained_clients(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        assert context is not None
        close = AsyncMock(side_effect=[RuntimeError("main close failed"), None])
        context.close = close
        with _facade(runtime) as lifecycle:
            with self.assertRaises(BrowserShutdownError):
                await Browser.shutdown()
            await Browser.retry_shutdown()
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.SUCCEEDED)
            self.assertIsNone(runtime.main_ctx)
            self.assertFalse(runtime.main_ctx_owned)
            self.assertEqual(runtime.contexts.count, 0)
            self.assertEqual(close.await_count, 2)

    async def test_concurrent_retries_share_owned_task(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        assert context is not None
        gate, entered = asyncio.Event(), asyncio.Event()

        async def retry_close() -> None:
            entered.set()
            await gate.wait()

        with _facade(runtime) as lifecycle:
            context.close = AsyncMock(side_effect=RuntimeError("main close failed"))
            with self.assertRaises(BrowserShutdownError):
                await Browser.shutdown()
            close = AsyncMock(side_effect=retry_close)
            context.close = close
            owner = asyncio.create_task(Browser.retry_shutdown())
            await asyncio.wait_for(entered.wait(), timeout=1)
            waiter = asyncio.create_task(Browser.retry_shutdown())
            await asyncio.sleep(0)
            gate.set()
            await asyncio.wait_for(asyncio.gather(owner, waiter), timeout=1)
            close.assert_awaited_once()
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.SUCCEEDED)

    async def test_cancelled_retry_waiter_leaves_owned_cleanup(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        assert context is not None
        gate, entered = asyncio.Event(), asyncio.Event()

        async def retry_close() -> None:
            entered.set()
            await gate.wait()

        with _facade(runtime) as lifecycle:
            context.close = AsyncMock(side_effect=RuntimeError("main close failed"))
            with self.assertRaises(BrowserShutdownError):
                await Browser.shutdown()
            context.close = AsyncMock(side_effect=retry_close)
            waiter = asyncio.create_task(Browser.retry_shutdown())
            await asyncio.wait_for(entered.wait(), timeout=1)
            owned = lifecycle._shutdown_task
            self.assertIsNotNone(owned)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            assert owned is not None
            self.assertFalse(owned.cancelled())
            gate.set()
            await asyncio.wait_for(asyncio.shield(owned), timeout=1)
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.SUCCEEDED)

    async def test_partial_retirement_keeps_local_ownership(self) -> None:
        runtime = _runtime()
        browser = runtime.cdp_browser
        assert browser is not None
        close = AsyncMock(side_effect=[RuntimeError("CDP close failed"), None])
        browser.close = close
        lifecycle = BrowserLifecycle(runtime)
        with self.assertRaises(RuntimeError):
            await runtime.rollback_start()
        self.assertIsNone(runtime.main_ctx)
        self.assertTrue(runtime.main_ctx_owned)
        self.assertIs(runtime.cdp_browser, browser)
        self.assertEqual(runtime.contexts.count, 1)
        await lifecycle._retire_stale_generation()
        self.assertFalse(runtime.main_ctx_owned)
        self.assertEqual(runtime.contexts.count, 0)
        self.assertEqual(close.await_count, 2)
        self.assertEqual(lifecycle.browser_restart_total, 1)

    async def test_remote_partial_retirement_refuses_local_launch(self) -> None:
        runtime = BrowserRuntimeState(max_groups=1)
        browser = MagicMock(spec=PWBrowser)
        browser.close = AsyncMock()
        runtime.cdp_browser = browser
        lifecycle = BrowserLifecycle(runtime)
        with self.assertRaisesRegex(BrowserStartError, "Reconnect explicitly"):
            await lifecycle._retire_stale_generation()
        browser.close.assert_not_awaited()

    async def test_failed_owned_profile_is_not_archived_by_sync_fallback(self) -> None:
        runtime = _runtime()
        lifecycle = BrowserLifecycle(runtime)
        lifecycle._owns_local_profile = True
        with patch.object(BrowserLifecycle, "pack_profile") as pack:
            lifecycle.do_sync_chores_before_exit()
        pack.assert_not_called()
        self.assertTrue(lifecycle._owns_local_profile)
        await runtime.rollback_start()

    async def test_pool_retry_uses_real_native_cleanup(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        assert context is not None
        close = AsyncMock(side_effect=[RuntimeError("main close failed"), None])
        context.close = close
        pool = EgressPool(BrowserConfig(), {"a": "socks5://127.0.0.1:10001"})
        with (
            _facade(runtime) as lifecycle,
            patch("prowl.browser.proxy.egress.create_egress_browser", return_value=Browser),
        ):
            await pool.acquire("a")
            with self.assertRaises(EgressError):
                await pool.aclose()
            self.assertEqual(pool.live_names(), ("a",))
            self.assertTrue(runtime.main_ctx_owned)
            await pool.aclose()
            self.assertEqual(pool.live_names(), ())
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.SUCCEEDED)
            self.assertEqual(close.await_count, 2)

    async def test_completed_attempt_keeps_error_after_successful_retry(self) -> None:
        runtime = _runtime()
        context = runtime.main_ctx
        assert context is not None
        gate, entered = asyncio.Event(), asyncio.Event()

        async def fail_close() -> None:
            entered.set()
            await gate.wait()
            message = "first attempt failed"
            raise RuntimeError(message)

        with _facade(runtime) as lifecycle:
            context.close = AsyncMock(side_effect=fail_close)
            waiter = asyncio.create_task(Browser.shutdown())
            await asyncio.wait_for(entered.wait(), timeout=1)
            original = lifecycle._shutdown_task
            assert original is not None
            gate.set()
            with self.assertRaises(BrowserShutdownError) as failure:
                await asyncio.wait_for(waiter, timeout=1)
            context.close = AsyncMock()
            await Browser.retry_shutdown()
            self.assertIs(original.result(), failure.exception)
            self.assertIsNone(lifecycle._shutdown_error)
            self.assertIs(lifecycle.shutdown_state, BrowserShutdownState.SUCCEEDED)
