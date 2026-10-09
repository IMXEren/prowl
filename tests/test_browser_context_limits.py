"""Hard-cap regressions for the Prowl-managed browser contexts of one identity.

These are mocked contract tests for the per-identity context cap: the shared persistent
context counts against it, a refused or failed creation consumes no capacity, reuse at the
cap works, and a failed close keeps its slot until a retry succeeds. Real-browser context
growth is verified separately; this file pins the accounting.
"""

from __future__ import annotations

import asyncio
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import BrowserContext as PWBrowserCtx
from pydoll.browser.chromium import Chrome

from prowl.browser.browser import Browser
from prowl.browser.config import DEFAULT_MAX_CONTEXTS, BrowserConfig, default_max_contexts
from prowl.browser.driver.contexts import BrowserContextManager
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.exceptions import BrowserContextError, BrowserStartError
from prowl.browser.lifecycle.startup import BrowserLifecycle

_WAIT_SECONDS = 5


class _FakeContext:
    """Playwright context stand-in that only reports whether it is closed."""

    def __init__(self) -> None:
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True


def _native_context() -> PWBrowserCtx:
    state = _FakeContext()
    native = Mock(spec=PWBrowserCtx)
    native.is_closed.side_effect = state.is_closed

    async def close() -> None:
        await state.close()

    native.close = AsyncMock(side_effect=close)
    return native


class _ContextFactory:
    """Creation double that records the contexts it actually created."""

    def __init__(self) -> None:
        self.created: list[PWBrowserCtx] = []

    async def __call__(self) -> PWBrowserCtx:
        context = _native_context()
        self.created.append(context)
        return context


def _bind_shared(manager: BrowserContextManager) -> PWBrowserCtx:
    """Bind a fresh fake persistent context to *manager* and return it."""
    return manager.bind_shared(_native_context()).context


class ContextCapManagerTests(IsolatedAsyncioTestCase):
    """The manager's own cap: the shared slot, refusal, and capacity accounting."""

    async def test_shared_context_is_the_only_context_when_the_cap_is_one(self) -> None:
        """Boundary: a cap of one admits the shared context and no isolated session."""
        manager = BrowserContextManager(max_contexts=1)
        shared = manager.bind_shared(_native_context())
        factory = _ContextFactory()

        with self.assertRaisesRegex(BrowserContextError, "the limit"):
            await manager.get_or_create("session-a", factory)

        self.assertIs(manager.shared(), shared)
        self.assertEqual(factory.created, [])
        self.assertEqual(manager.isolated_ids(), ())

    async def test_reuse_at_the_cap_returns_the_context_without_calling_the_factory(self) -> None:
        """Happy path: a live session is reused at the cap, and a new one is refused first."""
        manager = BrowserContextManager(max_contexts=2)
        _bind_shared(manager)
        factory = _ContextFactory()
        handle = await manager.get_or_create("session-a", factory)

        again = await manager.get_or_create("session-a", factory)

        self.assertIs(again, handle)
        self.assertEqual(len(factory.created), 1)
        with self.assertRaisesRegex(BrowserContextError, "the limit"):
            await manager.get_or_create("session-b", factory)
        self.assertEqual(len(factory.created), 1)

    async def test_concurrent_first_use_admits_only_up_to_the_cap(self) -> None:
        """Concurrency: distinct sessions racing on first use never exceed the cap."""
        manager = BrowserContextManager(max_contexts=3)
        _bind_shared(manager)
        factory = _ContextFactory()

        results = await asyncio.gather(
            *(manager.get_or_create(f"session-{index}", factory) for index in range(6)),
            return_exceptions=True,
        )

        admitted = [result for result in results if not isinstance(result, BaseException)]
        refused = [result for result in results if isinstance(result, BrowserContextError)]
        self.assertEqual(len(admitted), 2)
        self.assertEqual(len(refused), 4)
        self.assertEqual(len(factory.created), 2)
        self.assertEqual(len(manager.isolated_ids()), 2)

    async def test_failed_creation_consumes_no_capacity_and_can_retry(self) -> None:
        """Error path: a factory failure registers nothing and leaves the slot open."""
        manager = BrowserContextManager(max_contexts=2)
        _bind_shared(manager)
        attempts = 0

        async def create() -> PWBrowserCtx:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                msg = "creation failed"
                raise RuntimeError(msg)
            return _native_context()

        with self.assertRaisesRegex(RuntimeError, "creation failed"):
            await manager.get_or_create("session-a", create)

        self.assertEqual(manager.isolated_ids(), ())
        handle = await manager.get_or_create("session-a", create)

        self.assertEqual(attempts, 2)
        self.assertIs(manager.isolated("session-a"), handle)
        self.assertEqual(manager.isolated_ids(), ("session-a",))

    async def test_cancelled_creation_consumes_no_capacity(self) -> None:
        """Error path: a cancelled creation closes its context and holds no slot."""
        manager = BrowserContextManager(max_contexts=2)
        _bind_shared(manager)
        started, finish = asyncio.Event(), asyncio.Event()
        context = _native_context()

        async def create() -> PWBrowserCtx:
            started.set()
            await finish.wait()
            return context

        task = asyncio.create_task(manager.get_or_create("session-a", create))
        await asyncio.wait_for(started.wait(), timeout=_WAIT_SECONDS)
        task.cancel()
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=_WAIT_SECONDS)

        self.assertTrue(context.is_closed())
        self.assertEqual(manager.isolated_ids(), ())
        factory = _ContextFactory()
        self.assertIsNotNone(await manager.get_or_create("session-b", factory))

    async def test_failed_close_keeps_its_slot_until_a_retry_succeeds(self) -> None:
        """State transition: a failed close retains capacity; a successful retry releases it."""
        manager = BrowserContextManager(max_contexts=2)
        _bind_shared(manager)
        factory = _ContextFactory()
        handle = await manager.get_or_create("session-a", factory)

        with (
            patch.object(_FakeContext, "close", AsyncMock(side_effect=RuntimeError("close failed"))),
            self.assertRaisesRegex(RuntimeError, "close failed"),
        ):
            await manager.close_isolated("session-a")

        self.assertIs(manager.isolated("session-a"), handle)
        with self.assertRaisesRegex(BrowserContextError, "the limit"):
            await manager.get_or_create("session-b", factory)

        self.assertTrue(await manager.close_isolated("session-a"))
        self.assertIsNone(manager.isolated("session-a"))
        self.assertIsNotNone(await manager.get_or_create("session-b", factory))

    async def test_binding_the_shared_context_cannot_exceed_the_cap(self) -> None:
        """Error path: the shared context is refused when isolated contexts fill the cap."""
        manager = BrowserContextManager(max_contexts=1)
        factory = _ContextFactory()
        await manager.get_or_create("session-a", factory)

        with self.assertRaisesRegex(BrowserContextError, "the limit"):
            manager.bind_shared(_native_context())

        self.assertIsNone(manager.shared())


class ContextCapConfigTests(IsolatedAsyncioTestCase):
    """Configuration: the canonical default, env parsing, and the configure wiring."""

    def setUp(self) -> None:
        self.original_runtime = Browser._runtime
        self.original_lifecycle = Browser._lifecycle
        Browser._runtime = BrowserRuntimeState(max_groups=Browser._MAX_GROUPS)
        Browser._lifecycle = BrowserLifecycle(Browser._runtime)

    def tearDown(self) -> None:
        Browser._runtime = self.original_runtime
        Browser._lifecycle = self.original_lifecycle

    def test_default_cap_is_the_canonical_value(self) -> None:
        """Happy path: the dataclass default is the canonical cap."""
        self.assertEqual(DEFAULT_MAX_CONTEXTS, 8)
        self.assertEqual(BrowserConfig().max_contexts, DEFAULT_MAX_CONTEXTS)

    def test_env_sets_the_cap_and_rejects_values_below_one(self) -> None:
        """Input variation: the env value is honoured, unset is the default, bad is refused."""
        with patch.dict(os.environ, {"PROWL_MAX_CONTEXTS": "3"}):
            self.assertEqual(default_max_contexts(), 3)
            self.assertEqual(BrowserConfig.from_env().max_contexts, 3)
        with patch.dict(os.environ, {"PROWL_MAX_CONTEXTS": "   "}):
            self.assertEqual(default_max_contexts(), DEFAULT_MAX_CONTEXTS)
        for invalid in ["0", "-2", "many"]:
            with (
                patch.dict(os.environ, {"PROWL_MAX_CONTEXTS": invalid}),
                self.assertRaisesRegex(ValueError, "PROWL_MAX_CONTEXTS"),
            ):
                default_max_contexts()

    def test_configure_wires_the_cap_to_the_context_manager(self) -> None:
        """Happy path: configure adopts the launch config and the cap together."""
        config = BrowserConfig(
            proxy_url="socks5://127.0.0.1:1080",
            profile_dir="/opt/prowl/profile",
            max_contexts=5,
        )

        Browser.configure(config)

        self.assertEqual(Browser._lifecycle.proxy_url, "socks5://127.0.0.1:1080")
        self.assertEqual(Browser._lifecycle.profile_dir, "/opt/prowl/profile")
        self.assertEqual(Browser._runtime.contexts.max_contexts, 5)

    def test_invalid_cap_is_rejected_without_partial_mutation(self) -> None:
        """Error path: a cap below one mutates neither the lifecycle nor the manager."""
        config = BrowserConfig(
            proxy_url="socks5://127.0.0.1:1080",
            profile_dir="/opt/prowl/profile",
            max_contexts=0,
        )

        with self.assertRaisesRegex(BrowserStartError, "max_contexts"):
            Browser.configure(config)

        self.assertIsNone(Browser._lifecycle.proxy_url)
        self.assertNotEqual(Browser._lifecycle.profile_dir, "/opt/prowl/profile")
        self.assertEqual(Browser._runtime.contexts.max_contexts, DEFAULT_MAX_CONTEXTS)

    def test_configure_still_refuses_while_running_without_mutating(self) -> None:
        """Error path: a running browser is refused and the cap is left untouched."""
        state = Browser._runtime
        state.main_ctx = _native_context()
        state.shared_pd = Mock(spec=Chrome)

        with self.assertRaisesRegex(BrowserStartError, "running"):
            Browser.configure(BrowserConfig(profile_dir="/opt/prowl/profile", max_contexts=5))

        self.assertEqual(state.contexts.max_contexts, DEFAULT_MAX_CONTEXTS)

    def test_manager_rejects_a_cap_below_one(self) -> None:
        """Error path: the narrow configure-limit method refuses a value below one."""
        manager = BrowserContextManager()

        with self.assertRaisesRegex(BrowserContextError, "at least 1"):
            manager.configure_limit(0)

        self.assertEqual(manager.max_contexts, DEFAULT_MAX_CONTEXTS)


class ClosedContextCapacityTests(IsolatedAsyncioTestCase):
    async def test_closed_session_can_be_recreated_at_capacity(self) -> None:
        manager = BrowserContextManager(max_contexts=2)
        _bind_shared(manager)
        factory = _ContextFactory()
        first = await manager.get_or_create("session", factory)
        await first.context.close()
        second = await manager.get_or_create("session", factory)
        self.assertIsNot(first, second)
        self.assertEqual(len(factory.created), 2)
        await manager.close_all_isolated()

    async def test_invalid_constructor_cap_is_rejected(self) -> None:
        with self.assertRaises(BrowserContextError):
            BrowserContextManager(max_contexts=0)
