"""Ownership regressions for the shared persistent context and isolated session contexts.

These are mocked contract tests: they pin which context a group's pages are created in, that
an isolated context keeps its CDP browser context id on PyDoll tabs, that only owned isolated
contexts are ever closed, and that admission slots survive cancellation. Real-browser
fingerprint, storage and policy parity is a separate owner-run gate.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from playwright.async_api import Browser as PWBrowser

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.driver.contexts import BrowserContextManager
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.exceptions import BrowserContextError
from prowl.browser.lifecycle.startup import BrowserLifecycle

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext as PWBrowserCtx
    from playwright.async_api import Page as PWPage
    from pydoll.browser import Chrome

    from prowl.browser.driver.contexts import BrowserContextHandle

_MAIN_TARGET = "main-target"


class _FakeCdpSession:
    """CDP session that reports the target it is attached to."""

    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.detached = False

    async def send(self, command: str) -> dict[str, dict[str, str]]:
        if command != "Target.getTargetInfo":
            msg = f"unexpected command {command}"
            raise AssertionError(msg)
        info = {"targetId": self.page._target_id}
        if self.page._browser_context_id is not None:
            info["browserContextId"] = self.page._browser_context_id
        return {"targetInfo": info}

    async def detach(self) -> None:
        self.detached = True


class _FakePage:
    """Playwright page stand-in that knows its context and target ids."""

    def __init__(self, context: _FakeContext, target_id: str, browser_context_id: str | None) -> None:
        self.context = context
        self._target_id = target_id
        self._browser_context_id = browser_context_id
        self.opener_page: _FakePage | None = None
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def opener(self) -> _FakePage | None:
        return self.opener_page


class _FakeContext:
    """Playwright context stand-in with live targets and a close order log."""

    def __init__(  # noqa: PLR0913 - one test double with the knobs its cases vary
        self,
        label: str,
        events: list[str],
        *,
        browser: _FakeBrowser | None = None,
        browser_context_id: str | None = None,
        options: dict[str, Any] | None = None,
        target_prefix: str = "t",
    ) -> None:
        self.label = label
        self._events = events
        self.browser = browser
        self.options = options or {}
        self._browser_context_id = browser_context_id
        self._target_prefix = target_prefix
        self._next_target = 0
        self._closed = False
        self.popup_handlers: list[Any] = []
        self.close_count = 0

    @property
    def _impl_obj(self) -> SimpleNamespace:
        return SimpleNamespace(_options=self.options)

    def is_closed(self) -> bool:
        return self._closed

    def on(self, event: str, handler: Any) -> None:
        if event == "page":
            self.popup_handlers.append(handler)

    async def new_page(self) -> _FakePage:
        target_id = f"{self._target_prefix}{self._next_target}"
        self._next_target += 1
        return _FakePage(self, target_id, self._browser_context_id)

    async def new_cdp_session(self, page: _FakePage) -> _FakeCdpSession:
        return _FakeCdpSession(page)

    async def close(self) -> None:
        self.close_count += 1
        self._closed = True
        self._events.append(f"close:{self.label}")


class _FakeBrowser:
    """Playwright Browser stand-in that records the contexts created through it."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.new_context_kwargs: list[dict[str, Any]] = []
        self.created: list[_FakeContext] = []

    async def new_context(self, **kwargs: Any) -> _FakeContext:
        inspect.signature(PWBrowser.new_context).bind(self, **kwargs)
        self.new_context_kwargs.append(kwargs)
        index = len(self.created) + 1
        context = _FakeContext(
            f"isolated-{index}",
            self._events,
            browser_context_id=f"ctx-{index}",
            target_prefix=f"i{index}-",
        )
        self.created.append(context)
        return context


class _FakePdTab:
    """PyDoll tab stand-in that keeps the browser context id it was built with."""

    def __init__(self, _browser: Any, **kwargs: Any) -> None:
        self._target_id = kwargs["target_id"]
        self._browser_context_id = kwargs["browser_context_id"]


class _FakePydoll:
    """PyDoll client stand-in with the private tab cache the runtime writes to."""

    def __init__(self) -> None:
        self._tabs_opened: dict[str, Any] = {}
        self.resolved_tabs: list[Any] = []
        self.open_tabs_calls = 0
        self.closed = False
        self.tab_kwargs: list[dict[str, Any]] = []

    def _get_tab_kwargs(self, target_id: str, browser_context_id: str | None = None) -> dict[str, Any]:
        kwargs = {"target_id": target_id, "browser_context_id": browser_context_id}
        self.tab_kwargs.append(kwargs)
        return dict(kwargs)

    async def get_opened_tabs(self) -> list[Any]:
        self.open_tabs_calls += 1
        return list(self.resolved_tabs)

    async def close(self) -> None:
        self.closed = True


class _Fixture:
    """One running runtime with a shared persistent context and its driver clients."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.popups: list[_FakePage] = []
        self.last_page: _FakePage | None = None
        self.pd = _FakePydoll()
        self.browser = _FakeBrowser(self.events)
        self.main = _FakeContext(
            "main",
            self.events,
            browser=self.browser,
            options={
                "viewport": {"width": 1920, "height": 980},
                "colorScheme": "dark",
                "locale": "en-US",
                "userAgent": "persona-agent",
                "permissions": ["geolocation"],
                "args": ["--flag"],
                "env": [{"name": "X", "value": "1"}],
            },
            target_prefix="m",
        )
        self.state = BrowserRuntimeState(max_groups=3)
        self.state.main_ctx = cast("PWBrowserCtx", self.main)
        self.state.main_ctx_owned = True
        self.state.shared_pd = cast("Chrome", self.pd)
        self.state.popup_handler = self._record_popup
        self.state.contexts.bind_shared(cast("PWBrowserCtx", self.main))

    async def _record_popup(self, page: PWPage) -> None:
        self.popups.append(cast("_FakePage", page))

    async def isolated(self, session_id: str) -> BrowserContextHandle:
        return await self.state.isolated_context(session_id)


async def _noop_start() -> None:
    """Stand-in for the browser start coroutine that never starts anything."""


class BrowserContextRuntimeTests(IsolatedAsyncioTestCase):
    """Runtime ownership of contexts, groups, and CDP target identity."""

    def setUp(self) -> None:
        self.fixture = _Fixture()
        # Groups resolve their pages through the class that created them, so the fixture
        # runtime has to be the one the facade uses.
        self.original_runtime = Browser._runtime
        self.original_lifecycle = Browser._lifecycle
        Browser._runtime = self.fixture.state
        Browser._lifecycle = BrowserLifecycle(self.fixture.state)
        self._pdtab_patcher = patch("prowl.browser.driver.runtime.PDTab", _FakePdTab)
        self._pdtab_patcher.start()
        self.addCleanup(self._pdtab_patcher.stop)

    def tearDown(self) -> None:
        Browser._runtime = self.original_runtime
        Browser._lifecycle = self.original_lifecycle

    async def test_shared_context_wraps_the_persistent_context_and_is_stable(self) -> None:
        """Happy path: the shared handle is the persistent context and keeps its identity."""
        first = self.fixture.state.shared_context()

        self.assertTrue(first.is_shared)
        self.assertIs(first.context, self.fixture.main)
        self.assertIs(self.fixture.state.shared_context(), first)

    async def test_isolated_contexts_are_reused_per_session_and_distinct_between_sessions(self) -> None:
        """Input variation: one context per session id, created once and reused."""
        first = await self.fixture.isolated("session-a")
        again = await self.fixture.isolated("session-a")
        other = await self.fixture.isolated("session-b")

        self.assertIs(first, again)
        self.assertEqual(len(self.fixture.browser.created), 2)
        self.assertNotEqual(first.context, other.context)
        self.assertFalse(first.is_shared)
        self.assertEqual(first.session_id, "session-a")
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ("session-a", "session-b"))

    async def test_isolated_context_clones_only_persona_context_options(self) -> None:
        """Invariant: viewport and colour scheme are cloned, emulation and storage are not."""
        await self.fixture.isolated("session-a")

        self.assertEqual(
            self.fixture.browser.new_context_kwargs,
            [{"viewport": {"width": 1920, "height": 980}, "color_scheme": "dark"}],
        )

    async def test_isolated_context_clones_a_disabled_viewport(self) -> None:
        """Boundary: a persistent context with no default viewport passes that through."""
        self.fixture.main.options = {"noDefaultViewport": True, "colorScheme": "light"}

        await self.fixture.isolated("session-a")

        self.assertEqual(
            self.fixture.browser.new_context_kwargs,
            [{"no_viewport": True, "color_scheme": "light"}],
        )

    async def test_isolated_context_registers_popup_routing_and_humanize_patch(self) -> None:
        """Invariant: popup routing is registered per context and humanization is applied."""
        self.fixture.state.humanize_contexts = True
        with (
            patch("prowl.browser.driver.contexts.patch_context_async") as patch_context,
            patch("prowl.browser.driver.contexts.resolve_human_config", return_value="cfg") as resolve_config,
        ):
            handle = await self.fixture.isolated("session-a")

        self.assertEqual(cast("_FakeContext", handle.context).popup_handlers, [self.fixture.state.popup_handler])
        resolve_config.assert_called_once_with("default", None)
        patch_context.assert_called_once_with(handle.context, "cfg")

    async def test_isolated_context_skips_humanize_when_the_launch_did(self) -> None:
        """Boundary: humanization is only applied when the launch enabled it."""
        with patch("prowl.browser.driver.contexts.patch_context_async") as patch_context:
            await self.fixture.isolated("session-a")

        patch_context.assert_not_called()

    async def test_create_page_binds_the_group_to_the_chosen_context(self) -> None:
        """Happy path: the parent page and the group both belong to the selected context."""
        handle = await self.fixture.isolated("session-a")

        group = await self.fixture.state.create_group(TabGroup, handle)

        self.assertIs(group.context, handle)
        self.assertEqual(group.target_id, "i1-0")
        self.assertIs(self.fixture.state.target_to_page_map["i1-0"].context, handle.context)
        self.assertIs(self.fixture.state.page_to_group[self.fixture.state.target_to_page_map["i1-0"]], group)

    async def test_new_tab_of_a_group_stays_in_the_groups_context(self) -> None:
        """State transition: a child tab is created through the group's own context."""
        handle = await self.fixture.isolated("session-a")
        group = await self.fixture.state.create_group(TabGroup, handle)

        child = await group.new_tab()

        self.assertEqual(child._target_id, "i1-1")
        self.assertEqual(group.child_target_ids, ["i1-1"])
        self.assertIs(self.fixture.state.page_to_group[self.fixture.state.target_to_page_map["i1-1"]], group)

    async def test_pydoll_tabs_keep_the_cdp_browser_context_id(self) -> None:
        """Invariant: PyDoll tabs carry the target's browser context id, not None."""
        handle = await self.fixture.isolated("session-a")

        await self.fixture.state.create_group(TabGroup, handle)
        await self.fixture.state.create_group(TabGroup)

        self.assertEqual(
            [kwargs["browser_context_id"] for kwargs in self.fixture.pd.tab_kwargs],
            ["ctx-1", None],
        )

    async def test_popup_from_an_isolated_context_keeps_its_context_and_own_identity(self) -> None:
        """Invariant: a popup attaches through its own context and keeps its context id."""
        handle = await self.fixture.isolated("session-a")
        group = await self.fixture.state.create_group(TabGroup, handle)
        popup = cast("_FakePage", await handle.context.new_page())
        popup.opener_page = cast("_FakePage", self.fixture.state.target_to_page_map[group.target_id])

        await self.fixture.state.attach_popup_page(cast("PWPage", popup))

        self.assertEqual(group.child_target_ids, ["i1-1"])
        self.assertEqual(self.fixture.pd.tab_kwargs[-1], {"target_id": "i1-1", "browser_context_id": "ctx-1"})
        self.assertIs(self.fixture.state.page_to_group[cast("PWPage", popup)], group)

    async def test_get_pd_tab_prefers_the_registered_tab_over_target_resolution(self) -> None:
        """Invariant: the runtime resolves its own tab, which carries the context id."""
        await self.fixture.state.create_page(await self.fixture.isolated("session-a"))
        registered = self.fixture.pd._tabs_opened["i1-0"]
        self.fixture.pd.resolved_tabs = [object()]

        resolved = await self.fixture.state.get_pd_tab("i1-0")

        self.assertIs(resolved, registered)
        self.assertEqual(self.fixture.pd.open_tabs_calls, 0)

    async def test_foreign_handle_is_rejected(self) -> None:
        """Error path: another identity's context handle is never accepted."""
        foreign = BrowserContextManager().bind_shared(cast("PWBrowserCtx", _FakeContext("foreign", [])))

        with self.assertRaisesRegex(BrowserContextError, "different browser identity"):
            self.fixture.state.resolve_context(foreign)

    async def test_stale_handle_is_rejected_after_its_context_closed(self) -> None:
        """Error path: a handle whose context is gone is rejected instead of reused."""
        handle = await self.fixture.isolated("session-a")
        await self.fixture.state.close_isolated_context("session-a")

        with self.assertRaisesRegex(BrowserContextError, "closed"):
            self.fixture.state.resolve_context(handle)

    async def test_close_isolated_context_closes_only_that_context_and_is_idempotent(self) -> None:
        """State transition: one isolated context closes once; the shared one is untouched."""
        first = await self.fixture.isolated("session-a")
        second = await self.fixture.isolated("session-b")

        await self.fixture.state.close_isolated_context("session-a")
        await self.fixture.state.close_isolated_context("session-a")
        await self.fixture.state.close_isolated_context("unknown-session")

        self.assertEqual(cast("_FakeContext", first.context).close_count, 1)
        self.assertEqual(cast("_FakeContext", second.context).close_count, 0)
        self.assertEqual(self.fixture.main.close_count, 0)
        self.assertFalse(self.fixture.main.is_closed())
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ("session-b",))

    async def test_close_isolated_context_ends_the_groups_that_lived_in_it(self) -> None:
        """Invariant: closing a context closes its request groups and returns their slots."""
        handle = await self.fixture.isolated("session-a")
        groups = [
            await self.fixture.state.create_tab_group(
                TabGroup,
                _noop_start,
                lambda: True,
                handle,
            )
            for _ in range(2)
        ]
        parent_pages = [self.fixture.state.target_to_page_map[group.target_id] for group in groups]
        self.assertEqual(self.fixture.state.group_semaphore._value, 1)

        await self.fixture.state.close_isolated_context("session-a")

        self.assertEqual(self.fixture.state.active_groups, set())
        self.assertEqual(self.fixture.state.group_semaphore._value, 3)
        self.assertTrue(all(cast("_FakePage", page).closed for page in parent_pages))
        self.assertEqual(self.fixture.state.target_to_page_map, {})

    async def test_close_context_rejects_a_session_id_that_cannot_name_a_context(self) -> None:
        """Error path: an empty or non-string session id never reaches Playwright."""
        for invalid in [None, "", "   ", 7]:
            with self.assertRaisesRegex(BrowserContextError, "non-empty session id"):
                await self.fixture.state.close_isolated_context(invalid)

        self.assertEqual(self.fixture.browser.created, [])

    async def test_rollback_closes_isolated_contexts_before_the_persistent_one(self) -> None:
        """State transition: rollback tears isolated contexts down first, then the main one."""
        await self.fixture.isolated("session-a")

        await self.fixture.state.rollback_start()

        self.assertEqual(self.fixture.events, ["close:isolated-1", "close:main"])
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())
        self.assertIsNone(self.fixture.state.contexts.shared())

    async def test_rollback_leaves_a_caller_owned_remote_context_open(self) -> None:
        """Boundary: a remote attach rollback keeps the context it does not own."""
        self.fixture.state.main_ctx_owned = False
        await self.fixture.isolated("session-a")

        await self.fixture.state.rollback_start()

        self.assertEqual(self.fixture.events, ["close:isolated-1"])
        self.assertEqual(self.fixture.main.close_count, 0)
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())

    async def test_cancellation_during_page_creation_returns_the_admission_slot(self) -> None:
        """Error path: a cancelled admit leaks neither the slot nor a registered group."""
        with (
            patch.object(BrowserRuntimeState, "create_page", AsyncMock(side_effect=asyncio.CancelledError)),
            self.assertRaises(asyncio.CancelledError),
        ):
            await self.fixture.state.create_tab_group(
                TabGroup,
                _noop_start,
                lambda: True,
            )

        self.assertEqual(self.fixture.state.group_semaphore._value, 3)
        self.assertEqual(self.fixture.state.active_groups, set())

    async def test_cancellation_after_registration_returns_the_slot_exactly_once(self) -> None:
        """Boundary: a group registered before an interruption hands its slot back once."""
        group = TabGroup("parent", 1)
        page = _FakePage(self.fixture.main, "parent", None)
        self.fixture.state.target_to_page_map["parent"] = cast("PWPage", page)
        self.fixture.state.page_to_group[cast("PWPage", page)] = group

        async def _register_then_cancel(_factory: object, _context: object = None, registered: Any = None) -> None:
            self.fixture.state.active_groups.add(group)
            if registered is not None:
                registered.append(group)
            raise asyncio.CancelledError

        with (
            patch.object(BrowserRuntimeState, "create_group", AsyncMock(side_effect=_register_then_cancel)),
            self.assertRaises(asyncio.CancelledError),
        ):
            await self.fixture.state.create_tab_group(
                TabGroup,
                _noop_start,
                lambda: True,
            )

        self.assertEqual(self.fixture.state.group_semaphore._value, 3)
        self.assertEqual(self.fixture.state.active_groups, set())
        self.assertTrue(page.closed)
        self.assertEqual(self.fixture.state.target_to_page_map, {})
        self.assertEqual(self.fixture.state.page_to_group, {})

    async def test_failed_group_construction_closes_the_partial_page(self) -> None:
        """Error path: a page created for a group that cannot be built is closed and forgotten."""
        with (
            patch.object(BrowserRuntimeState, "create_page", AsyncMock(side_effect=self._page_then_fail)),
            self.assertRaisesRegex(RuntimeError, "factory failed"),
        ):
            await self.fixture.state.create_group(self._failing_factory)

        self.assertEqual(self.fixture.state.target_to_page_map, {})
        self.assertEqual(self.fixture.state.page_to_group, {})
        self.assertEqual(self.fixture.pd._tabs_opened, {})
        assert self.fixture.last_page is not None
        self.assertTrue(self.fixture.last_page.closed)

    async def test_failed_context_close_retains_ownership_and_fences_recreation(self) -> None:
        handle = await self.fixture.isolated("session-a")
        with (
            patch.object(_FakeContext, "close", AsyncMock(side_effect=RuntimeError("close failed"))),
            self.assertRaisesRegex(RuntimeError, "close failed"),
        ):
            await self.fixture.state.close_isolated_context("session-a")
        self.assertIs(self.fixture.state.contexts.isolated("session-a"), handle)
        with self.assertRaisesRegex(BrowserContextError, "still closing"):
            await self.fixture.isolated("session-a")
        await self.fixture.state.close_isolated_context("session-a")
        self.assertIsNone(self.fixture.state.contexts.isolated("session-a"))

    async def test_reset_rejects_previous_generation_even_if_remote_context_stays_open(self) -> None:
        handle = self.fixture.state.shared_context()
        self.fixture.state.contexts.reset()
        self.fixture.state.shared_context()
        with self.assertRaisesRegex(BrowserContextError, "stale"):
            self.fixture.state.resolve_context(handle)
        self.assertFalse(self.fixture.main.is_closed())

    async def test_human_patch_failure_closes_unregistered_context(self) -> None:
        self.fixture.state.humanize_contexts = True
        with (
            patch("prowl.browser.driver.runtime.apply_human_patch", side_effect=RuntimeError("patch failed")),
            self.assertRaisesRegex(RuntimeError, "patch failed"),
        ):
            await self.fixture.isolated("session-a")
        self.assertTrue(self.fixture.browser.created[0].is_closed())
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())

    async def test_real_human_config_resolver_returns_a_configuration(self) -> None:
        self.fixture.state.humanize_contexts = True
        with patch("prowl.browser.driver.contexts.patch_context_async") as patch_context:
            await self.fixture.isolated("session-a")
        self.assertIsInstance(patch_context.call_args.args[1].typing_delay, (int, float))

    async def test_target_discovery_failure_closes_page_and_detaches_cdp(self) -> None:
        page = await self.fixture.main.new_page()
        cdp = _FakeCdpSession(page)
        with (
            patch.object(_FakeContext, "new_page", AsyncMock(return_value=page)),
            patch.object(_FakeContext, "new_cdp_session", AsyncMock(return_value=cdp)),
            patch.object(_FakeCdpSession, "send", AsyncMock(side_effect=RuntimeError("cdp failed"))),
            self.assertRaisesRegex(RuntimeError, "cdp failed"),
        ):
            await self.fixture.state.create_tab_group(TabGroup, _noop_start, lambda: True)
        self.assertTrue(page.closed)
        self.assertTrue(cdp.detached)
        self.assertEqual(self.fixture.state.group_semaphore._value, 3)

    async def test_cancelled_context_creation_closes_context_after_creation_finishes(self) -> None:
        started, finish = asyncio.Event(), asyncio.Event()
        context = _FakeContext("pending", self.fixture.events)

        async def create() -> Any:
            started.set()
            await finish.wait()
            return context

        task = asyncio.create_task(self.fixture.state.contexts.get_or_create("pending", create))
        await started.wait()
        task.cancel()
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(context.is_closed())
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())

    async def test_child_close_failure_still_closes_parent_and_returns_slot(self) -> None:
        group = await self.fixture.state.create_tab_group(TabGroup, _noop_start, lambda: True)
        parent = group.ppage
        child = await self.fixture.main.new_page()
        self.fixture.state.target_to_page_map[child._target_id] = cast("PWPage", child)
        group.child_target_ids.append(child._target_id)
        close = _FakePage.close

        async def fail_child(page: _FakePage) -> None:
            if page is child:
                msg = "child close failed"
                raise RuntimeError(msg)
            await close(page)

        with patch.object(_FakePage, "close", fail_child), self.assertRaisesRegex(RuntimeError, "child close failed"):
            await group.quit()
        self.assertTrue(cast("_FakePage", parent).closed)
        self.assertEqual(self.fixture.state.group_semaphore._value, 3)
        self.assertIs(self.fixture.state.target_to_page_map[child._target_id], child)

    async def _page_then_fail(self, context: BrowserContextHandle | None = None) -> tuple[str, Any]:
        page = await self.fixture.main.new_page()
        self.fixture.last_page = page
        return page._target_id, page

    def _failing_factory(self, target_id: str, gid: int) -> TabGroup:
        msg = "factory failed"
        raise RuntimeError(msg)


class BrowserContextFacadeTests(IsolatedAsyncioTestCase):
    """The Browser facade API Stage 2 uses: get, create in, and close contexts."""

    def setUp(self) -> None:
        self.fixture = _Fixture()
        self.original_runtime = Browser._runtime
        self.original_lifecycle = Browser._lifecycle
        Browser._runtime = self.fixture.state
        Browser._lifecycle = BrowserLifecycle(self.fixture.state)
        self._pdtab_patcher = patch("prowl.browser.driver.runtime.PDTab", _FakePdTab)
        self._pdtab_patcher.start()
        self.addCleanup(self._pdtab_patcher.stop)

    def tearDown(self) -> None:
        Browser._runtime = self.original_runtime
        Browser._lifecycle = self.original_lifecycle

    async def test_default_none_session_returns_the_shared_persistent_context(self) -> None:
        """Happy path: no session id means the shared persistent context."""
        handle = await Browser.get_context()

        self.assertTrue(handle.is_shared)
        self.assertIs(handle.context, self.fixture.main)

    async def test_session_id_returns_a_reused_isolated_context(self) -> None:
        """Happy path: a session id maps to one long-lived isolated context."""
        first = await Browser.get_context("session-a")
        second = await Browser.get_context("session-a")

        self.assertIs(first, second)
        self.assertEqual(len(self.fixture.browser.created), 1)

    async def test_create_with_a_context_builds_the_group_there_and_closes_cleanly(self) -> None:
        """State transition: a request group is created in the session context only."""
        handle = await Browser.get_context("session-a")

        group = await Browser.create(handle)
        await group.quit()

        self.assertIs(group.context, handle)
        self.assertEqual(group.target_id, "i1-0")
        self.assertEqual(self.fixture.state.group_semaphore._value, 3)
        self.assertEqual(self.fixture.main._next_target, 0)

    async def test_create_without_a_context_keeps_using_the_shared_context(self) -> None:
        """Invariant: the default create path is unchanged for existing callers."""
        group = await Browser.create()

        self.assertTrue(group.context.is_shared)
        self.assertEqual(group.target_id, "m0")

    async def test_create_with_a_foreign_handle_surfaces_a_context_error(self) -> None:
        """Error path: a foreign handle is reported as a context error, not a start error."""
        foreign = BrowserContextManager().bind_shared(cast("PWBrowserCtx", _FakeContext("foreign", [])))

        with self.assertRaisesRegex(BrowserContextError, "different browser identity"):
            await Browser.create(foreign)

        self.assertEqual(self.fixture.state.active_groups, set())
        self.assertEqual(self.fixture.state.group_semaphore._value, 3)

    async def test_close_context_closes_one_session_and_leaves_the_rest_running(self) -> None:
        """State transition: closing one session context leaves the shared and other sessions."""
        first = await Browser.get_context("session-a")
        second = await Browser.get_context("session-b")

        await Browser.close_context("session-a")
        await Browser.close_context("session-a")

        self.assertEqual(cast("_FakeContext", first.context).close_count, 1)
        self.assertEqual(cast("_FakeContext", second.context).close_count, 0)
        self.assertFalse(self.fixture.main.is_closed())
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ("session-b",))

    async def test_lifecycle_cleanup_closes_isolated_contexts_before_the_persistent_one(self) -> None:
        """State transition: browser shutdown closes owned isolated contexts first."""
        await Browser.get_context("session-a")

        await Browser._lifecycle._cleanup_resources()

        self.assertEqual(self.fixture.events, ["close:isolated-1", "close:main"])
        self.assertEqual(self.fixture.pd.closed, True)
        self.assertIsNone(self.fixture.state.main_ctx)
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())
