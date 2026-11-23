"""Mocked native restoration contracts; real storage restoration is verified separately."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext as PWBrowserCtx
from pydoll.browser import Chrome

from prowl.browser.browser import Browser
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.exceptions import BrowserContextError
from prowl.browser.lifecycle.startup import BrowserLifecycle

if TYPE_CHECKING:
    from playwright.async_api import Page, StorageState

#: Wire-format launch options a persistent context exposes through ``_impl_obj._options``.
_MAIN_OPTIONS: dict[str, object] = {
    "viewport": {"width": 1920, "height": 980},
    "colorScheme": "dark",
    "locale": "en-US",
    "userAgent": "persona-agent",
    "permissions": ["geolocation"],
}


def _minimal_state() -> StorageState:
    """Return a typed zero-content storage state the native API accepts."""
    return {"cookies": [], "origins": []}


class _RuntimeFixture:
    """One running runtime whose native browser is a typed Playwright stand-in."""

    def __init__(self) -> None:
        self.created_contexts: list[Mock] = []
        self.popups: list[Page] = []
        self.browser = Mock(spec=PWBrowser)
        self.browser.is_connected.return_value = True
        self.browser.new_context = AsyncMock(side_effect=self._new_context)
        self.main = self._make_context(_MAIN_OPTIONS)
        self.state = BrowserRuntimeState(max_groups=3)
        self.state.main_ctx = self.main
        self.state.main_ctx_owned = True
        self.state.shared_pd = Mock(spec=Chrome)
        self.state.popup_handler = self._record_popup
        self.state.contexts.bind_shared(self.main)

    def _make_context(self, options: dict[str, object]) -> Mock:
        context = Mock(spec=PWBrowserCtx)
        context.browser = self.browser
        context.is_closed.return_value = False
        context.pages = []
        context._impl_obj = SimpleNamespace(_options=options)
        context.close = AsyncMock()
        return context

    def _new_context(self, **_kwargs: object) -> Mock:
        context = self._make_context({})
        self.created_contexts.append(context)
        return context

    async def _record_popup(self, page: Page) -> None:
        self.popups.append(page)


def _browser_identity(state: BrowserRuntimeState) -> type[Browser]:
    """Return a per-test Browser subclass owning only *state* and its lifecycle."""
    return type(
        "TestBrowser",
        (Browser,),
        {"_runtime": state, "_lifecycle": BrowserLifecycle(state)},
    )


class BrowserContextRestoreTests(IsolatedAsyncioTestCase):
    """The facade and runtime seams that carry native storage state into new contexts."""

    def setUp(self) -> None:
        self.fixture = _RuntimeFixture()
        self.browser_cls = _browser_identity(self.fixture.state)

    async def test_new_isolated_context_gets_persona_options_and_the_same_state(self) -> None:
        """Happy path: the native state passes through unchanged beside persona options."""
        state = _minimal_state()

        handle = await self.browser_cls.get_context("session-a", storage_state=state)

        self.assertEqual(self.fixture.browser.new_context.call_count, 1)
        kwargs = self.fixture.browser.new_context.call_args.kwargs
        self.assertEqual(kwargs["viewport"], {"width": 1920, "height": 980})
        self.assertEqual(kwargs["color_scheme"], "dark")
        self.assertIs(kwargs["storage_state"], state)
        self.assertNotIn("locale", kwargs)
        self.assertNotIn("user_agent", kwargs)
        self.assertIs(handle.context, self.fixture.created_contexts[0])

    async def test_new_isolated_context_registers_popup_routing_and_human_patch(self) -> None:
        """Invariant: popup routing and humanization still apply to a restored context."""
        self.fixture.state.humanize_contexts = True

        with patch("prowl.browser.driver.runtime.apply_human_patch") as human_patch:
            await self.browser_cls.get_context("session-a", storage_state=_minimal_state())

        context = self.fixture.created_contexts[0]
        context.on.assert_any_call("page", self.fixture.state.popup_handler)
        human_patch.assert_called_once_with(context)

    async def test_state_none_keeps_the_previous_keyword_free_native_call(self) -> None:
        """Boundary: without state the native call carries only the persona options."""
        await self.browser_cls.get_context("session-a")

        self.assertEqual(
            self.fixture.browser.new_context.call_args.kwargs,
            {"viewport": {"width": 1920, "height": 980}, "color_scheme": "dark"},
        )

    async def test_reuse_keeps_the_live_context_and_ignores_a_newer_state(self) -> None:
        """State transition: a live handle wins, so the factory is not called again."""
        first_state = _minimal_state()
        second_state: StorageState = {
            "cookies": [{"name": "x", "value": "1", "domain": "example.test", "path": "/"}],
            "origins": [],
        }

        first = await self.browser_cls.get_context("session-a", storage_state=first_state)
        again = await self.browser_cls.get_context("session-a", storage_state=second_state)

        self.assertIs(first, again)
        self.assertEqual(self.fixture.browser.new_context.call_count, 1)
        self.assertIs(self.fixture.browser.new_context.call_args.kwargs["storage_state"], first_state)

    async def test_shared_context_rejects_state_before_any_start(self) -> None:
        """Error path: supplied state for the shared context fails before launch."""
        with (
            patch.object(self.browser_cls, "start", AsyncMock()) as start,
            self.assertRaisesRegex(BrowserContextError, "shared persistent context"),
        ):
            await self.browser_cls.get_context(None, storage_state=_minimal_state())

        start.assert_not_called()
        self.assertEqual(self.fixture.browser.new_context.call_count, 0)

    async def test_shared_default_still_returns_the_persistent_context(self) -> None:
        """Boundary: the shared path is unchanged when no state is supplied."""
        handle = await self.browser_cls.get_context()

        self.assertTrue(handle.is_shared)
        self.assertIs(handle.context, self.fixture.main)
        self.assertEqual(self.fixture.browser.new_context.call_count, 0)

    async def test_failed_native_restore_leaves_no_handle_or_capacity_use(self) -> None:
        """Error path: a native restore failure registers nothing and consumes no slot."""
        self.fixture.browser.new_context = AsyncMock(side_effect=RuntimeError("native restore failed"))
        before = self.fixture.state.contexts.context_created_total

        with self.assertRaisesRegex(RuntimeError, "native restore failed"):
            await self.browser_cls.get_context("session-a", storage_state=_minimal_state())

        self.assertIsNone(self.fixture.state.contexts.isolated("session-a"))
        self.assertEqual(self.fixture.state.contexts.isolated_ids(), ())
        self.assertEqual(self.fixture.state.contexts.context_created_total, before)
        self.assertEqual(self.fixture.state.contexts.count, 1)
