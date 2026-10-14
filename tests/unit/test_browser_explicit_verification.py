"""Explicit verification clicks the widget and requires a fresh response."""

from __future__ import annotations

import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import ElementHandle, Frame, JSHandle, Keyboard, Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydoll.browser.tab import Tab
from turbohtml import Element, Text

from prowl.browser.browser import TabGroup
from prowl.browser.exceptions import PageLoadError
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.browser.solvers import CloudflareSolver


def _document(title_text: str) -> Element:
    """Build a real turbohtml document whose ``<title>`` carries *title_text*."""
    html = Element("html")
    head = Element("head")
    title = Element("title")
    title.append(Text(title_text))
    head.append(title)
    html.append(head)
    return html


def _handle(value: str = "token") -> Mock:
    handle = Mock(spec=JSHandle)
    handle.json_value = AsyncMock(return_value=value)
    handle.dispose = AsyncMock()
    return handle


def _page(
    *,
    count: int = 1,
    initial: str = "",
    wait_return: Mock | None = None,
) -> tuple[Mock, Mock]:
    """Return a native-spec page plus the widget locator it resolves."""
    page = Mock(spec=Page)
    widget = Mock(spec=Locator)
    widget.count = AsyncMock(return_value=count)
    widget.evaluate_all = AsyncMock(return_value=[initial] * count)
    page.locator = Mock(return_value=widget)
    page.keyboard = Mock(spec=Keyboard)
    page.keyboard.press = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    page.evaluate_handle = AsyncMock()
    page.wait_for_function = AsyncMock(return_value=wait_return)
    frame = Mock(spec=Frame)
    frame.url = "https://challenges.cloudflare.com/widget"
    element = Mock(spec=ElementHandle)
    element.is_visible = AsyncMock(return_value=True)
    element.evaluate = AsyncMock(return_value=["", ""])
    element.dispose = AsyncMock()
    frame.frame_element = AsyncMock(return_value=element)
    checkbox = Mock(spec=Locator)
    checkbox.count = AsyncMock(return_value=1)
    checkbox.nth = Mock(return_value=checkbox)
    checkbox.is_visible = AsyncMock(return_value=True)
    checkbox.click = AsyncMock()
    frame.get_by_role = Mock(return_value=checkbox)
    page.frames = [frame]
    return page, widget


def _site(group: Mock) -> PageHandler:
    """Return a PageHandler whose original request still has time left on the clock."""
    site = PageHandler(group)
    site.timeout = 30
    site.start = time.perf_counter()
    return site


class SiteVerifyTurnstileTests(IsolatedAsyncioTestCase):
    """Verification keeps native clicking first and never uses keyboard fallback."""

    async def test_absent_widget_returns_none_without_click(self) -> None:
        group = Mock(spec=TabGroup)
        page, widget = _page(count=0)
        group.ppage = page
        self.assertIsNone(await _site(group).verify_turnstile())
        page.frames[0].get_by_role.assert_not_called()
        widget.evaluate_all.assert_not_called()
        page.keyboard.press.assert_not_called()

    async def test_fallback_widens_frame_and_clicks_actual_checkbox(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("new-token")
        page, _ = _page(initial="old", wait_return=handle)
        group.ppage = page
        self.assertEqual(await _site(group).verify_turnstile(), "new-token")
        frame = page.frames[0]
        frame.get_by_role.assert_called_once_with("checkbox")
        frame.get_by_role.return_value.click.assert_awaited_once()
        element = frame.frame_element.return_value
        self.assertIn("300px", element.evaluate.await_args_list[1].args[0])
        self.assertEqual(element.evaluate.await_args_list[2].args[1], ["", ""])
        element.dispose.assert_awaited_once()
        page.keyboard.press.assert_not_called()
        page.evaluate_handle.assert_not_called()
        handle.dispose.assert_awaited_once()
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["old"])

    async def test_duplicate_response_inputs_track_each_initial_value(self) -> None:
        group = Mock(spec=TabGroup)
        page, widget = _page(count=2, wait_return=_handle("fresh"))
        widget.evaluate_all = AsyncMock(return_value=["", "previous"])
        group.ppage = page
        self.assertEqual(await _site(group).verify_turnstile(), "fresh")
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["", "previous"])
        expression = page.wait_for_function.await_args.args[0]
        self.assertIn("querySelectorAll", expression)
        self.assertIn("initial.includes(input.value)", expression)
        self.assertNotIn("previous", expression)

    async def test_owned_native_checkbox_waits_without_fallback(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page(wait_return=_handle("fresh"))
        group.ppage = page
        site = _site(group)
        tab = Mock(spec=Tab)
        site.tab = tab
        site._active_solver = (CloudflareSolver(), tab)
        with (
            patch("prowl.browser.page_handler.click_embedded_turnstile", AsyncMock(return_value=True)) as click,
            patch("prowl.browser.page_handler.find_and_click_turnstile", AsyncMock()) as fallback,
        ):
            self.assertEqual(await site.verify_turnstile(), "fresh")
        click.assert_awaited_once_with(tab, page)
        fallback.assert_not_awaited()
        page.keyboard.press.assert_not_called()

    async def test_owned_native_solver_accepts_token_already_issued_on_new_page(self) -> None:
        group = Mock(spec=TabGroup)
        page, widget = _page(count=2)
        widget.evaluate_all = AsyncMock(return_value=["", "issued"])
        page.evaluate = AsyncMock(return_value="issued")
        group.ppage = page
        site = _site(group)
        tab = Mock(spec=Tab)
        site.tab = tab
        site._active_solver = (CloudflareSolver(), tab)
        with patch("prowl.browser.page_handler.click_embedded_turnstile", AsyncMock()) as click:
            self.assertEqual(await site.verify_turnstile(), "issued")
        click.assert_not_awaited()
        page.keyboard.press.assert_not_called()
        page.wait_for_function.assert_not_awaited()

    async def test_stale_prefilled_input_requires_fresh_response(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page(initial="old-value", wait_return=_handle("new-value"))
        group.ppage = page
        site = _site(group)
        tab = Mock(spec=Tab)
        site.tab = tab
        site._active_solver = (CloudflareSolver(), tab)
        with patch("prowl.browser.page_handler.click_embedded_turnstile", AsyncMock(return_value=False)):
            self.assertEqual(await site.verify_turnstile(), "new-value")
        page.evaluate.assert_awaited_once()
        page.keyboard.press.assert_not_called()
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["old-value"])

    async def test_unchanged_value_expires_at_original_deadline(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()

        async def wait(*_args: object, **_kwargs: object) -> Mock:
            await asyncio.Event().wait()
            return _handle()

        page.wait_for_function = AsyncMock(side_effect=wait)
        group.ppage = page
        site = _site(group)
        site.timeout = 0.05
        with self.assertRaises(TimeoutError):
            await site.verify_turnstile()
        page.keyboard.press.assert_not_called()
        page.frames[0].frame_element.return_value.dispose.assert_awaited_once()

    async def test_selected_page_survives_group_swap(self) -> None:
        group = Mock(spec=TabGroup)
        page_one, _ = _page(wait_return=_handle("captured"))
        page_two, _ = _page()
        group.ppage = page_one

        async def swap(**_kwargs: object) -> None:
            group.ppage = page_two

        page_one.frames[0].get_by_role.return_value.click = AsyncMock(side_effect=swap)
        self.assertEqual(await _site(group).verify_turnstile(), "captured")
        page_two.frames[0].get_by_role.assert_not_called()
        page_two.wait_for_function.assert_not_called()

    async def test_cancelled_click_restores_width_and_disposes_frame(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        group.ppage = page
        started = asyncio.Event()

        async def block(**_kwargs: object) -> None:
            started.set()
            await asyncio.Event().wait()

        page.frames[0].get_by_role.return_value.click = AsyncMock(side_effect=block)
        task = asyncio.create_task(_site(group).verify_turnstile())
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        element = page.frames[0].frame_element.return_value
        self.assertEqual(element.evaluate.await_args_list[-1].args[1], ["", ""])
        element.dispose.assert_awaited_once()
        page.wait_for_function.assert_not_called()


class WidgetClickScopeTests(IsolatedAsyncioTestCase):
    """The fallback never guesses a target or clicks the response container."""

    async def test_spoofed_provider_frame_is_ignored(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        page.frames[0].url = "https://challenges.cloudflare.com.example.org/widget"
        group.ppage = page
        self.assertIsNone(await _site(group).verify_turnstile())
        page.frames[0].frame_element.assert_not_awaited()
        page.wait_for_function.assert_not_awaited()

    async def test_hidden_frame_is_ignored(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        element = page.frames[0].frame_element.return_value
        element.is_visible = AsyncMock(return_value=False)
        group.ppage = page
        self.assertIsNone(await _site(group).verify_turnstile())
        page.frames[0].get_by_role.assert_not_called()
        element.dispose.assert_awaited_once()

    async def test_multiple_visible_checkboxes_are_not_clicked(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        checkbox = page.frames[0].get_by_role.return_value
        checkbox.count = AsyncMock(return_value=2)
        group.ppage = page
        self.assertIsNone(await _site(group).verify_turnstile())
        checkbox.click.assert_not_awaited()
        page.frames[0].frame_element.return_value.evaluate.assert_not_awaited()

    async def test_closed_shadow_without_accessible_control_is_not_guessed(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        checkbox = page.frames[0].get_by_role.return_value
        checkbox.count = AsyncMock(return_value=0)
        group.ppage = page
        self.assertIsNone(await _site(group).verify_turnstile())
        checkbox.click.assert_not_awaited()
        page.keyboard.press.assert_not_called()

    async def test_click_failure_restores_existing_important_width(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page()
        element = page.frames[0].frame_element.return_value
        element.evaluate = AsyncMock(return_value=["125px", "important"])
        checkbox = page.frames[0].get_by_role.return_value
        checkbox.click = AsyncMock(side_effect=PlaywrightTimeoutError("click timed out"))
        group.ppage = page
        with self.assertRaises(PlaywrightTimeoutError):
            await _site(group).verify_turnstile()
        self.assertEqual(element.evaluate.await_args_list[-1].args[1], ["125px", "important"])
        element.dispose.assert_awaited_once()
        page.wait_for_function.assert_not_awaited()


class SiteGetVerificationTests(IsolatedAsyncioTestCase):
    """Enabled GET verifies once after navigation; defaults keep the legacy payload."""

    def _group(self) -> tuple[Mock, Mock]:
        group = Mock(spec=TabGroup)
        tab = Mock(spec=Tab)
        tab.go_to = AsyncMock()
        tab.enable_page_events = AsyncMock()
        tab.disable_page_events = AsyncMock()
        tab.disable_fetch_events = AsyncMock()
        tab.remove_callback = AsyncMock()
        future: asyncio.Future[Tab] = asyncio.get_running_loop().create_future()
        future.set_result(tab)
        group.ptab = future
        page = Mock(spec=Page)
        page.url = "https://example.com/verified"
        frame = Mock(spec=Frame)
        frame.url = "https://challenges.cloudflare.com/widget"
        page.frames = [frame]
        page.query_selector_all = AsyncMock(return_value=[])
        group.ppage = page
        return group, tab

    async def _run_get(self, group: Mock, *, solve_captcha: bool, tree: Element) -> PageResponse:
        site = _site(group)
        with (
            patch.object(PageHandler, "_add_network_listeners", AsyncMock()),
            patch.object(PageHandler, "_remove_network_listeners", AsyncMock()),
            patch.object(PageHandler, "_remove_scoped_headers", AsyncMock()),
            patch.object(PageHandler, "_check_if_loaded", AsyncMock(return_value=True)),
            patch.object(PageHandler, "_wait_page_load", AsyncMock()),
            patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=tree)),
        ):
            return await site.get("https://example.com/page", 30, solve_captcha=solve_captcha)

    async def test_default_get_does_not_verify_or_attach(self) -> None:
        group, tab = self._group()
        with patch.object(PageHandler, "verify_turnstile", AsyncMock()) as verify:
            result = await self._run_get(group, solve_captcha=False, tree=_document("Ready"))

        verify.assert_not_called()
        tab.enable_auto_solve_cloudflare_captcha.assert_not_awaited()
        self.assertIsNone(result.captcha_token)
        self.assertIsNone(result.url)
        self.assertEqual(tab.go_to.await_count, 1)

    async def test_enabled_get_navigates_once_then_verifies(self) -> None:
        group, tab = self._group()
        seen: list[bool] = []

        async def _go_to(*_args: object, **_kwargs: object) -> None:
            tab.enable_auto_solve_cloudflare_captcha.assert_awaited_once_with(
                time_before_click=1, time_to_wait_captcha=30
            )

        tab.go_to = AsyncMock(side_effect=_go_to)

        async def _verify() -> str:
            seen.append(True)
            self.assertEqual(tab.go_to.await_count, 1)
            return "tok"

        with patch.object(PageHandler, "verify_turnstile", AsyncMock(side_effect=_verify)):
            result = await self._run_get(group, solve_captcha=True, tree=_document("Ready"))

        self.assertEqual(seen, [True])
        self.assertEqual(tab.go_to.await_count, 1)
        self.assertEqual(result.captcha_provider, "turnstile")
        self.assertEqual(result.captcha_token, "tok")
        self.assertEqual(result.url, "https://example.com/verified")
        tab.disable_auto_solve_cloudflare_captcha.assert_awaited_once()

    async def test_explicit_page_load_defers_solver_cleanup_until_token_check(self) -> None:
        group, tab = self._group()
        site = _site(group)
        site.tab = tab
        site.response_found = True
        site._loaded.set()
        solver = CloudflareSolver()
        site._active_solver = (solver, tab)
        site.cf_auto_solve_enabled = True

        self.assertTrue(await site._check_if_loaded(cleanup=False))
        tab.disable_auto_solve_cloudflare_captcha.assert_not_awaited()
        await site._cleanup()
        tab.disable_auto_solve_cloudflare_captcha.assert_awaited_once()

    async def test_enabled_get_reports_bounded_token_timeout(self) -> None:
        group, _ = self._group()
        with patch.object(PageHandler, "verify_turnstile", AsyncMock(side_effect=TimeoutError)):
            result = await self._run_get(group, solve_captcha=True, tree=_document("Ready"))
        self.assertIsNone(result.captcha_token)
        self.assertIsNone(result.captcha_provider)

    async def test_enabled_get_still_rejects_challenge_title(self) -> None:
        group, _ = self._group()
        with (
            patch.object(PageHandler, "verify_turnstile", AsyncMock(return_value=None)),
            self.assertRaisesRegex(PageLoadError, "cloudflare protection"),
        ):
            await self._run_get(group, solve_captcha=True, tree=_document("Just a moment..."))


class SourceTurnstileSnapshotTests(IsolatedAsyncioTestCase):
    """Snapshot preserves a carried token and the legacy default stays ``None``."""

    async def test_snapshot_preserves_captcha_token(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        group.ppage = page
        site = _site(group)
        source = PageResponse(
            "<html></html>",
            status_code=200,
            url="https://example.com/orig",
            captcha_provider="turnstile",
            captcha_token="carried",  # noqa: S106
        )
        fresh = _document("Ready")

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=fresh)):
            result = await site.snapshot(source, wait_in_seconds=0.001)

        self.assertEqual(result.captcha_token, "carried")

    async def test_default_get_source_has_no_token(self) -> None:
        group = Mock(spec=TabGroup)
        group.ppage = Mock(spec=Page)
        site = _site(group)
        source = PageResponse("<html></html>", status_code=200)

        self.assertIsNone(source.captcha_token)
        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Ready"))):
            result = await site.snapshot(source, return_screenshot=False)

        self.assertIs(result, source)
        self.assertIsNone(result.captcha_token)
