"""Explicit verification preserves keyboard, token freshness and native cleanup ownership."""

from __future__ import annotations

import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, call, patch

from playwright.async_api import JSHandle, Keyboard, Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydoll.browser.tab import Tab
from turbohtml import Element, Text

from prowl.browser.browser import TabGroup
from prowl.browser.exceptions import PageLoadError
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.browser.solvers import CloudflareSolver

_ANCHOR_REMOVE = "node => node.remove()"
_UNCHANGED = "turnstile value unchanged"
_TABS = 3


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


def _anchor() -> Mock:
    anchor = Mock(spec=JSHandle)
    anchor.evaluate = AsyncMock()
    anchor.dispose = AsyncMock()
    return anchor


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
    return page, widget


def _site(group: Mock) -> PageHandler:
    """Return a PageHandler whose original request still has time left on the clock."""
    site = PageHandler(group)
    site.timeout = 30
    site.start = time.perf_counter()
    return site


class SiteVerifyTurnstileTests(IsolatedAsyncioTestCase):
    """``PageHandler.verify_turnstile`` presses native keys and returns only a changed token."""

    async def test_absent_widget_returns_none_without_keyboard_work(self) -> None:
        group = Mock(spec=TabGroup)
        page, widget = _page(count=0)
        group.ppage = page
        site = _site(group)

        result = await site.verify_turnstile(_TABS)

        self.assertIsNone(result)
        page.keyboard.press.assert_not_called()
        widget.evaluate_all.assert_not_called()
        page.wait_for_function.assert_not_called()
        page.evaluate_handle.assert_not_called()

    async def test_tab_and_space_sequence_returns_changed_value(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("new-token")
        page, _ = _page(initial="old", wait_return=handle)
        group.ppage = page
        site = _site(group)

        result = await site.verify_turnstile(_TABS)

        self.assertEqual(result, "new-token")
        self.assertEqual(page.keyboard.press.call_args_list, [call("Tab")] * _TABS + [call("Space")])
        handle.dispose.assert_awaited_once()
        page.evaluate_handle.assert_not_called()
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["old"])
        self.assertNotIn('"old"', page.wait_for_function.await_args.args[0])

    async def test_duplicate_response_inputs_track_each_initial_value(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("fresh")
        page, widget = _page(count=2, wait_return=handle)
        widget.evaluate_all = AsyncMock(return_value=["", "previous"])
        group.ppage = page

        result = await _site(group).verify_turnstile(0)

        self.assertEqual(result, "fresh")
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["", "previous"])
        expression = page.wait_for_function.await_args.args[0]
        self.assertIn("querySelectorAll", expression)
        self.assertIn("initial.includes(input.value)", expression)
        self.assertNotIn("previous", expression)

    async def test_owned_native_checkbox_waits_for_changed_token_without_keyboard(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("fresh")
        page, widget = _page(count=2, wait_return=handle)
        widget.evaluate_all = AsyncMock(return_value=["", ""])
        group.ppage = page
        site = _site(group)
        tab = Mock(spec=Tab)
        site.tab = tab
        site._active_solver = (CloudflareSolver(), tab)

        with patch("prowl.browser.page_handler.click_embedded_turnstile", AsyncMock(return_value=True)) as click:
            result = await site.verify_turnstile(2)

        self.assertEqual(result, "fresh")
        click.assert_awaited_once_with(tab, page)
        page.keyboard.press.assert_not_called()
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["", ""])
        handle.dispose.assert_awaited_once()

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
            result = await site.verify_turnstile(0)

        self.assertEqual(result, "issued")
        click.assert_not_awaited()
        page.keyboard.press.assert_not_called()
        page.wait_for_function.assert_not_awaited()

    async def test_stale_prefilled_input_is_not_claimed_as_solver_result(self) -> None:
        group = Mock(spec=TabGroup)
        page, _ = _page(initial="old-value", wait_return=_handle("new-value"))
        group.ppage = page
        site = _site(group)
        tab = Mock(spec=Tab)
        site.tab = tab
        site._active_solver = (CloudflareSolver(), tab)

        with patch("prowl.browser.page_handler.click_embedded_turnstile", AsyncMock(return_value=False)):
            result = await site.verify_turnstile(1)

        self.assertEqual(result, "new-value")
        page.evaluate.assert_awaited_once()
        self.assertEqual(page.keyboard.press.call_args_list, [call("Tab"), call("Space")])
        self.assertEqual(page.wait_for_function.await_args.kwargs["arg"], ["old-value"])

    async def test_native_wait_timeout_retries_from_one_owned_focus_node(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("retried")
        calls = {"n": 0}

        async def _wait(*_args: object, **_kwargs: object) -> Mock:
            calls["n"] += 1
            if calls["n"] == 1:
                raise PlaywrightTimeoutError(_UNCHANGED)
            return handle

        page, _ = _page(initial="old")
        page.wait_for_function = AsyncMock(side_effect=_wait)
        anchor = _anchor()
        page.evaluate_handle = AsyncMock(return_value=anchor)
        group.ppage = page
        site = _site(group)

        result = await site.verify_turnstile(_TABS)

        self.assertEqual(result, "retried")
        page.evaluate_handle.assert_awaited_once()
        self.assertEqual(page.keyboard.press.call_args_list, ([call("Tab")] * _TABS + [call("Space")]) * 2)
        anchor.evaluate.assert_any_await("node => node.focus()")
        anchor.evaluate.assert_any_await(_ANCHOR_REMOVE)
        anchor.dispose.assert_awaited_once()

    async def test_unchanged_value_deadline_removes_and_disposes_helper(self) -> None:
        group = Mock(spec=TabGroup)
        calls = {"n": 0}

        async def _wait(*_args: object, **_kwargs: object) -> Mock:
            calls["n"] += 1
            if calls["n"] == 1:
                raise PlaywrightTimeoutError(_UNCHANGED)
            await asyncio.Event().wait()
            return _handle()

        page, _ = _page(initial="")
        page.wait_for_function = AsyncMock(side_effect=_wait)
        anchor = _anchor()
        page.evaluate_handle = AsyncMock(return_value=anchor)
        group.ppage = page
        site = PageHandler(group)
        site.timeout = 1
        site.start = time.perf_counter()

        with self.assertRaises(TimeoutError):
            await site.verify_turnstile(_TABS)

        anchor.evaluate.assert_any_await(_ANCHOR_REMOVE)
        anchor.dispose.assert_awaited_once()

    async def test_selected_page_survives_group_swap(self) -> None:
        group = Mock(spec=TabGroup)
        handle = _handle("captured")
        page_one, _ = _page(initial="old", wait_return=handle)
        page_two, _ = _page(initial="old", wait_return=handle)
        group.ppage = page_one
        site = _site(group)

        async def _swap(*_args: object, **_kwargs: object) -> Mock:
            group.ppage = page_two
            return handle

        page_one.wait_for_function = AsyncMock(side_effect=_swap)

        result = await site.verify_turnstile(_TABS)

        self.assertEqual(result, "captured")
        self.assertEqual(page_one.keyboard.press.await_count, _TABS + 1)
        page_two.keyboard.press.assert_not_called()

    async def test_cancellation_cleans_up_owned_focus_node(self) -> None:
        group = Mock(spec=TabGroup)
        started = asyncio.Event()
        blocking = asyncio.Event()
        blocker = asyncio.Event()

        async def _wait(*_args: object, **_kwargs: object) -> Mock:
            if not started.is_set():
                started.set()
                raise PlaywrightTimeoutError(_UNCHANGED)
            blocking.set()
            await blocker.wait()
            return _handle()

        page, _ = _page(initial="old")
        page.wait_for_function = AsyncMock(side_effect=_wait)
        anchor = _anchor()
        page.evaluate_handle = AsyncMock(return_value=anchor)
        group.ppage = page
        site = _site(group)

        task = asyncio.create_task(site.verify_turnstile(_TABS))
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(blocking.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        anchor.evaluate.assert_any_await(_ANCHOR_REMOVE)
        anchor.dispose.assert_awaited_once()


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
        group.ppage = page
        return group, tab

    async def _run_get(self, group: Mock, *, tabs_till_verify: int | None, tree: Element) -> PageResponse:
        site = _site(group)
        with (
            patch.object(PageHandler, "_add_network_listeners", AsyncMock()),
            patch.object(PageHandler, "_remove_network_listeners", AsyncMock()),
            patch.object(PageHandler, "_remove_scoped_headers", AsyncMock()),
            patch.object(PageHandler, "_check_if_loaded", AsyncMock(return_value=True)),
            patch.object(PageHandler, "_wait_page_load", AsyncMock()),
            patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=tree)),
        ):
            return await site.get("https://example.com/page", 30, tabs_till_verify=tabs_till_verify)

    async def test_default_get_does_not_verify_or_attach(self) -> None:
        group, tab = self._group()
        with patch.object(PageHandler, "verify_turnstile", AsyncMock()) as verify:
            result = await self._run_get(group, tabs_till_verify=None, tree=_document("Ready"))

        verify.assert_not_called()
        self.assertIsNone(result.turnstile_token)
        self.assertIsNone(result.url)
        self.assertEqual(tab.go_to.await_count, 1)

    async def test_enabled_get_navigates_once_then_verifies(self) -> None:
        group, tab = self._group()
        seen: list[int] = []

        async def _go_to(*_args: object, **_kwargs: object) -> None:
            tab.enable_auto_solve_cloudflare_captcha.assert_awaited_once_with(
                time_before_click=1, time_to_wait_captcha=30
            )

        tab.go_to = AsyncMock(side_effect=_go_to)

        async def _verify(tabs: int) -> str:
            seen.append(tabs)
            self.assertEqual(tab.go_to.await_count, 1)
            return "tok"

        with patch.object(PageHandler, "verify_turnstile", AsyncMock(side_effect=_verify)):
            result = await self._run_get(group, tabs_till_verify=2, tree=_document("Ready"))

        self.assertEqual(seen, [2])
        self.assertEqual(tab.go_to.await_count, 1)
        self.assertEqual(result.turnstile_token, "tok")
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
        with (
            patch.object(PageHandler, "verify_turnstile", AsyncMock(side_effect=TimeoutError)),
            self.assertRaisesRegex(PageLoadError, "turnstile verification timed out"),
        ):
            await self._run_get(group, tabs_till_verify=1, tree=_document("Ready"))

    async def test_enabled_get_still_rejects_challenge_title(self) -> None:
        group, _ = self._group()
        with (
            patch.object(PageHandler, "verify_turnstile", AsyncMock(return_value=None)),
            self.assertRaisesRegex(PageLoadError, "cloudflare protection"),
        ):
            await self._run_get(group, tabs_till_verify=1, tree=_document("Just a moment..."))


class SourceTurnstileSnapshotTests(IsolatedAsyncioTestCase):
    """Snapshot preserves a carried token and the legacy default stays ``None``."""

    async def test_snapshot_preserves_turnstile_token(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        group.ppage = page
        site = _site(group)
        source = PageResponse(
            "<html></html>",
            status_code=200,
            url="https://example.com/orig",
            turnstile_token="carried",  # noqa: S106
        )
        fresh = _document("Ready")

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=fresh)):
            result = await site.snapshot(source, wait_in_seconds=0.001)

        self.assertEqual(result.turnstile_token, "carried")

    async def test_default_get_source_has_no_token(self) -> None:
        group = Mock(spec=TabGroup)
        group.ppage = Mock(spec=Page)
        site = _site(group)
        source = PageResponse("<html></html>", status_code=200)

        self.assertIsNone(source.turnstile_token)
        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Ready"))):
            result = await site.snapshot(source, return_screenshot=False)

        self.assertIs(result, source)
        self.assertIsNone(result.turnstile_token)
