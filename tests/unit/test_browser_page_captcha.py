"""Tests for opt-in selected-page checkbox solving on ``PageHandler.get``.

The page, frames, and locators are spec-constrained doubles so provider
selection, the exactly-one contract, the opt-in default, and snapshot
propagation are verified without a live browser.
"""

from __future__ import annotations

import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import ElementHandle, Frame, Locator, Page
from pydoll.browser.tab import Tab
from turbohtml import Element, Text

from prowl.browser.browser import TabGroup
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.browser.solvers import solve_visible_captcha, solve_visible_checkbox

_RECAPTCHA_URL = "https://www.google.com/recaptcha/api2/anchor?k=site"
_HCAPTCHA_URL = "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html"


def _checkbox(*, count: int = 1, visible: bool = True) -> Mock:
    locator = Mock(spec=Locator)
    locator.count = AsyncMock(return_value=count)
    locator.is_visible = AsyncMock(return_value=visible)
    locator.click = AsyncMock()
    return locator


def _frame(url: str, checkbox: Mock, *, frame_visible: bool = True) -> Mock:
    frame = Mock(spec=Frame)
    frame.url = url
    element = Mock(spec=ElementHandle)
    element.is_visible = AsyncMock(return_value=frame_visible)
    frame.frame_element = AsyncMock(return_value=element)
    frame.locator = Mock(return_value=checkbox)
    return frame


def _page(frames: list[Mock], evaluate: AsyncMock) -> Mock:
    page = Mock(spec=Page)
    page.frames = frames
    page.evaluate = evaluate
    page.query_selector_all = AsyncMock(return_value=[])
    return page


def _document(title_text: str) -> Element:
    html = Element("html")
    head = Element("head")
    title = Element("title")
    title.append(Text(title_text))
    head.append(title)
    html.append(head)
    return html


def _site(group: Mock) -> PageHandler:
    site = PageHandler(group)
    site.timeout = 30
    site.start = time.perf_counter()
    return site


def _group() -> tuple[Mock, Mock]:
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
    page.url = "https://example.com/page"
    group.ppage = page
    return group, tab


async def _run_get(
    group: Mock,
    *,
    tree: Element,
    solve_captcha: bool,
) -> PageResponse:
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


class VisibleCheckboxSelectionTests(IsolatedAsyncioTestCase):
    """Only one recognized provider frame is ever dispatched to a click."""

    async def test_absent_provider_returns_none_without_click(self) -> None:
        evaluate = AsyncMock(return_value=[])
        page = _page([_frame("https://example.com/plain", _checkbox())], evaluate)

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        evaluate.assert_not_awaited()

    async def test_single_recaptcha_frame_returns_fresh_token(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        evaluate = AsyncMock(side_effect=[[""], ["fresh-token"]])
        page = _page([frame], evaluate)

        self.assertEqual(await solve_visible_checkbox(page, lambda: 5.0), ("recaptcha", "fresh-token"))
        checkbox.click.assert_awaited_once()

    async def test_one_visible_checkbox_among_same_provider_frames(self) -> None:
        hidden = _checkbox()
        visible = _checkbox()
        page = _page(
            [_frame(_RECAPTCHA_URL, hidden, frame_visible=False), _frame(_RECAPTCHA_URL, visible)],
            AsyncMock(side_effect=[[""], ["fresh-token"]]),
        )
        self.assertEqual(await solve_visible_checkbox(page, lambda: 5.0), ("recaptcha", "fresh-token"))
        hidden.click.assert_not_awaited()
        visible.click.assert_awaited_once()

    async def test_two_visible_checkboxes_of_same_provider_are_ambiguous(self) -> None:
        first = _checkbox()
        second = _checkbox()
        page = _page([_frame(_RECAPTCHA_URL, first), _frame(_RECAPTCHA_URL, second)], AsyncMock())
        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        first.click.assert_not_awaited()
        second.click.assert_not_awaited()

    async def test_two_provider_frames_are_ambiguous_and_never_click(self) -> None:
        recaptcha = _checkbox()
        hcaptcha = _checkbox()
        frames = [_frame(_RECAPTCHA_URL, recaptcha), _frame(_HCAPTCHA_URL, hcaptcha)]
        page = _page(frames, AsyncMock(return_value=[]))

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        recaptcha.click.assert_not_awaited()
        hcaptcha.click.assert_not_awaited()


class GetCaptchaOptInTests(IsolatedAsyncioTestCase):
    """Ordinary GET stays inert; opted-in GET attaches the fresh provider pair."""

    async def test_default_get_does_not_solve_captcha(self) -> None:
        group, _ = _group()
        with patch("prowl.browser.page_handler.solve_visible_captcha", AsyncMock()) as solve:
            result = await _run_get(group, tree=_document("Ready"), solve_captcha=False)

        solve.assert_not_called()
        self.assertIsNone(result.captcha_provider)
        self.assertIsNone(result.captcha_token)

    async def test_opted_in_get_attaches_provider_and_token(self) -> None:
        group, _ = _group()
        solved = ("recaptcha", "fresh-token")
        with patch("prowl.browser.page_handler.solve_visible_captcha", AsyncMock(return_value=solved)) as solve:
            result = await _run_get(group, tree=_document("Ready"), solve_captcha=True)

        solve.assert_awaited_once()
        self.assertEqual(result.captcha_provider, "recaptcha")
        self.assertEqual(result.captcha_token, "fresh-token")


class CaptchaSnapshotPropagationTests(IsolatedAsyncioTestCase):
    """Snapshot copies a carried captcha pair unchanged."""

    async def test_snapshot_preserves_captcha_pair(self) -> None:
        group, _ = _group()
        site = _site(group)
        source = PageResponse(
            "<html></html>",
            status_code=200,
            captcha_provider="hcaptcha",
            captcha_token="h-token",  # noqa: S106
        )

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Ready"))):
            result = await site.snapshot(source, wait_in_seconds=0.001)

        self.assertEqual(result.captcha_provider, "hcaptcha")
        self.assertEqual(result.captcha_token, "h-token")


class VisibleCaptchaDispatchTests(IsolatedAsyncioTestCase):
    async def test_absent_or_multiple_provider_mounts_do_not_solve(self) -> None:
        page = _page([], AsyncMock())
        with patch("prowl.browser.solvers.dispatch.solve_altcha", AsyncMock()) as solve:
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_not_awaited()
            page.query_selector_all.side_effect = [[Mock()], [Mock()]]
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_not_awaited()

    async def test_late_friendly_widget_is_detected_within_the_same_deadline(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[], [], [], [Mock()]]
        with patch(
            "prowl.browser.solvers.dispatch.solve_friendly_captcha",
            AsyncMock(return_value="fresh-payload"),
        ) as solve:
            self.assertEqual(await solve_visible_captcha(page, lambda: 1.0), ("friendly", "fresh-payload"))
            solve.assert_awaited_once()

    async def test_altcha_dispatch_returns_only_a_fresh_result(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[Mock()], []]
        with patch("prowl.browser.solvers.dispatch.solve_altcha", AsyncMock(return_value="fresh-payload")) as solve:
            self.assertEqual(await solve_visible_captcha(page, lambda: 5.0), ("altcha", "fresh-payload"))
            solve.assert_awaited_once()

    async def test_friendly_dispatch_does_not_report_unsolved_widget(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[], [Mock()]]
        with patch("prowl.browser.solvers.dispatch.solve_friendly_captcha", AsyncMock(return_value=None)) as solve:
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_awaited_once()
