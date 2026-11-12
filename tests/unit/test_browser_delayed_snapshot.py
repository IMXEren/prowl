"""Delayed snapshots use the selected native page without replaying requests."""

from __future__ import annotations

import asyncio
import base64
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Page, Request, Route
from turbohtml import Element, Html, Text

from prowl.browser.browser import TabGroup
from prowl.browser.exceptions import PageLoadError
from prowl.browser.page_handler import PageHandler, PageResponse

_DOCTYPE_HEADER = "<!DOCTYPE html>\n"


def _document(title_text: str) -> Element:
    """Build a real turbohtml document whose ``<title>`` carries *title_text*."""
    html = Element("html")
    head = Element("head")
    title = Element("title")
    title.append(Text(title_text))
    head.append(title)
    html.append(head)
    return html


def _site_with_budget(group: Mock) -> PageHandler:
    """Return a PageHandler whose original request still has time left on the clock."""
    site = PageHandler(group)
    site.timeout = 30
    site.start = time.perf_counter()
    return site


class SiteSnapshotTests(IsolatedAsyncioTestCase):
    """``PageHandler.snapshot`` observes the live page without renavigating or replaying POST."""

    async def test_disabled_options_return_same_source_without_native_work(self) -> None:
        group = Mock(spec=TabGroup)
        site = _site_with_budget(group)
        source = PageResponse(
            "<html><head><title>Ready</title></head></html>",
            status_code=200,
            headers={"x-test": "1"},
            user_agent="UA",
            url="https://example.com/orig",
        )

        with (
            patch.object(
                PageHandler, "get_time_left", Mock(side_effect=AssertionError("timeout bookkeeping on fast path"))
            ),
            patch.object(PageHandler, "build_dom_tree", AsyncMock()) as build,
            patch("prowl.browser.page_handler.asyncio.sleep", AsyncMock()) as sleep_mock,
        ):
            result = await site.snapshot(source)

        self.assertIs(result, source)
        build.assert_not_called()
        sleep_mock.assert_not_called()
        self.assertIsNone(source.screenshot)
        self.assertEqual(source.url, "https://example.com/orig")

    async def test_wait_then_fresh_dom_uses_current_selected_url(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse(
            "<html><head><title>Old</title></head></html>",
            status_code=200,
            headers={"x-test": "1"},
            user_agent="UA",
        )
        fresh = _document("Fresh")

        with (
            patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=fresh)) as build,
            patch("prowl.browser.page_handler.asyncio.sleep", AsyncMock()) as sleep_mock,
        ):
            result = await site.snapshot(source, wait_in_seconds=0.5)

        self.assertIsNot(result, source)
        sleep_mock.assert_awaited_once_with(0.5)
        build.assert_awaited_once()
        self.assertEqual(result.text, _DOCTYPE_HEADER + fresh.serialize(Html()))
        self.assertEqual(result.url, "https://example.com/after")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(dict(result.headers), {"x-test": "1"})
        self.assertEqual(result.user_agent, "UA")
        self.assertIsNone(result.screenshot)
        self.assertEqual(source.text, "<html><head><title>Old</title></head></html>")
        self.assertIsNone(source.url)
        page.goto.assert_not_called()

    async def test_screenshot_is_base64_of_selected_page_png(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        png = b"\x89PNG\r\n\x1a\npixels"
        page.screenshot = AsyncMock(return_value=png)
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse("<html></html>", status_code=200, url="https://example.com/orig")
        fresh = _document("Ready")

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=fresh)):
            result = await site.snapshot(source, return_screenshot=True)

        self.assertEqual(result.screenshot, base64.b64encode(png).decode("ascii"))
        page.screenshot.assert_awaited_once_with(type="png")

    async def test_screenshot_failure_propagates_without_replay(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        page.screenshot = AsyncMock(side_effect=RuntimeError("native screenshot failed"))
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse("<html></html>", status_code=200, url="https://example.com/orig")
        fresh = _document("Ready")

        with (
            patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=fresh)) as build,
            patch.object(PageHandler, "get", AsyncMock()) as replay_get,
            patch.object(PageHandler, "post", AsyncMock()) as replay_post,
            self.assertRaisesRegex(RuntimeError, "native screenshot failed"),
        ):
            await site.snapshot(source, return_screenshot=True)

        build.assert_awaited_once()
        replay_get.assert_not_called()
        replay_post.assert_not_called()
        page.goto.assert_not_called()

    async def test_deadline_timeout_propagates(self) -> None:
        group = Mock(spec=TabGroup)
        group.ppage = Mock(spec=Page)
        site = PageHandler(group)
        site.timeout = 1
        site.start = time.perf_counter() - 0.95
        source = PageResponse("<html></html>", status_code=200, url="https://example.com/orig")
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def _never_returns() -> Element:
            started.set()
            await blocker.wait()
            return _document("Ready")

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(side_effect=_never_returns)):
            task = asyncio.create_task(site.snapshot(source, return_screenshot=True))
            await asyncio.wait_for(started.wait(), timeout=1)
            with self.assertRaises(TimeoutError):
                await task

    async def test_cancellation_propagates_without_replay(self) -> None:
        group = Mock(spec=TabGroup)
        group.ppage = Mock(spec=Page)
        site = _site_with_budget(group)
        source = PageResponse("<html></html>", status_code=200, url="https://example.com/orig")
        started = asyncio.Event()
        blocker = asyncio.Event()

        async def _never_returns() -> Element:
            started.set()
            await blocker.wait()
            return _document("Ready")

        with (
            patch.object(PageHandler, "build_dom_tree", AsyncMock(side_effect=_never_returns)),
            patch.object(PageHandler, "get", AsyncMock()) as replay_get,
            patch.object(PageHandler, "post", AsyncMock()) as replay_post,
        ):
            task = asyncio.create_task(site.snapshot(source, return_screenshot=True))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        replay_get.assert_not_called()
        replay_post.assert_not_called()

    async def test_delayed_challenge_title_is_rejected(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/after"
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse(
            "<html><head><title>Ready</title></head></html>",
            status_code=200,
            url="https://example.com/orig",
        )
        challenge = _document("Just a moment...")

        with (
            patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=challenge)),
            self.assertRaisesRegex(PageLoadError, "cloudflare protection"),
        ):
            await site.snapshot(source, return_screenshot=True)

        page.screenshot.assert_not_called()

    async def test_explicit_post_target_survives_snapshot_on_warmed_page(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/"
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse("<html></html>", url="https://example.com/submit")
        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Posted"))):
            result = await site.snapshot(source, wait_in_seconds=0.001)
        self.assertEqual(result.url, source.url)
        self.assertIsNone(result.status_code)
        page.goto.assert_not_called()

    async def test_post_capture_fulfills_fetched_body_without_an_origin_replay(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.url = "https://example.com/"
        page.screenshot = AsyncMock(return_value=b"PNG")
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse(
            "<html><head><title>Posted</title></head></html>",
            url="https://example.com/submit",
            status_code=201,
            headers={"Content-Type": "text/html", "Content-Encoding": "gzip", "Set-Cookie": "sid=old"},
        )
        route = Mock(spec=Route)
        request = Mock(spec=Request)
        request.is_navigation_request.return_value = True
        request.frame = page.main_frame
        route.request = request

        async def navigate(_url: str, **_options: object) -> None:
            handler = page.route.await_args.args[1]
            await handler(route)
            page.url = source.url

        page.goto = AsyncMock(side_effect=navigate)
        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Posted"))):
            result = await site.snapshot(source, post_response=True, return_screenshot=True)
        route.fulfill.assert_awaited_once_with(status=200, headers={"Content-Type": "text/html"}, body=source.text)
        route.fallback.assert_not_awaited()
        page.unroute.assert_awaited_once_with(source.url, page.route.await_args.args[1])
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.url, source.url)

    async def test_failed_post_materialization_removes_only_its_owned_route(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        page.goto = AsyncMock(side_effect=RuntimeError("capture failed"))
        group.ppage = page
        site = _site_with_budget(group)
        source = PageResponse("<html></html>", url="https://example.com/submit")
        with self.assertRaises(RuntimeError):
            await site.snapshot(source, post_response=True, return_screenshot=True)
        page.unroute.assert_awaited_once_with(source.url, page.route.await_args.args[1])
        page.screenshot.assert_not_called()
