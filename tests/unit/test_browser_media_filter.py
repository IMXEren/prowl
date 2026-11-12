"""Page-local media filtering preserves other routes and cleanup ownership."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from playwright.async_api import Page, Request, Route

from prowl.browser.browser import TabGroup
from prowl.browser.page_handler import PageHandler

_MEDIA_TYPES = ("image", "stylesheet", "font")
_OTHER_TYPES = ("document", "script", "fetch", "xhr", "media")


def _group_with_page() -> tuple[Mock, Mock]:
    """Return a TabGroup double whose selected page is a Page double."""
    group = Mock(spec=TabGroup)
    page = Mock(spec=Page)
    group.ppage = page
    return group, page


def _route(resource_type: str) -> Mock:
    """Return a Route double carrying a native resource-type value."""
    request = Mock(spec=Request)
    request.resource_type = resource_type
    route = Mock(spec=Route)
    route.request = request
    return route


class SiteMediaFilterTests(IsolatedAsyncioTestCase):
    """``PageHandler.media_filter`` aborts only media resources on the captured page."""

    async def test_media_types_abort_without_fallback(self) -> None:
        for resource_type in _MEDIA_TYPES:
            with self.subTest(resource_type=resource_type):
                group, page = _group_with_page()
                site = PageHandler(group)
                route = _route(resource_type)

                async with site.media_filter():
                    handler = page.route.await_args.args[1]
                    await handler(route)

                route.abort.assert_awaited_once_with()
                route.fallback.assert_not_awaited()

    async def test_non_media_types_fall_back_without_abort(self) -> None:
        for resource_type in _OTHER_TYPES:
            with self.subTest(resource_type=resource_type):
                group, page = _group_with_page()
                site = PageHandler(group)
                route = _route(resource_type)

                async with site.media_filter():
                    handler = page.route.await_args.args[1]
                    await handler(route)

                route.fallback.assert_awaited_once_with()
                route.abort.assert_not_awaited()

    async def test_only_selected_page_receives_the_route(self) -> None:
        group = Mock(spec=TabGroup)
        page = Mock(spec=Page)
        other_page = Mock(spec=Page)
        group.ppage = page
        site = PageHandler(group)

        async with site.media_filter():
            pass

        handler = page.route.await_args.args[1]
        page.route.assert_awaited_once_with("**/*", handler)
        page.unroute.assert_awaited_once_with("**/*", handler)
        other_page.route.assert_not_called()
        other_page.unroute.assert_not_called()
        group.new_tab.assert_not_called()
        group.context.assert_not_called()

    async def test_success_and_body_failure_unroute_the_same_handler_once(self) -> None:
        for body_fails in (False, True):
            with self.subTest(body_fails=body_fails):
                group, page = _group_with_page()
                site = PageHandler(group)

                if body_fails:
                    message = "body failed"
                    with self.assertRaisesRegex(RuntimeError, message):
                        async with site.media_filter():
                            handler = page.route.await_args.args[1]
                            raise RuntimeError(message)
                else:
                    async with site.media_filter():
                        handler = page.route.await_args.args[1]

                handler = page.route.await_args.args[1]
                page.route.assert_awaited_once_with("**/*", handler)
                page.unroute.assert_awaited_once_with("**/*", handler)
                page.unroute_all.assert_not_called()

    async def test_installation_failure_skips_body_and_cleanup(self) -> None:
        group, page = _group_with_page()
        page.route = AsyncMock(side_effect=RuntimeError("route install failed"))
        site = PageHandler(group)
        body_ran = False

        with self.assertRaisesRegex(RuntimeError, "route install failed"):
            async with site.media_filter():
                body_ran = True

        self.assertFalse(body_ran)
        page.unroute.assert_not_called()

    async def test_cleanup_failure_propagates_after_body(self) -> None:
        group, page = _group_with_page()
        page.unroute = AsyncMock(side_effect=RuntimeError("unroute failed"))
        site = PageHandler(group)
        body_ran = False

        with self.assertRaisesRegex(RuntimeError, "unroute failed"):
            async with site.media_filter():
                body_ran = True

        self.assertTrue(body_ran)
        page.route.assert_awaited_once()

    async def test_cancellation_in_body_unroutes_on_captured_page(self) -> None:
        group, page = _group_with_page()
        site = PageHandler(group)
        entered = asyncio.Event()
        blocker = asyncio.Event()

        async def body() -> None:
            async with site.media_filter():
                entered.set()
                await blocker.wait()

        task = asyncio.create_task(body())
        await asyncio.wait_for(entered.wait(), timeout=1)
        handler = page.route.await_args.args[1]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        page.unroute.assert_awaited_once_with("**/*", handler)

    async def test_page_swap_after_enter_unroutes_the_original_page(self) -> None:
        group = Mock(spec=TabGroup)
        original = Mock(spec=Page)
        replacement = Mock(spec=Page)
        group.ppage = original
        site = PageHandler(group)

        async with site.media_filter():
            group.ppage = replacement

        handler = original.route.await_args.args[1]
        original.route.assert_awaited_once_with("**/*", handler)
        original.unroute.assert_awaited_once_with("**/*", handler)
        replacement.route.assert_not_called()
        replacement.unroute.assert_not_called()

    async def test_filter_falls_back_to_earlier_owned_routes(self) -> None:
        group, page = _group_with_page()
        site = PageHandler(group)
        earlier = AsyncMock()
        await page.route("https://example.com/**", earlier)
        route = _route("script")

        async with site.media_filter():
            handler = page.route.await_args.args[1]
            await handler(route)

        self.assertIsNot(handler, earlier)
        self.assertEqual(page.route.await_count, 2)
        route.fallback.assert_awaited_once_with()
        route.continue_.assert_not_awaited()
        route.abort.assert_not_awaited()
        earlier.assert_not_called()
        page.unroute.assert_awaited_once_with("**/*", handler)
