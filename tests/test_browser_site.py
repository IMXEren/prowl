"""Tests for Site request fidelity: header isolation, cookie ordering, and POST preflight.

The tab group is replaced by a fake so that header isolation, origin-root warmup
ordering, and cancellation cleanup are verified without a live browser.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from turbohtml import Element

from prowl.browser.site import Site, Source


class FakePage:
    """Records scoped routes installed on the Playwright page."""

    def __init__(self) -> None:
        self.main_frame = object()
        self.routes: list[tuple[Any, Any]] = []
        self.unroutes: list[tuple[Any, Any]] = []

    async def route(self, pattern: Any, handler: Any) -> None:
        self.routes.append((pattern, handler))

    async def unroute(self, pattern: Any, handler: Any) -> None:
        self.unroutes.append((pattern, handler))


class FakeRequest:
    """A routable browser request with browser-generated headers."""

    def __init__(self, page: FakePage, url: str, *, navigation: bool = False, main_frame: bool = True) -> None:
        self.url = url
        self.frame = page.main_frame if main_frame else object()
        self._navigation = navigation

    def is_navigation_request(self) -> bool:
        return self._navigation

    async def all_headers(self) -> dict[str, str]:
        return {"accept": "text/html", "user-agent": "browser-owned"}


class FakeRoute:
    """Records whether a request continued with modified headers."""

    def __init__(self, request: FakeRequest) -> None:
        self.request = request
        self.continued_headers: dict[str, str] | None = None

    async def continue_(self, *, headers: dict[str, str] | None = None) -> None:
        self.continued_headers = headers


class FakeTab:
    """Minimal tab double recording navigation and listener calls."""

    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.cookies: list[dict[str, Any]] | None = None

    async def enable_page_events(self) -> None:
        self.events.append(("enable_page_events", ""))

    async def disable_page_events(self) -> None:
        self.events.append(("disable_page_events", ""))

    async def enable_fetch_events(self, **_kwargs: Any) -> None:
        return None

    async def disable_fetch_events(self) -> None:
        return None

    async def on(self, _event: Any, _callback: Any) -> int:
        return 1

    async def remove_callback(self, _callback_id: int) -> None:
        return None

    async def execute_script(self, _script: str, **_kwargs: Any) -> dict[str, Any]:
        return {"result": {"result": {"value": "complete"}}}

    async def go_to(self, url: str, timeout: int | None = None) -> None:
        self.events.append(("go_to", url))

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.cookies = list(cookies)


class FakeTabGroup:
    """Minimal tab group double exposing ptab and ppage."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self._tab = FakeTab(self.events)
        self._page = FakePage()

    @property
    def ptab(self) -> Any:
        async def _resolve() -> FakeTab:
            return self._tab

        return _resolve()

    @property
    def ppage(self) -> FakePage:
        return self._page


class FakeWarmSite:
    """Fake solver returned by resolve_site for the origin-root warmup."""

    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.get_calls: list[str] = []

    async def get(self, url: str, timeout: int, headers: dict[str, str] | None = None) -> Source:
        self.events.append(("warm_get", url))
        self.get_calls.append(url)
        return Source("<html></html>", status_code=200, url=url)


async def _post_result(_url: str, _post_data: str, _headers: dict[str, str]) -> dict[str, Any]:
    return {"status": 200, "headers": {"content-type": "text/html"}, "body": "<html></html>", "userAgent": "UA"}


class SiteGetHeaderIsolationTests(IsolatedAsyncioTestCase):
    """GET leaves headers alone by default and scopes explicit custom headers."""

    async def _get(
        self,
        group: FakeTabGroup,
        *,
        fail: bool = False,
        headers: dict[str, str] | None = None,
        header_scope: str | None = None,
    ) -> None:
        site = Site(group)
        build = AsyncMock(side_effect=RuntimeError("boom")) if fail else AsyncMock(return_value=Element("html"))
        with (
            patch.object(Site, "_add_network_listeners", AsyncMock()),
            patch.object(Site, "_check_if_loaded", AsyncMock(return_value=True)),
            patch.object(Site, "_wait_page_load", AsyncMock()),
            patch.object(Site, "build_dom_tree", build),
        ):
            if fail:
                with self.assertRaises(Exception):  # noqa: B017
                    await site.get(
                        "https://example.com/page",
                        30,
                        headers=headers,
                        header_scope=header_scope,
                    )
            else:
                await site.get(
                    "https://example.com/page",
                    30,
                    headers=headers,
                    header_scope=header_scope,
                )

    async def _route(self, group: FakeTabGroup, request: FakeRequest) -> FakeRoute:
        handler = group.ppage.routes[0][1]
        route = FakeRoute(request)
        await handler(route)
        return route

    async def test_get_leaves_page_headers_untouched(self) -> None:
        group = FakeTabGroup()
        await self._get(group)
        self.assertEqual(group.ppage.routes, [])

    async def test_document_scope_modifies_only_initial_main_frame(self) -> None:
        group = FakeTabGroup()
        await self._get(group, headers={"authorization": "Bearer token"}, header_scope="document")

        matcher = group.ppage.routes[0][0]
        self.assertTrue(matcher("https://example.com/page"))
        self.assertFalse(matcher("https://example.com/app.js"))
        self.assertFalse(matcher("https://cdn.example.com/page"))
        document = await self._route(
            group,
            FakeRequest(group.ppage, "https://example.com/page", navigation=True),
        )
        subresource = await self._route(group, FakeRequest(group.ppage, "https://example.com/app.js"))
        child_navigation = await self._route(
            group,
            FakeRequest(group.ppage, "https://example.com/page", navigation=True, main_frame=False),
        )
        repeated_navigation = await self._route(
            group,
            FakeRequest(group.ppage, "https://example.com/page", navigation=True),
        )
        self.assertEqual(document.continued_headers["authorization"], "Bearer token")
        self.assertEqual(document.continued_headers["user-agent"], "browser-owned")
        self.assertIsNone(subresource.continued_headers)
        self.assertIsNone(child_navigation.continued_headers)
        self.assertIsNone(repeated_navigation.continued_headers)

    async def test_origin_scope_never_modifies_another_origin(self) -> None:
        group = FakeTabGroup()
        await self._get(group, headers={"x-api-key": "token"}, header_scope="origin")

        matcher = group.ppage.routes[0][0]
        self.assertTrue(matcher("https://example.com/api"))
        self.assertTrue(matcher("https://example.com:443/asset"))
        self.assertFalse(matcher("https://cdn.example.com/asset"))
        self.assertFalse(matcher("https://example.com:444/asset"))
        self.assertFalse(matcher("https://challenges.example.net/widget.js"))
        same_origin = await self._route(group, FakeRequest(group.ppage, "https://example.com/api"))
        explicit_default_port = await self._route(group, FakeRequest(group.ppage, "https://example.com:443/asset"))
        self.assertEqual(same_origin.continued_headers["x-api-key"], "token")
        self.assertEqual(explicit_default_port.continued_headers["x-api-key"], "token")

    async def test_scoped_route_is_removed_after_success_and_failure(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail):
                group = FakeTabGroup()
                await self._get(group, fail=fail, headers={"authorization": "token"}, header_scope="document")
                self.assertEqual(group.ppage.unroutes, group.ppage.routes)

    async def test_core_rejects_unsafe_headers(self) -> None:
        for headers, message in (
            ({"user-agent": "fake"}, "browser-controlled"),
            ({"bad name": "value"}, "invalid header name"),
            ({"x-api-key": "value\r\ninjected: true"}, "must not contain"),
        ):
            with self.subTest(headers=headers):
                group = FakeTabGroup()
                with self.assertRaisesRegex(Exception, message):
                    await self._get(group, headers=headers, header_scope="origin")
                self.assertEqual(group.ppage.routes, [])


class SitePostPreflightTests(IsolatedAsyncioTestCase):
    """POST warms the origin root through the normal GET solver, then fetches."""

    async def test_warmup_delegates_through_get_solver_not_bare_navigation(self) -> None:
        group = FakeTabGroup()
        site = Site(group)
        warm = FakeWarmSite(group.events)

        async def capture(url: str, _data: str, _headers: dict[str, str]) -> dict[str, Any]:
            group.events.append(("fetch", url))
            return await _post_result(url, "", {})

        with (
            patch("prowl.browser.site.resolve_site", return_value=warm),
            patch.object(Site, "_post_via_fetch", AsyncMock(side_effect=capture)),
        ):
            await site.post("https://example.com/api/search?q=1", 30, post_data="q=1", headers={})

        self.assertEqual(warm.get_calls, ["https://example.com/"])
        warm_index = group.events.index(("warm_get", "https://example.com/"))
        fetch_index = next(i for i, event in enumerate(group.events) if event[0] == "fetch")
        self.assertLess(warm_index, fetch_index)
        self.assertEqual(group.events[fetch_index][1], "https://example.com/api/search?q=1")
        # POST must not bypass the solver with its own navigation.
        self.assertEqual([event for event in group.events if event[0] == "go_to"], [])

    async def test_post_does_not_install_its_own_page_listeners(self) -> None:
        group = FakeTabGroup()
        site = Site(group)
        warm = FakeWarmSite(group.events)
        with (
            patch("prowl.browser.site.resolve_site", return_value=warm),
            patch.object(Site, "_post_via_fetch", AsyncMock(side_effect=_post_result)),
        ):
            await site.post("https://example.com/api", 30, post_data="q=1", headers={})

        self.assertNotIn(("enable_page_events", ""), group.events)

    async def test_post_cleans_up_on_cancellation(self) -> None:
        group = FakeTabGroup()
        site = Site(group)
        warm = FakeWarmSite(group.events)
        with (
            patch("prowl.browser.site.resolve_site", return_value=warm),
            patch.object(Site, "_post_via_fetch", AsyncMock(side_effect=asyncio.CancelledError())),
            self.assertRaises(asyncio.CancelledError),
        ):
            await site.post("https://example.com/api", 30, post_data="q=1", headers={})

        self.assertTrue(site.cleanup_done)
