"""Tests for Site request fidelity: headers, cookie ordering, and POST preflight.

The tab group is replaced by a fake so header application/cleanup, origin-root
warmup ordering, and cancellation cleanup are verified without a live browser.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from turbohtml import Element

from prowl.browser.site import Site, Source


class FakePage:
    """Records extra HTTP header application."""

    def __init__(self) -> None:
        self.header_calls: list[dict[str, str]] = []

    async def set_extra_http_headers(self, headers: dict[str, str]) -> None:
        self.header_calls.append(dict(headers))


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


class SiteGetHeaderTests(IsolatedAsyncioTestCase):
    """GET applies caller headers and always clears them."""

    async def _get(self, group: FakeTabGroup, headers: dict[str, str] | None) -> None:
        site = Site(group)
        with (
            patch.object(Site, "_add_network_listeners", AsyncMock()),
            patch.object(Site, "_check_if_loaded", AsyncMock(return_value=True)),
            patch.object(Site, "_wait_page_load", AsyncMock()),
            patch.object(Site, "build_dom_tree", AsyncMock(return_value=Element("html"))),
        ):
            await site.get("https://example.com/page", 30, headers=headers)

    async def test_headers_are_applied_then_cleared(self) -> None:
        group = FakeTabGroup()
        await self._get(group, {"accept": "application/json"})
        self.assertEqual(group.ppage.header_calls, [{"accept": "application/json"}, {}])

    async def test_absent_headers_touch_nothing(self) -> None:
        group = FakeTabGroup()
        await self._get(group, None)
        self.assertEqual(group.ppage.header_calls, [])

    async def test_headers_cleared_on_failure(self) -> None:
        group = FakeTabGroup()
        site = Site(group)
        with (
            patch.object(Site, "_add_network_listeners", AsyncMock()),
            patch.object(Site, "_check_if_loaded", AsyncMock(return_value=True)),
            patch.object(Site, "_wait_page_load", AsyncMock()),
            patch.object(Site, "build_dom_tree", AsyncMock(side_effect=RuntimeError("boom"))),
            self.assertRaises(Exception),  # noqa: B017
        ):
            await site.get("https://example.com/page", 30, headers={"accept": "text/plain"})
        self.assertEqual(group.ppage.header_calls, [{"accept": "text/plain"}, {}])


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
