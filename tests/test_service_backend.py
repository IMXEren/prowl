"""Tests for the browser backend's per-fetch tab-group lifecycle.

The shared browser is replaced by a fake so group creation, cleanup,
cookie installation, and cancellation are exercised without a live browser
or network. Per-session serialization is owned by the session registry and is
tested at the service layer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from prowl.browser.site import Source
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest, _normalize_cookies

CF_COOKIE = {
    "name": "cf_clearance",
    "value": "token",
    "domain": ".example.com",
    "path": "/",
    "expires": -1,
    "secure": True,
    "httpOnly": True,
    "sameSite": "None",
}


class FakeTab:
    """Records cookie installation."""

    def __init__(self) -> None:
        self.cookies: list[dict[str, Any]] | None = None

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.cookies = list(cookies)


class FakeGroup:
    """A tab group double tracking quit calls."""

    def __init__(self, cookies: list[dict[str, Any]]) -> None:
        self.tab = FakeTab()
        self.quit_calls = 0
        self._cookies = cookies

    @property
    async def ptab(self) -> FakeTab:
        return self.tab

    def pd(self) -> FakeGroup:
        return self

    async def get_cookies(self) -> list[dict[str, Any]]:
        return self._cookies

    async def quit(self) -> None:
        self.quit_calls += 1


class FakeBrowser:
    """Fake shared-browser facade that hands out one group per create()."""

    def __init__(self, group: FakeGroup) -> None:
        self.group = group
        self.start_calls = 0
        self.create_calls = 0
        self.shutdown_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def create(self) -> FakeGroup:
        self.create_calls += 1
        return self.group

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeSite:
    """Fake Site recording get/post calls with optional concurrency tracking."""

    def __init__(self, source: Source, *, delay: float = 0.0, error: BaseException | None = None) -> None:
        self.source = source
        self.delay = delay
        self.error = error
        self.get_calls: list[tuple[str, int, dict[str, str] | None]] = []
        self.post_calls: list[tuple[str, int, str, dict[str, str] | None]] = []
        self.active = 0
        self.max_active = 0

    async def _run(self) -> Source:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return self.source
        finally:
            self.active -= 1

    async def get(self, url: str, timeout: int, headers: dict[str, str] | None = None) -> Source:
        self.get_calls.append((url, timeout, headers))
        return await self._run()

    async def post(
        self,
        url: str,
        timeout: int,
        *,
        post_data: str = "",
        headers: dict[str, str] | None = None,
    ) -> Source:
        self.post_calls.append((url, timeout, post_data, headers))
        return await self._run()


def _source() -> Source:
    return Source(
        source="<html><body>ok</body></html>",
        status_code=200,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Test)",
        url="https://example.com/",
    )


class BrowserBackendLifecycleTests(IsolatedAsyncioTestCase):
    """Every fetch owns a tab group and closes it on every path."""

    def _install(
        self,
        site: FakeSite,
        cookies: list[dict[str, Any]] | None = None,
    ) -> tuple[BrowserBackend, FakeBrowser, FakeGroup]:
        group = FakeGroup(cookies or [])
        browser = FakeBrowser(group)
        backend = BrowserBackend()
        patchers = [
            patch.object(backend_module, "Browser", browser),
            patch.object(backend_module, "resolve_site", lambda _group, _url: site),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, browser, group

    async def test_fetch_creates_and_quits_group_on_success(self) -> None:
        site = FakeSite(_source())
        backend, browser, group = self._install(site)
        result = await backend.fetch("s", FetchRequest(url="https://example.com/"))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(browser.create_calls, 1)
        self.assertEqual(group.quit_calls, 1)

    async def test_fetch_quits_group_on_error(self) -> None:
        site = FakeSite(_source(), error=RuntimeError("boom"))
        backend, _browser, group = self._install(site)
        with self.assertRaises(RuntimeError):
            await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual(group.quit_calls, 1)

    async def test_fetch_quits_group_on_cancellation(self) -> None:
        site = FakeSite(_source(), error=asyncio.CancelledError())
        backend, _browser, group = self._install(site)
        with self.assertRaises(asyncio.CancelledError):
            await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual(group.quit_calls, 1)

    async def test_no_group_is_retained_between_fetches(self) -> None:
        site = FakeSite(_source())
        backend, browser, group = self._install(site)
        await backend.fetch("s", FetchRequest(url="https://example.com/"))
        await backend.fetch("s", FetchRequest(url="https://example.com/"))
        self.assertEqual(browser.create_calls, 2)
        self.assertEqual(group.quit_calls, 2)

    async def test_cookies_installed_before_navigation(self) -> None:
        site = FakeSite(_source())
        cookies = [{"name": "a", "value": "b", "domain": ".example.com"}]
        backend, _browser, group = self._install(site, [CF_COOKIE])
        await backend.fetch(None, FetchRequest(url="https://example.com/", cookies=cookies))
        self.assertEqual(group.tab.cookies, cookies)

    async def test_get_headers_forwarded_to_site(self) -> None:
        site = FakeSite(_source())
        backend, _browser, _group = self._install(site)
        await backend.fetch(None, FetchRequest(url="https://example.com/", headers={"accept": "application/json"}))
        self.assertEqual(site.get_calls[0][2], {"accept": "application/json"})

    async def test_post_forwarded_with_body_and_headers(self) -> None:
        site = FakeSite(_source())
        backend, _browser, _group = self._install(site)
        await backend.fetch(
            None,
            FetchRequest(
                url="https://example.com/api",
                method="POST",
                post_data="q=1",
                headers={"content-type": "application/x-www-form-urlencoded"},
            ),
        )
        self.assertEqual(site.post_calls[0][2], "q=1")
        self.assertEqual(site.post_calls[0][3], {"content-type": "application/x-www-form-urlencoded"})

    async def test_anonymous_fetches_are_not_serialized_by_the_backend(self) -> None:
        site = FakeSite(_source(), delay=0.05)
        backend, _browser, _group = self._install(site)
        await asyncio.gather(
            backend.fetch(None, FetchRequest(url="https://example.com/a")),
            backend.fetch(None, FetchRequest(url="https://example.com/b")),
        )
        self.assertEqual(site.max_active, 2)

    async def test_same_session_fetches_are_not_serialized_by_the_backend(self) -> None:
        site = FakeSite(_source(), delay=0.05)
        backend, _browser, _group = self._install(site)
        await asyncio.gather(
            backend.fetch("s", FetchRequest(url="https://example.com/a")),
            backend.fetch("s", FetchRequest(url="https://example.com/b")),
        )
        self.assertEqual(site.max_active, 2)

    async def test_start_and_aclose_delegate_to_browser(self) -> None:
        site = FakeSite(_source())
        backend, browser, _group = self._install(site)
        await backend.start()
        await backend.aclose()
        self.assertEqual(browser.start_calls, 1)
        self.assertEqual(browser.shutdown_calls, 1)

    async def test_close_session_is_idempotent(self) -> None:
        backend = BrowserBackend()
        await backend.close_session("missing")
        await backend.close_session("missing")


class CookieShapeTests(IsolatedAsyncioTestCase):
    """Cookies are reported in the FlareSolverr shape."""

    async def test_normalize_keeps_all_flaresolverr_fields(self) -> None:
        normalized = _normalize_cookies([CF_COOKIE])
        self.assertEqual(normalized[0]["name"], "cf_clearance")
        for key in ("value", "domain", "path", "expires", "secure", "httpOnly", "sameSite"):
            self.assertIn(key, normalized[0])

    async def test_normalize_drops_incomplete_cookies(self) -> None:
        self.assertEqual(_normalize_cookies([{"name": "x"}, {"value": "y"}]), [])
