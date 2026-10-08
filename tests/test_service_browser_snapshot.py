"""Browser snapshot options preserve routing, cookie ordering and cleanup contracts."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import BrowserContext, Cookie
from pydoll.browser.tab import Tab

from prowl.browser import Browser, TabGroup
from prowl.browser.driver.contexts import BrowserContextManager
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest, FetchResult
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE

WARM_ROOT = "https://example.com/"
OPTION_ERROR = "browser response options require browser or auto mode"

CF_COOKIE: Cookie = {
    "name": "cf_clearance",
    "value": "token",
    "domain": ".example.com",
    "path": "/",
    "expires": -1,
    "secure": True,
    "httpOnly": True,
    "sameSite": "None",
}


def _source(
    *,
    body: str = "<html><body>ok</body></html>",
    url: str = WARM_ROOT,
    screenshot: str | None = None,
) -> PageResponse:
    return PageResponse(
        source=body,
        status_code=200,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url=url,
        screenshot=screenshot,
    )


class _SnapshotFixture(IsolatedAsyncioTestCase):
    def _backend(
        self,
        *,
        post_source: PageResponse | None = None,
    ) -> tuple[BrowserBackend, Mock, Mock, list[str]]:
        events: list[str] = []

        async def read_cookies() -> list[Cookie]:
            events.append("cookies")
            return [CF_COOKIE.copy()]

        async def read_profile_cookies(_group: TabGroup) -> list[Cookie]:
            return await read_cookies()

        context = Mock(spec=BrowserContext)
        context.cookies = AsyncMock(side_effect=read_cookies)
        handle = BrowserContextManager().bind_shared(context)
        group = Mock(spec=TabGroup)
        tab: asyncio.Future[Tab] = asyncio.get_running_loop().create_future()
        tab.set_result(Mock(spec=Tab))
        group.ptab = tab
        self.owner = Mock(spec=Browser)
        self.owner.start = AsyncMock()
        self.owner.create = AsyncMock(return_value=group)
        self.owner.get_context = AsyncMock(return_value=handle)
        self.handle = handle
        resolved = post_source or _source()
        site = Mock(spec=PageHandler)

        async def get(_url: str, _timeout: int, **_kwargs: object) -> PageResponse:
            events.append("get")
            return resolved

        async def post(_url: str, _timeout: int, **_kwargs: object) -> PageResponse:
            events.append("post")
            return resolved

        site.get = AsyncMock(side_effect=get)
        site.post = AsyncMock(side_effect=post)
        backend = BrowserBackend()
        for patcher in (
            patch.object(backend_module, "Browser", self.owner),
            patch.object(backend_module, "resolve_page_handler", Mock(return_value=site)),
            patch.object(backend_module, "_collect_cookies", AsyncMock(side_effect=read_profile_cookies)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, group, site, events


class LegacyDefaultsTests(_SnapshotFixture):
    """A request without capture options keeps the legacy browser fetch and result shape."""

    async def test_defaults_do_not_request_a_snapshot(self) -> None:
        backend, group, site, events = self._backend()
        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT))

        self.assertEqual(site.snapshot.await_count, 0)
        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.post.await_count, 0)
        self.assertEqual(events, ["get", "cookies"])
        self.assertEqual(result.response, _source().text)
        self.assertEqual(result.url, WARM_ROOT)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.user_agent, "Mozilla/5.0 (Browser)")
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["cf_clearance"])
        self.assertIsNone(result.screenshot)
        self.assertIsNone(result.body_bytes)
        self.assertIsNone(result.header_items)
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)
        self.assertEqual(group.quit.await_count, 1)


class CaptureRequestTests(_SnapshotFixture):
    """Requested browser work snapshots once, after the fetch and before the cookie read."""

    async def test_post_snapshot_runs_once_between_the_fetch_and_the_cookie_read(self) -> None:
        post_source = _source(body="<html><body>posted</body></html>")
        late = _source(body="<html><body>late</body></html>", url="https://example.com/final", screenshot="UE5H")
        backend, group, site, events = self._backend(post_source=post_source)

        async def _snapshot(_source: PageResponse, **_kwargs: object) -> PageResponse:
            events.append("snapshot")
            return late

        site.snapshot = AsyncMock(side_effect=_snapshot)
        result = await backend.fetch(
            None,
            FetchRequest(
                url=WARM_ROOT,
                method="POST",
                post_data="q=1",
                wait_in_seconds=0.5,
                return_screenshot=True,
            ),
        )

        self.assertEqual(site.post.await_count, 1)
        self.assertEqual(site.get.await_count, 0)
        self.assertEqual(site.snapshot.await_count, 1)
        self.assertEqual(events, ["post", "snapshot", "cookies"])
        self.assertEqual(site.snapshot.await_args.args[0], post_source)
        self.assertEqual(
            site.snapshot.await_args.kwargs, {"wait_in_seconds": 0.5, "return_screenshot": True, "post_response": True}
        )
        self.assertEqual(result.response, late.text)
        self.assertEqual(result.url, "https://example.com/final")
        self.assertEqual(result.screenshot, "UE5H")
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["cf_clearance"])
        self.assertEqual(group.quit.await_count, 1)

    async def test_wait_only_capture_passes_the_original_wait_and_drops_the_screenshot(self) -> None:
        late = _source(body="<html><body>late</body></html>", screenshot="SHOULD_NOT_LEAK")
        backend, _group, site, _events = self._backend()

        site.snapshot = AsyncMock(return_value=late)
        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, wait_in_seconds=0.25))

        self.assertEqual(site.snapshot.await_count, 1)
        self.assertEqual(
            site.snapshot.await_args.kwargs,
            {"wait_in_seconds": 0.25, "return_screenshot": False, "post_response": False},
        )
        self.assertEqual(result.response, late.text)
        self.assertIsNone(result.screenshot)


class SnapshotFailureTests(_SnapshotFixture):
    """A failed or cancelled native snapshot propagates and the group still closes."""

    async def test_snapshot_failure_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        site.snapshot = AsyncMock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, wait_in_seconds=1.0))

        self.assertEqual(site.snapshot.await_count, 1)
        self.assertEqual(group.quit.await_count, 1)

    async def test_snapshot_cancellation_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        site.snapshot = AsyncMock(side_effect=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, return_screenshot=True))

        self.assertEqual(site.snapshot.await_count, 1)
        self.assertEqual(group.quit.await_count, 1)


class ExplicitHttpCaptureGuardTests(_SnapshotFixture):
    """An explicit http request with capture options is refused before any acquisition."""

    async def test_http_mode_with_wait_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        pool_acquire = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for
        backend._pool.acquire = pool_acquire

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=HTTP_MODE, wait_in_seconds=0.5))

        self.assertEqual(str(caught.exception), OPTION_ERROR)
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()
        pool_acquire.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)

    async def test_http_mode_with_screenshot_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        backend._steal_least_recent = steal

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=HTTP_MODE, return_screenshot=True))

        self.assertEqual(str(caught.exception), OPTION_ERROR)
        steal.assert_not_awaited()

    async def test_http_raw_post_with_a_screenshot_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(
                None,
                FetchRequest(
                    url=WARM_ROOT,
                    method="POST",
                    mode=HTTP_MODE,
                    body_bytes=b"x",
                    return_screenshot=True,
                ),
            )

        self.assertEqual(str(caught.exception), OPTION_ERROR)
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()


class AutoCaptureRoutingTests(_SnapshotFixture):
    """An auto capture goes straight to the route's browser context without an HTTP attempt."""

    async def test_auto_capture_uses_the_route_context_with_no_http_stage(self) -> None:
        late = _source(body="<html><body>late</body></html>", url="https://example.com/final", screenshot="UE5H")
        backend, group, site, events = self._backend()

        async def _snapshot(_source: PageResponse, **_kwargs: object) -> PageResponse:
            events.append("snapshot")
            return late

        site.snapshot = AsyncMock(side_effect=_snapshot)
        http_stage = AsyncMock()
        backend._http_stage = http_stage

        result = await backend.fetch(
            None,
            FetchRequest(url=WARM_ROOT, mode=AUTO_MODE, wait_in_seconds=0.3, return_screenshot=True),
        )

        http_stage.assert_not_awaited()
        self.assertEqual(backend.metrics.http_fastpath_total, 0)
        self.assertEqual(backend.metrics.browser_escalations_total, 0)
        self.owner.get_context.assert_awaited_once_with(None)
        self.owner.create.assert_awaited_once_with(context=self.handle)
        self.assertEqual(events, ["get", "snapshot", "cookies"])
        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(
            site.snapshot.await_args.kwargs, {"wait_in_seconds": 0.3, "return_screenshot": True, "post_response": False}
        )
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertIsNotNone(result.classification)
        self.assertEqual(result.response, late.text)
        self.assertEqual(result.screenshot, "UE5H")
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["cf_clearance"])
        self.assertEqual(group.quit.await_count, 1)

    async def test_auto_without_capture_keeps_the_http_first_route(self) -> None:
        backend, _group, site, _events = self._backend()
        http_result = FetchResult(
            url=WARM_ROOT,
            status_code=200,
            headers={"content-type": "text/html"},
            response="ok",
            cookies=[],
            user_agent="Mozilla/5.0 (HTTP)",
            mode=HTTP_MODE,
        )
        http_stage = AsyncMock(return_value=http_result)
        backend._http_stage = http_stage

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=AUTO_MODE))

        self.assertEqual(http_stage.await_count, 1)
        self.owner.create.assert_not_awaited()
        self.assertEqual(site.snapshot.await_count, 0)
        self.assertIs(result, http_result)


class FetchOptionShapeTests(TestCase):
    """The appended fields keep the legacy positional and default shape."""

    def test_request_options_keep_their_order_and_verification_is_appended_last(self) -> None:
        request = FetchRequest("https://example.com/")

        self.assertEqual(request.wait_in_seconds, 0.0)
        self.assertFalse(request.return_screenshot)
        self.assertFalse(request.disable_media)
        self.assertIsNone(request.tabs_till_verify)
        fields = list(FetchRequest.__dataclass_fields__)
        self.assertEqual(
            fields[fields.index("wait_in_seconds") : fields.index("solve_captcha") + 1],
            ["wait_in_seconds", "return_screenshot", "disable_media", "tabs_till_verify", "solve_captcha"],
        )
        self.assertFalse(request.solve_captcha)

    def test_result_token_is_appended_after_the_screenshot(self) -> None:
        result = FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={},
            response="ok",
            cookies=[],
            user_agent="ua",
        )

        self.assertIsNone(result.screenshot)
        self.assertIsNone(result.turnstile_token)
        self.assertEqual(
            list(FetchResult.__dataclass_fields__)[-4:],
            ["screenshot", "turnstile_token", "captcha_provider", "captcha_token"],
        )
        self.assertIsNone(result.captcha_provider)
        self.assertIsNone(result.captcha_token)
