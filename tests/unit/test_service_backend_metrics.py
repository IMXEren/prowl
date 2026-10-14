"""Backend routing metrics: counters, timings and acquisition measured at their seams.

The semantics pinned here: an HTTP attempt is counted once the client is in hand, an escalation
only when a browser-required decision or a transfer refusal sends the fetch to the browser, and a
detected challenge once per response. Acquisition timing covers the tab group's own creation, on
success and failure alike. No browser is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

from prowl.browser.page_handler import PageResponse
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest
from prowl.service.http_transport import (
    HttpNetworkError,
    HttpResult,
    UnsupportedCookieError,
    UnsupportedIdentityError,
)
from prowl.service.metrics import Metrics
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from prowl.service.sessions import ISOLATED_MODE

_CAPTCHA_BODY = '<html><body><div class="g-recaptcha" data-sitekey="k"></div></body></html>'
_PLAIN_BODY = "<html><body>ok</body></html>"


def _source(*, body: str = _PLAIN_BODY, status: int = 200) -> PageResponse:
    return PageResponse(
        source=body,
        status_code=status,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url="https://example.com/",
    )


def _http_result(*, status: int = 200, body: str = _PLAIN_BODY) -> HttpResult:
    return HttpResult(
        url="https://example.com/",
        status_code=status,
        headers={"content-type": "text/html"},
        body=body,
        cookies=[],
        set_cookie_headers=(),
    )


def _request(**fields: Any) -> FetchRequest:
    values: dict[str, Any] = {"url": "https://example.com/"}
    values.update(fields)
    return FetchRequest(**values)


class _StubTab:
    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        return None


class _StubGroup:
    """A tab group double: its context, its page, a cookie reader and its close count."""

    def __init__(self, context: Any, reader: Any) -> None:
        self.context = context
        self.tab = _StubTab()
        self.quit_calls = 0
        self._reader = reader

    @property
    async def ptab(self) -> _StubTab:
        return self.tab

    def pd(self) -> Any:
        return self._reader

    async def quit(self) -> None:
        self.quit_calls += 1


class _CookieReader:
    """A shared-profile cookie reader double holding an empty jar."""

    async def get_cookies(self) -> list[dict[str, Any]]:
        return []


class _StubSite:
    def __init__(self, source: PageResponse) -> None:
        self.source = source

    async def get(self, url: str, timeout: int, **kwargs: Any) -> PageResponse:
        return self.source

    async def post(self, url: str, timeout: int, **kwargs: Any) -> PageResponse:
        return self.source


class _FakeContext:
    def __init__(self) -> None:
        self.session_id: str | None = None

    async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        return None

    async def cookies(self) -> list[dict[str, Any]]:
        return []


class _FakeHandle:
    def __init__(self, session_id: str | None) -> None:
        self.session_id = session_id
        self.context = _FakeContext()


class _StubHttpClient:
    def __init__(
        self,
        *,
        result: HttpResult | None = None,
        error: BaseException | None = None,
        user_agent: str = "Mozilla/5.0 (HTTP)",
    ) -> None:
        self.result = result
        self.error = error
        self.identity = SimpleNamespace(user_agent=user_agent)

    async def fetch(self, url: str, **kwargs: Any) -> HttpResult:
        if self.error is not None:
            raise self.error
        return self.result or _http_result()


class _StubClients:
    """An injectable ``BrowserHttpClients`` double."""

    def __init__(
        self,
        client: _StubHttpClient | None = None,
        *,
        client_error: BaseException | None = None,
    ) -> None:
        self.client_instance = client or _StubHttpClient()
        self.client_error = client_error
        self.client_calls = 0

    async def client(self, context: Any, proxy: str | None) -> _StubHttpClient:
        self.client_calls += 1
        if self.client_error is not None:
            raise self.client_error
        return self.client_instance

    async def close_context(self, context: Any) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _MetricsFixture(IsolatedAsyncioTestCase):
    """Builds a ``BrowserBackend`` whose browser class is a stub and whose HTTP clients are injected."""

    def _backend(
        self,
        *,
        http_client: _StubHttpClient | None = None,
        client_error: BaseException | None = None,
        browser_source: PageResponse | None = None,
        create_error: BaseException | None = None,
    ) -> tuple[
        BrowserBackend,
        Metrics,
        list[dict[str, Any]],
        list[_StubGroup],
        _StubClients,
        dict[str | None, _FakeHandle],
    ]:
        metrics = Metrics()
        reader = _CookieReader()
        creates: list[dict[str, Any]] = []
        groups: list[_StubGroup] = []
        handles: dict[str | None, _FakeHandle] = {}

        class _FakeBrowser:
            @classmethod
            def proxy_url(cls) -> None:
                return None

            @classmethod
            async def start(cls) -> None:
                return None

            @classmethod
            async def create(cls, context: Any = None) -> _StubGroup:
                creates.append({} if context is None else {"context": context})
                if create_error is not None:
                    raise create_error
                group = _StubGroup(context, reader)
                groups.append(group)
                return group

            @classmethod
            async def get_context(cls, session_id: str | None) -> _FakeHandle:
                handle = handles.get(session_id)
                if handle is None:
                    handle = _FakeHandle(session_id)
                    handles[session_id] = handle
                return handle

            @classmethod
            async def close_context(cls, session_id: str) -> None:
                handles.pop(session_id, None)

            @classmethod
            async def shutdown(cls) -> None:
                return None

            @classmethod
            def pd(cls) -> Any:
                return reader

        site = _StubSite(browser_source or _source())
        clients = _StubClients(http_client, client_error=client_error)
        backend = BrowserBackend(metrics=metrics)
        backend._http = Mock(spec=backend_module.BrowserHttpClients, wraps=clients)
        for patcher in (
            patch.object(backend_module, "Browser", _FakeBrowser),
            patch.object(backend_module, "resolve_page_handler", lambda _group, _url: site),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, metrics, creates, groups, clients, handles


class ConstructorMetricTests(_MetricsFixture):
    """The metrics keyword is optional and each backend defaults to its own instance."""

    async def test_a_backend_without_metrics_owns_a_fresh_instance(self) -> None:
        first = BrowserBackend().metrics
        second = BrowserBackend().metrics
        self.assertIsInstance(first, Metrics)
        self.assertIsNot(first, second)

    async def test_a_supplied_metrics_instance_is_used_as_given(self) -> None:
        metrics = Metrics()
        backend = BrowserBackend(metrics=metrics)
        self.assertIs(backend.metrics, metrics)


class HttpFastpathMetricTests(_MetricsFixture):
    """An HTTP answer counts one attempt and one success, with no group acquisition."""

    async def test_http_success_counts_attempt_and_success_without_acquisition(self) -> None:
        backend, metrics, creates, _groups, clients, _handles = self._backend(
            http_client=_StubHttpClient(result=_http_result(status=200)),
        )
        result = await backend.fetch(None, _request(mode=HTTP_MODE))
        self.assertEqual(result.mode, HTTP_MODE)
        self.assertEqual(clients.client_calls, 1)
        self.assertEqual(metrics.requests_total, 1)
        self.assertEqual(metrics.requests_active, 0)
        self.assertEqual(metrics.http_fastpath_total, 1)
        self.assertEqual(metrics.http_fastpath_success_total, 1)
        self.assertEqual(metrics.browser_escalations_total, 0)
        self.assertEqual(metrics.challenge_detected_total, 0)
        self.assertEqual(metrics.request_duration_seconds.count, 1)
        self.assertEqual(creates, [])
        self.assertEqual(metrics.browser_acquire_seconds.count, 0)
        await backend.aclose()


class AutoChallengeMetricTests(_MetricsFixture):
    """An auto challenge counts its HTTP attempt, its escalation and each detected response."""

    async def test_auto_challenge_counts_attempt_escalation_and_unsolved_browser_response(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, metrics, creates, groups, _clients, _handles = self._backend(
            http_client=client,
            browser_source=_source(status=403, body=_CAPTCHA_BODY),
        )
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(metrics.http_fastpath_total, 1)
        self.assertEqual(metrics.http_fastpath_success_total, 0)
        self.assertEqual(metrics.browser_escalations_total, 1)
        self.assertEqual(metrics.challenge_detected_total, 2)
        self.assertEqual(len(creates), 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(metrics.browser_acquire_seconds.count, 1)
        await backend.aclose()

    async def test_auto_solved_browser_response_adds_no_second_challenge(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, metrics, _creates, _groups, _clients, _handles = self._backend(
            http_client=client,
            browser_source=_source(status=200),
        )
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(metrics.browser_escalations_total, 1)
        self.assertEqual(metrics.challenge_detected_total, 1)
        await backend.aclose()


class EscalationBoundaryMetricTests(_MetricsFixture):
    """A refusal before the client is a bypass; a refusal from the transfer is an escalation."""

    async def test_identity_refusal_before_the_client_is_a_bypass_not_an_attempt(self) -> None:
        backend, metrics, creates, _groups, _clients, _handles = self._backend(
            client_error=UnsupportedIdentityError("no profile"),
        )
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(metrics.http_fastpath_total, 0)
        self.assertEqual(metrics.browser_escalations_total, 0)
        self.assertEqual(metrics.challenge_detected_total, 0)
        self.assertEqual(len(creates), 1)
        await backend.aclose()

    async def test_a_transfer_cookie_refusal_counts_an_attempt_and_an_escalation(self) -> None:
        client = _StubHttpClient(error=UnsupportedCookieError("partitioned"))
        backend, metrics, _creates, _groups, clients, _handles = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(clients.client_calls, 1)
        self.assertEqual(metrics.http_fastpath_total, 1)
        self.assertEqual(metrics.browser_escalations_total, 1)
        self.assertEqual(metrics.challenge_detected_total, 0)
        await backend.aclose()


class TransportFailureMetricTests(_MetricsFixture):
    """A transport failure counts an attempt and never an escalation."""

    async def test_a_network_error_counts_an_attempt_but_no_escalation(self) -> None:
        client = _StubHttpClient(error=HttpNetworkError("no response"))
        backend, metrics, creates, _groups, _clients, _handles = self._backend(http_client=client)
        with self.assertRaises(HttpNetworkError):
            await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(metrics.requests_total, 1)
        self.assertEqual(metrics.requests_active, 0)
        self.assertEqual(metrics.http_fastpath_total, 1)
        self.assertEqual(metrics.http_fastpath_success_total, 0)
        self.assertEqual(metrics.browser_escalations_total, 0)
        self.assertEqual(metrics.challenge_detected_total, 0)
        self.assertEqual(creates, [])
        await backend.aclose()


class RequestLifecycleMetricTests(_MetricsFixture):
    """A fetch that never returns still records its duration and releases the active gauge."""

    async def test_a_cancelled_fetch_records_its_duration_and_releases_active(self) -> None:
        entered = asyncio.Event()

        class _BlockingClient(_StubHttpClient):
            async def fetch(self, url: str, **kwargs: Any) -> HttpResult:
                entered.set()
                await asyncio.Event().wait()
                return _http_result()

        backend, metrics, _creates, _groups, _clients, _handles = self._backend(
            http_client=_BlockingClient(),
        )
        task = asyncio.create_task(backend.fetch(None, _request(mode=HTTP_MODE)))
        await asyncio.wait_for(entered.wait(), timeout=1)
        self.assertEqual(metrics.http_fastpath_total, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(metrics.requests_total, 1)
        self.assertEqual(metrics.requests_active, 0)
        self.assertEqual(metrics.request_duration_seconds.count, 1)
        await backend.aclose()


class GroupAcquisitionMetricTests(_MetricsFixture):
    """Acquisition timing covers the group's own creation, on success and failure alike."""

    async def test_shared_group_creation_forwards_no_context_keyword(self) -> None:
        backend, metrics, creates, _groups, _clients, _handles = self._backend()
        result = await backend.fetch(None, _request())
        self.assertIsNone(result.mode)
        self.assertEqual(creates, [{}])
        self.assertEqual(metrics.browser_acquire_seconds.count, 1)
        self.assertGreaterEqual(metrics.browser_acquire_seconds.sum, 0.0)
        await backend.aclose()

    async def test_isolated_group_creation_forwards_the_session_context(self) -> None:
        backend, metrics, creates, groups, _clients, handles = self._backend()
        await backend.fetch(None, _request(session_id="s1", session_mode=ISOLATED_MODE))
        handle = handles["s1"]
        self.assertEqual(creates, [{"context": handle}])
        self.assertIs(groups[0].context, handle)
        self.assertEqual(metrics.browser_acquire_seconds.count, 1)
        await backend.close_session("s1")
        await backend.aclose()

    async def test_a_failed_group_creation_is_still_timed(self) -> None:
        backend, metrics, creates, _groups, _clients, _handles = self._backend(
            create_error=RuntimeError("no tab"),
        )
        with self.assertRaises(RuntimeError):
            await backend.fetch(None, _request())
        self.assertEqual(creates, [{}])
        self.assertEqual(metrics.browser_acquire_seconds.count, 1)
        self.assertEqual(metrics.requests_active, 0)
        self.assertEqual(metrics.request_duration_seconds.count, 1)
        await backend.aclose()


class DefaultBrowserShapeTests(_MetricsFixture):
    """The default browser fetch keeps its legacy result shape while counting one request."""

    async def test_browser_mode_keeps_its_legacy_result_shape(self) -> None:
        backend, metrics, _creates, _groups, _clients, _handles = self._backend()
        result = await backend.fetch(None, _request())
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)
        self.assertEqual(result.url, "https://example.com/")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers, {"content-type": "text/html"})
        self.assertEqual(result.response, _PLAIN_BODY)
        self.assertEqual(result.cookies, [])
        self.assertEqual(result.user_agent, "Mozilla/5.0 (Browser)")
        self.assertEqual(metrics.requests_total, 1)
        self.assertEqual(metrics.challenge_detected_total, 0)
        await backend.aclose()
