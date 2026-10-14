"""Backend execution-mode routing regressions with stub browsers and injectable HTTP clients.

The default ``browser`` mode must keep its legacy result shape and its fake-owner contract. The
``http`` and ``auto`` modes resolve one owner and context per route, hold a single named egress
claim across an HTTP attempt and a browser escalation, seed supplied cookies once, and classify
the final response honestly. No browser is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.browser.page_handler import PageResponse
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest
from prowl.service.classification import CAPTCHA, RATE_LIMITED, SUCCESS, UNKNOWN_BLOCK
from prowl.service.http_transport import (
    HttpNetworkError,
    HttpResult,
    UnsupportedCookieError,
    UnsupportedIdentityError,
)
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from prowl.service.sessions import ISOLATED_MODE

_EGRESS_URL = "socks5://127.0.0.1:10001"
_DEFAULT_PROXY = "socks5://127.0.0.1:10002"
_CAPTCHA_BODY = '<html><body><div class="g-recaptcha" data-sitekey="k"></div></body></html>'
_PLAIN_BODY = "<html><body>Not Found</body></html>"


def _ok_source(*, body: str = "<html><body>ok</body></html>", status: int = 200) -> PageResponse:
    return PageResponse(
        source=body,
        status_code=status,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url="https://example.com/",
    )


def _http_result(
    *,
    status: int = 200,
    body: str = "<html><body>ok</body></html>",
    cookies: list[dict[str, Any]] | None = None,
) -> HttpResult:
    return HttpResult(
        url="https://example.com/",
        status_code=status,
        headers={"content-type": "text/html"},
        body=body,
        cookies=cookies or [],
        set_cookie_headers=(),
    )


def _request(**fields: Any) -> FetchRequest:
    values: dict[str, Any] = {"url": "https://example.com/"}
    values.update(fields)
    return FetchRequest(**values)


async def _wait_until(predicate: Any, *, attempts: int = 400) -> None:
    """Yield until *predicate* holds, failing rather than hanging when it never does."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    msg = "the expected state was not reached in time"
    raise AssertionError(msg)


class _FakeContext:
    """A native context double holding its own cookie writes."""

    def __init__(self) -> None:
        self.added: list[list[dict[str, Any]]] = []
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def cookies(self) -> list[dict[str, Any]]:
        return []

    async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.added.append([dict(cookie) for cookie in cookies])

    async def close(self) -> None:
        self.closed = True


class _FakeHandle:
    """The context handle the backend resolves and passes to ``create``."""

    def __init__(self, session_id: str | None, context: _FakeContext) -> None:
        self.session_id = session_id
        self.context = context


class _StubTab:
    def __init__(self) -> None:
        self.cookies: list[dict[str, Any]] | None = None

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.cookies = list(cookies)


class _StubGroup:
    """A tab group double recording its context and closes."""

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


class _StubSite:
    """A PageHandler double returning one configurable source."""

    def __init__(self, source: PageResponse | None = None) -> None:
        self.source = source or _ok_source()
        self.get_calls: list[tuple[str, int, dict[str, Any]]] = []
        self.post_calls: list[tuple[str, int, dict[str, Any]]] = []

    async def get(self, url: str, timeout: int, **kwargs: Any) -> PageResponse:
        self.get_calls.append((url, timeout, kwargs))
        return self.source

    async def post(self, url: str, timeout: int, **kwargs: Any) -> PageResponse:
        self.post_calls.append((url, timeout, kwargs))
        return self.source


class _Recorder:
    """Records context, group, claim and shutdown activity for a stub browser class."""

    def __init__(self) -> None:
        self.contexts: dict[str | None, _FakeHandle] = {}
        self.get_context_calls: list[str | None] = []
        self.closed_contexts: list[str] = []
        self.groups: list[_StubGroup] = []
        self.started: list[str] = []
        self.shutdown_log: list[str] = []
        self.profile_cookies: list[dict[str, Any]] = []
        self.events: list[tuple[str, Any]] = []

    async def get_context(self, session_id: str | None) -> _FakeHandle:
        self.get_context_calls.append(session_id)
        handle = self.contexts.get(session_id)
        if handle is None:
            handle = _FakeHandle(session_id, _FakeContext())
            self.contexts[session_id] = handle
        return handle

    async def close_context(self, session_id: str) -> None:
        self.events.append(("native-close", session_id))
        self.closed_contexts.append(session_id)
        self.contexts.pop(session_id, None)

    async def create(self, context: Any = None) -> _StubGroup:
        group = _StubGroup(context, self)
        self.groups.append(group)
        return group

    async def get_cookies(self) -> list[dict[str, Any]]:
        return list(self.profile_cookies)


class _StubIdentity:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent


class _StubHttpClient:
    """An injectable HTTP client double."""

    def __init__(
        self,
        *,
        result: HttpResult | None = None,
        error: BaseException | None = None,
        user_agent: str = "Mozilla/5.0 (HTTP)",
    ) -> None:
        self.result = result
        self.error = error
        self.identity = _StubIdentity(user_agent)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def fetch(self, url: str, **kwargs: Any) -> HttpResult:
        self.calls.append((url, kwargs))
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
        events: list[tuple[str, Any]] | None = None,
    ) -> None:
        self.client_instance = client or _StubHttpClient()
        self.client_error = client_error
        self.client_calls: list[tuple[Any, str | None]] = []
        self.closed_contexts: list[Any] = []
        self.aclose_calls = 0
        self._events = events if events is not None else []

    async def client(self, context: Any, proxy: str | None) -> _StubHttpClient:
        self.client_calls.append((context, proxy))
        if self.client_error is not None:
            raise self.client_error
        return self.client_instance

    async def close_context(self, context: Any) -> None:
        self._events.append(("http-close", context))
        self.closed_contexts.append(context)

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _RoutingFixture(IsolatedAsyncioTestCase):
    """Builds a ``BrowserBackend`` whose browsers are stubs and whose HTTP clients are injected."""

    def _backend(
        self,
        *,
        http_client: _StubHttpClient | None = None,
        client_error: BaseException | None = None,
        browser_source: PageResponse | None = None,
        default_proxy: str | None = None,
        egress_idle_seconds: float = 60.0,
    ) -> tuple[BrowserBackend, _Recorder, _StubClients, _StubSite]:
        recorder = _Recorder()
        proxies: dict[str, str | None] = {"default": default_proxy, "decodo": _EGRESS_URL}

        def _browser_class(name: str) -> type[Any]:
            class FakeBrowser:
                _lifecycle = SimpleNamespace(proxy_url=proxies.get(name))

                @classmethod
                def proxy_url(cls) -> str | None:
                    return cls._lifecycle.proxy_url

                @classmethod
                async def start(cls) -> None:
                    recorder.started.append(name)

                @classmethod
                async def create(cls, context: Any = None) -> _StubGroup:
                    return await recorder.create(context)

                @classmethod
                async def get_context(cls, session_id: str | None) -> _FakeHandle:
                    return await recorder.get_context(session_id)

                @classmethod
                async def close_context(cls, session_id: str) -> None:
                    await recorder.close_context(session_id)

                @classmethod
                async def shutdown(cls) -> None:
                    recorder.shutdown_log.append(name)

                @classmethod
                def pd(cls) -> Any:
                    return recorder

            return FakeBrowser

        def _create_browser(**kwargs: Any) -> type[Any]:
            return _browser_class(str(kwargs["name"]))

        backend = BrowserBackend(egresses={"decodo": _EGRESS_URL}, egress_idle_seconds=egress_idle_seconds)
        site = _StubSite(browser_source)
        clients = _StubClients(http_client, client_error=client_error, events=recorder.events)
        backend._http = Mock(spec=backend_module.BrowserHttpClients, wraps=clients)
        for patcher in (
            patch.object(backend_module, "Browser", _browser_class("default")),
            patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=_create_browser),
            patch.object(backend_module, "resolve_page_handler", lambda _group, _url: site),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, recorder, clients, site


def _counting(backend: BrowserBackend) -> tuple[list[str], list[str]]:
    """Count the named pool's acquire and release calls in order."""
    acquires: list[str] = []
    releases: list[str] = []
    original_acquire = backend._pool.acquire
    original_release = backend._pool.release

    async def acquire(name: str) -> type[Any]:
        acquires.append(name)
        return await original_acquire(name)

    async def release(name: str) -> None:
        releases.append(name)
        await original_release(name)

    backend._pool.acquire = acquire
    backend._pool.release = release
    return acquires, releases


class BrowserModeBypassTests(_RoutingFixture):
    """The default browser mode keeps its legacy shape and never reaches the HTTP path."""

    async def test_browser_mode_never_uses_the_http_path(self) -> None:
        backend, recorder, clients, _site = self._backend()
        result = await backend.fetch(None, _request(mode=BROWSER_MODE))
        self.assertEqual(clients.client_calls, [])
        self.assertEqual(recorder.get_context_calls, [])
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)
        self.assertEqual(len(recorder.groups), 1)
        self.assertEqual(recorder.groups[0].quit_calls, 1)
        await backend.aclose()

    async def test_http_mode_makes_no_group_and_does_not_steal_a_tab(self) -> None:
        backend, recorder, _clients, _site = self._backend(http_client=_StubHttpClient(result=_http_result()))
        with patch.object(backend, "_steal_least_recent", new=AsyncMock()) as steal:
            result = await backend.fetch(None, _request(mode=HTTP_MODE))
        steal.assert_not_awaited()
        self.assertEqual(recorder.groups, [])
        self.assertEqual(result.mode, HTTP_MODE)
        self.assertEqual(result.status_code, 200)
        assert result.classification is not None
        self.assertEqual(result.classification.category, SUCCESS)
        await backend.aclose()

    async def test_http_mode_reports_context_cookies_and_the_client_identity(self) -> None:
        cookies = [{"name": "a", "value": "b", "domain": ".example.com", "path": "/", "expires": -1}]
        client = _StubHttpClient(result=_http_result(cookies=cookies), user_agent="UA-HTTP")
        backend, _recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=HTTP_MODE))
        self.assertEqual(result.user_agent, "UA-HTTP")
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["a"])
        await backend.aclose()


class HttpModeGuardsTests(_RoutingFixture):
    """An explicit http fetch fails clearly before the network when it cannot run."""

    async def test_http_mode_forwards_headers_with_their_scope(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(
            None,
            _request(mode=HTTP_MODE, headers={"authorization": "Bearer x"}, header_scope="origin"),
        )
        self.assertEqual(result.mode, HTTP_MODE)
        self.assertEqual(client.calls[0][1]["headers"], {"authorization": "Bearer x"})
        self.assertEqual(client.calls[0][1]["header_scope"], "origin")
        self.assertEqual(recorder.groups, [])
        await backend.aclose()

    async def test_http_mode_refuses_a_partitioned_cookie(self) -> None:
        backend, _recorder, clients, _site = self._backend(http_client=_StubHttpClient(result=_http_result()))
        partitioned = [{"name": "a", "value": "b", "domain": ".example.com", "partitionKey": "https://x"}]
        with self.assertRaises(UnsupportedCookieError):
            await backend.fetch(None, _request(mode=HTTP_MODE, cookies=partitioned))
        self.assertEqual(clients.client_calls, [])
        await backend.aclose()


class AutoRoutingTests(_RoutingFixture):
    """Auto tries HTTP and escalates for exactly the classifications and refusals that need one."""

    async def test_auto_challenge_escalates_and_classifies_the_browser_result(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, recorder, clients, _site = self._backend(
            http_client=client,
            browser_source=_ok_source(status=403, body=_CAPTCHA_BODY),
        )
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(len(clients.client_calls), 1)
        self.assertEqual(len(recorder.groups), 1)
        self.assertEqual(result.mode, BROWSER_MODE)
        assert result.classification is not None
        self.assertEqual(result.classification.category, CAPTCHA)
        await backend.aclose()

    async def test_auto_ordinary_refusal_does_not_escalate(self) -> None:
        client = _StubHttpClient(result=_http_result(status=404, body=_PLAIN_BODY))
        backend, recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(recorder.groups, [])
        self.assertEqual(result.mode, HTTP_MODE)
        assert result.classification is not None
        self.assertEqual(result.classification.category, UNKNOWN_BLOCK)
        self.assertFalse(result.classification.browser_required)
        await backend.aclose()

    async def test_auto_rate_limit_does_not_escalate(self) -> None:
        client = _StubHttpClient(result=_http_result(status=429, body="slow down"))
        backend, recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(recorder.groups, [])
        assert result.classification is not None
        self.assertEqual(result.classification.category, RATE_LIMITED)
        await backend.aclose()

    async def test_auto_network_error_does_not_escalate(self) -> None:
        client = _StubHttpClient(error=HttpNetworkError("no response"))
        backend, recorder, _clients, _site = self._backend(http_client=client)
        with self.assertRaises(HttpNetworkError):
            await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(recorder.groups, [])
        await backend.aclose()

    async def test_auto_success_answers_over_http(self) -> None:
        client = _StubHttpClient(result=_http_result(status=200))
        backend, recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(recorder.groups, [])
        self.assertEqual(result.mode, HTTP_MODE)
        await backend.aclose()

    async def test_http_mode_never_escalates_on_a_challenge(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, recorder, _clients, _site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=HTTP_MODE))
        self.assertEqual(recorder.groups, [])
        self.assertEqual(result.mode, HTTP_MODE)
        assert result.classification is not None
        self.assertEqual(result.classification.category, CAPTCHA)
        await backend.aclose()

    async def test_auto_unsupported_identity_escalates(self) -> None:
        backend, recorder, clients, _site = self._backend(client_error=UnsupportedIdentityError("no profile"))
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(len(clients.client_calls), 1)
        self.assertEqual(len(recorder.groups), 1)
        self.assertEqual(result.mode, BROWSER_MODE)
        await backend.aclose()

    async def test_auto_unsupported_cookie_escalates(self) -> None:
        backend, recorder, _clients, _site = self._backend(client_error=UnsupportedCookieError("partitioned"))
        result = await backend.fetch(None, _request(mode=AUTO_MODE))
        self.assertEqual(len(recorder.groups), 1)
        self.assertEqual(result.mode, BROWSER_MODE)
        await backend.aclose()

    async def test_auto_post_goes_straight_to_the_browser(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, recorder, clients, site = self._backend(http_client=client)
        result = await backend.fetch(None, _request(mode=AUTO_MODE, method="POST", post_data="q=1"))
        self.assertEqual(clients.client_calls, [])
        self.assertEqual(len(site.get_calls), 0)
        self.assertEqual(len(site.post_calls), 1)
        self.assertEqual(len(recorder.groups), 1)
        self.assertEqual(result.mode, BROWSER_MODE)
        await backend.aclose()

    async def test_auto_with_headers_uses_scoped_http(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, site = self._backend(http_client=client)
        result = await backend.fetch(
            None,
            _request(mode=AUTO_MODE, headers={"authorization": "Bearer x"}, header_scope="document"),
        )
        self.assertEqual(len(clients.client_calls), 1)
        self.assertEqual(site.get_calls, [])
        self.assertEqual(client.calls[0][1]["headers"], {"authorization": "Bearer x"})
        self.assertEqual(client.calls[0][1]["header_scope"], "document")
        self.assertEqual(result.mode, HTTP_MODE)
        await backend.aclose()


class SharedRouteOwnershipTests(_RoutingFixture):
    """A shared route holds one claim and one context across HTTP and an escalation."""

    async def test_one_claim_and_one_context_across_the_fallback(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, recorder, clients, _site = self._backend(
            http_client=client,
            browser_source=_ok_source(status=403, body=_CAPTCHA_BODY),
        )
        acquires, releases = _counting(backend)
        result = await backend.fetch(None, _request(mode=AUTO_MODE, egress="decodo"))
        self.assertEqual(acquires, ["decodo"])
        self.assertEqual(releases, ["decodo"])
        self.assertEqual(len(clients.client_calls), 1)
        self.assertEqual(recorder.get_context_calls, [None])
        self.assertEqual(len(recorder.groups), 1)
        self.assertIs(recorder.groups[0].context, recorder.contexts[None])
        self.assertIs(clients.client_calls[0][0], recorder.contexts[None].context)
        self.assertEqual(result.mode, BROWSER_MODE)
        await backend.aclose()

    async def test_auto_seeds_the_context_once_and_does_not_reapply_on_fallback(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, recorder, _clients, _site = self._backend(
            http_client=client,
            browser_source=_ok_source(status=403, body=_CAPTCHA_BODY),
        )
        cookies = [{"name": "a", "value": "b", "domain": ".example.com", "path": "/"}]
        await backend.fetch(None, _request(mode=AUTO_MODE, cookies=cookies))
        context = recorder.contexts[None].context
        self.assertEqual(len(context.added), 1)
        self.assertEqual(context.added[0][0]["name"], "a")
        self.assertIsNone(recorder.groups[0].tab.cookies)
        await backend.aclose()


class IsolatedRouteTests(_RoutingFixture):
    """An isolated route reuses the session's context and keeps its lifetime claim."""

    async def test_isolated_auto_keeps_the_lifetime_claim_and_context(self) -> None:
        client = _StubHttpClient(result=_http_result(status=403, body=_CAPTCHA_BODY))
        backend, recorder, clients, _site = self._backend(
            http_client=client,
            browser_source=_ok_source(status=403, body=_CAPTCHA_BODY),
        )
        result = await backend.fetch(
            None,
            _request(mode=AUTO_MODE, egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
        )
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertTrue(backend._isolated["s1"].claim_held)
        self.assertEqual(recorder.closed_contexts, [])
        self.assertEqual(len(recorder.groups), 1)
        self.assertIs(recorder.groups[0].context, recorder.contexts["s1"])
        self.assertIs(clients.client_calls[0][0], recorder.contexts["s1"].context)
        self.assertEqual(result.mode, BROWSER_MODE)
        await backend.close_session("s1")
        await backend.aclose()

    async def test_close_session_closes_the_http_client_before_the_context(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, recorder, clients, _site = self._backend(http_client=client)
        await backend.fetch(None, _request(mode=HTTP_MODE, session_id="s1", session_mode=ISOLATED_MODE))
        context = recorder.contexts["s1"].context
        await backend.close_session("s1")
        self.assertEqual(clients.closed_contexts, [context])
        self.assertEqual(recorder.closed_contexts, ["s1"])
        self.assertEqual(recorder.events, [("http-close", context), ("native-close", "s1")])
        await backend.aclose()

    async def test_shared_session_destroy_keeps_shared_clients(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, recorder, clients, _site = self._backend(http_client=client)
        await backend.fetch(None, _request(mode=HTTP_MODE))
        await backend.close_session("shared-id")
        self.assertEqual(clients.closed_contexts, [])
        self.assertEqual(recorder.closed_contexts, [])
        await backend.aclose()
        self.assertEqual(clients.aclose_calls, 1)

    async def test_aclose_drains_the_http_clients(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, _site = self._backend(http_client=client)
        await backend.fetch(None, _request(mode=HTTP_MODE))
        await backend.aclose()
        self.assertEqual(clients.aclose_calls, 1)


class HttpProxyTests(_RoutingFixture):
    """The HTTP client's proxy matches the identity's egress exactly."""

    async def test_default_egress_uses_the_browser_proxy(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, _site = self._backend(http_client=client, default_proxy=_DEFAULT_PROXY)
        await backend.fetch(None, _request(mode=HTTP_MODE))
        self.assertEqual(clients.client_calls[0][1], _DEFAULT_PROXY)
        await backend.aclose()

    async def test_default_egress_without_a_proxy_binds_none(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, _site = self._backend(http_client=client)
        await backend.fetch(None, _request(mode=HTTP_MODE))
        self.assertIsNone(clients.client_calls[0][1])
        await backend.aclose()

    async def test_named_egress_uses_its_configured_proxy(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, _site = self._backend(http_client=client)
        await backend.fetch(None, _request(mode=HTTP_MODE, egress="decodo"))
        self.assertEqual(clients.client_calls[0][1], _EGRESS_URL)
        await backend.aclose()


class RouteCancellationTests(_RoutingFixture):
    """The route's budget bounds the whole route and cancellation releases its transient claim."""

    async def test_cancellation_releases_the_transient_claim(self) -> None:
        class _BlockingClient(_StubHttpClient):
            async def fetch(self, url: str, **kwargs: Any) -> HttpResult:
                self.calls.append((url, kwargs))
                await asyncio.Event().wait()
                return _http_result()

        backend, recorder, clients, _site = self._backend(http_client=_BlockingClient())
        _acquires, releases = _counting(backend)
        task = asyncio.create_task(backend.fetch(None, _request(mode=HTTP_MODE, egress="decodo")))
        await _wait_until(lambda: bool(clients.client_calls))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(releases, ["decodo"])
        self.assertEqual(recorder.groups, [])
        await backend.aclose()

    async def test_the_route_budget_bounds_a_never_finishing_http_stage(self) -> None:
        class _BlockingClient(_StubHttpClient):
            async def fetch(self, url: str, **kwargs: Any) -> HttpResult:
                await asyncio.Event().wait()
                return _http_result()

        backend, recorder, _clients, _site = self._backend(http_client=_BlockingClient())
        with self.assertRaises(TimeoutError):
            await backend.fetch(None, _request(mode=HTTP_MODE, egress="decodo", timeout_seconds=1))
        self.assertEqual(recorder.groups, [])
        await backend.aclose()


class RoutingReviewRegressionTests(_RoutingFixture):
    async def test_explicit_http_post_preserves_the_body(self) -> None:
        client = _StubHttpClient()
        backend, recorder, _, _ = self._backend(http_client=client)
        try:
            await backend.fetch(None, _request(mode=HTTP_MODE, method="POST", post_data="message=hello"))
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.calls[0][1]["content"], "message=hello")
            self.assertEqual(recorder.groups, [])
        finally:
            await backend.aclose()

    async def test_successful_auto_http_does_not_steal_an_interactive_tab(self) -> None:
        backend, recorder, _, _ = self._backend()
        steal = AsyncMock()
        backend._steal_least_recent = steal
        try:
            result = await backend.fetch(None, _request(mode=AUTO_MODE))
            self.assertEqual(result.mode, HTTP_MODE)
            steal.assert_not_awaited()
            self.assertEqual(recorder.groups, [])
        finally:
            await backend.aclose()

    async def test_auto_steals_only_when_it_actually_enters_the_browser(self) -> None:
        client = _StubHttpClient(
            result=HttpResult(
                url="https://example.com/",
                status_code=403,
                headers={"cf-mitigated": "challenge"},
                body="challenge",
                cookies=[],
                set_cookie_headers=(),
            )
        )
        backend, recorder, _, _ = self._backend(http_client=client)
        steal = AsyncMock()
        backend._steal_least_recent = steal
        try:
            result = await backend.fetch(None, _request(mode=AUTO_MODE))
            self.assertEqual(result.mode, BROWSER_MODE)
            steal.assert_awaited_once()
            self.assertEqual(len(recorder.groups), 1)
        finally:
            await backend.aclose()
