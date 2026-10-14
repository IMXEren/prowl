"""Backend request-cookie bridge: a raw ``Cookie`` header resolved against the native context.

The bridge turns a caller's ``Cookie`` header into an update on the native context the request
actually runs in, once, before the request's own cookie changes can be overwritten. These tests
drive a real :class:`BrowserContextManager` and a PW-spec native context double, so the handle
contract is the production one. No browser is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlsplit

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext as PWBrowserContext
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import StorageState
from pydoll.browser.tab import Tab

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.driver.contexts import BrowserContextHandle, BrowserContextManager
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle.startup import BrowserLifecycle
from prowl.browser.page_handler import PageResponse
from prowl.browser.proxy.cookies import CookieHeaderError
from prowl.service import backend as backend_module
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, BrowserBackend, FetchRequest, FetchResult
from prowl.service.http_transport import HttpResult
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, CMD_REQUEST_GET, ExecutionMode, FetchCommand
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE, SessionMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright._impl._api_structures import Cookie, SetCookieParam

_EGRESS = "decodo"
_EGRESS_URL = "socks5://127.0.0.1:10001"
_URL = "https://example.test/app"
_MESSAGE = "request cookie header cannot be applied safely"
_CAPTCHA_BODY = '<html><body><div class="g-recaptcha" data-sitekey="k"></div></body></html>'


class _NativeContext:
    """A PW-spec context double recording every cookie read and write the bridge makes."""

    def __init__(self, cookies: list[Cookie] | None = None) -> None:
        self.raw: list[Cookie] = [cookie.copy() for cookie in cookies or []]
        #: One entry per ``cookies()`` call: the scoping URLs, or ``None`` for an unscoped read.
        self.reads: list[list[str] | None] = []
        #: One entry per ``add_cookies()`` call: the updates the caller wrote to the context.
        self.added: list[list[SetCookieParam]] = []
        browser = Mock(spec=PWBrowser)
        browser.is_connected.return_value = True
        self.context = Mock(spec=PWBrowserContext)
        self.context.browser = browser
        self.context.pages = []
        self.context.is_closed.return_value = False
        self.context.close = AsyncMock(side_effect=self._close)
        self.context.cookies = AsyncMock(side_effect=self._read)
        self.context.add_cookies = AsyncMock(side_effect=self._write)

    def _close(self) -> None:
        self.context.is_closed.return_value = True

    async def _read(self, urls: list[str] | None = None) -> list[Cookie]:
        self.reads.append(urls)
        if urls is None:
            return [cookie.copy() for cookie in self.raw]
        path = urlsplit(urls[0]).path or "/"
        return [
            cookie.copy()
            for cookie in self.raw
            if path == cookie.get("path") or path.startswith(cookie.get("path", "/").rstrip("/") + "/")
        ]

    async def _write(self, updates: list[SetCookieParam]) -> None:
        self.added.append([update.copy() for update in updates])
        for update in updates:
            value = update.get("value")
            if value is None:
                msg = "cookie update must include its value"
                raise AssertionError(msg)
            for cookie in self.raw:
                if (cookie.get("name"), cookie.get("domain"), cookie.get("path")) == (
                    update.get("name"),
                    update.get("domain"),
                    update.get("path"),
                ):
                    cookie["value"] = value
                    break
            else:
                parsed = urlsplit(update.get("url") or _URL)
                expires = update.get("expires")
                secure = update.get("secure")
                self.raw.append(
                    {
                        "name": update.get("name") or "",
                        "value": value,
                        "domain": update.get("domain") or parsed.hostname or "",
                        "path": update.get("path") or parsed.path.rpartition("/")[0] + "/",
                        "expires": expires if expires is not None else -1,
                        "httpOnly": update.get("httpOnly") or False,
                        "secure": secure if secure is not None else parsed.scheme == "https",
                        "sameSite": update.get("sameSite") or "Lax",
                    }
                )


class _ProfileReader:
    """The shared pydoll connection double the profile cookie read goes through."""

    async def get_cookies(self) -> list[dict[str, object]]:
        return []


class _StubSite:
    """A PageHandler double returning one configurable source and recording every navigation."""

    def __init__(self, source: PageResponse) -> None:
        self.source = source
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []

    async def get(self, url: str, timeout: int, **kwargs: object) -> PageResponse:
        self.get_calls.append(url)
        return self.source

    async def post(self, url: str, timeout: int, **kwargs: object) -> PageResponse:
        self.post_calls.append(url)
        return self.source


class _StubIdentity:
    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent


class _StubHttpClient:
    """An injectable identity-matched HTTP client double."""

    def __init__(self, result: HttpResult, on_fetch: Callable[[], None] | None = None) -> None:
        self.result = result
        self.on_fetch = on_fetch
        self.identity = _StubIdentity("Mozilla/5.0 (HTTP)")
        self.calls: list[str] = []

    async def fetch(self, url: str, **kwargs: object) -> HttpResult:
        self.calls.append(url)
        if self.on_fetch is not None:
            # Stand in for the HTTP transport mirroring the response's cookie change.
            self.on_fetch()
        return self.result


class _StubClients:
    """An injectable ``BrowserHttpClients`` double holding one client."""

    def __init__(self, client: _StubHttpClient) -> None:
        self.client_instance = client
        self.client_calls: list[object] = []

    async def client(self, context: object, proxy: str | None) -> _StubHttpClient:
        self.client_calls.append(context)
        return self.client_instance

    async def close_context(self, context: object) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _Identity:
    """One browser identity: a real context manager, its native doubles, and its activity log."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.manager = BrowserContextManager()
        self.profile = _ProfileReader()
        self.shared: _NativeContext | None = None
        self.isolated: dict[str, _NativeContext] = {}
        self.isolated_seed: list[Cookie] = []
        self.get_context_calls: list[str | None] = []
        self.closed_contexts: list[tuple[str, bool]] = []
        self.start_calls = 0
        self.shutdown_calls = 0
        self.groups: list[TabGroup] = []
        self.acquires: list[str] = []
        self.releases: list[str] = []

    def bind_shared(self, cookies: list[Cookie] | None = None) -> _NativeContext:
        native = _NativeContext(cookies)
        self.shared = native
        self.manager.bind_shared(native.context)
        return native

    def native(self, session_id: str | None) -> _NativeContext:
        native = self.shared if session_id is None else self.isolated.get(session_id)
        if native is None:
            msg = f"no native context was created for {session_id!r}"
            raise AssertionError(msg)
        return native

    async def get_context(
        self,
        session_id: str | None = None,
        storage_state: StorageState | None = None,
    ) -> BrowserContextHandle:
        self.get_context_calls.append(session_id)
        if session_id is None:
            handle = self.manager.shared()
            if handle is None:
                msg = "the shared context was never bound"
                raise AssertionError(msg)
            return handle

        async def _make() -> PWBrowserContext:
            native = _NativeContext(self.isolated_seed)
            self.isolated[session_id] = native
            return native.context

        return await self.manager.get_or_create(session_id, _make)

    def create_group(self, context: BrowserContextHandle | None) -> TabGroup:
        group = Mock(spec=TabGroup)
        group.context = context
        group.pd.return_value = self.profile
        group.quit = AsyncMock()

        async def _tab() -> Tab:
            return Mock(spec=Tab)

        group.ptab = _tab()
        self.groups.append(group)
        return group


def _stub_browser(identity: _Identity) -> type[Browser]:
    """Build a real Browser subclass whose contexts are this identity's manager handles."""
    runtime = BrowserRuntimeState(max_groups=2)
    runtime.contexts = identity.manager

    class StubBrowser(Browser):
        _runtime = runtime
        _lifecycle = BrowserLifecycle(runtime)

        @classmethod
        async def start(cls) -> None:
            identity.start_calls += 1

        @classmethod
        async def get_context(
            cls,
            session_id: str | None = None,
            *,
            storage_state: StorageState | None = None,
        ) -> BrowserContextHandle:
            return await identity.get_context(session_id, storage_state)

        @classmethod
        async def close_context(cls, session_id: str, *, evicted: bool = False) -> None:
            identity.closed_contexts.append((session_id, evicted))
            await identity.manager.close_isolated(session_id, evicted=evicted)

        @classmethod
        async def create(cls, context: BrowserContextHandle | None = None) -> TabGroup:
            return identity.create_group(context)

        @classmethod
        async def shutdown(cls) -> None:
            identity.shutdown_calls += 1

    return StubBrowser


def _native_cookie(name: str, value: str, *, path: str = "/app") -> Cookie:
    return {
        "name": name,
        "value": value,
        "domain": ".example.test",
        "path": path,
        "expires": 2100000000.5,
        "httpOnly": True,
        "secure": True,
        "sameSite": "Strict",
    }


def _ok_source(*, status: int = 200, body: str = "<html><body>ok</body></html>") -> PageResponse:
    return PageResponse(
        source=body,
        status_code=status,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url=_URL,
    )


def _http_result(*, status: int = 200, body: str = "<html><body>ok</body></html>") -> HttpResult:
    return HttpResult(
        url=_URL,
        status_code=status,
        headers={"content-type": "text/html"},
        body=body,
        cookies=[],
        set_cookie_headers=(),
    )


def _fetch_request(
    *,
    mode: ExecutionMode = BROWSER_MODE,
    session_id: str | None = None,
    session_mode: SessionMode = SHARED_MODE,
    egress: str = "default",
    cookie_header: str | None = None,
) -> FetchRequest:
    return replace(
        FetchRequest(_URL),
        mode=mode,
        session_id=session_id,
        session_mode=session_mode,
        egress=egress,
        cookie_header=cookie_header,
    )


@dataclass(slots=True)
class _Bridge:
    backend: BrowserBackend
    default: _Identity
    egress: _Identity
    site: _StubSite
    clients: _StubClients


class _BridgeFixture(IsolatedAsyncioTestCase):
    """Builds a backend whose browser classes are real subclasses over real context managers."""

    def _backend(
        self,
        *,
        http_client: _StubHttpClient | None = None,
        default_cookies: list[Cookie] | None = None,
        egress_cookies: list[Cookie] | None = None,
        browser_source: PageResponse | None = None,
    ) -> _Bridge:
        backend = BrowserBackend(egresses={_EGRESS: _EGRESS_URL}, egress_idle_seconds=60.0)
        default = _Identity("default")
        egress = _Identity(_EGRESS)
        default.bind_shared(default_cookies)
        egress.bind_shared(egress_cookies)
        site = _StubSite(browser_source or _ok_source())
        clients = _StubClients(http_client or _StubHttpClient(_http_result()))
        backend._http = Mock(spec=backend_module.BrowserHttpClients, wraps=clients)

        original_acquire = backend._pool.acquire
        original_release = backend._pool.release

        async def _acquire(name: str) -> type[Browser]:
            egress.acquires.append(name)
            return await original_acquire(name)

        async def _release(name: str) -> None:
            egress.releases.append(name)
            await original_release(name)

        patchers = [
            patch.object(backend_module, "Browser", _stub_browser(default)),
            patch(
                "prowl.browser.proxy.egress.create_egress_browser", side_effect=lambda **_kwargs: _stub_browser(egress)
            ),
            patch.object(backend._pool, "acquire", new=_acquire),
            patch.object(backend._pool, "release", new=_release),
            patch.object(backend_module, "resolve_page_handler", lambda _group, _url: site),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return _Bridge(backend=backend, default=default, egress=egress, site=site, clients=clients)


class IsolatedBridgeTests(_BridgeFixture):
    """An isolated fetch resolves the header against its own context, URL-scoped and attribute-preserving."""

    async def test_the_isolated_context_gets_one_url_scoped_read_and_a_full_update(self) -> None:
        fixture = self._backend()
        fixture.default.isolated_seed = [_native_cookie("sid", "old")]

        result = await fixture.backend.fetch(
            "s1",
            _fetch_request(
                mode=BROWSER_MODE,
                session_id="s1",
                session_mode=ISOLATED_MODE,
                cookie_header="sid=new",
            ),
        )

        native = fixture.default.native("s1")
        self.assertEqual(native.reads.count([_URL]), 1)
        self.assertEqual(len(native.added), 1)
        update = native.added[0][0]
        self.assertEqual(
            update,
            {
                "name": "sid",
                "value": "new",
                "domain": ".example.test",
                "path": "/app",
                "expires": 2100000000.5,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Strict",
            },
        )
        self.assertNotIn("url", update)
        self.assertIs(fixture.default.groups[0].context, fixture.default.manager.isolated("s1"))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.cookies[0]["value"], "new")
        await fixture.backend.close_session("s1")
        await fixture.backend.aclose()


class SharedBridgeTests(_BridgeFixture):
    """A legacy shared browser request only resolves a handle when a header is present."""

    async def test_a_shared_header_uses_the_bound_manager_handle_once(self) -> None:
        fixture = self._backend(default_cookies=[_native_cookie("sid", "old")])

        await fixture.backend.fetch(None, _fetch_request(mode=BROWSER_MODE, cookie_header="sid=new"))

        native = fixture.default.native(None)
        self.assertEqual(fixture.default.get_context_calls, [None])
        self.assertEqual(native.reads, [[_URL], None])
        self.assertEqual(len(native.added), 1)
        self.assertEqual(native.added[0][0].get("value"), "new")
        self.assertIs(fixture.default.groups[0].context, fixture.default.manager.shared())
        await fixture.backend.aclose()

    async def test_a_headerless_shared_request_keeps_the_legacy_path(self) -> None:
        fixture = self._backend(default_cookies=[_native_cookie("sid", "old")])

        await fixture.backend.fetch(None, _fetch_request(mode=BROWSER_MODE))

        native = fixture.default.native(None)
        self.assertEqual(fixture.default.get_context_calls, [])
        self.assertEqual(native.reads, [])
        self.assertEqual(native.added, [])
        self.assertIsNone(fixture.default.groups[0].context)
        await fixture.backend.aclose()

    async def test_an_unchanged_header_writes_nothing_and_an_empty_header_reads_nothing(self) -> None:
        fixture = self._backend(default_cookies=[_native_cookie("sid", "old")])

        await fixture.backend.fetch(None, _fetch_request(mode=BROWSER_MODE, cookie_header="sid=old"))
        native = fixture.default.native(None)
        self.assertEqual(native.reads, [[_URL], None])
        self.assertEqual(native.added, [])

        await fixture.backend.fetch(None, _fetch_request(mode=BROWSER_MODE, cookie_header=""))
        self.assertEqual(native.reads, [[_URL], None])
        self.assertEqual(native.added, [])
        await fixture.backend.aclose()


class AutoEscalationBridgeTests(_BridgeFixture):
    """An auto route applies the header once and never overwrites the HTTP response's cookie changes."""

    async def test_the_header_is_applied_once_across_the_escalation(self) -> None:
        client = _StubHttpClient(_http_result(status=403, body=_CAPTCHA_BODY))
        fixture = self._backend(http_client=client, default_cookies=[_native_cookie("sid", "old")])
        native = fixture.default.native(None)

        def _mirror_response_cookie() -> None:
            native.raw = [_native_cookie("sid", "from-http")]

        client.on_fetch = _mirror_response_cookie
        result = await fixture.backend.fetch(None, _fetch_request(mode=AUTO_MODE, cookie_header="sid=new"))

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(fixture.default.groups), 1)
        self.assertEqual(native.reads, [[_URL], None])
        self.assertEqual(len(native.added), 1)
        self.assertEqual(native.added[0][0].get("value"), "new")
        self.assertEqual(native.raw[0].get("value"), "from-http")
        self.assertEqual(result.cookies[0]["value"], "from-http")
        await fixture.backend.aclose()


class AmbiguousHeaderTests(_BridgeFixture):
    """An ambiguous header fails before the group and the wire, preserving each route's ownership."""

    async def test_an_ambiguous_header_keeps_an_isolated_session_and_its_claim(self) -> None:
        fixture = self._backend()
        fixture.egress.isolated_seed = [
            _native_cookie("sid", "one", path="/app"),
            _native_cookie("sid", "two", path="/"),
        ]

        with self.assertRaises(CookieHeaderError) as caught:
            await fixture.backend.fetch(
                "s1",
                _fetch_request(
                    mode=BROWSER_MODE,
                    egress=_EGRESS,
                    session_id="s1",
                    session_mode=ISOLATED_MODE,
                    cookie_header="sid=one",
                ),
            )

        self.assertEqual(str(caught.exception), _MESSAGE)
        self.assertEqual(fixture.egress.groups, [])
        self.assertEqual(fixture.clients.client_calls, [])
        self.assertEqual(fixture.egress.acquires, [_EGRESS])
        self.assertTrue(fixture.backend._isolated["s1"].claim_held)
        await fixture.backend.close_session("s1")
        await fixture.backend.aclose()

    async def test_an_ambiguous_header_releases_a_shared_claim(self) -> None:
        fixture = self._backend(
            egress_cookies=[
                _native_cookie("sid", "one", path="/app"),
                _native_cookie("sid", "two", path="/"),
            ],
        )

        with self.assertRaises(CookieHeaderError) as caught:
            await fixture.backend.fetch(
                None,
                _fetch_request(mode=BROWSER_MODE, egress=_EGRESS, cookie_header="sid=one"),
            )

        self.assertEqual(str(caught.exception), _MESSAGE)
        self.assertEqual(fixture.egress.groups, [])
        self.assertEqual(fixture.clients.client_calls, [])
        self.assertEqual(fixture.egress.acquires, [_EGRESS])
        self.assertEqual(fixture.egress.releases, [_EGRESS])
        await fixture.backend.aclose()


class NativeWriteFailureTests(_BridgeFixture):
    """A native write failure is caller-safe; a cancellation is not hidden as validation."""

    async def test_a_native_write_failure_is_generic_and_echoes_no_native_detail(self) -> None:
        fixture = self._backend(default_cookies=[_native_cookie("sid", "old")])
        native = fixture.default.native(None)
        native.context.add_cookies = AsyncMock(side_effect=PlaywrightError("rejected SECRET-VALUE"))

        with self.assertRaises(CookieHeaderError) as caught:
            await fixture.backend.fetch(None, _fetch_request(mode=BROWSER_MODE, cookie_header="sid=new"))

        self.assertEqual(str(caught.exception), _MESSAGE)
        self.assertNotIn("SECRET-VALUE", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(fixture.default.groups, [])
        await fixture.backend.aclose()

    async def test_a_native_cancellation_propagates_and_still_releases_the_claim(self) -> None:
        fixture = self._backend(egress_cookies=[_native_cookie("sid", "old")])
        native = fixture.egress.native(None)
        native.context.add_cookies = AsyncMock(side_effect=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await fixture.backend.fetch(
                None,
                _fetch_request(mode=BROWSER_MODE, egress=_EGRESS, cookie_header="sid=new"),
            )

        self.assertEqual(fixture.egress.groups, [])
        self.assertEqual(fixture.egress.releases, [_EGRESS])
        await fixture.backend.aclose()


class ServiceCookieHeaderTests(IsolatedAsyncioTestCase):
    """``Service.fetch`` carries the raw header beside the wire headers, never inside them."""

    def _service(self) -> tuple[Service, list[FetchRequest]]:
        captured: list[FetchRequest] = []
        backend = Mock(spec=Backend)

        async def _fetch(_session_id: str | None, request: FetchRequest) -> FetchResult:
            captured.append(request)
            return FetchResult(
                url=request.url,
                status_code=200,
                headers={},
                response="ok",
                cookies=[],
                user_agent="ua",
            )

        backend.fetch = _fetch
        return Service(ServiceConfig(), backend), captured

    async def test_the_raw_header_travels_separately_from_custom_headers(self) -> None:
        service, captured = self._service()
        command = FetchCommand(
            cmd=CMD_REQUEST_GET,
            url=_URL,
            timeout_ms=60000,
            session="s",
            headers={"authorization": "Bearer x"},
        )

        await service.fetch(command, cookie_header="sid=abc")

        self.assertEqual(captured[0].cookie_header, "sid=abc")
        self.assertEqual(captured[0].headers, {"authorization": "Bearer x"})
        self.assertEqual(captured[0].session_id, "s")

    async def test_an_omitted_header_defaults_to_none(self) -> None:
        service, captured = self._service()
        command = FetchCommand(cmd=CMD_REQUEST_GET, url=_URL, timeout_ms=60000)

        await service.fetch(command)

        self.assertIsNone(captured[0].cookie_header)
