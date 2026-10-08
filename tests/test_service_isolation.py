"""Isolated-session regressions: protocol, dispatcher, and backend context ownership.

An opt-in ``sessionMode: isolated`` gives a logical session its own browser context inside the
same browser process. These tests exercise the wire contract, the service's use of the lease's
metadata, and the backend's one-context-per-id ownership, held egress claim, interactive-tab
scoping, and cleanup with stub browsers. No process is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from prowl.browser.page_handler import PageResponse
from prowl.service import backend as backend_module
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import (
    BrowserBackend,
    CookieQuery,
    FetchRequest,
    FetchResult,
    InteractiveOpenResult,
    InteractiveRequest,
    InteractiveTab,
)
from prowl.service.errors import SessionError
from prowl.service.protocol import (
    BrowserOpenCommand,
    CookiesListCommand,
    FetchCommand,
    ProtocolError,
    SessionsCreateCommand,
    parse_request,
)
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE

if TYPE_CHECKING:
    from collections.abc import Callable

_EGRESS_URL = "socks5://127.0.0.1:10001"
_OTHER_URL = "socks5://127.0.0.1:10002"
_PAGE = "<html><head><title>Example Domain</title></head><body>ok</body></html>"


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-test-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-test-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


class SessionModeParsingTests(TestCase):
    """``sessionMode`` is an opt-in selector with a required session id."""

    def test_absent_session_mode_stays_none(self) -> None:
        command = parse_request({"cmd": "sessions.create", "session": "s"})
        assert isinstance(command, SessionsCreateCommand)
        self.assertIsNone(command.session_mode)
        self.assertIsNone(command.proxy)

    def test_create_accepts_isolated_with_a_proxy(self) -> None:
        command = parse_request(
            {"cmd": "sessions.create", "session": "s", "sessionMode": "isolated", "proxy": {"name": "decodo"}},
        )
        assert isinstance(command, SessionsCreateCommand)
        self.assertEqual(command.session_mode, ISOLATED_MODE)
        assert command.proxy is not None
        self.assertEqual(command.proxy.name, "decodo")

    def test_fetch_and_open_accept_a_session_mode(self) -> None:
        fetch = parse_request(
            {"cmd": "request.get", "url": "https://example.com/", "session": "s", "sessionMode": "isolated"}
        )
        assert isinstance(fetch, FetchCommand)
        self.assertEqual(fetch.session_mode, ISOLATED_MODE)
        opened = parse_request(
            {"cmd": "browser.open", "url": "https://example.com/", "session": "s", "sessionMode": "shared"}
        )
        assert isinstance(opened, BrowserOpenCommand)
        self.assertEqual(opened.session_mode, SHARED_MODE)

    def test_an_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "sessions.create", "session": "s", "sessionMode": "banana"})

    def test_isolated_requires_a_session_on_every_implicit_creator(self) -> None:
        payloads = [
            {"cmd": "sessions.create", "sessionMode": "isolated"},
            {"cmd": "request.get", "url": "https://example.com/", "sessionMode": "isolated"},
            {"cmd": "browser.open", "url": "https://example.com/", "sessionMode": "isolated"},
        ]
        for payload in payloads:
            with self.assertRaises(ProtocolError, msg=payload):
                parse_request(payload)

    def test_cookies_list_accepts_an_existing_session_selector(self) -> None:
        command = parse_request({"cmd": "cookies.list", "session": "s"})
        assert isinstance(command, CookiesListCommand)
        self.assertEqual(command.session, "s")

    def test_cookies_list_rejects_a_session_mode(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "cookies.list", "session": "s", "sessionMode": "isolated"})


class FakeBackend:
    """In-memory backend double recording every command it receives."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self.requests: list[tuple[str | None, FetchRequest]] = []
        self.opened: list[InteractiveRequest] = []
        self.queries: list[CookieQuery] = []
        self.closed_sessions: list[str] = []
        self.closed = False
        self.started = False
        self.close_session_error: BaseException | None = None
        self._delay = delay

    async def start(self) -> None:
        self.started = True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)
        if self.close_session_error is not None:
            raise self.close_session_error

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        self.requests.append((session_id, request))
        if self._delay:
            await asyncio.sleep(self._delay)
        return FetchResult(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/html"},
            response=_PAGE,
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        self.opened.append(request)
        tab = InteractiveTab(tab_id="tab-1", url=request.url, title="", status_code=200, egress=request.egress)
        return InteractiveOpenResult(tab=tab)

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        return [] if tab_id is None else [tab_id]

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        return []

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        self.queries.append(query)
        return []

    async def aclose(self) -> None:
        self.closed = True


async def _fetch(service: Service, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"cmd": "request.get", "url": "https://example.com/"}
    payload.update(fields)
    _status, body = await service.handle(payload)
    return body


async def _settle(*, ticks: int = 20) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0.005)


class IsolatedDispatchTests(IsolatedAsyncioTestCase):
    """The service builds each request from the lease's own metadata, not an earlier ensure."""

    def _service(self, **overrides: Any) -> tuple[Service, FakeBackend]:
        backend = FakeBackend()
        return Service(_config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}, **overrides), backend), backend

    async def test_isolated_create_binds_the_named_egress(self) -> None:
        service, _backend = self._service()
        _status, created = await service.handle(
            {"cmd": "sessions.create", "session": "s", "sessionMode": "isolated", "proxy": {"name": "decodo"}},
        )
        self.assertEqual(created["status"], "ok")
        self.assertEqual(created["session"], "s")
        _status, listed = await service.handle({"cmd": "sessions.list"})
        self.assertEqual(listed["sessions"], ["s"])

    async def test_isolated_fetch_carries_the_mode_and_id_to_the_backend(self) -> None:
        service, backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "sessionMode": "isolated"})
        body = await _fetch(service, session="s")
        self.assertEqual(body["status"], "ok")
        session_id, request = backend.requests[0]
        self.assertEqual(session_id, "s")
        self.assertEqual(request.session_mode, ISOLATED_MODE)
        self.assertEqual(request.session_id, "s")
        self.assertEqual(request.egress, "default")

    async def test_a_no_proxy_request_reuses_a_sessions_bound_egress(self) -> None:
        service, backend = self._service()
        await _fetch(service, session="s", proxy={"name": "decodo"})
        body = await _fetch(service, session="s")
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.requests[1][1].egress, "decodo")

    async def test_a_conflicting_proxy_is_rejected(self) -> None:
        service, backend = self._service()
        await _fetch(service, session="s", proxy={"name": "decodo"})
        body = await _fetch(service, session="s", proxy={"name": "warp"})
        self.assertEqual(body["status"], "error")
        self.assertIn("egress", body["message"])
        self.assertEqual(len(backend.requests), 1)

    async def test_cookies_list_reads_the_isolated_session_context(self) -> None:
        service, backend = self._service()
        await service.handle(
            {"cmd": "sessions.create", "session": "s", "sessionMode": "isolated", "proxy": {"name": "decodo"}}
        )
        _status, body = await service.handle({"cmd": "cookies.list", "session": "s"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual(backend.queries[0].session_id, "s")
        self.assertEqual(backend.queries[0].session_mode, ISOLATED_MODE)
        self.assertEqual(backend.queries[0].egress, "decodo")

    async def test_cookies_list_does_not_create_an_unknown_session(self) -> None:
        service, backend = self._service()
        _status, body = await service.handle({"cmd": "cookies.list", "session": "missing"})
        self.assertEqual(body["status"], "error")
        self.assertIn("unknown session", body["message"])
        self.assertEqual(backend.queries, [])
        self.assertEqual(await service.sessions.list_sessions(), [])

    async def test_destroy_routes_cleanup_through_the_registry_callback(self) -> None:
        service, backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "sessionMode": "isolated"})
        _status, body = await service.handle({"cmd": "sessions.destroy", "session": "s"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.closed_sessions, ["s"])

    async def test_the_periodic_sweep_expires_idle_sessions(self) -> None:
        backend = FakeBackend()
        service = Service(
            _config(egresses={"decodo": _EGRESS_URL}),
            backend,
            sweep_interval_seconds=0.01,
        )
        await service.handle({"cmd": "sessions.create", "session": "s", "sessionMode": "isolated"})
        service.sessions._entries["s"].expires_at = 0.0
        service.start_periodic_cleanup()
        try:
            await _settle()
        finally:
            await service.stop_periodic_cleanup()
        self.assertEqual(backend.closed_sessions, ["s"])
        self.assertEqual(await service.sessions.list_sessions(), [])

    async def test_aclose_stops_periodic_work_and_closes_the_backend(self) -> None:
        service, backend = self._service()
        service.start_periodic_cleanup()
        await service.aclose()
        self.assertIsNone(service._sweep_task)
        self.assertTrue(backend.closed)


class _FakeContext:
    """An isolated Playwright context double holding its own cookie jar."""

    def __init__(self, cookies: list[dict[str, Any]], error: BaseException | None = None) -> None:
        self._cookies = cookies
        self._error = error
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def cookies(self) -> list[dict[str, Any]]:
        if self._error is not None:
            raise self._error
        return list(self._cookies)

    async def close(self) -> None:
        self.closed = True


class _FakeHandle:
    """The context handle the backend receives and passes to ``create``."""

    def __init__(self, session_id: str, context: _FakeContext) -> None:
        self.session_id = session_id
        self.context = context


class _StubTab:
    def __init__(self) -> None:
        self.cookies: list[dict[str, Any]] | None = None

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        self.cookies = list(cookies)


class _StubGroup:
    """A tab group double recording its context and closes.

    ``pd()`` returns the one shared profile reader, as the real ``TabGroup`` delegates to its
    owner's process-wide PyDoll connection; it is deliberately not the group's own context.
    """

    def __init__(self, context: Any, reader: Any) -> None:
        self.context = context
        self.tab = _StubTab()
        self.quit_calls = 0
        self.quit_error: BaseException | None = None
        self._reader = reader

    @property
    async def ptab(self) -> _StubTab:
        return self.tab

    def pd(self) -> Any:
        return self._reader

    async def quit(self) -> None:
        self.quit_calls += 1
        if self.quit_error is not None:
            raise self.quit_error


class _StubSite:
    async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
        return PageResponse(source=_PAGE, status_code=200, headers={}, user_agent="Mozilla/5.0 (Test)", url=url)


class _Recorder:
    """Records context, group, and shutdown activity for a stub browser class."""

    def __init__(self) -> None:
        self.contexts: dict[str, _FakeHandle] = {}
        self.get_context_calls: list[str] = []
        self.closed_contexts: list[str] = []
        self.groups: list[_StubGroup] = []
        self.shutdown_log: list[str] = []
        #: The one process-wide profile jar, as the shared PyDoll connection reports it.
        self.profile_cookies: list[dict[str, Any]] = []
        #: Each isolated context's own jar, keyed by session id.
        self.context_cookies: dict[str, list[dict[str, Any]]] = {}
        self.context_cookie_error: BaseException | None = None
        self.get_context_gate: asyncio.Event | None = None
        self.get_context_error: BaseException | None = None
        self.close_context_failures = 0
        self.quit_error: BaseException | None = None

    async def get_context(self, session_id: str) -> _FakeHandle:
        self.get_context_calls.append(session_id)
        if self.get_context_gate is not None:
            await self.get_context_gate.wait()
        if self.get_context_error is not None:
            raise self.get_context_error
        handle = self.contexts.get(session_id)
        if handle is None:
            handle = _FakeHandle(
                session_id,
                _FakeContext(self.context_cookies.setdefault(session_id, []), self.context_cookie_error),
            )
            self.contexts[session_id] = handle
        return handle

    async def close_context(self, session_id: str) -> None:
        if self.close_context_failures > 0:
            self.close_context_failures -= 1
            msg = "context close failed"
            raise RuntimeError(msg)
        self.closed_contexts.append(session_id)
        self.contexts.pop(session_id, None)

    async def create(self, context: Any = None) -> _StubGroup:
        group = _StubGroup(context, self)
        group.quit_error = self.quit_error
        self.groups.append(group)
        return group

    async def get_cookies(self) -> list[dict[str, Any]]:
        return list(self.profile_cookies)


class _BackendFixture(IsolatedAsyncioTestCase):
    """Builds a ``BrowserBackend`` whose browsers are stubs, returning the recorder."""

    def _backend(self, *, egress_idle_seconds: float = 60.0) -> tuple[BrowserBackend, _Recorder]:
        recorder = _Recorder()

        def _browser_class(name: str) -> type[Any]:
            class FakeBrowser:
                @classmethod
                async def start(cls) -> None:
                    """No-op, as the real browser is already running."""

                @classmethod
                async def create(cls, context: Any = None) -> _StubGroup:
                    return await recorder.create(context)

                @classmethod
                async def get_context(cls, session_id: str) -> _FakeHandle:
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

        backend = BrowserBackend(
            egresses={"decodo": _EGRESS_URL},
            egress_idle_seconds=egress_idle_seconds,
        )
        for patcher in (
            patch.object(backend_module, "Browser", _browser_class("default")),
            patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=_create_browser),
            patch.object(backend_module, "resolve_page_handler", lambda _group, _url: _StubSite()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, recorder


class BackendIsolatedContextTests(_BackendFixture):
    """One isolated context per id, held with its egress claim until explicit cleanup."""

    async def test_a_fetch_uses_the_session_context_and_keeps_it(self) -> None:
        backend, recorder = self._backend()
        await backend.fetch(None, FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE))
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertEqual(len(recorder.groups), 1)
        self.assertIs(recorder.groups[0].context, recorder.contexts["s1"])
        self.assertEqual(recorder.groups[0].quit_calls, 1)
        self.assertEqual(recorder.closed_contexts, [])
        await backend.aclose()

    async def test_the_same_id_reuses_one_context(self) -> None:
        backend, recorder = self._backend()
        await backend.fetch(
            None, FetchRequest(url="https://example.com/a", session_id="s1", session_mode=ISOLATED_MODE)
        )
        await backend.fetch(
            None, FetchRequest(url="https://example.com/b", session_id="s1", session_mode=ISOLATED_MODE)
        )
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertEqual(len(recorder.groups), 2)
        await backend.aclose()

    async def test_distinct_ids_get_distinct_contexts(self) -> None:
        backend, recorder = self._backend()
        await backend.fetch(
            None, FetchRequest(url="https://example.com/a", session_id="s1", session_mode=ISOLATED_MODE)
        )
        await backend.fetch(
            None, FetchRequest(url="https://example.com/b", session_id="s2", session_mode=ISOLATED_MODE)
        )
        self.assertEqual(recorder.get_context_calls, ["s1", "s2"])
        self.assertIsNot(recorder.contexts["s1"], recorder.contexts["s2"])
        await backend.aclose()

    async def test_a_shared_fetch_keeps_the_shared_owner(self) -> None:
        backend, recorder = self._backend()
        await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual(recorder.get_context_calls, [])
        self.assertEqual(len(recorder.groups), 1)
        self.assertIsNone(recorder.groups[0].context)
        await backend.aclose()

    async def test_close_session_closes_the_tabs_and_context_then_releases_the_claim(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        await backend.fetch(
            None,
            FetchRequest(url="https://example.com/", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
        )
        self.assertEqual(recorder.shutdown_log, [])
        await backend.close_session("s1")
        self.assertEqual(recorder.closed_contexts, ["s1"])
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_close_session_of_a_shared_id_is_a_no_op(self) -> None:
        backend, recorder = self._backend()
        await backend.fetch(None, FetchRequest(url="https://example.com/"))
        await backend.close_session("shared-id")
        self.assertEqual(recorder.closed_contexts, [])
        await backend.aclose()

    async def test_a_failed_context_close_retains_the_session_for_retry(self) -> None:
        backend, recorder = self._backend()
        recorder.close_context_failures = 1
        await backend.fetch(None, FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE))
        with self.assertRaises(RuntimeError):
            await backend.close_session("s1")
        self.assertIn("s1", backend._isolated)
        await backend.close_session("s1")
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_concurrent_first_use_initializes_one_context(self) -> None:
        backend, recorder = self._backend()
        recorder.get_context_gate = asyncio.Event()
        first = asyncio.create_task(
            backend.fetch(None, FetchRequest(url="https://example.com/a", session_id="s1", session_mode=ISOLATED_MODE)),
        )
        second = asyncio.create_task(
            backend.fetch(None, FetchRequest(url="https://example.com/b", session_id="s1", session_mode=ISOLATED_MODE)),
        )
        await _settle(ticks=3)
        recorder.get_context_gate.set()
        await asyncio.gather(first, second)
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertEqual(len(recorder.groups), 2)
        await backend.aclose()

    async def test_a_cancelled_initialization_rolls_back_the_claim(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        recorder.get_context_gate = asyncio.Event()
        task = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        await _settle(ticks=3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn("s1", backend._isolated)
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])
        await backend.aclose()


class BackendIsolatedInteractiveTests(_BackendFixture):
    """Isolated tabs live in the session context and reuse is scoped to their session."""

    async def test_an_isolated_open_uses_the_session_context(self) -> None:
        backend, recorder = self._backend()
        await backend.open_interactive(
            InteractiveRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
        )
        self.assertIs(recorder.groups[0].context, recorder.contexts["s1"])
        await backend.aclose()

    async def test_reuse_is_scoped_to_the_session_and_context(self) -> None:
        backend, _recorder = self._backend()
        shared = (await backend.open_interactive(InteractiveRequest(url="https://example.com/"))).tab
        isolated = (
            await backend.open_interactive(
                InteractiveRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
            )
        ).tab
        self.assertNotEqual(shared.tab_id, isolated.tab_id)

        again = await backend.open_interactive(
            InteractiveRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
        )
        self.assertTrue(again.reused)
        self.assertEqual(again.tab.tab_id, isolated.tab_id)

        other = await backend.open_interactive(
            InteractiveRequest(url="https://example.com/", session_id="s2", session_mode=ISOLATED_MODE),
        )
        self.assertFalse(other.reused)
        self.assertNotEqual(other.tab.tab_id, isolated.tab_id)
        await backend.aclose()

    async def test_isolated_cleanup_closes_only_its_own_tabs_and_context(self) -> None:
        backend, recorder = self._backend()
        shared = (await backend.open_interactive(InteractiveRequest(url="https://example.com/shared"))).tab
        isolated = (
            await backend.open_interactive(
                InteractiveRequest(url="https://example.com/isolated", session_id="s1", session_mode=ISOLATED_MODE),
            )
        ).tab
        await backend.close_session("s1")
        remaining = [tab.tab_id for tab in await backend.list_interactive()]
        self.assertEqual(remaining, [shared.tab_id])
        self.assertEqual(recorder.closed_contexts, ["s1"])
        self.assertNotIn(isolated.tab_id, remaining)
        await backend.aclose()


class BackendIsolatedCookieTests(_BackendFixture):
    """A session cookie read is scoped to that session's own context."""

    async def test_context_cookies_are_read_for_an_isolated_session(self) -> None:
        backend, recorder = self._backend()
        recorder.context_cookies["s1"] = [
            {"name": "session", "value": "abc", "domain": ".example.com", "path": "/", "expires": -1},
            {"name": "other", "value": "x", "domain": ".other.example", "path": "/"},
        ]
        listed = await backend.list_cookies(
            CookieQuery(session_id="s1", session_mode=ISOLATED_MODE, url="https://sub.example.com/"),
        )
        self.assertEqual([cookie["name"] for cookie in listed], ["session"])
        await backend.close_session("s1")
        await backend.aclose()

    async def test_a_shared_read_is_unaffected_by_isolated_ownership(self) -> None:
        backend, recorder = self._backend()
        recorder.profile_cookies = [{"name": "a", "value": "b", "domain": ".example.com", "path": "/"}]
        await backend.fetch(None, FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE))
        listed = await backend.list_cookies(CookieQuery())
        self.assertEqual([cookie["name"] for cookie in listed], ["a"])
        self.assertEqual(recorder.closed_contexts, [])
        await backend.close_session("s1")
        await backend.aclose()

    async def test_a_failed_isolated_cookie_read_propagates(self) -> None:
        backend, recorder = self._backend()
        recorder.context_cookie_error = RuntimeError("context cookies failed")
        with self.assertRaises(RuntimeError):
            await backend.list_cookies(CookieQuery(session_id="s1", session_mode=ISOLATED_MODE))
        await backend.close_session("s1")
        await backend.aclose()


async def _wait_until(predicate: Callable[[], bool], *, attempts: int = 400) -> None:
    """Yield until *predicate* holds, failing rather than hanging when it never does."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    msg = "the expected state was not reached in time"
    raise AssertionError(msg)


class _GatedBackend(FakeBackend):
    """A backend that parks inside a fetch until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        self.requests.append((session_id, request))
        self.entered.set()
        await self.release.wait()
        return FetchResult(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/html"},
            response=_PAGE,
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )


class BackendFetchCookieScopeTests(_BackendFixture):
    """A fetch response reports the cookies of the context the request ran in."""

    async def test_an_isolated_fetch_never_returns_the_shared_profile_cookies(self) -> None:
        backend, recorder = self._backend()
        # The shared PyDoll singleton holds a profile jar; the isolated context holds its own.
        recorder.profile_cookies = [{"name": "shared", "value": "1", "domain": ".example.com", "path": "/"}]
        recorder.context_cookies["s1"] = [{"name": "isolated", "value": "2", "domain": ".example.com", "path": "/"}]
        result = await backend.fetch(
            None,
            FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
        )
        names = [cookie["name"] for cookie in result.cookies]
        self.assertEqual(names, ["isolated"])
        self.assertNotIn("shared", names)
        await backend.aclose()

    async def test_a_shared_fetch_still_reads_the_profile_cookies(self) -> None:
        backend, recorder = self._backend()
        recorder.profile_cookies = [{"name": "shared", "value": "1", "domain": ".example.com", "path": "/"}]
        result = await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["shared"])
        await backend.aclose()

    async def test_a_failed_isolated_fetch_cookie_read_propagates(self) -> None:
        backend, recorder = self._backend()
        recorder.context_cookie_error = RuntimeError("context cookies failed")
        with self.assertRaises(RuntimeError):
            await backend.fetch(
                None,
                FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
            )
        await backend.aclose()


class BackendInteractiveOwnershipTests(_BackendFixture):
    """Interactive registration and close failures are owned, not leaked or hidden."""

    async def test_a_shared_named_open_reuses_a_shared_tab(self) -> None:
        backend, _recorder = self._backend()
        first = await backend.open_interactive(
            InteractiveRequest(url="https://example.com/", session_id="s", session_mode=SHARED_MODE),
        )
        again = await backend.open_interactive(
            InteractiveRequest(url="https://example.com/", session_id="s", session_mode=SHARED_MODE),
        )
        self.assertTrue(again.reused)
        self.assertEqual(again.tab.tab_id, first.tab.tab_id)
        await backend.aclose()

    async def test_cancelling_while_registering_closes_the_group_and_releases_the_claim(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        # Skip reuse lookup and the LRU takeover so the task parks on the tab registry itself.
        backend._steal_least_recent_enabled = False
        await backend._tabs_lock.acquire()
        task = asyncio.create_task(
            backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo", new_tab=True)),
        )
        await _wait_until(lambda: bool(recorder.groups))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        backend._tabs_lock.release()
        self.assertEqual(recorder.groups[0].quit_calls, 1)
        self.assertEqual(backend._tabs, {})
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_close_interactive_surfaces_a_group_close_failure(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        recorder.quit_error = RuntimeError("quit failed")
        await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
        with self.assertRaises(RuntimeError):
            await backend.close_interactive(None)
        self.assertEqual(backend._tabs, {})
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])
        await backend.aclose()


class BackendIsolatedInitializationTests(_BackendFixture):
    """One initialization task per generation; a failure is shared and never re-run detached."""

    async def test_concurrent_first_callers_share_one_failed_initialization(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        recorder.get_context_gate = asyncio.Event()
        recorder.get_context_error = RuntimeError("context failed")
        first = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/a", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        second = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/b", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        await _settle(ticks=3)
        recorder.get_context_gate.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual([type(result).__name__ for result in results], ["RuntimeError", "RuntimeError"])
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertNotIn("s1", backend._isolated)
        # The failed initialization released its egress claim.
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])

        # A later caller starts a fresh generation and succeeds.
        recorder.get_context_error = None
        recorder.get_context_gate = None
        result = await backend.fetch(
            None,
            FetchRequest(url="https://example.com/c", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(recorder.get_context_calls, ["s1", "s1"])
        self.assertIn("s1", backend._isolated)
        await backend.close_session("s1")
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_cancelling_one_first_caller_cancels_the_shared_initialization(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        recorder.get_context_gate = asyncio.Event()
        first = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/a", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        second = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/b", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        await _wait_until(lambda: recorder.get_context_calls == ["s1"])
        await _settle(ticks=3)
        first.cancel()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results), results)
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertNotIn("s1", backend._isolated)
        await _settle()
        self.assertEqual(recorder.shutdown_log, ["decodo"])

    async def test_close_session_waits_for_initialization_to_settle(self) -> None:
        backend, recorder = self._backend(egress_idle_seconds=0.01)
        recorder.get_context_gate = asyncio.Event()
        fetch_task = asyncio.create_task(
            backend.fetch(
                None,
                FetchRequest(url="https://example.com/", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
            ),
        )
        await _wait_until(lambda: recorder.get_context_calls == ["s1"])
        close_task = asyncio.create_task(backend.close_session("s1"))
        await _settle(ticks=3)
        self.assertFalse(close_task.done())
        recorder.get_context_gate.set()
        await fetch_task
        await asyncio.wait_for(close_task, timeout=2.0)
        self.assertEqual(recorder.closed_contexts, ["s1"])
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_a_different_egress_for_an_existing_isolated_session_is_rejected(self) -> None:
        backend, _recorder = self._backend()
        await backend.fetch(
            None,
            FetchRequest(url="https://example.com/", egress="decodo", session_id="s1", session_mode=ISOLATED_MODE),
        )
        with self.assertRaises(SessionError):
            await backend.fetch(
                None,
                FetchRequest(url="https://example.com/", session_id="s1", session_mode=ISOLATED_MODE),
            )
        await backend.close_session("s1")
        await backend.aclose()


class ServiceTtlPreservationTests(IsolatedAsyncioTestCase):
    """An omitted request TTL is unspecified and preserves the configured TTL."""

    def _service(self) -> tuple[Service, FakeBackend]:
        backend = FakeBackend()
        return Service(_config(egresses={"decodo": _EGRESS_URL}), backend), backend

    async def test_a_fetch_omitting_the_ttl_preserves_it(self) -> None:
        service, _backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "session_ttl_minutes": 5})
        await _fetch(service, session="s")
        entry = service.sessions._entries["s"]
        self.assertEqual(entry.ttl_minutes, 5)
        self.assertIsNotNone(entry.expires_at)

    async def test_browser_open_omitting_the_ttl_preserves_it(self) -> None:
        service, _backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "session_ttl_minutes": 5})
        await service.handle({"cmd": "browser.open", "url": "https://example.com/", "session": "s"})
        self.assertEqual(service.sessions._entries["s"].ttl_minutes, 5)

    async def test_cookies_list_omitting_the_ttl_preserves_it(self) -> None:
        service, _backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "session_ttl_minutes": 5})
        await service.handle({"cmd": "cookies.list", "session": "s"})
        self.assertEqual(service.sessions._entries["s"].ttl_minutes, 5)

    async def test_a_long_active_fetch_does_not_expire_on_completion(self) -> None:
        backend = _GatedBackend()
        service = Service(_config(), backend)
        await service.handle({"cmd": "sessions.create", "session": "s", "session_ttl_minutes": 5})
        task = asyncio.create_task(_fetch(service, session="s"))
        await backend.entered.wait()
        service.sessions._entries["s"].expires_at = time.monotonic() - 1
        backend.release.set()
        await task
        self.assertGreater(service.sessions._entries["s"].expires_at, time.monotonic())
        self.assertEqual(await service.sessions.list_sessions(), ["s"])


class ServiceLeaseValidationTests(IsolatedAsyncioTestCase):
    """The lease validates the command's explicit mode and egress against the real generation."""

    def _service(self) -> tuple[Service, FakeBackend]:
        backend = FakeBackend()
        return Service(_config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}), backend), backend

    async def test_an_incompatible_mode_does_not_reach_the_backend(self) -> None:
        service, backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s"})
        body = await _fetch(service, session="s", sessionMode="isolated")
        self.assertEqual(body["status"], "error")
        self.assertEqual(backend.requests, [])

    async def test_a_recreated_generation_rejects_the_previous_mode(self) -> None:
        service, backend = self._service()
        await service.handle({"cmd": "sessions.create", "session": "s", "sessionMode": "isolated"})
        await service.handle({"cmd": "sessions.destroy", "session": "s"})
        await service.handle({"cmd": "sessions.create", "session": "s"})
        body = await _fetch(service, session="s", sessionMode="isolated")
        self.assertEqual(body["status"], "error")
        self.assertEqual(backend.requests, [])

    async def test_a_conflicting_egress_does_not_reach_the_backend(self) -> None:
        service, backend = self._service()
        await _fetch(service, session="s", proxy={"name": "decodo"})
        body = await _fetch(service, session="s", proxy={"name": "warp"})
        self.assertEqual(body["status"], "error")
        self.assertEqual(len(backend.requests), 1)

    async def test_an_omitted_proxy_binds_the_default_egress(self) -> None:
        service, backend = self._service()
        await _fetch(service, session="s")
        self.assertEqual(backend.requests[0][1].egress, "default")
        info = await service.sessions.ensure("s")
        self.assertEqual(info.egress, "default")

    async def test_a_bound_session_reuses_its_egress_when_the_proxy_is_omitted(self) -> None:
        service, backend = self._service()
        await _fetch(service, session="s", proxy={"name": "decodo"})
        await _fetch(service, session="s")
        self.assertEqual(backend.requests[1][1].egress, "decodo")


class _SlowCloseBackend(FakeBackend):
    """A backend whose session cleanup parks until the test releases it."""

    def __init__(self) -> None:
        super().__init__()
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)
        self.close_entered.set()
        await self.close_release.wait()


class ServiceShutdownTests(IsolatedAsyncioTestCase):
    """Shutdown drains session cleanup before the backend, and reports failures honestly."""

    async def test_a_session_cleanup_failure_still_closes_the_backend(self) -> None:
        backend = FakeBackend()
        backend.close_session_error = RuntimeError("session close failed")
        service = Service(_config(), backend)
        await service.handle({"cmd": "sessions.create", "session": "s"})
        with self.assertRaises(RuntimeError):
            await service.aclose()
        self.assertTrue(backend.closed)

    async def test_a_cancelled_shutdown_waiter_does_not_close_the_backend_early(self) -> None:
        backend = _SlowCloseBackend()
        service = Service(_config(), backend)
        await service.handle({"cmd": "sessions.create", "session": "s"})
        waiter = asyncio.create_task(service.aclose())
        await backend.close_entered.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        # Session cleanup is still in progress, so the backend has not been closed yet.
        self.assertFalse(backend.closed)
        backend.close_release.set()
        await asyncio.wait_for(service.aclose(), timeout=2.0)
        self.assertTrue(backend.closed)
