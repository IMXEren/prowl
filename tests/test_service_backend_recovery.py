"""Isolated-session generation recovery: stale contexts are refreshed, not reused.

Native recovery retires a dead browser generation and relaunches it, so a live isolated session's
cached context handle can point at a context that no longer exists. These tests pin the backend's
response with stub browsers: only a live native context fast-returns, a stale one is refreshed with
one tracked task on the existing session that reuses its owner and held egress claim, a failed or
cancelled refresh stays retryable instead of losing the session, and cleanup drains an in-flight
refresh before it closes native resources. No process is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from prowl.browser.page_handler import PageResponse
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest
from prowl.service.sessions import ISOLATED_MODE

if TYPE_CHECKING:
    from collections.abc import Callable

_EGRESS_URL = "socks5://127.0.0.1:10001"
_PAGE = "<html><head><title>Example Domain</title></head><body>ok</body></html>"


class _FakeBrowser:
    """The native browser projection a context reports."""

    def __init__(self, *, connected: bool = True) -> None:
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected


class _FakeContext:
    """A native context double that can be aged into a stale generation."""

    def __init__(self) -> None:
        self._closed = False
        self._browser: _FakeBrowser | None = _FakeBrowser()
        self.cookie_jar: list[dict[str, Any]] = []

    def is_closed(self) -> bool:
        return self._closed

    @property
    def browser(self) -> _FakeBrowser | None:
        return self._browser

    def go_stale(self, *, disconnected: bool = False) -> None:
        """Age this context as a browser restart would: closed, or an open context on a dead browser."""
        self._closed = not disconnected
        self._browser = _FakeBrowser(connected=False)

    async def close(self) -> None:
        self._closed = True

    async def cookies(self) -> list[dict[str, Any]]:
        return list(self.cookie_jar)


class _FakeHandle:
    """A context handle as the backend sees and passes it around."""

    def __init__(self, session_id: str, context: _FakeContext) -> None:
        self.session_id = session_id
        self.context = context


class _StubTab:
    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        del cookies


class _StubGroup:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.tab = _StubTab()
        self.quit_calls = 0

    @property
    async def ptab(self) -> _StubTab:
        return self.tab

    async def quit(self) -> None:
        self.quit_calls += 1


class _StubSite:
    async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
        return PageResponse(source=_PAGE, status_code=200, headers={}, user_agent="Mozilla/5.0 (Test)", url=url)


class _Recorder:
    """The one process-wide stub browser's recorded native generation activity."""

    def __init__(self) -> None:
        self.handles: list[_FakeHandle] = []
        self.get_context_calls: list[str] = []
        self.closed_contexts: list[str] = []
        self.groups: list[_StubGroup] = []
        self.start_calls = 0
        self.shutdown_log: list[str] = []
        self.get_context_gate: asyncio.Event | None = None
        self.get_context_error: BaseException | None = None

    async def start(self) -> None:
        self.start_calls += 1

    async def get_context(self, session_id: str) -> _FakeHandle:
        self.get_context_calls.append(session_id)
        if self.get_context_gate is not None:
            await self.get_context_gate.wait()
        if self.get_context_error is not None:
            raise self.get_context_error
        handle = _FakeHandle(session_id, _FakeContext())
        self.handles.append(handle)
        return handle

    async def close_context(self, session_id: str) -> None:
        self.closed_contexts.append(session_id)

    async def create(self, context: Any = None) -> _StubGroup:
        group = _StubGroup(context)
        self.groups.append(group)
        return group

    async def get_cookies(self) -> list[dict[str, Any]]:
        return []


class _RecoveryFixture(IsolatedAsyncioTestCase):
    """Builds a ``BrowserBackend`` whose browsers are stubs, returning the recorder and a trace."""

    def _backend(self) -> tuple[BrowserBackend, _Recorder, dict[str, list[Any]]]:
        recorder = _Recorder()
        trace: dict[str, list[Any]] = {"retired": [], "acquired": [], "released": []}

        def _browser_class() -> type[Any]:
            class FakeBrowser:
                @classmethod
                async def start(cls) -> None:
                    await recorder.start()

                @classmethod
                async def get_context(cls, session_id: str) -> _FakeHandle:
                    return await recorder.get_context(session_id)

                @classmethod
                async def close_context(cls, session_id: str) -> None:
                    await recorder.close_context(session_id)

                @classmethod
                async def create(cls, context: Any = None) -> _StubGroup:
                    return await recorder.create(context)

                @classmethod
                async def shutdown(cls) -> None:
                    recorder.shutdown_log.append("browser")

                @classmethod
                def pd(cls) -> Any:
                    return recorder

            return FakeBrowser

        def _create_browser(**_kwargs: Any) -> type[Any]:
            return _browser_class()

        async def _retire(context: Any) -> None:
            trace["retired"].append(context)

        async def _acquire(egress: str) -> type[Any]:
            trace["acquired"].append(egress)
            return await original_acquire(egress)

        async def _release(egress: str) -> None:
            trace["released"].append(egress)
            await original_release(egress)

        backend = BrowserBackend(egresses={"decodo": _EGRESS_URL})
        original_acquire = backend._pool.acquire
        original_release = backend._pool.release

        patchers = [
            patch.object(backend_module, "Browser", _browser_class()),
            patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=_create_browser),
            patch.object(backend_module, "resolve_page_handler", lambda _group, _url: _StubSite()),
            patch.object(backend._http, "close_context", new=_retire),
            patch.object(backend._pool, "acquire", new=_acquire),
            patch.object(backend._pool, "release", new=_release),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, recorder, trace


async def _settle(*, ticks: int = 20) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0.005)


async def _wait_until(predicate: Callable[[], bool], *, attempts: int = 400) -> None:
    """Yield until *predicate* holds, failing rather than hanging when it never does."""
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.005)
    msg = "the expected state was not reached in time"
    raise AssertionError(msg)


def _isolated(url: str = "https://example.com/") -> FetchRequest:
    return FetchRequest(url=url, egress="decodo", session_id="s1", session_mode=ISOLATED_MODE)


class IsolatedContextRecoveryTests(_RecoveryFixture):
    """A stale native context is rebuilt on the existing session, never silently reused."""

    async def test_a_closed_context_is_refreshed_for_the_same_session(self) -> None:
        backend, recorder, trace = self._backend()
        await backend.fetch(None, _isolated())
        stale = recorder.handles[0]
        stale.context.go_stale()

        await backend.fetch(None, _isolated())

        self.assertEqual(recorder.get_context_calls, ["s1", "s1"])
        self.assertEqual(len(recorder.handles), 2)
        self.assertIs(recorder.groups[-1].context, recorder.handles[1])
        self.assertEqual(trace["retired"], [stale.context])
        self.assertEqual(recorder.start_calls, 1)
        self.assertEqual(trace["acquired"], ["decodo"])
        self.assertTrue(backend._isolated["s1"].claim_held)
        await backend.aclose()

    async def test_a_disconnected_browser_refreshes_an_open_context(self) -> None:
        backend, recorder, _trace = self._backend()
        await backend.fetch(None, _isolated())
        recorder.handles[0].context.go_stale(disconnected=True)

        await backend.fetch(None, _isolated())

        self.assertEqual(recorder.get_context_calls, ["s1", "s1"])
        await backend.aclose()

    async def test_a_live_context_reuses_the_session_without_a_new_task(self) -> None:
        backend, recorder, trace = self._backend()
        await backend.fetch(None, _isolated())
        session = backend._isolated["s1"]
        settled = session.task

        await backend.fetch(None, _isolated())

        self.assertIs(session.task, settled)
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertEqual(recorder.start_calls, 0)
        self.assertEqual(trace["acquired"], ["decodo"])
        self.assertEqual(trace["released"], [])
        await backend.aclose()

    async def test_concurrent_callers_share_one_refresh_and_one_claim(self) -> None:
        backend, recorder, trace = self._backend()
        await backend.fetch(None, _isolated())
        recorder.handles[0].context.go_stale()
        recorder.get_context_gate = asyncio.Event()

        first = asyncio.create_task(backend.fetch(None, _isolated()))
        second = asyncio.create_task(backend.fetch(None, _isolated()))
        await _wait_until(lambda: recorder.get_context_calls == ["s1", "s1"])
        recorder.get_context_gate.set()
        await asyncio.gather(first, second)

        self.assertEqual(recorder.get_context_calls, ["s1", "s1"])
        self.assertEqual(trace["acquired"], ["decodo"])
        self.assertEqual(trace["released"], [])
        session = backend._isolated["s1"]
        self.assertTrue(session.claim_held)
        self.assertTrue(session.ready)
        await backend.aclose()

    async def test_a_failed_refresh_retains_the_claim_and_is_retried(self) -> None:
        backend, recorder, trace = self._backend()
        await backend.fetch(None, _isolated())
        recorder.handles[0].context.go_stale()
        recorder.get_context_error = RuntimeError("native generation is gone")

        with self.assertRaises(RuntimeError):
            await backend.fetch(None, _isolated())

        session = backend._isolated["s1"]
        self.assertTrue(session.claim_held)
        self.assertFalse(session.ready)
        self.assertEqual(trace["released"], [])

        recorder.get_context_error = None
        await backend.fetch(None, _isolated())

        self.assertEqual(recorder.get_context_calls, ["s1", "s1", "s1"])
        self.assertTrue(session.ready)
        self.assertEqual(trace["acquired"], ["decodo"])
        await backend.aclose()

    async def test_a_cancelled_refresh_retains_the_entry_and_close_cleans_it(self) -> None:
        backend, recorder, _trace = self._backend()
        await backend.fetch(None, _isolated())
        recorder.handles[0].context.go_stale()
        recorder.get_context_gate = asyncio.Event()

        refresh = asyncio.create_task(backend.fetch(None, _isolated()))
        await _wait_until(lambda: recorder.get_context_calls == ["s1", "s1"])
        refresh.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await refresh

        session = backend._isolated["s1"]
        self.assertTrue(session.claim_held)
        self.assertFalse(session.ready)
        self.assertIsNotNone(session.owner)

        recorder.get_context_gate = None
        await backend.close_session("s1")
        self.assertEqual(recorder.closed_contexts, ["s1"])
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_close_session_drains_an_in_flight_refresh_before_native_close(self) -> None:
        backend, recorder, _trace = self._backend()
        await backend.fetch(None, _isolated())
        recorder.handles[0].context.go_stale()
        recorder.get_context_gate = asyncio.Event()

        refresh = asyncio.create_task(backend._isolated_session("s1", "decodo"))
        await _wait_until(lambda: recorder.get_context_calls == ["s1", "s1"])
        close = asyncio.create_task(backend.close_session("s1"))
        await _settle()
        self.assertFalse(close.done())
        self.assertEqual(recorder.closed_contexts, [])

        recorder.get_context_gate.set()
        await asyncio.wait_for(refresh, timeout=2.0)
        await asyncio.wait_for(close, timeout=2.0)

        self.assertEqual(recorder.closed_contexts, ["s1"])
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_failed_http_retirement_keeps_old_handle_for_cleanup_retry(self) -> None:
        backend, recorder, trace = self._backend()
        await backend.fetch(None, _isolated())
        stale = recorder.handles[0]
        stale.context.go_stale()
        attempts = 0

        async def retire(context: Any) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                message = "HTTP close failed"
                raise RuntimeError(message)
            trace["retired"].append(context)

        with patch.object(backend._http, "close_context", new=retire):
            with self.assertRaisesRegex(RuntimeError, "HTTP close failed"):
                await backend.fetch(None, _isolated())
            session = backend._isolated["s1"]
            self.assertIs(session.context, stale)
            self.assertTrue(session.claim_held)
            await backend.close_session("s1")
        self.assertEqual(attempts, 2)
        self.assertEqual(trace["retired"], [stale.context])
        self.assertEqual(trace["released"], ["decodo"])
        await backend.aclose()
