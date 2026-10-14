"""Automatic context evictions: the marker the registry carries and the count it drives.

An idle TTL expiry or an admission victim retires a session's native context automatically.
These tests pin the marker on the cleanup callback, that an explicit destroy or shutdown is
never marked, that a failed automatic cleanup keeps its marker across a retry, and that only a
genuine successful automatic close reaches the native manager's eviction counter. No process is
launched and no network is touched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

from playwright.async_api import BrowserContext as PWBrowserCtx

from prowl.browser.browser import Browser
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle import BrowserLifecycle
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
from prowl.service.sessions import ISOLATED_MODE, SessionInfo, SessionRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from prowl.browser.driver.contexts import BrowserContextHandle


class _CleanupError(Exception):
    """A backend cleanup failure used to exercise the retained-tombstone path."""


class _CleanupRecorder:
    """Record the metadata every cleanup receives, optionally failing the first ones."""

    def __init__(self, *, failures: int = 0) -> None:
        self.calls: list[SessionInfo] = []
        self._failures = failures

    async def __call__(self, info: SessionInfo) -> None:
        self.calls.append(info)
        if len(self.calls) <= self._failures:
            msg = "cleanup failed"
            raise _CleanupError(msg)


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-evict-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-evict-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


class _LegacyBackend:
    """A backend double that keeps the original positional ``close_session`` contract."""

    def __init__(self) -> None:
        self.closed: list[str] = []

    async def start(self) -> None:
        """No-op for the double."""

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        raise NotImplementedError

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        raise NotImplementedError

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        raise NotImplementedError

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        raise NotImplementedError

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        """No-op for the double."""


class SessionEvictedMarkerTests(IsolatedAsyncioTestCase):
    """The cleanup callback carries whether the close is automatic retirement."""

    async def test_ttl_expiry_marks_the_cleanup_automatic(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=4, cleanup=recorder)
        await registry.create("s", 5, mode=ISOLATED_MODE)
        registry._entries["s"].expires_at = 0.0

        self.assertEqual(await registry.purge_expired(), ["s"])

        self.assertEqual([info.evicted for info in recorder.calls], [True])

    async def test_admission_eviction_is_automatic_while_destroy_and_shutdown_are_not(self) -> None:
        recorder = _CleanupRecorder()
        registry = SessionRegistry(max_sessions=2, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a", mode=ISOLATED_MODE, egress="east")
        await registry.create("b", mode=ISOLATED_MODE, egress="east")

        self.assertEqual([info.evicted for info in recorder.calls], [True])

        await registry.destroy("b")
        self.assertEqual([info.evicted for info in recorder.calls], [True, False])

        shutdown = _CleanupRecorder()
        other = SessionRegistry(max_sessions=2, cleanup=shutdown)
        await other.create("s")
        await other.aclose()
        self.assertEqual([info.evicted for info in shutdown.calls], [False])

    async def test_a_failed_automatic_cleanup_keeps_its_marker_across_a_destroy_retry(self) -> None:
        recorder = _CleanupRecorder(failures=1)
        registry = SessionRegistry(max_sessions=1, cleanup=recorder, isolated_per_egress_limit=1)
        await registry.create("a")

        with self.assertRaises(_CleanupError):
            await registry.create("b")
        await registry.destroy("a")

        self.assertEqual([info.evicted for info in recorder.calls], [True, True])


class _OwnerRecorder:
    def __init__(self) -> None:
        self.closes: list[tuple[str, bool]] = []
        self.runtime = BrowserRuntimeState(max_groups=3)


def _owner_browser(recorder: _OwnerRecorder) -> type[Browser]:
    base = _counting_browser(recorder.runtime, _fresh)

    class OwnerBrowser(base):
        @classmethod
        async def close_context(cls, session_id: str, *, evicted: bool = False) -> None:
            recorder.closes.append((session_id, evicted))
            await super().close_context(session_id, evicted=evicted)

    return OwnerBrowser


class BackendCloseSessionContractTests(IsolatedAsyncioTestCase):
    """The concrete backend forwards the flag; a legacy backend keeps the old contract."""

    async def test_the_concrete_backend_forwards_the_eviction_flag_only_when_set(self) -> None:
        recorder = _OwnerRecorder()
        backend = BrowserBackend()
        patcher = patch.object(backend_module, "Browser", _owner_browser(recorder))
        patcher.start()
        self.addCleanup(patcher.stop)

        await backend.list_cookies(CookieQuery(session_id="auto", session_mode=ISOLATED_MODE))
        await backend.close_session("auto", evicted=True)
        await backend.list_cookies(CookieQuery(session_id="explicit", session_mode=ISOLATED_MODE))
        await backend.close_session("explicit")

        self.assertEqual(recorder.closes, [("auto", True), ("explicit", False)])

    async def test_a_legacy_backend_keeps_the_positional_close_contract(self) -> None:
        backend = _LegacyBackend()
        service = Service(_config(max_contexts=2), backend)

        await service.sessions.create("first", mode=ISOLATED_MODE, egress="east")
        await service.sessions.create("second", mode=ISOLATED_MODE, egress="east")

        self.assertEqual(backend.closed, ["first"])
        await service.aclose()


class _NativeContext:
    """A Playwright context double whose native close can fail a fixed number of times."""

    def __init__(self, *, close_failures: int = 0) -> None:
        self._closed = False
        self._remaining = close_failures

    def is_closed(self) -> bool:
        return self._closed

    async def cookies(self) -> list[dict[str, Any]]:
        return []

    async def close(self) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            msg = "context close failed"
            raise RuntimeError(msg)
        self._closed = True


def _native_context(*, close_failures: int = 0) -> Any:
    return Mock(spec=PWBrowserCtx, wraps=_NativeContext(close_failures=close_failures))


async def _fresh(_session_id: str) -> Any:
    return _native_context()


async def _flaky(_session_id: str) -> Any:
    return _native_context(close_failures=1)


def _counting_browser(runtime: BrowserRuntimeState, create: Callable[[str], Awaitable[Any]]) -> type[Browser]:
    """Return a real Browser subclass whose facade closes real managed contexts.

    The process is never launched: a session id's context is registered through the real runtime
    context manager, and the inherited ``close_context`` forwards to it exactly as in production,
    so the native eviction counter is driven by the genuine facade path.
    """

    class CountingBrowser(Browser):
        @classmethod
        async def get_context(cls, session_id: str | None = None) -> BrowserContextHandle:
            if session_id is None:
                shared = cls._runtime.contexts.shared()
                assert shared is not None
                return shared
            return await cls._runtime.contexts.get_or_create(session_id, lambda: create(session_id))

    CountingBrowser._runtime = runtime
    CountingBrowser._lifecycle = BrowserLifecycle(runtime)
    return CountingBrowser


class ServiceContextEvictionIntegrationTests(IsolatedAsyncioTestCase):
    """Only a genuine successful automatic close reaches the native eviction counter."""

    def _patch_facade(self, runtime: BrowserRuntimeState, create: Callable[[str], Awaitable[Any]]) -> None:
        patcher = patch.object(backend_module, "Browser", _counting_browser(runtime, create))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_a_successful_automatic_close_counts_once_while_metadata_and_explicit_do_not(self) -> None:
        runtime = BrowserRuntimeState(max_groups=3)
        runtime.contexts.bind_shared(_native_context())
        self._patch_facade(runtime, _fresh)
        backend = BrowserBackend()
        service = Service(_config(max_contexts=4), backend)

        # Metadata-only: an isolated session that never used a context is evicted, but there is
        # no native context to remove, so nothing is counted.
        await service.sessions.create("ghost", mode=ISOLATED_MODE)
        service.sessions._entries["ghost"].expires_at = 0.0
        self.assertEqual(await service.sessions.purge_expired(), ["ghost"])
        self.assertEqual(runtime.contexts.context_evicted_total, 0)

        # A live isolated context is created, then its session expires: the automatic close
        # removes the native handle and counts exactly one eviction.
        await service.sessions.create("live", mode=ISOLATED_MODE)
        await backend.list_cookies(CookieQuery(session_id="live", session_mode=ISOLATED_MODE))
        self.assertIsNotNone(runtime.contexts.isolated("live"))
        service.sessions._entries["live"].expires_at = 0.0
        self.assertEqual(await service.sessions.purge_expired(), ["live"])
        self.assertEqual(runtime.contexts.context_evicted_total, 1)
        self.assertIsNone(runtime.contexts.isolated("live"))

        # An explicit destroy removes its context but is never counted as an eviction.
        await service.sessions.create("explicit", mode=ISOLATED_MODE)
        await backend.list_cookies(CookieQuery(session_id="explicit", session_mode=ISOLATED_MODE))
        await service.sessions.destroy("explicit")
        self.assertEqual(runtime.contexts.context_evicted_total, 1)
        self.assertIsNone(runtime.contexts.isolated("explicit"))

        # No automatic close ever touched the shared persistent context.
        self.assertIsNotNone(runtime.contexts.shared())

    async def test_a_failed_automatic_close_counts_only_after_a_successful_retry(self) -> None:
        runtime = BrowserRuntimeState(max_groups=3)
        self._patch_facade(runtime, _flaky)
        backend = BrowserBackend()
        service = Service(_config(max_contexts=4), backend)

        await service.sessions.create("flaky", mode=ISOLATED_MODE)
        await backend.list_cookies(CookieQuery(session_id="flaky", session_mode=ISOLATED_MODE))
        service.sessions._entries["flaky"].expires_at = 0.0

        with self.assertRaises(RuntimeError):
            await service.sessions.purge_expired()
        self.assertEqual(runtime.contexts.context_evicted_total, 0)
        self.assertIsNotNone(runtime.contexts.isolated("flaky"))

        await service.sessions.retry_cleanup("flaky")
        self.assertEqual(runtime.contexts.context_evicted_total, 1)
        self.assertIsNone(runtime.contexts.isolated("flaky"))
