"""Backend isolated-session storage-state persistence.

With a :class:`ContextStateStore` configured the backend seeds a newly created isolated context
from its binding's snapshot, refreshes a stale context from the same snapshot without reacquiring
the egress claim, snapshots a live context on eviction or shutdown, and forgets the binding only on
an explicit destroy. The store is real (a temporary directory) and the browser facade is an
isolated subclass whose ``get_context`` hands back a handle the real
:class:`BrowserContextManager` actually owns. No process is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext as PWBrowserContext
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import StorageState

from prowl.browser.browser import Browser
from prowl.browser.driver.contexts import BrowserContextManager
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle.startup import BrowserLifecycle
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend
from prowl.service.context_state import ContextStateError, ContextStateStore
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE, SessionInfo

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from prowl.browser.driver.contexts import BrowserContextHandle

_EGRESS = "decodo"
_EGRESS_URL = "socks5://127.0.0.1:10001"


def _snapshot(*, name: str = "seed") -> StorageState:
    return StorageState(
        cookies=[{"name": name, "value": "v", "domain": "example.test", "path": "/"}],
        origins=[],
    )


def _destroy(session_id: str) -> SessionInfo:
    return SessionInfo(id=session_id, mode=ISOLATED_MODE, egress=_EGRESS, ttl_minutes=None, close_reason="destroy")


def _evicted(session_id: str) -> SessionInfo:
    return SessionInfo(
        id=session_id,
        mode=ISOLATED_MODE,
        egress=_EGRESS,
        ttl_minutes=None,
        evicted=True,
        close_reason="evicted",
    )


def _shared(session_id: str) -> SessionInfo:
    return SessionInfo(id=session_id, mode=SHARED_MODE, egress=_EGRESS, ttl_minutes=None, close_reason="destroy")


class _RecordingStore(ContextStateStore):
    """A real store that records every binding operation, so tests can prove IO did or did not run."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.loads: list[tuple[str, str]] = []
        self.saves: list[tuple[str, str, StorageState]] = []
        self.deletes: list[tuple[str, str]] = []

    def load(self, egress: str, session_id: str) -> StorageState | None:
        self.loads.append((egress, session_id))
        return super().load(egress, session_id)

    def save(self, egress: str, session_id: str, state: StorageState) -> None:
        self.saves.append((egress, session_id, state))
        super().save(egress, session_id, state)

    def delete(self, egress: str, session_id: str) -> None:
        self.deletes.append((egress, session_id))
        super().delete(egress, session_id)


def _seed(store: _RecordingStore, session_id: str, state: StorageState) -> None:
    """Write a binding's file directly and reset the recording counters."""
    store.save(_EGRESS, session_id, state)
    store.loads.clear()
    store.saves.clear()
    store.deletes.clear()


def _native_context(state: StorageState) -> Mock:
    browser = Mock(spec=PWBrowser)
    browser.is_connected.return_value = True
    context = Mock(spec=PWBrowserContext)
    context.browser = browser
    context.pages = []
    context.is_closed.return_value = False
    context.storage_state = AsyncMock(return_value=state)

    async def close() -> None:
        context.is_closed.return_value = True

    context.close = AsyncMock(side_effect=close)
    return context


class _Recorder:
    """The one stub browser's recorded native activity, keyed by session."""

    def __init__(self, manager: BrowserContextManager) -> None:
        self.manager = manager
        self.get_context_calls: list[str | None] = []
        self.restored_states: list[StorageState | None] = []
        self.native: dict[str, Mock] = {}
        self.start_calls = 0
        self.shutdown_log: list[str] = []
        self.closed_contexts: list[tuple[str, bool]] = []
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.get_context_error: BaseException | None = None

    async def get_context(self, session_id: str | None, storage_state: StorageState | None) -> BrowserContextHandle:
        self.get_context_calls.append(session_id)
        self.restored_states.append(storage_state)
        if self.get_context_error is not None:
            raise self.get_context_error
        state = storage_state or StorageState(cookies=[], origins=[])

        async def _make() -> PWBrowserContext:
            native = _native_context(state)
            self.native[session_id or ""] = native
            return native

        return await self.manager.get_or_create(session_id or "", _make)


class _PersistenceFixture(IsolatedAsyncioTestCase):
    """Builds a browser backend whose native contexts are real manager handles over PW-spec doubles."""

    def _backend(
        self,
        *,
        make_store: Callable[[Path], ContextStateStore] | None = _RecordingStore,
        egress_idle_seconds: float = 60.0,
    ) -> tuple[BrowserBackend, _Recorder, ContextStateStore | None]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        store = None if make_store is None else make_store(root)
        manager = BrowserContextManager()
        recorder = _Recorder(manager)

        def _stub_browser(label: str) -> type[Browser]:
            runtime = BrowserRuntimeState(max_groups=1)
            if label == "egress":
                runtime.contexts = manager

            class StubBrowser(Browser):
                _runtime = runtime
                _lifecycle = BrowserLifecycle(runtime)

                @classmethod
                async def start(cls) -> None:
                    recorder.start_calls += 1

                @classmethod
                async def get_context(
                    cls,
                    session_id: str | None = None,
                    *,
                    storage_state: StorageState | None = None,
                ) -> BrowserContextHandle:
                    return await recorder.get_context(session_id, storage_state)

                @classmethod
                async def close_context(cls, session_id: str, *, evicted: bool = False) -> None:
                    recorder.closed_contexts.append((session_id, evicted))
                    await recorder.manager.close_isolated(session_id, evicted=evicted)

                @classmethod
                async def shutdown(cls) -> None:
                    recorder.shutdown_log.append(label)

                @classmethod
                async def retry_shutdown(cls) -> None:
                    recorder.shutdown_log.append(f"{label}-retry")

            return StubBrowser

        default_stub = _stub_browser("default")
        egress_stub = _stub_browser("egress")

        backend = BrowserBackend(
            egresses={_EGRESS: _EGRESS_URL},
            egress_idle_seconds=egress_idle_seconds,
            state_store=store,
        )

        original_acquire = backend._pool.acquire
        original_release = backend._pool.release

        async def _acquire(egress: str) -> type[Browser]:
            recorder.acquired.append(egress)
            return await original_acquire(egress)

        async def _release(egress: str) -> None:
            recorder.released.append(egress)
            await original_release(egress)

        patchers = [
            patch.object(backend_module, "Browser", default_stub),
            patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=lambda **_kwargs: egress_stub),
            patch.object(backend._pool, "acquire", new=_acquire),
            patch.object(backend._pool, "release", new=_release),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return backend, recorder, store


class ContextPersistenceRestoreTests(_PersistenceFixture):
    """A configured store seeds and refreshes an isolated context exactly once per generation."""

    async def test_initialization_loads_once_and_a_ready_context_reuses_it(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())

        session = await backend._isolated_session("s1", _EGRESS)

        self.assertEqual(store.loads, [(_EGRESS, "s1")])
        self.assertEqual(recorder.get_context_calls, ["s1"])
        self.assertEqual(recorder.restored_states, [_snapshot()])
        self.assertTrue(session.ready)

        settled = session.task
        await backend._isolated_session("s1", _EGRESS)

        self.assertIs(session.task, settled)
        self.assertEqual(store.loads, [(_EGRESS, "s1")])
        self.assertEqual(recorder.get_context_calls, ["s1"])
        await backend.aclose()

    async def test_refresh_restores_the_saved_snapshot_without_reacquiring_the_claim(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        await backend._isolated_session("s1", _EGRESS)
        recorder.native["s1"].is_closed.return_value = True
        recorder.native["s1"].browser.is_connected.return_value = False
        store.loads.clear()

        session = await backend._isolated_session("s1", _EGRESS)

        self.assertEqual(store.loads, [(_EGRESS, "s1")])
        self.assertEqual(recorder.get_context_calls, ["s1", "s1"])
        self.assertEqual(recorder.restored_states, [_snapshot(), _snapshot()])
        self.assertTrue(session.ready)
        self.assertTrue(session.claim_held)
        self.assertEqual(recorder.acquired, [_EGRESS])
        self.assertEqual(recorder.released, [])
        await backend.aclose()


class ContextPersistenceCleanupTests(_PersistenceFixture):
    """Cleanup snapshots or forgets a binding according to the reason it was handed."""

    async def test_eviction_snapshots_the_live_context_and_retains_the_file(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        await backend._isolated_session("s1", _EGRESS)
        native = recorder.native["s1"]

        await backend.close_session("s1", evicted=True, cleanup=_evicted("s1"))

        native.storage_state.assert_awaited_once_with(indexed_db=True)
        self.assertEqual(store.saves, [(_EGRESS, "s1", _snapshot())])
        self.assertEqual(recorder.closed_contexts, [("s1", True)])
        self.assertNotIn("s1", backend._isolated)
        self.assertEqual(store.load(_EGRESS, "s1"), _snapshot())
        await backend.aclose()

    async def test_explicit_destroy_forgets_only_its_own_binding(self) -> None:
        backend, _recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        await backend._isolated_session("s1", _EGRESS)

        _seed(store, "s2", _snapshot())
        await backend.close_session("s1", cleanup=_destroy("s1"))
        self.assertIsNone(store.load(_EGRESS, "s1"))
        self.assertEqual(store.load(_EGRESS, "s2"), _snapshot())

        # A shared cleanup never deletes an isolated file, even for a missing session.
        _seed(store, "missing", _snapshot())
        await backend.close_session("missing", cleanup=_shared("missing"))
        self.assertEqual(store.load(_EGRESS, "missing"), _snapshot())
        self.assertEqual(store.load(_EGRESS, "s2"), _snapshot())

        # A destroy of a session never initialized this generation still erases its binding.
        _seed(store, "s4", _snapshot())
        await backend.close_session("s4", cleanup=_destroy("s4"))
        self.assertIsNone(store.load(_EGRESS, "s4"))

        # An eviction of an uninitialized binding retains it.
        _seed(store, "s5", _snapshot())
        await backend.close_session("s5", cleanup=_evicted("s5"))
        self.assertEqual(store.load(_EGRESS, "s5"), _snapshot())
        await backend.aclose()

    async def test_aclose_snapshots_the_remaining_sessions_as_shutdown(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        await backend._isolated_session("s1", _EGRESS)

        await backend.aclose()

        recorder.native["s1"].storage_state.assert_awaited_once_with(indexed_db=True)
        self.assertEqual(store.saves, [(_EGRESS, "s1", _snapshot())])
        self.assertEqual(store.load(_EGRESS, "s1"), _snapshot())
        self.assertEqual(recorder.shutdown_log, ["default", "egress"])


class ContextPersistenceFailureTests(_PersistenceFixture):
    """A failed save keeps ownership for a retry; shutdown waits for that retry first."""

    async def test_a_failed_save_retains_ownership_and_a_retry_succeeds(self) -> None:
        class _FailingStore(_RecordingStore):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self.fail_saves = 0

            def save(self, egress: str, session_id: str, state: StorageState) -> None:
                if self.fail_saves > 0:
                    self.fail_saves -= 1
                    msg = "save failed"
                    raise ContextStateError(msg)
                super().save(egress, session_id, state)

        backend, recorder, store = self._backend(make_store=_FailingStore)
        assert isinstance(store, _FailingStore)
        store.fail_saves = 1
        await backend._isolated_session("s1", _EGRESS)

        with self.assertRaises(ContextStateError):
            await backend.close_session("s1", evicted=True, cleanup=_evicted("s1"))

        self.assertIn("s1", backend._isolated)
        self.assertTrue(backend._isolated["s1"].claim_held)
        self.assertEqual(recorder.closed_contexts, [])

        await backend.close_session("s1", evicted=True, cleanup=_evicted("s1"))
        self.assertNotIn("s1", backend._isolated)
        self.assertEqual(recorder.closed_contexts, [("s1", True)])
        self.assertIsNotNone(store.load(_EGRESS, "s1"))
        await backend.aclose()

    async def test_aclose_after_a_failed_save_keeps_the_owner_alive_for_retry(self) -> None:
        class _FailingStore(_RecordingStore):
            def __init__(self, root: Path) -> None:
                super().__init__(root)
                self.fail_saves = 0

            def save(self, egress: str, session_id: str, state: StorageState) -> None:
                if self.fail_saves > 0:
                    self.fail_saves -= 1
                    msg = "save failed"
                    raise ContextStateError(msg)
                super().save(egress, session_id, state)

        backend, recorder, store = self._backend(make_store=_FailingStore)
        assert isinstance(store, _FailingStore)
        store.fail_saves = 1
        await backend._isolated_session("s1", _EGRESS)

        with self.assertRaises(ContextStateError):
            await backend.aclose()

        # The native owner is still up, and the session keeps its context and claim for retry.
        self.assertEqual(recorder.shutdown_log, [])
        self.assertIn("s1", backend._isolated)
        self.assertTrue(backend._isolated["s1"].claim_held)

        await backend.aclose()
        self.assertEqual(recorder.shutdown_log, ["default", "egress"])
        self.assertNotIn("s1", backend._isolated)

    async def test_failed_delete_keeps_the_closed_handle_and_claim_until_retry(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        session = await backend._isolated_session("s1", _EGRESS)
        native = recorder.native["s1"]

        with (
            patch.object(store, "delete", side_effect=ContextStateError("delete failed")),
            self.assertRaises(ContextStateError),
        ):
            await backend.close_session("s1", cleanup=_destroy("s1"))
        self.assertIs(backend._isolated["s1"], session)
        self.assertTrue(session.claim_held)
        self.assertTrue(native.is_closed())
        self.assertEqual(recorder.released, [])
        self.assertIsNotNone(store.load(_EGRESS, "s1"))

        await backend.close_session("s1", cleanup=_destroy("s1"))
        native.close.assert_awaited_once()
        self.assertIsNone(store.load(_EGRESS, "s1"))
        self.assertNotIn("s1", backend._isolated)
        self.assertEqual(recorder.released, [_EGRESS])
        await backend.aclose()

    async def test_dead_context_retirement_preserves_the_last_saved_snapshot(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        await backend._isolated_session("s1", _EGRESS)
        native = recorder.native["s1"]
        native.is_closed.return_value = True
        native.browser.is_connected.return_value = False

        await backend.close_session("s1", evicted=True, cleanup=_evicted("s1"))
        native.storage_state.assert_not_awaited()
        self.assertEqual(store.saves, [])
        self.assertEqual(store.load(_EGRESS, "s1"), _snapshot())
        await backend.aclose()


class ContextPersistenceCancellationTests(_PersistenceFixture):
    """An in-flight write is joined before a cancellation escapes, and native failures stay safe."""

    async def test_cancelling_a_close_waits_for_the_inflight_write(self) -> None:
        gate = threading.Event()
        started = threading.Event()
        finished = threading.Event()

        class _GateStore(_RecordingStore):
            def save(self, egress: str, session_id: str, state: StorageState) -> None:
                started.set()
                gate.wait(timeout=5)
                super().save(egress, session_id, state)
                finished.set()

        backend, recorder, _store = self._backend(make_store=_GateStore)
        await backend._isolated_session("s1", _EGRESS)

        draining = asyncio.Event()
        redraining = asyncio.Event()
        original_shield = asyncio.shield
        calls = 0

        def shield(awaitable: Awaitable[object]) -> asyncio.Future[object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                draining.set()
            elif calls == 3:
                redraining.set()
            return original_shield(awaitable)

        with patch.object(backend_module.asyncio, "shield", side_effect=shield):
            task = asyncio.create_task(backend.close_session("s1", evicted=True, cleanup=_evicted("s1")))
            self.assertTrue(await asyncio.wait_for(asyncio.to_thread(started.wait, 5), timeout=5))
            task.cancel()
            await asyncio.wait_for(draining.wait(), timeout=2)
            task.cancel()
            await asyncio.wait_for(redraining.wait(), timeout=2)
            self.assertFalse(task.done())
            self.assertFalse(finished.is_set())
            gate.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)

        self.assertTrue(finished.is_set())
        self.assertIn("s1", backend._isolated)
        self.assertTrue(backend._isolated["s1"].claim_held)
        self.assertEqual(recorder.closed_contexts, [])

        await backend.close_session("s1", evicted=True, cleanup=_evicted("s1"))
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_a_native_restore_error_becomes_caller_safe_and_rolls_back(self) -> None:
        backend, recorder, store = self._backend()
        assert isinstance(store, _RecordingStore)
        _seed(store, "s1", _snapshot())
        recorder.get_context_error = PlaywrightError("native restore failed")

        with self.assertRaises(ContextStateError) as error:
            await backend._isolated_session("s1", _EGRESS)

        self.assertEqual(str(error.exception), "persisted context state could not be restored")
        self.assertIsNone(error.exception.__cause__)
        self.assertNotIn("s1", backend._isolated)
        await backend.aclose()

    async def test_a_native_error_without_persistence_passes_through(self) -> None:
        backend, recorder, _store = self._backend(make_store=None)
        recorder.get_context_error = PlaywrightError("plain context failure")

        with self.assertRaises(PlaywrightError):
            await backend._isolated_session("s1", _EGRESS)
