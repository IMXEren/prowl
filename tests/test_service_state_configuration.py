"""Service wiring for optional isolated-session state persistence.

These are pure tests for the ``session_state_dir`` seam: the config default and env
parsing, the store ``run_server`` builds for its own backend, an injected backend left
untouched, the exact cleanup metadata a persistence-enabled concrete backend receives,
and the positional contracts kept by a disabled concrete or legacy backend. No browser
process, no network, and no startup runs.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.service import app as app_module
from prowl.service.app import Service, ServiceConfig, run_server
from prowl.service.backend import Backend, BrowserBackend
from prowl.service.context_state import ContextStateStore
from prowl.service.sessions import ISOLATED_MODE, SessionCloseReason

if TYPE_CHECKING:
    from aiohttp import web

_STATE_DIR_ENV = "PROWL_SESSION_STATE_DIR"


def _run_server_app(config: ServiceConfig, backend: Backend | None = None) -> web.Application:
    """Run ``run_server`` with the server loop stubbed and return the app it built."""
    with patch.object(app_module.web, "run_app") as run_app:
        run_server(config, backend)
    return run_app.call_args.args[0]


class ServiceConfigStateDirTests(TestCase):
    """The state directory is the last config field, disabled by default and blank-env."""

    def test_appended_field_defaults_to_disabled(self) -> None:
        """The state field retains its positional slot and disabled default."""
        self.assertEqual([f.name for f in dataclasses.fields(ServiceConfig)].index("session_state_dir"), 15)
        self.assertIsNone(ServiceConfig().session_state_dir)

    def test_a_blank_env_value_stays_disabled(self) -> None:
        """Input variation: whitespace-only env cannot enable persistence."""
        with patch.dict(os.environ, {_STATE_DIR_ENV: "   "}):
            self.assertIsNone(ServiceConfig.from_env().session_state_dir)

    def test_a_configured_env_value_becomes_the_field(self) -> None:
        """Happy path: the env value is carried through unchanged."""
        with patch.dict(os.environ, {_STATE_DIR_ENV: "/srv/prowl/state"}):
            self.assertEqual(ServiceConfig.from_env().session_state_dir, "/srv/prowl/state")


class RunServerStateDirTests(TestCase):
    """``run_server`` builds a store for its own backend only when a directory is set."""

    def test_a_configured_directory_becomes_a_store_without_creating_it(self) -> None:
        """Happy path: a real store points at the configured root and nothing is written."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            app = _run_server_app(ServiceConfig(session_state_dir=str(root)))
            self.assertFalse(root.exists())

        backend = app[app_module._BACKEND_KEY]
        assert isinstance(backend, BrowserBackend)
        self.assertTrue(backend.state_persistence_enabled)
        store = backend._state_store
        assert isinstance(store, ContextStateStore)
        self.assertEqual(store._root, root)
        self.assertFalse(root.exists())

    def test_an_unset_directory_leaves_the_backend_without_a_store(self) -> None:
        """Boundary: the default config builds no store and creates no directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            app = _run_server_app(ServiceConfig())
            self.assertFalse(root.exists())

        backend = app[app_module._BACKEND_KEY]
        assert isinstance(backend, BrowserBackend)
        self.assertFalse(backend.state_persistence_enabled)
        self.assertIsNone(backend._state_store)
        self.assertFalse(root.exists())

    def test_an_injected_backend_is_used_as_is(self) -> None:
        """Boundary: an explicit backend is never replaced and its directory is never created."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            injected = Mock(spec=Backend)
            with patch.object(app_module, "BrowserBackend") as backend_cls:
                app = _run_server_app(ServiceConfig(session_state_dir=str(root)), injected)
            self.assertFalse(root.exists())

        backend_cls.assert_not_called()
        self.assertIs(app[app_module._BACKEND_KEY], injected)
        self.assertFalse(root.exists())


class ServicePersistenceCleanupTests(IsolatedAsyncioTestCase):
    """A persistence-enabled concrete backend receives the registry's exact cleanup metadata."""

    async def test_destroy_eviction_and_shutdown_forward_the_exact_info(self) -> None:
        """Happy path: every cleanup origin carries its reason and the matching evicted flag."""
        with tempfile.TemporaryDirectory() as tmp:
            origins: tuple[SessionCloseReason, ...] = ("destroy", "evicted", "shutdown")
            for origin in origins:
                with self.subTest(origin=origin):
                    backend = BrowserBackend(state_store=ContextStateStore(Path(tmp) / origin))
                    backend.close_session = AsyncMock()
                    service = Service(ServiceConfig(), backend)

                    info = await service.sessions.ensure("s", 5, mode=ISOLATED_MODE, egress="east")
                    if origin == "destroy":
                        await service.sessions.destroy("s")
                    elif origin == "evicted":
                        service.sessions._entries["s"].expires_at = 0.0
                        await service.sessions.purge_expired()
                    else:
                        await service.sessions.aclose()

                    expected = dataclasses.replace(info, evicted=origin == "evicted", close_reason=origin)
                    close = backend.close_session
                    self.assertEqual(close.await_count, 1)
                    call = close.await_args
                    assert call is not None
                    self.assertEqual(call.args, ("s",))
                    self.assertEqual(call.kwargs, {"evicted": expected.evicted, "cleanup": expected})


class ServiceDisabledCleanupTests(IsolatedAsyncioTestCase):
    """Without persistence the reviewed positional close contracts are preserved unchanged."""

    async def test_a_disabled_concrete_backend_keeps_the_positional_and_evicted_contracts(self) -> None:
        """Boundary: a disabled concrete backend sees the id alone, plus the eviction flag."""
        backend = BrowserBackend()
        backend.close_session = AsyncMock()
        service = Service(ServiceConfig(), backend)
        self.assertFalse(backend.state_persistence_enabled)

        await service.sessions.ensure("d", 5, mode=ISOLATED_MODE, egress="east")
        await service.sessions.destroy("d")
        backend.close_session.assert_awaited_once_with("d")

        await service.sessions.ensure("e", 5, mode=ISOLATED_MODE, egress="east")
        service.sessions._entries["e"].expires_at = 0.0
        await service.sessions.purge_expired()
        backend.close_session.assert_awaited_with("e", evicted=True)

    async def test_a_legacy_backend_only_ever_sees_the_positional_id(self) -> None:
        """Boundary: a legacy double is handed no keyword metadata for any origin."""
        backend = Mock(spec=Backend)
        backend.close_session = AsyncMock()
        service = Service(ServiceConfig(), backend)

        await service.sessions.ensure("d", 5, mode=ISOLATED_MODE, egress="east")
        await service.sessions.destroy("d")
        await service.sessions.ensure("e", 5, mode=ISOLATED_MODE, egress="east")
        service.sessions._entries["e"].expires_at = 0.0
        await service.sessions.purge_expired()

        self.assertEqual(backend.close_session.await_count, 2)
        self.assertEqual([call.args for call in backend.close_session.await_args_list], [("d",), ("e",)])
        self.assertTrue(all(not call.kwargs for call in backend.close_session.await_args_list))
