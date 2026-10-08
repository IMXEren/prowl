"""Service wiring for the browser context cap.

These are pure tests for the ``max_contexts`` seam: the service config default, env
parsing and validation, the registry budget the service derives from it, the browser
config ``run_server`` forwards, and the per-egress runtime limit the named-egress
factory applies before any launch. No browser process or network is involved.
"""

from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from prowl.browser.browser import Browser
from prowl.browser.config import DEFAULT_MAX_CONTEXTS, BrowserConfig, default_max_contexts
from prowl.browser.proxy.egress import EgressPool, create_egress_browser
from prowl.service import app as app_module
from prowl.service.app import Service, ServiceConfig, run_server
from prowl.service.backend import Backend
from prowl.service.errors import SessionLimitError
from prowl.service.sessions import ISOLATED_MODE

_EGRESS_URL = "socks5://127.0.0.1:1080"


class _RecordingBackend:
    """Backend double that records every cleanup the registry drives."""

    def __init__(self) -> None:
        self.closed_sessions: list[str] = []

    async def start(self) -> None:
        """No-op for the double."""

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    async def aclose(self) -> None:
        """No-op for the double."""


class ServiceConfigCapTests(TestCase):
    """The config default and env parsing share the canonical cap helper."""

    def test_default_is_the_canonical_cap(self) -> None:
        """Happy path: an unset cap is the browser's canonical default."""
        self.assertEqual(ServiceConfig().max_contexts, DEFAULT_MAX_CONTEXTS)

    def test_env_sets_the_cap_through_the_shared_helper(self) -> None:
        """Input variation: the env value is honoured and equals the helper's result."""
        with patch.dict(os.environ, {"PROWL_MAX_CONTEXTS": "4"}):
            self.assertEqual(ServiceConfig.from_env().max_contexts, 4)
            self.assertEqual(default_max_contexts(), 4)

    def test_a_cap_below_one_is_refused_everywhere(self) -> None:
        """Error path: a bad env value and a directly built bad config both raise."""
        for invalid in ["0", "-2", "many"]:
            with (
                patch.dict(os.environ, {"PROWL_MAX_CONTEXTS": invalid}),
                self.assertRaisesRegex(ValueError, "PROWL_MAX_CONTEXTS"),
            ):
                ServiceConfig.from_env()
        with self.assertRaisesRegex(ValueError, "at least 1"):
            ServiceConfig(max_contexts=0)


class ServiceRegistryCapTests(IsolatedAsyncioTestCase):
    """The service derives one registry slot per context beyond the shared one."""

    async def test_a_cap_of_one_admits_no_isolated_session(self) -> None:
        """Boundary: the shared context consumes the only slot, so isolated is disabled."""
        backend = _RecordingBackend()
        service = Service(ServiceConfig(max_contexts=1), Mock(spec=Backend, wraps=backend))
        self.assertEqual(service.sessions.isolated_per_egress_limit, 0)

        await service.sessions.create("shared")
        with self.assertRaises(SessionLimitError):
            await service.sessions.create("isolated", mode=ISOLATED_MODE)
        self.assertEqual(await service.sessions.list_sessions(), ["shared"])
        await service.aclose()

    async def test_an_idle_isolated_session_is_evicted_through_cleanup(self) -> None:
        """Happy path: a cap of two evicts the oldest idle isolated session on admission."""
        backend = _RecordingBackend()
        service = Service(ServiceConfig(max_contexts=2), Mock(spec=Backend, wraps=backend))
        self.assertEqual(service.sessions.isolated_per_egress_limit, 1)

        await service.sessions.create("first", mode=ISOLATED_MODE, egress="east")
        await service.sessions.create("second", mode=ISOLATED_MODE, egress="east")

        self.assertEqual(backend.closed_sessions, ["first"])
        self.assertEqual(await service.sessions.list_sessions(), ["second"])
        await service.aclose()


class RunServerCapTests(TestCase):
    """``run_server`` forwards the configured cap into the browser's launch config."""

    def test_run_server_passes_the_cap_into_the_browser_config(self) -> None:
        """Happy path: the browser config the backend is built with carries the cap."""
        with (
            patch.object(app_module, "BrowserBackend") as backend_cls,
            patch.object(app_module.web, "run_app") as run_app,
        ):
            run_server(ServiceConfig(max_contexts=3))

        config = backend_cls.call_args.args[0]
        self.assertIsInstance(config, BrowserConfig)
        self.assertEqual(config.max_contexts, 3)
        run_app.assert_called_once()


class NamedEgressCapTests(IsolatedAsyncioTestCase):
    """A named egress obeys the same configured cap as the default identity."""

    def test_the_factory_sets_the_runtime_limit_without_launching(self) -> None:
        """Happy path: the factory configures its own manager's cap and starts nothing."""
        browser = create_egress_browser(
            name="east",
            proxy_url=_EGRESS_URL,
            profile_dir="/tmp/prowl-ctx-profile/east",  # noqa: S108
            profile_archive="/tmp/prowl-ctx-profile-east.zip",  # noqa: S108
            preferred_cdp_port=9300,
            max_contexts=5,
        )

        self.assertEqual(browser._runtime.contexts.max_contexts, 5)
        self.assertIsNot(browser._runtime, Browser._runtime)
        self.assertFalse(browser.is_running())

    async def test_the_pool_forwards_the_configured_cap_to_the_egress(self) -> None:
        """Happy path: the pool's created browser carries the base config's cap."""
        pool = EgressPool(
            BrowserConfig(
                profile_dir="/tmp/prowl-ctx-profile",  # noqa: S108
                profile_archive="/tmp/prowl-ctx-profile.zip",  # noqa: S108
                max_contexts=3,
            ),
            {"east": _EGRESS_URL},
            idle_seconds=0.01,
        )
        try:
            browser = await pool.acquire("east")
            self.assertEqual(browser._runtime.contexts.max_contexts, 3)
        finally:
            await pool.aclose()
