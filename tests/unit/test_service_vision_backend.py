"""Opt-in GeeTest and cloud VLM lifecycle and selected-page bindings."""

from __future__ import annotations

import asyncio
import unittest
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Page

from prowl.browser.page_handler import PageHandler
from prowl.browser.solvers.vision.config import VisionSettings, build_cloud_vlm_settings, build_vision_settings
from prowl.browser.solvers.vision.hcaptcha import PAGE_NOT_ACCEPTED_REASON
from prowl.browser.solvers.vision.schemas import CaptchaAttemptResult, VisionOutcome
from prowl.service import app as app_module
from prowl.service import backend as backend_module
from prowl.service.app import ServiceConfig, run_server
from prowl.service.backend import BrowserBackend

if TYPE_CHECKING:
    from prowl.browser.browser import TabGroup


def _vlm_settings(*, effort: str | None = None, headers: str | None = None) -> VisionSettings:
    endpoint = "responses" if effort else "chat/completions"
    return build_vision_settings(
        vlm=build_cloud_vlm_settings(
            endpoint=f"http://127.0.0.1:9090/v1/{endpoint}",
            model="test-model",
            api_key="test-key",
            reasoning_effort=effort,
            headers=headers,
        )
    )


class DisabledBackendTests(unittest.TestCase):
    def test_default_backend_has_no_vision_client_or_provider_adapters(self) -> None:
        backend = BrowserBackend()
        site = PageHandler(cast("TabGroup", object()))
        backend._bind_vision(site)
        self.assertIsNone(backend._vision_vlm)
        self.assertFalse(site.vision_captcha_solvers)

    def test_local_geetest_binding_needs_no_model(self) -> None:
        backend = BrowserBackend(vision_settings=build_vision_settings())
        site = PageHandler(cast("TabGroup", object()))
        backend._bind_vision(site)
        self.assertEqual(set(site.vision_captcha_solvers), {"geetest"})


class CloudStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_constructed_offline_and_reused(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings(effort="minimal"))
        with patch.object(backend_module.Browser, "start", new=AsyncMock()) as start:
            await backend.start()
            await backend.start()
        self.assertEqual(start.await_count, 2)
        client = backend._vision_vlm
        assert client is not None
        self.assertIsNone(client._session)
        self.assertEqual(client._reasoning_effort, "minimal")
        await client.aclose()

    async def test_generic_header_is_bound_without_a_provider_specific_name(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings(headers='{"X-Request-Nonce":"$uuid"}'))
        with patch.object(backend_module.Browser, "start", new=AsyncMock()):
            await backend.start()
        client = backend._vision_vlm
        assert client is not None
        self.assertEqual(client._headers, {"X-Request-Nonce": "$uuid"})
        await client.aclose()

    async def test_client_failure_does_not_start_browser(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings())
        with (
            patch("prowl.vision.vlm.CloudVlmClient", side_effect=RuntimeError("constructor failed")),
            patch.object(backend_module.Browser, "start", new=AsyncMock()) as start,
            self.assertRaisesRegex(RuntimeError, "constructor failed"),
        ):
            await backend.start()
        start.assert_not_awaited()
        self.assertIsNone(backend._vision_vlm)

    async def test_browser_start_failure_closes_the_new_client(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings())
        client = Mock(aclose=AsyncMock())
        with (
            patch.object(backend_module.Browser, "start", new=AsyncMock(side_effect=RuntimeError("boom"))),
            patch("prowl.vision.vlm.CloudVlmClient", return_value=client),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            await backend.start()
        client.aclose.assert_awaited_once()
        self.assertIsNone(backend._vision_vlm)


class VisionBindingTests(unittest.TestCase):
    def test_vlm_binding_exposes_only_the_supported_providers(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings())
        backend._vision_vlm = Mock()
        site = PageHandler(cast("TabGroup", object()))
        backend._bind_vision(site)
        self.assertEqual(set(site.vision_captcha_solvers), {"geetest", "hcaptcha", "recaptcha"})

    def test_hcaptcha_attempt_gates_non_success_tokens_and_records_outcome(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings())
        backend._vision_vlm = Mock()
        site = PageHandler(cast("TabGroup", object()))
        site.tab = Mock()
        rejected = CaptchaAttemptResult(
            outcome=VisionOutcome(status="needs_human", reason=PAGE_NOT_ACCEPTED_REASON),
            token="stale",  # noqa: S106
        )
        adapter = AsyncMock(return_value=rejected)
        page = Mock(spec=Page)
        initial = frozenset({"stale"})
        with patch("prowl.browser.solvers.vision.hcaptcha.solve_hcaptcha_challenge", new=adapter):
            backend._bind_vision(site)
            solve = site._vision_captcha_attempt(page, "hcaptcha")
            assert solve is not None

            async def attempt() -> str | None:
                return await solve(initial)

            token = asyncio.run(attempt())
        adapter.assert_awaited_once_with(
            page, site.get_time_left, vlm_client=backend._vision_vlm, initial_tokens=initial, native_tab=site.tab
        )
        self.assertIsNone(token)
        self.assertEqual(site.vision_outcome, rejected.outcome)

    def test_foreign_site_is_not_mutated(self) -> None:
        backend = BrowserBackend(vision_settings=_vlm_settings())
        foreign: Any = object()
        backend._bind_vision(foreign)
        self.assertFalse(hasattr(foreign, "vision_captcha_solvers"))


class VisionShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_closed_before_browser_even_if_tab_cleanup_fails(self) -> None:
        order: list[str] = []
        backend = BrowserBackend()
        backend._vision_vlm = Mock(aclose=AsyncMock(side_effect=lambda: order.append("vlm")))
        with (
            patch.object(backend, "close_interactive", new=AsyncMock(side_effect=RuntimeError("tabs"))),
            patch.object(
                backend_module.Browser,
                "shutdown",
                new=AsyncMock(side_effect=lambda: order.append("browser")),
            ),
            self.assertRaises(RuntimeError),
        ):
            await backend.aclose()
        self.assertEqual(order, ["vlm", "browser"])
        self.assertIsNone(backend._vision_vlm)

    async def test_failed_client_close_retains_ownership(self) -> None:
        backend = BrowserBackend()
        client = Mock(aclose=AsyncMock(side_effect=RuntimeError("close failed")))
        backend._vision_vlm = client
        with patch.object(backend_module.Browser, "shutdown", new=AsyncMock()), self.assertRaises(RuntimeError):
            await backend.aclose()
        self.assertIs(backend._vision_vlm, client)


class RunServerVisionWiringTests(unittest.TestCase):
    def test_default_backend_receives_geetest_settings(self) -> None:
        captured: dict[str, object] = {}
        with (
            patch.object(
                app_module, "create_app", side_effect=lambda _config, backend: captured.update(backend=backend)
            ),
            patch.object(app_module.web, "run_app"),
        ):
            run_server(ServiceConfig(vision_enabled=True))
        built = captured["backend"]
        assert isinstance(built, BrowserBackend)
        settings = built._vision_settings
        assert settings is not None
        self.assertIsNone(settings.vlm)

    def test_injected_backend_is_not_reconfigured(self) -> None:
        injected = BrowserBackend()
        with (
            patch.object(app_module, "create_app", return_value=object()),
            patch.object(app_module.web, "run_app"),
        ):
            run_server(ServiceConfig(vision_enabled=True), injected)
        self.assertIsNone(injected._vision_settings)
