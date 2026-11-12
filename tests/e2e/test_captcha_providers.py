"""Opt-in live CAPTCHA checks through Prowl's real browser fetch path."""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock
from uuid import uuid4

import pytest

if os.getenv("PROWL_E2E") != "1":
    pytest.skip("live CAPTCHA E2E requires PROWL_E2E=1", allow_module_level=True)

from tests.integration.verify_isolated_sessions import _installed_launch, _resolve_binary_path

from prowl.browser import Browser, BrowserConfig
from prowl.browser import page_handler as page_handler_module
from prowl.browser.driver import runtime as driver_runtime
from prowl.browser.lifecycle import startup as lifecycle_startup
from prowl.browser.solvers.vision.config import (
    VISION_VLM_HEADERS_ENV,
    VisionSettings,
    build_cloud_vlm_settings,
    build_vision_settings,
)
from prowl.service.backend import BrowserBackend, FetchRequest

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page, Response


@dataclass(frozen=True, slots=True)
class CaptchaCase:
    """One public provider demo and its expected Prowl token source."""

    name: str
    url: str
    provider: str
    vlm: bool = False
    timeout: int = 90


CASES = (
    CaptchaCase("turnstile", "https://2captcha.com/demo/cloudflare-turnstile", "turnstile"),
    CaptchaCase("altcha", "https://playground.altcha.org/#/widget", "altcha"),
    CaptchaCase("friendly", "https://developer.friendlycaptcha.com/playground", "friendly"),
    CaptchaCase(
        "recaptcha-enterprise", "https://2captcha.com/demo/recaptcha-v2-enterprise", "recaptcha", vlm=True, timeout=140
    ),
    CaptchaCase(
        "recaptcha-callback", "https://2captcha.com/demo/recaptcha-v2-callback", "recaptcha", vlm=True, timeout=140
    ),
    CaptchaCase("hcaptcha-tiles", "https://nopecha.com/demo/hcaptcha", "hcaptcha", vlm=True, timeout=140),
    CaptchaCase(
        "hcaptcha-canvas",
        "https://maximedrn.github.io/hcaptcha-solver-python-selenium/",
        "hcaptcha",
        vlm=True,
        timeout=140,
    ),
    CaptchaCase("geetest-v3", "https://2captcha.com/demo/geetest", "geetest"),
    CaptchaCase("geetest-v4", "https://2captcha.com/demo/geetest-v4", "geetest"),
)


def _recording_path(case: CaptchaCase) -> Path | None:
    if os.getenv("PROWL_E2E_RECORD") != "1":
        return None
    return Path(__file__).resolve().parents[1] / "e2e-records" / f"{case.name}-{uuid4().hex[:8]}.webm"


def _vision_settings(case: CaptchaCase) -> VisionSettings | None:
    if case.vlm:
        endpoint = os.getenv("PROWL_VISION_VLM_ENDPOINT")
        model = os.getenv("PROWL_VISION_VLM_MODEL")
        if not endpoint or not model:
            pytest.skip(f"{case.name}: set PROWL_VISION_VLM_ENDPOINT and PROWL_VISION_VLM_MODEL")
        vlm = build_cloud_vlm_settings(
            endpoint=endpoint,
            model=model,
            api_key=os.getenv("PROWL_VISION_VLM_API_KEY"),
            reasoning_effort=os.getenv("PROWL_VISION_VLM_REASONING_EFFORT"),
            headers=os.getenv(VISION_VLM_HEADERS_ENV),
        )
        return build_vision_settings(vlm=vlm)
    return build_vision_settings() if case.provider == "geetest" else None


async def _run(  # noqa: PLR0915 - one owned browser lifecycle
    case: CaptchaCase, *, interstitial: bool = False
) -> tuple[str | None, bool, int | None, bool, str | None]:
    with tempfile.TemporaryDirectory(prefix=f"prowl-e2e-{case.name}-") as directory:
        root = Path(directory)
        profile = root / "profile"
        config = BrowserConfig(
            profile_dir=str(profile), profile_archive=str(root / "profile.zip"), extensions_dir=None, policy_dir=None
        )
        video = _recording_path(case)
        pages: list[Page] = []
        challenged = False
        result_provider: str | None = None
        token_seen = False
        status: int | None = None
        challenge_left = False
        vision_reason: str | None = None

        async def launch(*args: Any, **kwargs: Any) -> BrowserContext:
            if video is not None:
                kwargs["record_video_dir"] = str(root / "recordings")
                kwargs["record_video_size"] = {"width": 1280, "height": 720}
            return await _installed_launch(*args, **kwargs)

        def record_response(response: Response) -> None:
            nonlocal challenged
            if response.headers.get("cf-mitigated") == "challenge":
                challenged = True

        def record_page(page: Page) -> None:
            pages.append(page)
            if interstitial:
                page.on("response", record_response)

        backend = BrowserBackend(config, vision_settings=_vision_settings(case))
        original_solver = page_handler_module.solve_visible_captcha

        async def activate_friendly(page: Page, *args: Any, **kwargs: Any) -> tuple[str, str] | None:
            mode = page.locator("select").filter(has=page.locator('option[value="smart"]'))
            await mode.select_option(label="One-click - User must click once", timeout=5000)
            await page.get_by_placeholder("Your full name").focus(timeout=5000)
            await page.locator(".frc-captcha").wait_for(state="attached", timeout=12000)
            return await original_solver(page, *args, **kwargs)

        setup = (
            mock.patch.object(page_handler_module, "solve_visible_captcha", activate_friendly)
            if case.name == "friendly"
            else nullcontext()
        )
        try:
            with (
                setup,
                mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": _resolve_binary_path()}),
                mock.patch.object(lifecycle_startup, "launch_persistent_context_async", launch),
                mock.patch.object(driver_runtime, "launch_persistent_context_async", launch),
            ):
                Browser.configure(config)
                if Browser._webdata_path().parent.parent.resolve() != profile.resolve():
                    pytest.fail("isolated E2E profile ownership check failed", pytrace=False)
                await backend.start()
                context = (await Browser.get_context(None)).context
                context.on("page", record_page)
                try:
                    result = await backend.fetch(
                        None,
                        FetchRequest(url=case.url, mode="browser", solve_captcha=True, timeout_seconds=case.timeout),
                    )
                    result_provider, token_seen, status = (
                        result.captcha_provider,
                        bool(result.captcha_token),
                        result.status_code,
                    )
                    vision_reason = result.vision_reason
                    challenge_left = "Just a moment..." in result.response or "cf-chl" in result.response
                finally:
                    if video is not None:
                        if not pages or pages[0].video is None:
                            pytest.fail(f"{case.name}: challenge-tab recording missing", pytrace=False)
                        video.parent.mkdir(parents=True, exist_ok=True)
                        await pages[0].video.save_as(str(video))
        finally:
            await backend.aclose()
        return result_provider, token_seen, status, challenged and not challenge_left, vision_reason


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_captcha_provider(case: CaptchaCase) -> None:
    """Require a fresh provider response; never treat a visual state as success."""
    try:
        provider, token, _status, _challenged, reason = asyncio.run(_run(case))
    except Exception as exc:  # noqa: BLE001 - redact potentially token-bearing provider errors
        pytest.fail(f"{case.name}: {type(exc).__name__} during browser E2E", pytrace=False)
    if provider != case.provider or not token:
        pytest.fail(
            f"{case.name}: no fresh {case.provider} response "
            f"(observed {provider or 'none'}; reason {reason or 'none'})",
            pytrace=False,
        )


def test_cloudflare_interstitial() -> None:
    """Require an observed Cloudflare interstitial and a cleared target page."""
    target = os.getenv("PROWL_E2E_INTERSTITIAL_URL")
    if not target:
        pytest.skip("set PROWL_E2E_INTERSTITIAL_URL to a site that reliably serves a Cloudflare interstitial")
    case = CaptchaCase("cloudflare-interstitial", target, "turnstile", timeout=120)
    try:
        _provider, _token, status, cleared, _reason = asyncio.run(_run(case, interstitial=True))
    except Exception as exc:  # noqa: BLE001 - redact potentially token-bearing provider errors
        pytest.fail(f"cloudflare-interstitial: {type(exc).__name__} during browser E2E", pytrace=False)
    if not cleared or status != 200:
        pytest.fail("Cloudflare interstitial was not observed and cleared", pytrace=False)
