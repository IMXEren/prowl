"""Select one local CAPTCHA provider on the requested browser page."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from prowl.browser.solvers.checkbox import _provider_for_frame, solve_visible_checkbox
from prowl.browser.solvers.pow import solve_altcha, solve_friendly_captcha

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from playwright.async_api import Page

    from prowl.browser.solvers.checkbox import CaptchaAttempt


async def solve_visible_captcha(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    provider_attempts: Mapping[str, CaptchaAttempt] | None = None,
) -> tuple[str, str] | None:
    """Attempt one unambiguous provider and return only a fresh provider response."""
    remaining = seconds_remaining()
    if remaining <= 0:
        return None
    try:
        async with asyncio.timeout(remaining):
            loading_geetest = (
                provider_attempts
                and "geetest" in provider_attempts
                and await page.query_selector_all('[class*="geetest"],[id*="geetest"]')
            )
            detection_deadline = time.monotonic() + min(12.0 if loading_geetest else 3.0, remaining)
            while True:
                checkbox_providers = {
                    provider for frame in page.frames if (provider := _provider_for_frame(frame.url)) is not None
                }
                altcha = await page.query_selector_all("altcha-widget")
                friendly = await page.query_selector_all(".frc-captcha")
                geetest_count = len(
                    [widget for widget in await page.query_selector_all(".geetest_holder") if await widget.is_visible()]
                )
                turnstile = any(urlsplit(frame.url).hostname == "challenges.cloudflare.com" for frame in page.frames)
                count = len(checkbox_providers) + len(altcha) + len(friendly) + geetest_count + int(turnstile)
                if count > 1:
                    return None
                if count == 1:
                    break
                wait = min(0.1, detection_deadline - time.monotonic(), seconds_remaining())
                if wait <= 0:
                    return None
                await asyncio.sleep(wait)
            if turnstile or geetest_count:
                provider = "turnstile" if turnstile else "geetest"
                attempt = provider_attempts.get(provider) if provider_attempts else None
                response = await attempt(frozenset()) if attempt is not None else None
                return (provider, response) if response else None
            if checkbox_providers:
                return await solve_visible_checkbox(
                    page,
                    seconds_remaining,
                    challenge_attempt=(
                        provider_attempts.get(next(iter(checkbox_providers))) if provider_attempts else None
                    ),
                )
            provider, solver = ("altcha", solve_altcha) if altcha else ("friendly", solve_friendly_captcha)
            response = await solver(page, seconds_remaining)
            return (provider, response) if response is not None else None
    except (TimeoutError, PlaywrightTimeoutError):
        return None
