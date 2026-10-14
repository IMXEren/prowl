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
    from collections.abc import Awaitable, Callable

    from playwright.async_api import Page


async def solve_visible_captcha(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    turnstile_solver: Callable[[], Awaitable[str | None]] | None = None,
) -> tuple[str, str] | None:
    """Attempt one unambiguous provider and return only a fresh response value."""
    remaining = seconds_remaining()
    if remaining <= 0:
        return None
    try:
        async with asyncio.timeout(remaining):
            detection_deadline = time.monotonic() + min(3.0, remaining)
            while True:
                checkbox_providers = {
                    provider for frame in page.frames if (provider := _provider_for_frame(frame.url)) is not None
                }
                altcha = await page.query_selector_all("altcha-widget")
                friendly = await page.query_selector_all(".frc-captcha")
                turnstile = any(urlsplit(frame.url).hostname == "challenges.cloudflare.com" for frame in page.frames)
                count = len(checkbox_providers) + len(altcha) + len(friendly) + int(turnstile)
                if count > 1:
                    return None
                if count == 1:
                    break
                wait = min(0.1, detection_deadline - time.monotonic(), seconds_remaining())
                if wait <= 0:
                    return None
                await asyncio.sleep(wait)
            if turnstile:
                response = await turnstile_solver() if turnstile_solver is not None else None
                return ("turnstile", response) if response else None
            if checkbox_providers:
                return await solve_visible_checkbox(page, seconds_remaining)
            provider, solver = ("altcha", solve_altcha) if altcha else ("friendly", solve_friendly_captcha)
            response = await solver(page, seconds_remaining)
            return (provider, response) if response is not None else None
    except (TimeoutError, PlaywrightTimeoutError):
        return None
