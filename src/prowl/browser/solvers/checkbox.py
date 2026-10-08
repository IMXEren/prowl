"""Provider checkbox solvers for reCAPTCHA v2 and hCaptcha.

Each solver operates only on the supplied page: it clicks exactly one visible
provider checkbox and returns the provider response token only when the token
is nonempty and fresh relative to every response value already present before
the click. A checked box is never treated as a solution, and there is no
transcription, submission, or fallback click path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Locator, Page

_POLL_INTERVAL_SECONDS: float = 0.2
_MILLISECONDS_PER_SECOND: int = 1000
_READ_RESPONSE_VALUES = "selector => Array.from(document.querySelectorAll(selector), element => element.value)"


@dataclass(frozen=True)
class _CheckboxWidget:
    hostnames: frozenset[str]
    frame_path_markers: tuple[str, ...]
    checkbox_selector: str
    response_selector: str


_RECAPTCHA_V2 = _CheckboxWidget(
    hostnames=frozenset({"www.google.com", "www.recaptcha.net"}),
    frame_path_markers=("/recaptcha/", "/anchor"),
    checkbox_selector="#recaptcha-anchor",
    response_selector='textarea[name="g-recaptcha-response"], input[name="g-recaptcha-response"]',
)

_HCAPTCHA = _CheckboxWidget(
    hostnames=frozenset({"newassets.hcaptcha.com", "assets.hcaptcha.com", "hcaptcha.com", "www.hcaptcha.com"}),
    frame_path_markers=("/captcha/", "/static/hcaptcha.html"),
    checkbox_selector="#checkbox",
    response_selector='textarea[name="h-captcha-response"], input[name="h-captcha-response"]',
)


def _matches_widget_frame(url: str, widget: _CheckboxWidget) -> bool:
    parts = urlsplit(url)
    if parts.hostname not in widget.hostnames:
        return False
    return all(marker in parts.path for marker in widget.frame_path_markers)


async def _locate_checkbox(page: Page, widget: _CheckboxWidget) -> tuple[Locator | None, bool]:
    candidates: list[Locator] = []
    for frame in page.frames:
        if not _matches_widget_frame(frame.url, widget):
            continue
        element = await frame.frame_element()
        if element is None or not await element.is_visible():
            continue
        checkbox = frame.locator(widget.checkbox_selector)
        count = await checkbox.count()
        if count > 1:
            return None, True
        if count == 1 and await checkbox.is_visible():
            candidates.append(checkbox)
    return (candidates[0], False) if len(candidates) == 1 else (None, len(candidates) > 1)


async def _read_response_values(page: Page, selector: str) -> list[str]:
    values = await page.evaluate(_READ_RESPONSE_VALUES, selector)
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str)]


def _fresh_token(values: list[str], initial: list[str]) -> str | None:
    return next((value for value in values if value and value not in initial), None)


async def _wait_for_fresh_token(page: Page, selector: str, initial: list[str], deadline: float) -> str | None:
    while True:
        token = _fresh_token(await _read_response_values(page, selector), initial)
        if token is not None:
            return token
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _solve_checkbox(page: Page, seconds_remaining: Callable[[], float], widget: _CheckboxWidget) -> str | None:
    remaining = seconds_remaining()
    if remaining <= 0:
        return None
    try:
        async with asyncio.timeout(remaining):
            if not any(_matches_widget_frame(frame.url, widget) for frame in page.frames):
                return None
            deadline = time.monotonic() + remaining
            ready_deadline = min(deadline, time.monotonic() + 3.0)
            while True:
                checkbox, ambiguous = await _locate_checkbox(page, widget)
                if ambiguous:
                    return None
                if checkbox is not None:
                    break
                wait = min(_POLL_INTERVAL_SECONDS, ready_deadline - time.monotonic(), seconds_remaining())
                if wait <= 0:
                    return None
                await asyncio.sleep(wait)
            initial = await _read_response_values(page, widget.response_selector)
            await checkbox.click(timeout=max(1, seconds_remaining() * _MILLISECONDS_PER_SECOND))
            return await _wait_for_fresh_token(page, widget.response_selector, initial, deadline)
    except (TimeoutError, PlaywrightTimeoutError):
        return None


async def solve_recaptcha_checkbox(page: Page, seconds_remaining: Callable[[], float]) -> str | None:
    """Click one visible reCAPTCHA v2 checkbox and return a fresh token.

    Returns ``None`` when no single visible provider checkbox is present, when
    the deadline expires, or when no fresh ``g-recaptcha-response`` appears.
    Cancellation propagates to the caller and the site form is never submitted.
    """
    return await _solve_checkbox(page, seconds_remaining, _RECAPTCHA_V2)


async def solve_hcaptcha_checkbox(page: Page, seconds_remaining: Callable[[], float]) -> str | None:
    """Click one visible hCaptcha checkbox and return a fresh token.

    Returns ``None`` when no single visible provider checkbox is present, when
    the deadline expires, or when no fresh ``h-captcha-response`` appears.
    Cancellation propagates to the caller and the site form is never submitted.
    """
    return await _solve_checkbox(page, seconds_remaining, _HCAPTCHA)


def _provider_for_frame(url: str) -> str | None:
    """Return the recognized provider name for a frame URL, or ``None``."""
    if _matches_widget_frame(url, _RECAPTCHA_V2):
        return "recaptcha"
    if _matches_widget_frame(url, _HCAPTCHA):
        return "hcaptcha"
    return None


async def solve_visible_checkbox(page: Page, seconds_remaining: Callable[[], float]) -> tuple[str, str] | None:
    """Solve the single recognized provider checkbox among *page* frames.

    Exactly one provider must match an existing strict frame matcher, and that
    provider must have one visible checkbox; this dispatches and returns
    ``("recaptcha" | "hcaptcha", fresh_token)`` only when a fresh response token
    actually appears. When no recognized frame is present, or when more than
    one is, it returns ``None`` without clicking. Cancellation propagates to the
    caller and the site form is never submitted.
    """
    providers = {provider for frame in page.frames if (provider := _provider_for_frame(frame.url))}
    if len(providers) != 1:
        return None
    provider = providers.pop()
    solver = solve_recaptcha_checkbox if provider == "recaptcha" else solve_hcaptcha_checkbox
    token = await solver(page, seconds_remaining)
    if token is None:
        return None
    return provider, token
