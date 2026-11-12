# pyright: reportPrivateUsage=false
"""Provider checkbox solvers for reCAPTCHA v2 and hCaptcha.

Each solver operates only on the supplied page: it clicks exactly one visible
provider checkbox and returns the provider response token only when the token
is nonempty and fresh relative to every response value already present before
the click. A checked box is never treated as a solution. The hCaptcha solver also
accepts an optional vision challenge attempt: a challenge frame that is already
present is handed straight to it, and a challenge that appears after the one
checkbox click interrupts token polling for that same attempt, while the
reCAPTCHA path is unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from playwright.async_api import Locator, Page

    #: Bound challenge attempt: initial response values in, an accepted fresh token or ``None``.
    type CaptchaAttempt = Callable[[frozenset[str]], Awaitable[str | None]]

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


async def _active_challenge_frame_count(page: Page, widget: _CheckboxWidget) -> int:
    """Count only a visible challenge for the selected checkbox provider."""
    if widget is _RECAPTCHA_V2:
        from prowl.browser.solvers.vision.recaptcha import _is_challenge_frame  # noqa: PLC0415 - enabled-only

        selector = "#rc-imageselect"
    else:
        # Adapter-only lazy import.
        from prowl.browser.solvers.vision.hcaptcha import (  # noqa: PLC0415
            _is_challenge_frame,
        )

        selector = ".challenge-view"
    frames = [frame for frame in page.frames if _is_challenge_frame(frame.url)]
    if len(frames) != 1:
        return len(frames)
    view = frames[0].locator(selector)
    return int(await view.count() == 1 and await view.is_visible())


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


async def _challenge_token(
    page: Page,
    widget: _CheckboxWidget,
    challenge_attempt: CaptchaAttempt,
    initial: list[str] | None = None,
) -> str | None:
    """Run the wired provider challenge attempt and return only a fresh token.

    The attempt's own contract already yields a token only for its ``handled``/``page_accepted``
    outcome; this still refuses any value already present in the pre-attempt response values, so
    a stale or reused token is never reported as a solution.
    """
    if await _active_challenge_frame_count(page, widget) != 1:
        return None
    values = initial if initial is not None else await _read_response_values(page, widget.response_selector)
    token = await challenge_attempt(frozenset(values))
    return token if token and token not in values else None


def _fresh_token(values: list[str], initial: list[str]) -> str | None:
    return next((value for value in values if value and value not in initial), None)


async def _wait_for_fresh_token(
    page: Page,
    selector: str,
    initial: list[str],
    deadline: float,
    *,
    challenge_observed: Callable[[], Awaitable[int]] | None = None,
) -> tuple[str | None, bool]:
    """Poll for a fresh response value, or report an observed challenge frame instead.

    Returns ``(token, False)`` when a fresh token appears, ``(None, True)`` when no token appears
    but ``challenge_observed`` reports one valid challenge frame, and ``(None, False)`` at the
    deadline. The token is checked first on every poll, so an already accepted token still wins
    over a challenge frame.
    """
    while True:
        token = _fresh_token(await _read_response_values(page, selector), initial)
        if token is not None:
            return token, False
        if challenge_observed is not None and await challenge_observed():
            return None, True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None, False
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _solve_checkbox(
    page: Page,
    seconds_remaining: Callable[[], float],
    widget: _CheckboxWidget,
    *,
    challenge_attempt: CaptchaAttempt | None = None,
) -> str | None:
    remaining = seconds_remaining()
    if remaining <= 0:
        return None
    try:
        async with asyncio.timeout(remaining):
            if not any(_matches_widget_frame(frame.url, widget) for frame in page.frames):
                return None
            deadline = time.monotonic() + remaining
            if challenge_attempt is not None and await _active_challenge_frame_count(page, widget):
                return await _challenge_token(page, widget, challenge_attempt)
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
            interrupt = partial(_active_challenge_frame_count, page, widget) if challenge_attempt is not None else None
            token, challenge_seen = await _wait_for_fresh_token(
                page,
                widget.response_selector,
                initial,
                deadline,
                challenge_observed=interrupt,
            )
            if challenge_seen and challenge_attempt is not None:
                return await _challenge_token(page, widget, challenge_attempt, initial)
            return token
    except (TimeoutError, PlaywrightTimeoutError):
        return None


async def solve_recaptcha_checkbox(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    challenge_attempt: CaptchaAttempt | None = None,
) -> str | None:
    """Click one visible reCAPTCHA v2 checkbox and return a fresh token.

    Returns ``None`` when no single visible provider checkbox is present, when
    the deadline expires, or when no fresh ``g-recaptcha-response`` appears.
    Cancellation propagates to the caller and the site form is never submitted.
    """
    return await _solve_checkbox(page, seconds_remaining, _RECAPTCHA_V2, challenge_attempt=challenge_attempt)


async def solve_hcaptcha_checkbox(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    challenge_attempt: CaptchaAttempt | None = None,
) -> str | None:
    """Resolve one visible hCaptcha checkbox or its optional vision challenge to a fresh token.

    Without ``challenge_attempt`` this clicks one visible checkbox and returns a fresh token
    exactly as before. With it, an already present challenge frame is handed straight to the
    vision adapter; otherwise the checkbox is clicked once, and if no token appears but a
    visible challenge does, the adapter runs against the same recorded baseline. A checked box
    is never treated as a solution.

    Returns ``None`` when no single visible provider checkbox is present, when
    the deadline expires, or when no fresh ``h-captcha-response`` appears.
    Cancellation propagates to the caller and the site form is never submitted.
    """
    return await _solve_checkbox(page, seconds_remaining, _HCAPTCHA, challenge_attempt=challenge_attempt)


def _provider_for_frame(url: str) -> str | None:
    """Return the recognized provider name for a frame URL, or ``None``."""
    if _matches_widget_frame(url, _RECAPTCHA_V2):
        return "recaptcha"
    if _matches_widget_frame(url, _HCAPTCHA):
        return "hcaptcha"
    return None


async def solve_visible_checkbox(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    challenge_attempt: CaptchaAttempt | None = None,
) -> tuple[str, str] | None:
    """Solve the single recognized provider checkbox among *page* frames.

    Exactly one provider must match an existing strict frame matcher, and that
    provider must have one visible checkbox; this dispatches and returns
    ``("recaptcha" | "hcaptcha", fresh_token)`` only when a fresh response token
    actually appears. The optional ``challenge_attempt`` is offered only to the selected provider;
    without one, both checkbox paths are unchanged. When no recognized frame is present, or when more than
    one is, it returns ``None`` without clicking. Cancellation propagates to the caller and the
    site form is never submitted.
    """
    providers = {provider for frame in page.frames if (provider := _provider_for_frame(frame.url))}
    if len(providers) != 1:
        return None
    provider = providers.pop()
    if provider == "recaptcha":
        token = await solve_recaptcha_checkbox(page, seconds_remaining, challenge_attempt=challenge_attempt)
    else:
        token = await solve_hcaptcha_checkbox(page, seconds_remaining, challenge_attempt=challenge_attempt)
    if token is None:
        return None
    return provider, token
