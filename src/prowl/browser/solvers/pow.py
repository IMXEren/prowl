"""Local proof-of-work solvers for ALTCHA and Friendly Captcha widgets.

Each solver operates only on the supplied main page. It verifies that exactly one
provider widget is mounted, captures every response value already present, then
triggers the widget's own public API (or clicks the single visible widget-owned
control when no API exists), and polls for a nonempty fresh response token while
the widget reports a solved state. A locally produced token is not server
acceptance, so no form is submitted and no network endpoint is contacted.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Locator, Page

_POLL_INTERVAL_SECONDS: float = 0.25

_ALTCHA_START_JS = """() => {
  const widget = document.querySelector('altcha-widget');
  if (!widget || typeof widget.verify !== 'function') {
    return false;
  }
  widget.verify();
  return true;
}"""

_ALTCHA_SNAPSHOT_JS = """() => {
  const widget = document.querySelector('altcha-widget');
  let state = null;
  if (widget && typeof widget.getState === 'function') {
    state = widget.getState();
  }
  const values = Array.from(
    document.querySelectorAll('input[name="altcha"]'),
    (element) => element.value,
  );
  return { state: state, values: values };
}"""

_FRIENDLY_START_JS = """() => {
  const mount = document.querySelector('.frc-captcha');
  if (!mount) {
    return false;
  }
  const widget = window.frcWidget || mount;
  if (typeof widget.start !== 'function') {
    return false;
  }
  widget.start();
  return true;
}"""

_FRIENDLY_SNAPSHOT_JS = """() => {
  const mount = document.querySelector('.frc-captcha');
  const widget = window.frcWidget || mount;
  let state = null;
  if (widget && typeof widget.getState === 'function') {
    state = widget.getState();
  } else if (mount && mount.dataset && typeof mount.dataset.state === 'string') {
    state = mount.dataset.state;
  }
  const values = Array.from(
    document.querySelectorAll('input[name="frc-captcha-response"], [name="frc-captcha-solution"]'),
    (element) => element.value,
  );
  return { state: state, values: values };
}"""


@dataclass(frozen=True)
class _PowWidget:
    mount_selector: str
    target_selector: str
    start_js: str
    snapshot_js: str
    accepted_states: frozenset[str]
    failed_states: frozenset[str]
    min_token_length: int
    reject_prefix: str


_ALTCHA = _PowWidget(
    mount_selector="altcha-widget",
    target_selector='altcha-widget input[type="checkbox"]',
    start_js=_ALTCHA_START_JS,
    snapshot_js=_ALTCHA_SNAPSHOT_JS,
    accepted_states=frozenset({"verified"}),
    failed_states=frozenset({"error"}),
    min_token_length=1,
    reject_prefix="",
)

_FRIENDLY = _PowWidget(
    mount_selector=".frc-captcha",
    target_selector=".frc-captcha .frc-button",
    start_js=_FRIENDLY_START_JS,
    snapshot_js=_FRIENDLY_SNAPSHOT_JS,
    accepted_states=frozenset({"completed"}),
    failed_states=frozenset({"error"}),
    min_token_length=21,
    reject_prefix=".",
)


@dataclass(frozen=True)
class _Snapshot:
    state: str | None
    values: list[str]


def _parse_snapshot(raw: Any) -> _Snapshot:
    if not isinstance(raw, dict):
        return _Snapshot(state=None, values=[])
    state = raw.get("state")
    values = raw.get("values")
    return _Snapshot(
        state=state if isinstance(state, str) else None,
        values=[value for value in values if isinstance(value, str)] if isinstance(values, list) else [],
    )


def _fresh_token(values: list[str], initial: list[str], widget: _PowWidget) -> str | None:
    for value in values:
        if not value or value in initial:
            continue
        if len(value) < widget.min_token_length:
            continue
        if widget.reject_prefix and value.startswith(widget.reject_prefix):
            continue
        return value
    return None


async def _friendly_frame_button(page: Page) -> Locator | None:
    candidates: list[Locator] = []
    for frame in page.frames:
        url = urlsplit(frame.url)
        host = url.hostname or ""
        if not host.endswith(".frcapi.com") or url.path != "/api/v2/captcha/widget":
            continue
        element = await frame.frame_element()
        if element is None or not await element.is_visible():
            continue
        button = frame.locator("button.button[aria-label]")
        if await button.count() == 1 and await button.is_visible():
            candidates.append(button)
    return candidates[0] if len(candidates) == 1 else None


async def _click_target(page: Page, widget: _PowWidget) -> bool:
    target = page.locator(widget.target_selector)
    count = await target.count()
    if count == 1 and await target.is_visible():
        await target.click()
        return True
    if widget is _FRIENDLY and count == 0:
        button = await _friendly_frame_button(page)
        if button is not None:
            await button.click()
            return True
    return False


async def _poll_token(page: Page, widget: _PowWidget, initial: list[str], deadline: float) -> str | None:
    while True:
        snapshot = _parse_snapshot(await page.evaluate(widget.snapshot_js))
        if snapshot.state in widget.failed_states:
            return None
        token = _fresh_token(snapshot.values, initial, widget)
        if token is not None and (snapshot.state is None or snapshot.state in widget.accepted_states):
            return token
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _solve(page: Page, seconds_remaining: Callable[[], float], widget: _PowWidget) -> str | None:
    remaining = seconds_remaining()
    if remaining <= 0:
        return None
    deadline = time.monotonic() + remaining
    try:
        async with asyncio.timeout(remaining):
            mounts = await page.query_selector_all(widget.mount_selector)
            if len(mounts) != 1:
                return None
            initial = _parse_snapshot(await page.evaluate(widget.snapshot_js)).values
            ready_deadline = min(deadline, time.monotonic() + 3.0)
            while True:
                started = await page.evaluate(widget.start_js)
                if started is True or await _click_target(page, widget):
                    break
                wait = min(_POLL_INTERVAL_SECONDS, ready_deadline - time.monotonic())
                if wait <= 0:
                    return None
                await asyncio.sleep(wait)
            return await _poll_token(page, widget, initial, deadline)
    except (TimeoutError, PlaywrightTimeoutError):
        return None


async def solve_altcha(page: Page, seconds_remaining: Callable[[], float]) -> str | None:
    """Trigger one mounted ALTCHA widget and return a fresh token.

    Returns ``None`` when no single ``altcha-widget`` is mounted, when the widget
    reports a failed state, when the deadline expires, or when no fresh nonempty
    ``input[name="altcha"]`` value appears while the widget reports ``verified``.
    A locally produced token is not server acceptance, so the form is never
    submitted, and cancellation propagates to the caller.
    """
    return await _solve(page, seconds_remaining, _ALTCHA)


async def solve_friendly_captcha(page: Page, seconds_remaining: Callable[[], float]) -> str | None:
    """Trigger one mounted Friendly Captcha widget and return a fresh token.

    Returns ``None`` when no single ``.frc-captcha`` is mounted, when the widget
    reports a failed state, when the deadline expires, or when no fresh response
    of more than twenty characters (and not a ``.``-prefixed placeholder) appears
    while the widget reports ``completed``. A locally produced token is not
    server acceptance, so the form is never submitted, and cancellation
    propagates to the caller.
    """
    return await _solve(page, seconds_remaining, _FRIENDLY)
