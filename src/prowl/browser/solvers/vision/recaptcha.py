"""Opt-in reCAPTCHA v2 image challenge adapter for the selected browser page."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from prowl.browser.solvers.checkbox import _RECAPTCHA_V2, _fresh_token, _read_response_values
from prowl.browser.solvers.vision.origins import _canonical_origin
from prowl.browser.solvers.vision.schemas import (
    ERROR_REASON,
    PAGE_ACCEPTED_REASON,
    TIMED_OUT_REASON,
    CaptchaAttemptResult,
    OutcomeStatus,
    VisionOutcome,
)
from prowl.vision.schemas import VisionError
from prowl.vision.vlm import REQUEST_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Frame, Locator, Page
    from pydoll.browser.tab import Tab

    from prowl.vision.vlm import CloudVlmClient

_CHALLENGE_HOSTS: Final = frozenset({"www.google.com", "www.recaptcha.net"})
_CHALLENGE_PATHS: Final = frozenset({"/recaptcha/api2/bframe", "/recaptcha/enterprise/bframe"})
_TARGET: Final = "#rc-imageselect-target"
_PROMPT: Final = ".rc-imageselect-desc-wrapper"
_VERIFY: Final = "#recaptcha-verify-button"
_REJECTION: Final = (
    ".rc-imageselect-error-select-more, .rc-imageselect-error-dynamic-more, "
    ".rc-imageselect-error-select-something, .rc-imageselect-error-invalid-response"
)
_TILES_READY: Final = """() => {
  const target = document.querySelector('#rc-imageselect-target');
  const tiles = target ? Array.from(target.querySelectorAll('td')) : [];
  let imgs = 0, ready = 0;
  for (const tile of tiles) {
    for (const img of tile.querySelectorAll('img')) {
      imgs += 1;
      if (img.complete && img.naturalWidth > 0) ready += 1;
    }
  }
  return {imgs: imgs, ready: ready};
}"""
_MAX_PAGES: Final = 8
_MAX_EXTRA_PAGES: Final = 2
_MAX_REPLACEMENTS: Final = 8
_MAX_PROMPT_LENGTH: Final = 1024
_POLL_SECONDS: Final = 0.15
_TOKEN_POLL_SECONDS: Final = 0.05
_TRANSITION_SECONDS: Final = 1.0
_READY_SECONDS: Final = 5.0
_READY_GRACE: Final = 0.5


def _is_challenge_frame(url: str) -> bool:
    parts = urlsplit(url)
    return (
        parts.scheme == "https"
        and parts.hostname in _CHALLENGE_HOSTS
        and parts.path in _CHALLENGE_PATHS
        and parts.port in (None, 443)
        and not parts.username
        and not parts.password
    )


@dataclass(frozen=True, slots=True)
class _Grid:
    frame: Frame
    frame_url: str
    prompt: str
    rows: int
    tiles: tuple[Locator, ...]
    target: Locator
    verify: Locator
    image: bytes


@dataclass(slots=True)
class _Context:
    page: Page
    origin: str
    client: CloudVlmClient
    initial_tokens: frozenset[str]
    deadline: float
    native_tab: Tab | None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()


def _outcome(status: OutcomeStatus, reason: str) -> CaptchaAttemptResult:
    return CaptchaAttemptResult(outcome=VisionOutcome(status=status, reason=reason), token=None)


def _current(context: _Context, grid: _Grid | None = None) -> bool:
    if _canonical_origin(context.page.url) != context.origin or context.remaining() <= 0:
        return False
    if grid is None:
        return True
    return grid.frame.url == grid.frame_url and grid.frame in context.page.frames


async def _visible_tiles(target: Locator) -> tuple[Locator, ...] | None:
    rows = target.locator("tr")
    count = await rows.count()
    if count not in (3, 4) or await target.locator("td").count() != count * count:
        return None
    for row in range(count):
        if await rows.nth(row).locator("td").count() != count:
            return None
    tiles = tuple(target.locator("td").nth(index) for index in range(count * count))
    for tile in tiles:
        if not await tile.is_visible():
            return None
    return tiles


async def _wait_for_tiles_ready(context: _Context, frame: Frame) -> bool:
    """Wait for decoded imagery, not just tile elements.

    Layouts without image elements get a short grace period instead.
    """
    started = time.monotonic()
    stop = min(context.deadline, started + _READY_SECONDS)
    while True:
        try:
            state = await frame.evaluate(_TILES_READY)
        except PlaywrightError:
            return False
        imgs = int(state.get("imgs") or 0)
        ready = int(state.get("ready") or 0)
        if imgs and ready == imgs:
            return True
        now = time.monotonic()
        if not imgs and now - started >= _READY_GRACE:
            return True
        if now >= stop:
            return False
        await asyncio.sleep(min(_POLL_SECONDS, max(stop - now, 0)))


async def _read_grid(context: _Context) -> _Grid | CaptchaAttemptResult:
    if not _current(context):
        return _outcome("needs_human", "origin_lost")
    frames = [frame for frame in context.page.frames if _is_challenge_frame(frame.url)]
    if len(frames) != 1:
        return _outcome("needs_human", "challenge_unavailable")
    frame = frames[0]
    element = await frame.frame_element()
    if element is None or not await element.is_visible():
        return _outcome("needs_human", "challenge_unavailable")
    view = frame.locator("#rc-imageselect")
    target = frame.locator(_TARGET)
    prompt = frame.locator(_PROMPT)
    verify = frame.locator(_VERIFY)
    for locator in (view, target, prompt, verify):
        if await locator.count() != 1 or not await locator.is_visible():
            return _outcome("needs_human", "challenge_unavailable")
    tiles = await _visible_tiles(target)
    if tiles is None:
        return _outcome("unsupported", "unsupported_layout")
    if not await _wait_for_tiles_ready(context, frame):
        return _outcome("needs_human", "challenge_unavailable")
    instruction = " ".join((await prompt.inner_text()).split())
    if not instruction or len(instruction) > _MAX_PROMPT_LENGTH:
        return _outcome("unsupported", "unsupported_layout")
    image = await target.screenshot(scale="css", timeout=max(1, context.remaining() * 1000))
    if not _current(context):
        return _outcome("needs_human", "origin_lost")
    return _Grid(frame, frame.url, instruction, math.isqrt(len(tiles)), tiles, target, verify, image)


async def _tile_signature(tile: Locator) -> str:
    """Track source imagery, ignoring the selected-checkbox CSS state."""
    return await tile.evaluate(
        """el => Array.from(el.querySelectorAll('img')).map(img =>
          [img.currentSrc.length, img.currentSrc.slice(-256)].join(':')).join('|') + '|' +
          Array.from(el.querySelectorAll('div')).map(div => {
            const image = getComputedStyle(div).backgroundImage;
            return [image.length, image.slice(-256)].join(':');
          }).join('|')"""
    )


async def _click(context: _Context, grid: _Grid, locator: Locator) -> bool:
    if not _current(context, grid) or not await locator.is_visible():
        return False
    if context.native_tab is None:
        await locator.click(timeout=max(1, context.remaining() * 1000))
        return _current(context, grid)
    await locator.scroll_into_view_if_needed(timeout=max(1, context.remaining() * 1000))
    box = await locator.bounding_box()
    if box is None or any(not math.isfinite(box[key]) for key in ("x", "y", "width", "height")):
        return False
    if box["width"] <= 0 or box["height"] <= 0 or not _current(context, grid):
        return False
    prompt = grid.frame.locator(_PROMPT)
    if await prompt.count() != 1 or " ".join((await prompt.inner_text()).split()) != grid.prompt:
        return False
    if await grid.target.locator("td").count() != len(grid.tiles):
        return False
    mouse = context.native_tab.mouse
    await mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, humanize=True)
    if not _current(context, grid) or await locator.bounding_box() != box:
        return False
    await mouse.down()
    try:
        await asyncio.sleep(min(0.08, max(context.remaining(), 0)))
    finally:
        await mouse.up()
    return _current(context, grid)


async def _fresh_response(context: _Context) -> CaptchaAttemptResult | None:
    if not _current(context):
        return _outcome("needs_human", "origin_lost")
    try:
        values = await _read_response_values(context.page, _RECAPTCHA_V2.response_selector)
    except PlaywrightError:
        # The provider navigates away as soon as it accepts an answer, which destroys the
        # execution context mid-read. That is a page transition, not a read failure.
        return None
    token = _fresh_token(values, list(context.initial_tokens))
    if token is None:
        return None
    return CaptchaAttemptResult(outcome=VisionOutcome(status="handled", reason=PAGE_ACCEPTED_REASON), token=token)


async def _rejection_visible(frame: Frame, *, include_text: bool = False) -> bool:
    errors = frame.locator(_REJECTION)
    for index in range(await errors.count()):
        if await errors.nth(index).is_visible():
            return True
    if not include_text:
        return False
    # Google's /sorry/ page can show this after the replacement puzzle appears.
    retry = frame.get_by_text("Please try again")
    return any([await retry.nth(index).is_visible() for index in range(await retry.count())])


async def _next_page_visible(grid: _Grid, previous_sources: tuple[str, ...]) -> bool:
    prompt = grid.frame.locator(_PROMPT)
    if await prompt.count() != 1 or not await prompt.is_visible():
        return False
    if " ".join((await prompt.inner_text()).split()) != grid.prompt:
        return True
    if await grid.frame.locator(".rc-imageselect-tileselected").count():
        return False
    target = grid.frame.locator(_TARGET)
    if await target.count() != 1 or await target.locator("td").count() != len(grid.tiles):
        return False
    return tuple([await _tile_signature(tile) for tile in grid.tiles]) != previous_sources


async def _wait_after_verify(
    context: _Context, grid: _Grid, previous_sources: tuple[str, ...]
) -> CaptchaAttemptResult | Literal["next", "rejected"]:
    """Poll the brief token window before slower page-transition checks.

    A delayed structural check can still miss a token; this is not guaranteed capture.
    """
    stop = min(context.deadline, time.monotonic() + 6.0)
    rejected = False
    next_check = time.monotonic() + _TRANSITION_SECONDS
    while time.monotonic() < stop:
        accepted = await _fresh_response(context)
        if isinstance(accepted, CaptchaAttemptResult):
            return accepted
        if not _current(context, grid):
            return _outcome("needs_human", "stale_revision")
        now = time.monotonic()
        if now >= next_check:
            next_check = now + _TRANSITION_SECONDS
            try:
                rejected |= await _rejection_visible(grid.frame)
                if await _next_page_visible(grid, previous_sources):
                    return "rejected" if rejected else "next"
            except PlaywrightError:
                return _outcome("needs_human", "stale_revision")
        await asyncio.sleep(min(_TOKEN_POLL_SECONDS, max(stop - time.monotonic(), 0)))
    return _outcome("needs_human", "page_not_accepted")


async def _wait_replacements(context: _Context, grid: _Grid, sources: dict[int, str]) -> bool:
    stop = min(context.deadline, time.monotonic() + 5.0)
    original = hashlib.sha256(grid.image).digest()
    while time.monotonic() < stop:
        if not _current(context, grid):
            return False
        if all([await _tile_signature(grid.tiles[index]) != before for index, before in sources.items()]):
            ready = all(
                [
                    await grid.tiles[index].evaluate(
                        "el => [...el.querySelectorAll('img')].every(img => img.complete && img.naturalWidth > 0)"
                    )
                    for index in sources
                ]
            )
            if ready and not await grid.frame.locator(".rc-imageselect-tileselected").count():
                image = await grid.target.screenshot(scale="css", timeout=max(1, context.remaining() * 1000))
                if hashlib.sha256(image).digest() != original:
                    return True
        await asyncio.sleep(min(_POLL_SECONDS, max(stop - time.monotonic(), 0)))
    return False


async def _apply_selection(
    context: _Context, grid: _Grid, indexes: tuple[int, ...], clicked: set[int]
) -> tuple[bool, bool] | CaptchaAttemptResult:
    elimination = "none left" in grid.prompt.lower()
    sources: dict[int, str] = {}
    for index in indexes:
        if index in clicked:
            continue
        tile = grid.tiles[index]
        if elimination:
            sources[index] = await _tile_signature(tile)
        if not await _click(context, grid, tile):
            return _outcome("needs_human", "stale_revision")
        clicked.add(index)
    if not sources:
        return False, bool(indexes)
    if not await _wait_replacements(context, grid, sources):
        return _outcome("needs_human", "replacement_unavailable")
    clicked.difference_update(sources)
    return True, True


async def _plan_tiles(context: _Context, grid: _Grid) -> tuple[int, ...] | CaptchaAttemptResult:
    plan = await context.client.select_points(
        grid.image,
        f"{grid.prompt}. Select centers of matching {grid.rows}x{grid.rows} image tiles; no other points.",
        context.remaining(),
    )
    if not _current(context, grid):
        return _outcome("needs_human", "stale_revision")
    indexes = tuple(sorted({int(point.y * grid.rows) * grid.rows + int(point.x * grid.rows) for point in plan.points}))
    if len(indexes) != len(plan.points):
        return _outcome("needs_human", "ambiguous_selection")
    return indexes


async def _complete_selection(
    context: _Context, grid: _Grid, indexes: tuple[int, ...], clicked: set[int], *, interacted: bool
) -> tuple[Literal["refresh", "verify"], bool] | CaptchaAttemptResult:
    elimination = "none left" in grid.prompt.lower()
    if elimination and not indexes:
        if not interacted:
            return _outcome("needs_human", "empty_selection")
        if await grid.frame.locator(".rc-imageselect-tileselected").count():
            return _outcome("needs_human", "replacement_unavailable")
        return "verify", interacted
    applied = await _apply_selection(context, grid, indexes, clicked)
    if isinstance(applied, CaptchaAttemptResult):
        return applied
    refreshed, newly_interacted = applied
    interacted |= newly_interacted
    if refreshed:
        return "refresh", interacted
    if not interacted:
        return _outcome("needs_human", "empty_selection")
    if elimination:
        return _outcome("needs_human", "replacement_unavailable")
    return "verify", interacted


async def _solve_page(
    context: _Context, first_grid: _Grid | None = None
) -> CaptchaAttemptResult | Literal["next", "rejected"]:
    seen: set[bytes] = set()
    clicked: set[int] = set()
    interacted = False
    for _ in range(_MAX_REPLACEMENTS + 1):
        grid = first_grid if first_grid is not None else await _read_grid(context)
        first_grid = None
        if isinstance(grid, CaptchaAttemptResult):
            return grid
        digest = hashlib.sha256(grid.image).digest()
        if digest in seen:
            return _outcome("needs_human", "stale_revision")
        seen.add(digest)
        indexes = await _plan_tiles(context, grid)
        if isinstance(indexes, CaptchaAttemptResult):
            return indexes
        decision = await _complete_selection(context, grid, indexes, clicked, interacted=interacted)
        if isinstance(decision, CaptchaAttemptResult):
            return decision
        action, interacted = decision
        if action == "refresh":
            continue
        previous_sources = tuple([await _tile_signature(tile) for tile in grid.tiles])
        if not await _click(context, grid, grid.verify):
            return _outcome("needs_human", "stale_revision")
        return await _wait_after_verify(context, grid, previous_sources)
    return _outcome("needs_human", "replacement_limit")


async def _run_pages(context: _Context) -> CaptchaAttemptResult:
    extra_pages = 0
    after_rejection = False
    first_grid: _Grid | None = None
    for _ in range(_MAX_PAGES):
        if first_grid is not None:
            if not _current(context, first_grid):
                return _outcome("needs_human", "stale_revision")
            after_rejection |= await _rejection_visible(first_grid.frame, include_text=True)
        if after_rejection:
            extra_pages += 1
            if extra_pages > _MAX_EXTRA_PAGES:
                return _outcome("needs_human", "page_limit")
        result = await _solve_page(context, first_grid)
        first_grid = None
        if isinstance(result, CaptchaAttemptResult):
            return result
        after_rejection |= result == "rejected"
        if result == "next" and not after_rejection:
            next_grid = await _read_grid(context)
            if isinstance(next_grid, CaptchaAttemptResult):
                return next_grid
            first_grid = next_grid
    return _outcome("needs_human", "page_limit")


async def solve_recaptcha_challenge(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    vlm_client: CloudVlmClient,
    initial_tokens: frozenset[str],
    native_tab: Tab | None = None,
) -> CaptchaAttemptResult:
    """Attempt bounded 3x3/4x4 reCAPTCHA image pages on the existing tab.

    Only a fresh ``g-recaptcha-response`` can be handled. After an observed rejection, at
    most two extra fresh provider pages are attempted, including normal follow-up pages;
    earlier required pages do not spend that budget. Unobserved rejection cannot be counted.
    Unsupported layouts, unclear transitions, stale origin/frame or empty first plans stop.
    """
    origin = _canonical_origin(page.url)
    if origin is None:
        return _outcome("needs_human", "origin_lost")
    remaining = seconds_remaining()
    if remaining <= 0:
        return _outcome("timed_out", TIMED_OUT_REASON)
    context = _Context(page, origin, vlm_client, initial_tokens, time.monotonic() + remaining, native_tab)
    try:
        async with asyncio.timeout(remaining):
            return await _run_pages(context)
    except (TimeoutError, PlaywrightTimeoutError):
        return _outcome("timed_out", TIMED_OUT_REASON)
    except VisionError as error:
        return (
            _outcome("timed_out", TIMED_OUT_REASON)
            if error.code == REQUEST_TIMEOUT
            else _outcome("error", ERROR_REASON)
        )
    except Exception:  # noqa: BLE001 - keep browser and VLM failure categories fixed
        return _outcome("error", ERROR_REASON)


__all__ = ["solve_recaptcha_challenge"]
