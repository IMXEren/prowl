# pyright: reportPrivateUsage=false
"""hCaptcha tile and canvas challenges in the selected page."""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import parse_qsl, urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from prowl.browser.solvers.checkbox import _HCAPTCHA, _fresh_token, _read_response_values
from prowl.browser.solvers.vision.origins import _canonical_origin, _page_origin_allowed
from prowl.browser.solvers.vision.schemas import (
    ERROR_REASON,
    PAGE_ACCEPTED_REASON,
    TIMED_OUT_REASON,
    CaptchaAttemptResult,
    VisionOutcome,
)
from prowl.vision.schemas import VisionError
from prowl.vision.vlm import (
    MAX_INSTRUCTION_CHARS,
    REQUEST_TIMEOUT,
    CanvasActionPlan,
    CanvasClick,
    CanvasDrag,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Frame, Locator, Page, Position
    from pydoll.browser.tab import Tab

    from prowl.vision.vlm import CloudVlmClient, Point

VLM_INSTRUCTION: Final = "Select every tile matching the visible challenge instruction in the image"

EMPTY_SELECTION_REASON: Final = "empty_selection"
STALE_REVISION_REASON: Final = "stale_revision"
PAGE_NOT_ACCEPTED_REASON: Final = "page_not_accepted"
ORIGIN_LOST_REASON: Final = "origin_lost"
CHALLENGE_UNAVAILABLE_REASON: Final = "challenge_unavailable"

_CHALLENGE_HOSTNAME: Final = "newassets.hcaptcha.com"
_CHALLENGE_PATH_PREFIX: Final = "/captcha/v1/"

_CHALLENGE_VIEW_SELECTOR: Final = ".challenge-view"
_TASK_SELECTOR: Final = "div.task"
_CANVAS_SELECTOR: Final = "canvas[role=img]"
_PROMPT_SELECTOR: Final = "h2.prompt-text"
_SUBMIT_SELECTOR: Final = ".button-submit.button"
_BREADCRUMB_SELECTOR: Final = ".challenge-breadcrumbs .Crumb"
_MAX_ROUNDS: Final = 8
_MAX_EXTRA_PAGES: Final = 2
_ARIA_LABEL_ATTRIBUTE: Final = "aria-label"
_SKIP_LABEL: Final = "skip"
_TASK_INDEXES: Final = tuple(range(1, 10))
_MAX_CANVAS_DIMENSION: Final = 10_000.0

_POLL_INTERVAL_SECONDS: Final = 0.2
_POLL_MARGIN_SECONDS: Final = 0.05
_MILLISECONDS_PER_SECOND: Final = 1000
_TIMEOUT_CODES: Final = frozenset({REQUEST_TIMEOUT})


@dataclass(frozen=True, slots=True)
class _TaskGrid:
    """The nine visible challenge tiles keyed by their 1-based aria-label index."""

    locators: dict[int, Locator]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GridChallenge:
    """One read challenge iframe carrying the nine-tile grid and the view screenshot."""

    frame_url: str
    grid: _TaskGrid
    submit: Locator
    screenshot: bytes


@dataclass(frozen=True, slots=True)
class _CanvasChallenge:
    """One read challenge iframe carrying a single canvas, its prompt and CSS click surface."""

    frame_url: str
    prompt: str
    canvas: Locator
    submit: Locator
    screenshot: bytes
    width: float
    height: float


_Challenge = _GridChallenge | _CanvasChallenge
_RoundMarker = tuple[int, int]


@dataclass(slots=True)
class _ExtraPageBudget:
    """New provider puzzle pages permitted after a rejected answer."""

    remaining: int

    def spend(self) -> bool:
        """Consume one extra page, returning ``False`` once the pool is exhausted."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


@dataclass(frozen=True, slots=True)
class _RejectedPage:
    """A fresh provider puzzle after a breadcrumb reset or changed same-marker prompt."""

    marker: _RoundMarker


@dataclass(frozen=True, slots=True)
class _RunContext:
    """One selected page and its bound origins, VLM client and absolute deadline."""

    page: Page
    allowed: frozenset[str]
    client: CloudVlmClient
    initial_tokens: frozenset[str]
    deadline: float
    seconds_remaining: Callable[[], float]
    native_tab: Tab | None


def _needs_human(reason: str) -> CaptchaAttemptResult:
    return CaptchaAttemptResult(outcome=VisionOutcome(status="needs_human", reason=reason), token=None)


def _timed_out() -> CaptchaAttemptResult:
    return CaptchaAttemptResult(outcome=VisionOutcome(status="timed_out", reason=TIMED_OUT_REASON), token=None)


def _error() -> CaptchaAttemptResult:
    return CaptchaAttemptResult(outcome=VisionOutcome(status="error", reason=ERROR_REASON), token=None)


def _origin_ok(context: _RunContext) -> bool:
    return _page_origin_allowed(context.page, context.allowed)


def _guard_origin(context: _RunContext) -> CaptchaAttemptResult | None:
    return None if _origin_ok(context) else _needs_human(ORIGIN_LOST_REASON)


def _is_image_bytes(value: object) -> bool:
    return isinstance(value, bytes) and bool(value)


def _is_challenge_frame(url: object) -> bool:
    """Accept an HTTPS hCaptcha frame with one challenge marker in its query or fragment."""
    if not isinstance(url, str) or not url:
        return False
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or parts.hostname != _CHALLENGE_HOSTNAME:
        return False
    if not parts.path.startswith(_CHALLENGE_PATH_PREFIX):
        return False
    markers = [value for data in (parts.query, parts.fragment) for key, value in parse_qsl(data) if key == "frame"]
    return markers == ["challenge"]


async def _find_challenge_frame(page: Page) -> Frame | None:
    """Return the single matching challenge iframe, or ``None`` when it is absent or ambiguous."""
    frames = [frame for frame in page.frames if _is_challenge_frame(frame.url)]
    return frames[0] if len(frames) == 1 else None


async def _single_visible(frame: Frame, selector: str) -> Locator | None:
    locator = frame.locator(selector)
    if await locator.count() != 1:
        return None
    return locator if await locator.is_visible() else None


async def _has_visible_task(frame: Frame) -> bool:
    tasks = frame.locator(_TASK_SELECTOR)
    for position in range(await tasks.count()):
        if await tasks.nth(position).is_visible():
            return True
    return False


def _is_canvas_dimension(value: object) -> bool:
    """Return whether *value* is a finite, positive, bounded CSS pixel dimension."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and 0.0 < value <= _MAX_CANVAS_DIMENSION
    )


async def _read_canvas_size(context: _RunContext, canvas: Locator) -> tuple[float, float] | CaptchaAttemptResult:
    bounds = await canvas.bounding_box()
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not isinstance(bounds, dict):
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    width = bounds.get("width")
    height = bounds.get("height")
    if not _is_canvas_dimension(width) or not _is_canvas_dimension(height):
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    return float(width), float(height)


async def _read_prompt(context: _RunContext, frame: Frame) -> str | CaptchaAttemptResult:
    """Return the one visible prompt within the instruction bound, or a fixed failure."""
    prompt = await _single_visible(frame, _PROMPT_SELECTOR)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if prompt is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    text = await prompt.inner_text()
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not isinstance(text, str):
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_INSTRUCTION_CHARS:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    return stripped


async def _read_grid(frame: Frame) -> _TaskGrid | None:
    """Read the nine visible tiles with unique aria-label indexes 1 to 9, or ``None``."""
    tasks = frame.locator(_TASK_SELECTOR)
    total = await tasks.count()
    entries: dict[int, tuple[Locator, str]] = {}
    visible_count = 0
    for position in range(total):
        task = tasks.nth(position)
        if not await task.is_visible():
            continue
        visible_count += 1
        label = await task.get_attribute(_ARIA_LABEL_ATTRIBUTE)
        if not isinstance(label, str):
            return None
        matched = re.findall(r"(?<!\d)[1-9](?!\d)", label)
        if len(matched) != 1:
            return None
        index = int(matched[0])
        if index in entries:
            return None
        entries[index] = (task, label)
    if visible_count != len(_TASK_INDEXES) or set(entries) != set(_TASK_INDEXES):
        return None
    locators = {index: entries[index][0] for index in _TASK_INDEXES}
    labels = tuple(entries[index][1] for index in _TASK_INDEXES)
    return _TaskGrid(locators=locators, labels=labels)


async def _read_challenge(context: _RunContext, frame: Frame) -> _Challenge | CaptchaAttemptResult:
    """Read the one visible grid or canvas challenge, rechecking origin throughout.

    A frame with any visible ``div.task`` keeps the nine-tile contract and never falls back;
    a frame with none is read as one canvas, its prompt and its CSS click surface.
    """
    view = await _single_visible(frame, _CHALLENGE_VIEW_SELECTOR)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if view is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    task_present = await _has_visible_task(frame)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if task_present:
        return await _read_grid_challenge(context, frame, view)
    return await _read_canvas_challenge(context, frame)


async def _read_grid_challenge(context: _RunContext, frame: Frame, view: Locator) -> _Challenge | CaptchaAttemptResult:
    grid = await _read_grid(frame)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if grid is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    submit = await _single_visible(frame, _SUBMIT_SELECTOR)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if submit is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    screenshot = await view.screenshot(type="png")
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not _is_image_bytes(screenshot):
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    return _GridChallenge(
        frame_url=frame.url,
        grid=grid,
        submit=submit,
        screenshot=screenshot,
    )


async def _read_canvas_challenge(context: _RunContext, frame: Frame) -> _Challenge | CaptchaAttemptResult:
    canvas = await _single_visible(frame, _CANVAS_SELECTOR)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if canvas is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    prompt = await _read_prompt(context, frame)
    if isinstance(prompt, CaptchaAttemptResult):
        return prompt
    submit = await _single_visible(frame, _SUBMIT_SELECTOR)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if submit is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    size = await _read_canvas_size(context, canvas)
    if isinstance(size, CaptchaAttemptResult):
        return size
    screenshot = await canvas.screenshot(type="png", scale="css")
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not _is_image_bytes(screenshot):
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    return _CanvasChallenge(
        frame_url=frame.url,
        prompt=prompt,
        canvas=canvas,
        submit=submit,
        screenshot=screenshot,
        width=size[0],
        height=size[1],
    )


async def _discover(context: _RunContext) -> _Challenge | CaptchaAttemptResult:
    if (lost := _guard_origin(context)) is not None:
        return lost
    frame = await _find_challenge_frame(context.page)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if frame is None:
        return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
    return await _read_challenge(context, frame)


async def _round_marker(context: _RunContext) -> _RoundMarker | None:
    """Read the unique active breadcrumb, or return None when progress is unavailable."""
    frame = await _find_challenge_frame(context.page)
    if frame is None or not _origin_ok(context):
        return None
    widths = await frame.locator(_BREADCRUMB_SELECTOR).evaluate_all("nodes => nodes.map(node => node.style.width)")
    if not isinstance(widths, list) or not 1 <= len(widths) <= _MAX_ROUNDS:
        return None
    parsed: list[float] = []
    for width in widths:
        if not isinstance(width, str) or not width.endswith("px"):
            return None
        try:
            value = float(width[:-2])
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        parsed.append(value)
    largest = max(parsed)
    return (len(parsed), parsed.index(largest)) if parsed.count(largest) == 1 else None


async def _fresh_revision(context: _RunContext, baseline: _Challenge) -> _Challenge | CaptchaAttemptResult:
    """Re-read origin, frame URL and the grid or canvas revision, failing closed on any change."""
    if (lost := _guard_origin(context)) is not None:
        return lost
    frame = await _find_challenge_frame(context.page)
    if frame is None:
        return _needs_human(STALE_REVISION_REASON)
    challenge = await _read_challenge(context, frame)
    if isinstance(challenge, CaptchaAttemptResult):
        if challenge.outcome.reason == ORIGIN_LOST_REASON:
            return challenge
        return _needs_human(STALE_REVISION_REASON)
    if not _same_revision(baseline, challenge):
        return _needs_human(STALE_REVISION_REASON)
    return challenge


async def _replan_canvas(
    context: _RunContext,
    baseline: _CanvasChallenge,
) -> tuple[_CanvasChallenge, CanvasActionPlan] | CaptchaAttemptResult:
    """Replan once only when an empty plan preceded a repaint of the same canvas task."""
    current = await _discover(context)
    if isinstance(current, CaptchaAttemptResult):
        if current.outcome.reason == ORIGIN_LOST_REASON:
            return current
        return _needs_human(STALE_REVISION_REASON)
    if not isinstance(current, _CanvasChallenge) or not _same_canvas_identity(baseline, current):
        return _needs_human(STALE_REVISION_REASON)
    if current.screenshot == baseline.screenshot:
        return _needs_human(EMPTY_SELECTION_REASON)
    remaining = context.seconds_remaining()
    if remaining <= 0:
        return _timed_out()
    plan = await context.client.plan_canvas(current.screenshot, current.prompt, remaining)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not isinstance(plan, CanvasActionPlan):
        return _error()
    return current, plan


def _same_revision(first: _Challenge, second: _Challenge) -> bool:
    """Return whether *second* is the same challenge variant and revision as *first*.

    The nine-tile grid also requires identical pixels, while a canvas is compared by its
    structural identity because the provider repaints the same canvas in place.
    """
    if isinstance(first, _GridChallenge):
        return (
            isinstance(second, _GridChallenge)
            and second.frame_url == first.frame_url
            and second.screenshot == first.screenshot
            and second.grid.labels == first.grid.labels
        )
    return isinstance(second, _CanvasChallenge) and _same_canvas_identity(first, second)


def _revision_failure(revision: _Challenge | CaptchaAttemptResult) -> CaptchaAttemptResult:
    return revision if isinstance(revision, CaptchaAttemptResult) else _needs_human(STALE_REVISION_REASON)


def _click_timeout(context: _RunContext) -> float:
    return max(1.0, context.seconds_remaining() * _MILLISECONDS_PER_SECOND)


async def _guard_challenge_frame(context: _RunContext, challenge: _Challenge) -> CaptchaAttemptResult | None:
    frame = await _find_challenge_frame(context.page)
    if frame is None or frame.url != challenge.frame_url:
        return _needs_human(STALE_REVISION_REASON)
    return _guard_origin(context)


async def _click_locator(context: _RunContext, challenge: _Challenge, locator: Locator) -> CaptchaAttemptResult | None:
    """Use native input when available; fall back to the frame locator otherwise."""
    native = context.native_tab
    if native is None:
        await locator.click(timeout=_click_timeout(context))
        return None
    await locator.scroll_into_view_if_needed(timeout=_click_timeout(context))
    bounds = await locator.bounding_box()
    if (lost := await _guard_challenge_frame(context, challenge)) is not None:
        return lost
    if not isinstance(bounds, dict) or not all(_is_canvas_dimension(bounds.get(name)) for name in ("width", "height")):
        return _needs_human(STALE_REVISION_REASON)
    x, y = bounds.get("x"), bounds.get("y")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) for value in (x, y)
    ):
        return _needs_human(STALE_REVISION_REASON)
    center = (float(x) + bounds["width"] / 2, float(y) + bounds["height"] / 2)
    await native.mouse.move(*center, humanize=True)
    if (lost := await _guard_challenge_frame(context, challenge)) is not None:
        return lost
    if await locator.bounding_box() != bounds:
        return _needs_human(STALE_REVISION_REASON)
    if locator is challenge.submit and await _offers_skip(locator):
        return _needs_human(PAGE_NOT_ACCEPTED_REASON)
    await native.mouse.down()
    try:
        await asyncio.sleep(0.08)
    finally:
        await native.mouse.up()
    return None


async def _click_cells(
    context: _RunContext,
    challenge: _GridChallenge,
    cells: tuple[int, ...],
) -> CaptchaAttemptResult | None:
    """Click only the selected tiles through native locators, rechecking origin before each."""
    for cell in cells:
        if (lost := _guard_origin(context)) is not None:
            return lost
        task = challenge.grid.locators.get(cell + 1)
        if task is None:
            return _needs_human(STALE_REVISION_REASON)
        if (failed := await _click_locator(context, challenge, task)) is not None:
            return failed
    return None


def _canvas_position(point: Point, challenge: _CanvasChallenge) -> Position:
    return {"x": point.x * challenge.width, "y": point.y * challenge.height}


async def _canvas_ready(context: _RunContext, challenge: _CanvasChallenge) -> CaptchaAttemptResult | None:
    if (failed := await _guard_challenge_frame(context, challenge)) is not None:
        return failed
    return _timed_out() if context.seconds_remaining() <= 0 else None


async def _native_canvas_origin(
    context: _RunContext, challenge: _CanvasChallenge
) -> tuple[float, float] | CaptchaAttemptResult:
    """Map the screenshot canvas to viewport CSS coordinates before native input."""
    await challenge.canvas.scroll_into_view_if_needed(timeout=_click_timeout(context))
    bounds = await challenge.canvas.bounding_box()
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    if (
        not isinstance(bounds, dict)
        or bounds.get("width") != challenge.width
        or bounds.get("height") != challenge.height
    ):
        return _needs_human(STALE_REVISION_REASON)
    x, y = bounds.get("x"), bounds.get("y")
    if not all(
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) for value in (x, y)
    ):
        return _needs_human(STALE_REVISION_REASON)
    return float(x), float(y)


async def _click_canvas_point(
    context: _RunContext,
    challenge: _CanvasChallenge,
    point: Point,
) -> CaptchaAttemptResult | None:
    """Click one normalized point through the canvas locator, once and inside its CSS box."""
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    native = context.native_tab
    if native is None:
        await challenge.canvas.click(position=_canvas_position(point, challenge), timeout=_click_timeout(context))
        return None
    origin = await _native_canvas_origin(context, challenge)
    if isinstance(origin, CaptchaAttemptResult):
        return origin
    await native.mouse.move(
        origin[0] + point.x * challenge.width, origin[1] + point.y * challenge.height, humanize=True
    )
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    if await _native_canvas_origin(context, challenge) != origin:
        return _needs_human(STALE_REVISION_REASON)
    await native.mouse.down()
    try:
        await asyncio.sleep(0.08)
    finally:
        await native.mouse.up()
    return None


async def _drag_canvas(
    context: _RunContext,
    challenge: _CanvasChallenge,
    drag: CanvasDrag,
) -> CaptchaAttemptResult | None:
    """Drag the piece's center to its target, releasing the button on every exit path."""
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    native = context.native_tab
    if native is not None:
        return await _drag_canvas_native(context, challenge, drag, native)
    await challenge.canvas.hover(position=_canvas_position(drag.start, challenge), timeout=_click_timeout(context))
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    await context.page.mouse.down()
    try:
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
        await asyncio.sleep(drag.hold_ms / _MILLISECONDS_PER_SECOND)
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
        await challenge.canvas.hover(
            position=_canvas_position(drag.release, challenge), timeout=_click_timeout(context)
        )
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
    finally:
        await context.page.mouse.up()
    return None


async def _drag_canvas_native(
    context: _RunContext, challenge: _CanvasChallenge, drag: CanvasDrag, native: Tab
) -> CaptchaAttemptResult | None:
    """Humanize the unpressed approach; keep the pressed drag within canvas bounds."""
    origin = await _native_canvas_origin(context, challenge)
    if isinstance(origin, CaptchaAttemptResult):
        return origin
    start = (origin[0] + drag.start.x * challenge.width, origin[1] + drag.start.y * challenge.height)
    release = (origin[0] + drag.release.x * challenge.width, origin[1] + drag.release.y * challenge.height)
    await native.mouse.move(*start, humanize=True)
    if (failed := await _canvas_ready(context, challenge)) is not None:
        return failed
    if await _native_canvas_origin(context, challenge) != origin:
        return _needs_human(STALE_REVISION_REASON)
    await native.mouse.down()
    try:
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
        await asyncio.sleep(drag.hold_ms / _MILLISECONDS_PER_SECOND)
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
        await native.mouse.move(*release, humanize=False)
        if (failed := await _canvas_ready(context, challenge)) is not None:
            return failed
    finally:
        await native.mouse.up()
    return None


async def _run_canvas_actions(
    context: _RunContext,
    challenge: _CanvasChallenge,
    actions: tuple[CanvasClick | CanvasDrag, ...],
) -> CaptchaAttemptResult | None:
    for action in actions:
        if isinstance(action, CanvasClick):
            failed = await _click_canvas_point(context, challenge, action.point)
        elif isinstance(action, CanvasDrag):
            failed = await _drag_canvas(context, challenge, action)
        else:
            return _error()
        if failed is not None:
            return failed
    return None


def _same_canvas_identity(first: _CanvasChallenge, second: _CanvasChallenge) -> bool:
    return (
        second.frame_url == first.frame_url
        and second.prompt == first.prompt
        and second.width == first.width
        and second.height == first.height
    )


def _is_skip_label(value: object) -> bool:
    return isinstance(value, str) and value.strip().casefold() == _SKIP_LABEL


async def _offers_skip(submit: Locator) -> bool:
    if _is_skip_label(await submit.inner_text()):
        return True
    return _is_skip_label(await submit.get_attribute(_ARIA_LABEL_ATTRIBUTE))


async def _submit(context: _RunContext, challenge: _Challenge) -> CaptchaAttemptResult | None:
    """Click the single submit control once, refusing a control that only offers the skip escape."""
    if (lost := _guard_origin(context)) is not None:
        return lost
    if await _offers_skip(challenge.submit):
        return _needs_human(PAGE_NOT_ACCEPTED_REASON)
    if (lost := _guard_origin(context)) is not None:
        return lost
    return await _click_locator(context, challenge, challenge.submit)


async def _fresh_response(context: _RunContext) -> CaptchaAttemptResult | None:
    token = _fresh_token(
        await _read_response_values(context.page, _HCAPTCHA.response_selector), sorted(context.initial_tokens)
    )
    if (lost := _guard_origin(context)) is not None:
        return lost
    if token is None:
        return None
    return CaptchaAttemptResult(outcome=VisionOutcome(status="handled", reason=PAGE_ACCEPTED_REASON), token=token)


async def _changed_canvas_prompt(
    context: _RunContext, baseline: _CanvasChallenge, marker: _RoundMarker
) -> _RejectedPage | CaptchaAttemptResult | None:
    """Recognize a fresh task prompt without rescreenshoting on every token poll."""
    frame = await _find_challenge_frame(context.page)
    if frame is None or frame.url != baseline.frame_url:
        return _needs_human(STALE_REVISION_REASON)
    prompt = await _read_prompt(context, frame)
    if isinstance(prompt, CaptchaAttemptResult):
        return prompt if prompt.outcome.reason == ORIGIN_LOST_REASON else None
    if prompt != baseline.prompt and await _round_marker(context) == marker:
        return _RejectedPage(marker=marker)
    return None


def _breadcrumb_transition(
    marker: _RoundMarker, current: _RoundMarker | None
) -> CaptchaAttemptResult | _RoundMarker | _RejectedPage | None:
    if current is None or current == marker:
        return None
    if current == (marker[0], marker[1] + 1):
        return current
    if marker[1] > 0 and current == (marker[0], 0):
        return _RejectedPage(marker=current)
    return _needs_human(STALE_REVISION_REASON)


async def _poll_token_or_round(
    context: _RunContext, marker: _RoundMarker | None, baseline: _Challenge
) -> CaptchaAttemptResult | _RoundMarker | _RejectedPage:
    """Accept a fresh token, observed next breadcrumb, or changed same-marker canvas task.

    A reset to the first breadcrumb and a new canvas prompt at the same marker each mean
    the provider presented a different puzzle after submission, not model success.
    """
    next_prompt_check = 0.0
    while True:
        if (lost := _guard_origin(context)) is not None:
            return lost
        if (accepted := await _fresh_response(context)) is not None:
            return accepted
        if marker is not None:
            current = await _round_marker(context)
            if (lost := _guard_origin(context)) is not None:
                return lost
            if (transition := _breadcrumb_transition(marker, current)) is not None:
                return transition
            if current == marker and isinstance(baseline, _CanvasChallenge) and time.monotonic() >= next_prompt_check:
                next_prompt_check = time.monotonic() + 1.0
                if (replacement := await _changed_canvas_prompt(context, baseline, marker)) is not None:
                    return replacement
        remaining = context.deadline - time.monotonic() - _POLL_MARGIN_SECONDS
        if remaining <= 0:
            return _needs_human(PAGE_NOT_ACCEPTED_REASON)
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _wait_rejected_page(  # noqa: C901 - one fail-closed poll for a fresh provider challenge
    context: _RunContext, marker: _RoundMarker, rejected: _Challenge
) -> _Challenge | CaptchaAttemptResult:
    """Wait for a fresh provider challenge at the observed breadcrumb, never rejected pixels.

    A further marker change, a lost origin or an expired deadline fails closed; only a
    discoverable challenge whose pixels differ from the rejected one is accepted.
    """
    while True:
        if (lost := _guard_origin(context)) is not None:
            return lost
        if (accepted := await _fresh_response(context)) is not None:
            return accepted
        current = await _round_marker(context)
        if (lost := _guard_origin(context)) is not None:
            return lost
        if current is not None and current != marker:
            return _needs_human(STALE_REVISION_REASON)
        if current == marker:
            challenge = await _discover(context)
            if isinstance(challenge, CaptchaAttemptResult):
                if challenge.outcome.reason == ORIGIN_LOST_REASON:
                    return challenge
            elif challenge.screenshot != rejected.screenshot and await _round_marker(context) == marker:
                return challenge
        remaining = context.deadline - time.monotonic() - _POLL_MARGIN_SECONDS
        if remaining <= 0:
            return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _wait_next_round(
    context: _RunContext, marker: _RoundMarker, previous: _Challenge
) -> _Challenge | CaptchaAttemptResult:
    """Wait for the next round's visible task, not just its progress indicator."""
    while True:
        if (lost := _guard_origin(context)) is not None:
            return lost
        if (accepted := await _fresh_response(context)) is not None:
            return accepted
        current = await _round_marker(context)
        if current is not None and current != marker:
            return _needs_human(STALE_REVISION_REASON)
        if current == marker:
            challenge = await _discover(context)
            if (
                isinstance(challenge, _GridChallenge | _CanvasChallenge)
                and (challenge.screenshot != previous.screenshot or not _same_revision(previous, challenge))
                and await _round_marker(context) == marker
            ):
                return challenge
        remaining = context.deadline - time.monotonic() - _POLL_MARGIN_SECONDS
        if remaining <= 0:
            return _needs_human(CHALLENGE_UNAVAILABLE_REASON)
        await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


async def _run(context: _RunContext) -> CaptchaAttemptResult:  # noqa: C901, PLR0912 - bounded round loop
    """Solve observed rounds under one deadline, allowing two extra pages after a rejection."""
    extra_pages = _ExtraPageBudget(remaining=_MAX_EXTRA_PAGES)
    expected: _RoundMarker | None = None
    previous: _Challenge | None = None
    pending: _Challenge | None = None
    after_rejection = False
    for _ in range(_MAX_ROUNDS):
        if pending is not None:
            baseline, pending = pending, None
        elif expected is not None and previous is not None:
            baseline = await _wait_next_round(context, expected, previous)
            if not isinstance(baseline, CaptchaAttemptResult) and after_rejection and not extra_pages.spend():
                return _needs_human(PAGE_NOT_ACCEPTED_REASON)
        else:
            baseline = await _discover(context)
        if isinstance(baseline, CaptchaAttemptResult):
            return baseline
        marker = await _round_marker(context)
        if (lost := _guard_origin(context)) is not None:
            return lost
        if expected is not None and marker != expected:
            return _needs_human(STALE_REVISION_REASON)
        remaining = context.seconds_remaining()
        if remaining <= 0:
            return _timed_out()
        submitted = (
            await _run_grid(context, baseline, remaining)
            if isinstance(baseline, _GridChallenge)
            else await _run_canvas(context, baseline, remaining)
        )
        if submitted is not None:
            return submitted
        next_step = await _poll_token_or_round(context, marker, baseline)
        if isinstance(next_step, CaptchaAttemptResult):
            return next_step
        if isinstance(next_step, _RejectedPage):
            if not extra_pages.spend():
                return _needs_human(PAGE_NOT_ACCEPTED_REASON)
            restarted = await _wait_rejected_page(context, next_step.marker, baseline)
            if isinstance(restarted, CaptchaAttemptResult):
                return restarted
            pending = restarted
            expected = None
            previous = None
            after_rejection = True
            continue
        previous, expected = baseline, next_step
    return _needs_human(PAGE_NOT_ACCEPTED_REASON)


async def _run_grid(context: _RunContext, baseline: _GridChallenge, remaining: float) -> CaptchaAttemptResult | None:
    cells = await context.client.select_grid(baseline.screenshot, VLM_INSTRUCTION, remaining)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not cells.cells:
        return _needs_human(EMPTY_SELECTION_REASON)
    fresh = await _fresh_revision(context, baseline)
    if not isinstance(fresh, _GridChallenge):
        return _revision_failure(fresh)
    if context.seconds_remaining() <= 0:
        return _timed_out()
    clicked = await _click_cells(context, fresh, cells.cells)
    if clicked is not None:
        return clicked
    return await _submit(context, fresh)


async def _run_canvas(
    context: _RunContext, baseline: _CanvasChallenge, remaining: float
) -> CaptchaAttemptResult | None:
    """Run one canvas round, replanning once only if an empty-plan image repaints."""
    plan = await context.client.plan_canvas(baseline.screenshot, baseline.prompt, remaining)
    if (lost := _guard_origin(context)) is not None:
        return lost
    if not isinstance(plan, CanvasActionPlan):
        return _error()
    if not plan.actions:
        replanned = await _replan_canvas(context, baseline)
        if isinstance(replanned, CaptchaAttemptResult):
            return replanned
        baseline, plan = replanned
        if not plan.actions:
            return _needs_human(EMPTY_SELECTION_REASON)
    fresh = await _fresh_revision(context, baseline)
    if not isinstance(fresh, _CanvasChallenge):
        return _revision_failure(fresh)
    if context.seconds_remaining() <= 0:
        return _timed_out()
    failed = await _run_canvas_actions(context, fresh, plan.actions)
    if failed is not None:
        return failed
    return await _submit(context, fresh)


async def solve_hcaptcha_challenge(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    vlm_client: CloudVlmClient,
    initial_tokens: frozenset[str],
    native_tab: Tab | None = None,
) -> CaptchaAttemptResult:
    """Solve selected-page tile or canvas puzzles within the request deadline.

    An observed rejection allows two extra provider pages; normal round progression is free.
    Only a fresh ``h-captcha-response`` absent from *initial_tokens* counts as success.
    """
    origin = _canonical_origin(page.url)
    if origin is None:
        return _needs_human(ORIGIN_LOST_REASON)
    remaining = seconds_remaining()
    if remaining <= 0:
        return _timed_out()
    context = _RunContext(
        page=page,
        allowed=frozenset({origin}),
        client=vlm_client,
        initial_tokens=initial_tokens,
        deadline=time.monotonic() + remaining,
        seconds_remaining=seconds_remaining,
        native_tab=native_tab,
    )
    try:
        async with asyncio.timeout(remaining):
            return await _run(context)
    except (TimeoutError, PlaywrightTimeoutError):
        return _timed_out()
    except VisionError as error:
        return _timed_out() if error.code in _TIMEOUT_CODES else _error()
    except Exception:  # noqa: BLE001 - any adapter failure is reported as one static category
        return _error()


__all__ = [
    "CHALLENGE_UNAVAILABLE_REASON",
    "EMPTY_SELECTION_REASON",
    "ORIGIN_LOST_REASON",
    "PAGE_ACCEPTED_REASON",
    "PAGE_NOT_ACCEPTED_REASON",
    "STALE_REVISION_REASON",
    "VLM_INSTRUCTION",
    "solve_hcaptcha_challenge",
]
