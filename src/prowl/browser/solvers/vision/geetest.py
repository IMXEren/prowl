"""Opt-in GeeTest slide adapter over the selected page's v3 or v4 widget."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from prowl.browser.solvers.vision.geetest_image import RgbaImage, match_geetest_piece
from prowl.browser.solvers.vision.schemas import (
    ERROR_REASON,
    LOW_CONFIDENCE_REASON,
    PAGE_ACCEPTED_REASON,
    TIMED_OUT_REASON,
    CaptchaAttemptResult,
    OutcomeStatus,
    VisionOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Page
    from pydoll.browser.tab import Tab

_WIDGET = ".geetest_holder:visible"
_BG = ".geetest_canvas_bg:visible"
_PIECE = ".geetest_canvas_slice:visible"
_HANDLE = ".geetest_slider_button:visible"
_RAIL = ".geetest_slider:visible"
_TOKEN_NAMES = ("geetest_challenge", "geetest_validate", "geetest_seccode")
_RECT_LENGTH = 4
_RGBA_CHANNELS = 4
_MAX_BYTE = 255
_MIN_CANVAS_SIDE = 50
_MAX_CANVAS_WIDTH = 600
_MAX_CANVAS_HEIGHT = 300
_MIN_PIECE_SIDE = 5
_MAX_PIECE_SIDE = 150
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_MAX_SCALE_SKEW = 0.05
_READ_TOKENS = """names => names.map(name => {
  const fields = [...document.querySelectorAll('input[name="' + name + '"]')];
  return fields.length === 1 ? fields[0].value : null;
})"""
_READ_SLIDE = """() => {
  const one = s => { const a = [...document.querySelectorAll(s)].filter(e => {
    const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0;
  }); return a.length === 1 ? a[0] : null; };
  const bg = one('.geetest_canvas_bg'), piece = one('.geetest_canvas_slice');
  const handle = one('.geetest_slider_button'), rail = one('.geetest_slider');
  if (!bg || !piece || !handle || !rail || bg.width !== piece.width || bg.height !== piece.height ||
      bg.width < 50 || bg.width > 600 || bg.height < 50 || bg.height > 300) return null;
  const a = bg.getContext('2d', {willReadFrequently:true}).getImageData(0,0,bg.width,bg.height).data;
  const b = piece.getContext('2d', {willReadFrequently:true}).getImageData(0,0,piece.width,piece.height).data;
  let left=piece.width, top=piece.height, right=-1, bottom=-1;
  for (let y=0; y<piece.height; y++) for (let x=0; x<piece.width; x++) {
    if (b[(y*piece.width+x)*4+3] >= 128) {
      left=Math.min(left,x); top=Math.min(top,y); right=Math.max(right,x); bottom=Math.max(bottom,y);
    }
  }
  if (right < left || bottom < top || right-left+1 > 150 || bottom-top+1 > 150) return null;
  const cropped = piece.getContext('2d').getImageData(left,top,right-left+1,bottom-top+1).data;
  const rect = e => {const r=e.getBoundingClientRect(); return [r.x,r.y,r.width,r.height]};
  return {width:bg.width,height:bg.height,background:Array.from(a),
    piece_width:right-left+1,piece_height:bottom-top+1,piece:Array.from(cropped),
    left,top,bg_rect:rect(bg),handle_rect:rect(handle),rail_rect:rect(rail)};
}"""


@dataclass(frozen=True)
class _Slide:
    background: RgbaImage
    piece: RgbaImage
    left: int
    top: int
    bg_rect: tuple[float, float, float, float]
    handle_rect: tuple[float, float, float, float]
    rail_rect: tuple[float, float, float, float]


def _outcome(status: OutcomeStatus, reason: str, token: str | None = None) -> CaptchaAttemptResult:
    return CaptchaAttemptResult(VisionOutcome(status=status, reason=reason), token)


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme in {"http", "https"} and parts.hostname else None


def _rect(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != _RECT_LENGTH:
        return None
    if not all(
        isinstance(number, int | float) and not isinstance(number, bool) and math.isfinite(number) for number in value
    ):
        return None
    x, y, width, height = value
    return (float(x), float(y), float(width), float(height)) if width > 0 and height > 0 else None


def _pixels(value: Any, width: int, height: int) -> bytes | None:
    if not isinstance(value, list) or len(value) != width * height * _RGBA_CHANNELS:
        return None
    if not all(isinstance(n, int) and not isinstance(n, bool) and 0 <= n <= _MAX_BYTE for n in value):
        return None
    return bytes(value)


def _slide(value: Any) -> _Slide | None:
    if not isinstance(value, dict):
        return None
    dimensions = (
        value.get("width"),
        value.get("height"),
        value.get("piece_width"),
        value.get("piece_height"),
        value.get("left"),
        value.get("top"),
    )
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in dimensions):
        return None
    width, height, pw, ph, left, top = cast("tuple[int, int, int, int, int, int]", dimensions)
    if not (
        _MIN_CANVAS_SIDE <= width <= _MAX_CANVAS_WIDTH
        and _MIN_CANVAS_SIDE <= height <= _MAX_CANVAS_HEIGHT
        and _MIN_PIECE_SIDE <= pw <= _MAX_PIECE_SIDE
        and _MIN_PIECE_SIDE <= ph <= _MAX_PIECE_SIDE
    ):
        return None
    if not (0 <= left <= width - pw and 0 <= top <= height - ph):
        return None
    background = _pixels(value.get("background"), width, height)
    piece = _pixels(value.get("piece"), pw, ph)
    bg_rect = _rect(value.get("bg_rect"))
    handle_rect = _rect(value.get("handle_rect"))
    rail_rect = _rect(value.get("rail_rect"))
    if background is None or piece is None or bg_rect is None or handle_rect is None or rail_rect is None:
        return None
    return _Slide(
        RgbaImage(width, height, background), RgbaImage(pw, ph, piece), left, top, bg_rect, handle_rect, rail_rect
    )


async def _read_tokens(page: Page) -> tuple[str | None, str | None, str | None] | None:
    values = await page.evaluate(_READ_TOKENS, _TOKEN_NAMES)
    if (
        not isinstance(values, list)
        or len(values) != len(_TOKEN_NAMES)
        or any(v is not None and not isinstance(v, str) for v in values)
    ):
        return None
    return values[0], values[1], values[2]


def _fresh_validation(
    current: tuple[str | None, str | None, str | None] | None,
    initial: tuple[str | None, str | None, str | None],
) -> str | None:
    if current is None:
        return None
    challenge, validate, seccode = current
    return validate if challenge and validate and seccode and validate != initial[1] and seccode != initial[2] else None


def _positions(slide: _Slide, gap: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
    _, _, bg_width, bg_height = slide.bg_rect
    hx, hy, hw, hh = slide.handle_rect
    rx, ry, rw, rh = slide.rail_rect
    scale = bg_width / slide.background.width
    if not (_MIN_SCALE <= scale <= _MAX_SCALE and abs(bg_height / slide.background.height - scale) <= _MAX_SCALE_SKEW):
        return None
    if not (rx <= hx + hw / 2 <= rx + rw and ry - hh <= hy + hh / 2 <= ry + rh + hh):
        return None
    delta = (gap - slide.left) * scale
    start = (hx + hw / 2, hy + hh / 2)
    end = (start[0] + delta, start[1])
    if not (delta > 0 and end[0] <= rx + rw - hw / 2 and 0 <= gap <= slide.background.width - slide.piece.width):
        return None
    return start, end


async def _show_slider(
    page: Page, seconds_remaining: Callable[[], float], native_tab: Tab | None
) -> CaptchaAttemptResult | None:
    if (
        await page.locator(_BG).count()
        == await page.locator(_PIECE).count()
        == await page.locator(_HANDLE).count()
        == 1
    ):
        return None
    trigger = page.locator(".geetest_radar_tip:visible")
    if await trigger.count() != 1 or native_tab is None:
        return _outcome("unsupported", "challenge_unavailable")
    box = await trigger.bounding_box()
    if box is None:
        return _outcome("unsupported", "challenge_unavailable")
    await native_tab.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, humanize=True)
    await native_tab.mouse.down()
    try:
        await asyncio.sleep(0.05)
    finally:
        await native_tab.mouse.up()
    await page.locator(_BG).wait_for(state="visible", timeout=min(15000, max(1, int(seconds_remaining() * 1000))))
    if await page.locator(_PIECE).count() != 1 or await page.locator(_HANDLE).count() != 1:
        return _outcome("unsupported", "challenge_unavailable")
    return None


async def _drag_slider(
    page: Page,
    native_tab: Tab,
    origin: str,
    positions: tuple[tuple[float, float], tuple[float, float]],
    *,
    handle_selector: str = _HANDLE,
) -> CaptchaAttemptResult | None:
    if _origin(page.url) != origin or await page.locator(handle_selector).count() != 1:
        return _outcome("needs_human", "stale_revision")
    start, end = positions
    await native_tab.mouse.move(*start, humanize=True)
    if _origin(page.url) != origin or await page.locator(handle_selector).count() != 1:
        return _outcome("needs_human", "stale_revision")
    await native_tab.mouse.down()
    try:
        await asyncio.sleep(0.12)
        await native_tab.mouse.move(*end, humanize=True)
        await asyncio.sleep(0.1)
    finally:
        await native_tab.mouse.up()
    return None


async def _wait_validation(
    page: Page,
    origin: str,
    initial: tuple[str | None, str | None, str | None],
    seconds_remaining: Callable[[], float],
) -> CaptchaAttemptResult:
    deadline = time.monotonic() + min(7.0, max(0.0, seconds_remaining()))
    while time.monotonic() < deadline:
        if _origin(page.url) != origin:
            return _outcome("needs_human", "origin_lost")
        if (token := _fresh_validation(await _read_tokens(page), initial)) is not None:
            return _outcome("handled", PAGE_ACCEPTED_REASON, token)
        await asyncio.sleep(0.1)
    return _outcome("needs_human", "page_not_accepted")


async def _read_painted_slide(page: Page, seconds_remaining: Callable[[], float]) -> _Slide | None:
    deadline = time.monotonic() + min(4.0, max(0.0, seconds_remaining()))
    while time.monotonic() < deadline:
        if (slide := _slide(await page.evaluate(_READ_SLIDE))) is not None:
            return slide
        await asyncio.sleep(0.1)
    return None


async def _attempt_slide(
    page: Page,
    origin: str,
    initial: tuple[str | None, str | None, str | None],
    native_tab: Tab | None,
    seconds_remaining: Callable[[], float],
) -> CaptchaAttemptResult:
    if (failed := await _show_slider(page, seconds_remaining, native_tab)) is not None:
        return failed
    if _origin(page.url) != origin:
        return _outcome("needs_human", "origin_lost")
    slide = await _read_painted_slide(page, seconds_remaining)
    if slide is None:
        return _outcome("unsupported", "detection_mapping_unsupported")
    gap = await asyncio.to_thread(match_geetest_piece, slide.background, slide.piece, slide.top)
    if gap is None:
        return _outcome("needs_human", LOW_CONFIDENCE_REASON)
    positions = _positions(slide, gap)
    if positions is None or native_tab is None:
        return _outcome("needs_human", "detection_mapping_unsupported")
    if (failed := await _drag_slider(page, native_tab, origin, positions)) is not None:
        return failed
    return await _wait_validation(page, origin, initial, seconds_remaining)


async def solve_geetest_slide(
    page: Page,
    seconds_remaining: Callable[[], float],
    *,
    initial_tokens: frozenset[str],
    native_tab: Tab | None = None,
) -> CaptchaAttemptResult:
    """Attempt one visible slide; return only a fresh provider validation value."""
    del initial_tokens  # GeeTest reads its own provider validation fields.
    origin = _origin(page.url)
    if origin is None:
        return _outcome("needs_human", "origin_lost")
    remaining = seconds_remaining()
    if remaining <= 0:
        return _outcome("timed_out", TIMED_OUT_REASON)
    try:
        async with asyncio.timeout(remaining):
            if await page.locator(_WIDGET).count() != 1:
                return _outcome("unsupported", "challenge_unavailable")
            if await page.locator(".geetest_btn_click:visible").count() == 1:
                # v4 shares v3 geometry.
                from prowl.browser.solvers.vision.geetest_v4 import (  # noqa: PLC0415
                    solve_geetest_v4,
                )

                return await solve_geetest_v4(page, seconds_remaining, native_tab=native_tab)
            initial = await _read_tokens(page)
            if initial is None:
                return _outcome("unsupported", "challenge_unavailable")
            return await _attempt_slide(page, origin, initial, native_tab, seconds_remaining)
    except (TimeoutError, PlaywrightTimeoutError):
        return _outcome("timed_out", TIMED_OUT_REASON)
    except Exception:  # noqa: BLE001 - fixed public outcome; never return provider/internal text
        return _outcome("error", ERROR_REASON)
