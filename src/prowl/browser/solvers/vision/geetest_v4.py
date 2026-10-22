"""GeeTest v4 slide adapter for the selected browser page."""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from prowl.browser.solvers.vision.geetest import (
    _MAX_CANVAS_HEIGHT,
    _MAX_CANVAS_WIDTH,
    _MAX_PIECE_SIDE,
    _MAX_SCALE,
    _MAX_SCALE_SKEW,
    _MIN_CANVAS_SIDE,
    _MIN_PIECE_SIDE,
    _MIN_SCALE,
    _drag_slider,
    _origin,
    _outcome,
    _positions,
    _rect,
    _Slide,
)
from prowl.browser.solvers.vision.geetest_image import RgbaImage, match_geetest_piece
from prowl.browser.solvers.vision.schemas import (
    ERROR_REASON,
    LOW_CONFIDENCE_REASON,
    PAGE_ACCEPTED_REASON,
    TIMED_OUT_REASON,
    CaptchaAttemptResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Page, Response
    from pydoll.browser.tab import Tab

_BG = ".geetest_bg:visible"
_PIECE = ".geetest_slice_bg:visible"
_HANDLE = ".geetest_btn:visible"
_TRIGGER = ".geetest_btn_click:visible"
_MAX_IMAGE_BYTES = 500_000
_MAX_VERIFY_BYTES = 65_536
_MAX_TOKEN_LENGTH = 4096
_OK_HTTP = 200
_LAYOUT_SETTLE_SECONDS = 0.3
_CALLBACK = re.compile(r"[A-Za-z_$][\w$]*\(")
_READ_LAYOUT = """() => {
  const one = s => { const a = [...document.querySelectorAll(s)].filter(e => {
    const r=e.getBoundingClientRect(); return r.width > 0 && r.height > 0;
  }); return a.length === 1 ? a[0] : null; };
  const bg=one('.geetest_bg'), piece=one('.geetest_slice_bg');
  const handle=one('.geetest_btn'), rail=one('.geetest_slider');
  if (!bg || !piece || !handle || !rail) return null;
  const rect=e=>{const r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height]};
  return {bg_url:getComputedStyle(bg).backgroundImage, piece_url:getComputedStyle(piece).backgroundImage,
    bg_rect:rect(bg),piece_rect:rect(piece),handle_rect:rect(handle),rail_rect:rect(rail)};
}"""


@dataclass(frozen=True)
class _Layout:
    bg_url: str
    piece_url: str
    bg_rect: tuple[float, float, float, float]
    piece_rect: tuple[float, float, float, float]
    handle_rect: tuple[float, float, float, float]
    rail_rect: tuple[float, float, float, float]


def _asset_url(style: Any, kind: str) -> str | None:
    if not isinstance(style, str) or not style.startswith("url(") or not style.endswith(")"):
        return None
    url = style[4:-1].strip("\"'")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname != "static.geetest.com"
        or parts.port not in (None, 443)
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or not parts.path.startswith("/pictures/v4_pic/")
        or f"/{kind}/" not in parts.path
        or not parts.path.endswith(".png")
    ):
        return None
    return url


def _layout(value: Any) -> _Layout | None:
    if not isinstance(value, dict):
        return None
    bg_url = _asset_url(value.get("bg_url"), "bg")
    piece_url = _asset_url(value.get("piece_url"), "slide")
    bg_rect = _rect(value.get("bg_rect"))
    piece_rect = _rect(value.get("piece_rect"))
    handle_rect = _rect(value.get("handle_rect"))
    rail_rect = _rect(value.get("rail_rect"))
    if (
        bg_url is None
        or piece_url is None
        or bg_rect is None
        or piece_rect is None
        or handle_rect is None
        or rail_rect is None
    ):
        return None
    if bg_url.replace("/bg/", "/slide/", 1) != piece_url:
        return None
    return _Layout(bg_url, piece_url, bg_rect, piece_rect, handle_rect, rail_rect)


async def _asset(page: Page, url: str, *, background: bool) -> RgbaImage | None:
    response = await page.request.get(url, timeout=8000)
    if response.status != _OK_HTTP or int(response.headers.get("content-length", "0")) > _MAX_IMAGE_BYTES:
        return None
    payload = await response.body()
    if not payload or len(payload) > _MAX_IMAGE_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if background and not (
                _MIN_CANVAS_SIDE <= width <= _MAX_CANVAS_WIDTH and _MIN_CANVAS_SIDE <= height <= _MAX_CANVAS_HEIGHT
            ):
                return None
            if not background and not (
                _MIN_PIECE_SIDE <= width <= _MAX_PIECE_SIDE and _MIN_PIECE_SIDE <= height <= _MAX_PIECE_SIDE
            ):
                return None
            return RgbaImage(width, height, image.convert("RGBA").tobytes())
    except (OSError, UnidentifiedImageError, ValueError):
        return None


async def _slide(page: Page, layout: _Layout) -> _Slide | None:
    background, piece = await asyncio.gather(
        _asset(page, layout.bg_url, background=True), _asset(page, layout.piece_url, background=False)
    )
    if background is None or piece is None:
        return None
    scale = layout.bg_rect[2] / background.width
    top = round((layout.piece_rect[1] - layout.bg_rect[1]) / scale)
    if (
        not math.isfinite(scale)
        or not _MIN_SCALE <= scale <= _MAX_SCALE
        or not 0 <= top <= background.height - piece.height
        or abs(layout.piece_rect[2] / piece.width - scale) > _MAX_SCALE_SKEW
    ):
        return None
    return _Slide(background, piece, 0, top, layout.bg_rect, layout.handle_rect, layout.rail_rect)


def _validation(body: bytes) -> str | None:
    if not body or len(body) > _MAX_VERIFY_BYTES:
        return None
    try:
        text = body.decode("utf-8").strip()
        if text.startswith("{"):
            raw = text
        else:
            left = text.find("{")
            right = text.rfind("}")
            if (
                left < 0
                or right <= left
                or not _CALLBACK.fullmatch(text[:left])
                or text[right + 1 :]
                not in (
                    ")",
                    ");",
                )
            ):
                return None
            raw = text[left : right + 1]
        envelope = json.loads(raw)
    except (UnicodeError, ValueError):
        return None
    if not isinstance(envelope, dict) or envelope.get("status") != "success":
        return None
    data = envelope.get("data")
    if not isinstance(data, dict) or data.get("result") != "success" or not isinstance(data.get("lot_number"), str):
        return None
    seccode = data.get("seccode")
    if not isinstance(seccode, dict):
        return None
    token = seccode.get("pass_token")
    return token if isinstance(token, str) and 0 < len(token) <= _MAX_TOKEN_LENGTH and data["lot_number"] else None


def _verify_response(response: Response) -> bool:
    parts = urlsplit(response.url)
    return (
        parts.scheme == "https"
        and parts.hostname == "gcaptcha4.geetest.com"
        and parts.path == "/verify"
        and response.status == _OK_HTTP
    )


async def _show_slide(page: Page, native_tab: Tab, seconds_remaining: Callable[[], float]) -> bool:
    trigger = page.locator(_TRIGGER)
    if await trigger.count() != 1 or (box := await trigger.bounding_box()) is None:
        return False
    await native_tab.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, humanize=True)
    await native_tab.mouse.down()
    try:
        await asyncio.sleep(0.05)
    finally:
        await native_tab.mouse.up()
    timeout = min(15000, max(1, int(seconds_remaining() * 1000)))
    await page.locator(_BG).wait_for(state="visible", timeout=timeout)
    return await page.locator(_PIECE).count() == await page.locator(_HANDLE).count() == 1


async def _read_layout(page: Page, seconds_remaining: Callable[[], float]) -> _Layout | None:
    deadline = time.monotonic() + min(4.0, max(0.0, seconds_remaining()))
    previous: _Layout | None = None
    stable_since: float | None = None
    while time.monotonic() < deadline:
        layout = _layout(await page.evaluate(_READ_LAYOUT))
        now = time.monotonic()
        if layout != previous:
            previous, stable_since = layout, now if layout is not None else None
        elif layout is not None and stable_since is not None and now - stable_since >= _LAYOUT_SETTLE_SECONDS:
            return layout
        await asyncio.sleep(min(0.1, max(0.0, deadline - now)))
    return None


async def _prepare_drag(
    page: Page, origin: str, native_tab: Tab, seconds_remaining: Callable[[], float]
) -> CaptchaAttemptResult | None:
    if not await _show_slide(page, native_tab, seconds_remaining):
        return _outcome("unsupported", "challenge_unavailable")
    if _origin(page.url) != origin:
        return _outcome("needs_human", "origin_lost")
    layout = await _read_layout(page, seconds_remaining)
    if layout is None or (slide := await _slide(page, layout)) is None:
        return _outcome("unsupported", "detection_mapping_unsupported")
    gap = await asyncio.to_thread(match_geetest_piece, slide.background, slide.piece, slide.top)
    if gap is None:
        return _outcome("needs_human", LOW_CONFIDENCE_REASON)
    positions = _positions(slide, gap)
    if positions is None:
        return _outcome("needs_human", "detection_mapping_unsupported")
    if _layout(await page.evaluate(_READ_LAYOUT)) != layout:
        return _outcome("needs_human", "stale_revision")
    return await _drag_slider(page, native_tab, origin, positions, handle_selector=_HANDLE)


async def _wait_verified(
    page: Page, origin: str, observed: list[str], seconds_remaining: Callable[[], float]
) -> CaptchaAttemptResult:
    deadline = time.monotonic() + min(7.0, max(0.0, seconds_remaining()))
    while time.monotonic() < deadline:
        if _origin(page.url) != origin:
            return _outcome("needs_human", "origin_lost")
        if observed:
            return _outcome("handled", PAGE_ACCEPTED_REASON, observed[0])
        await asyncio.sleep(0.1)
    return _outcome("needs_human", "page_not_accepted")


async def _attempt(
    page: Page, origin: str, native_tab: Tab, seconds_remaining: Callable[[], float]
) -> CaptchaAttemptResult:
    observed: list[str] = []
    pending: set[asyncio.Task[None]] = set()

    async def parse_response(response: Response) -> None:
        try:
            if (token := _validation(await response.body())) is not None:
                observed.append(token)
        except Exception:  # noqa: BLE001 - unsuccessful provider read cannot fabricate acceptance
            return

    def on_response(response: Response) -> None:
        if _verify_response(response) and response.request.frame == page.main_frame:
            task = asyncio.create_task(parse_response(response))
            pending.add(task)
            task.add_done_callback(pending.discard)

    page.on("response", on_response)
    try:
        if (failed := await _prepare_drag(page, origin, native_tab, seconds_remaining)) is not None:
            return failed
        return await _wait_verified(page, origin, observed, seconds_remaining)
    finally:
        page.remove_listener("response", on_response)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def solve_geetest_v4(
    page: Page, seconds_remaining: Callable[[], float], *, native_tab: Tab | None
) -> CaptchaAttemptResult:
    """Return a pass_token only after a successful same-frame provider response."""
    origin = _origin(page.url)
    if origin is None:
        return _outcome("needs_human", "origin_lost")
    remaining = seconds_remaining()
    if remaining <= 0:
        return _outcome("timed_out", TIMED_OUT_REASON)
    if native_tab is None:
        return _outcome("unsupported", "challenge_unavailable")
    try:
        async with asyncio.timeout(remaining):
            return await _attempt(page, origin, native_tab, seconds_remaining)
    except TimeoutError:
        return _outcome("timed_out", TIMED_OUT_REASON)
    except Exception:  # noqa: BLE001 - fixed public outcome; never return provider/internal text
        return _outcome("error", ERROR_REASON)
