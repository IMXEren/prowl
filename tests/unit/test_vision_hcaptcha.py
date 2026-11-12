# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportPrivateUsage=false
"""Offline tests for the isolated hCaptcha grid and canvas adapter.

The page, frames, locators and VLM client are spec-constrained doubles, so origin gating,
challenge-frame discovery, the nine-tile index mapping, the canvas point mapping, bounded
round submits, token freshness, redirects, deadlines and cancellation are verified without a live
browser or network.
"""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import IsolatedAsyncioTestCase, TestCase

from prowl.browser.solvers.vision.hcaptcha import (
    _BREADCRUMB_SELECTOR,
    _CANVAS_SELECTOR,
    _CHALLENGE_VIEW_SELECTOR,
    _PROMPT_SELECTOR,
    _SUBMIT_SELECTOR,
    _TASK_SELECTOR,
    CHALLENGE_UNAVAILABLE_REASON,
    EMPTY_SELECTION_REASON,
    ORIGIN_LOST_REASON,
    PAGE_ACCEPTED_REASON,
    PAGE_NOT_ACCEPTED_REASON,
    STALE_REVISION_REASON,
    VLM_INSTRUCTION,
    solve_hcaptcha_challenge,
)
from prowl.browser.solvers.vision.schemas import ERROR_REASON, TIMED_OUT_REASON, CaptchaAttemptResult, VisionOutcome
from prowl.vision.schemas import VisionError
from prowl.vision.vlm import (
    MAX_INSTRUCTION_CHARS,
    REQUEST_FAILED,
    REQUEST_TIMEOUT,
    CanvasActionPlan,
    CanvasClick,
    CanvasDrag,
    GridSelection,
    Point,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydoll.browser.tab import Tab

_EMPTY_TOKENS: frozenset[str] = frozenset()
_PAGE_URL = "http://127.0.0.1:8080/fixture.html"
_OFF_ORIGIN_URL = "http://127.0.0.1:9999/fixture.html"

_CHALLENGE_URL = "https://newassets.hcaptcha.com/captcha/v1/abcd/static/hcaptcha.html#frame=challenge"
_QUERY_CHALLENGE_URL = "https://newassets.hcaptcha.com/captcha/v1/abcd/static/hcaptcha.html?frame=challenge"
_CHECKBOX_URL = "https://newassets.hcaptcha.com/captcha/v1/abcd/static/hcaptcha.html#frame=checkbox"
_OTHER_CHALLENGE_URL = "https://newassets.hcaptcha.com/captcha/v1/efgh/static/hcaptcha.html#frame=challenge"

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_OTHER_PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 16
_THIRD_PNG = b"\x89PNG\r\n\x1a\n" + b"\x02" * 16

_CANVAS_BOUNDS = {"x": 0.0, "y": 0.0, "width": 120.0, "height": 80.0}
_PROMPT_TEXT = "Select every tile showing a crosswalk"


def _png_with_size(width: int, height: int) -> bytes:
    """Return a byte string with a real PNG signature and an IHDR carrying *width* and *height*."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)


def _widen(bounds: dict[str, float]) -> bytes:
    """Widen *bounds* in place and return the default PNG bytes, simulating a resized canvas."""
    bounds["width"] = 240.0
    return _PNG


class _FakeTask:
    """One ``div.task`` element double carrying its aria-label index and click count."""

    def __init__(self, index: int, *, visible: bool = True, on_click: Callable[[], None] | None = None) -> None:
        self.index = index
        self._visible = visible
        self._on_click = on_click
        self.clicks = 0

    async def is_visible(self) -> bool:
        return self._visible

    async def get_attribute(self, name: str) -> str | None:
        return f"tile-{self.index}" if name == "aria-label" else None

    async def bounding_box(self) -> dict[str, float]:
        return {"x": self.index * 10.0, "y": 20.0, "width": 9.0, "height": 9.0}

    async def scroll_into_view_if_needed(self, **_kwargs: object) -> None:
        return None

    async def click(self, timeout: float | None = None) -> None:
        self.clicks += 1
        if self._on_click is not None:
            self._on_click()


class _FakeTasks:
    """A ``div.task`` collection locator double."""

    def __init__(self, tasks: list[_FakeTask]) -> None:
        self._tasks = list(tasks)

    async def count(self) -> int:
        return len(self._tasks)

    def nth(self, index: int) -> _FakeTask:
        return self._tasks[index]


class _FakeBreadcrumbs:
    def __init__(self, widths: Callable[[], list[str]] | None = None) -> None:
        self._widths = widths or list

    async def evaluate_all(self, expression: str) -> list[str]:
        return self._widths()


class _ResetBreadcrumbs:
    """Breadcrumb widths with one active crumb that follows a fixed reset and advance script."""

    def __init__(self, count: int, active: int, sequence: tuple[int, ...]) -> None:
        self.count = count
        self.active = active
        self._sequence = sequence
        self._next = 0

    def widths(self) -> list[str]:
        return ["24px" if index == self.active else "8px" for index in range(self.count)]

    def on_submit(self) -> None:
        if self._next < len(self._sequence):
            self.active = self._sequence[self._next]
            self._next += 1


class _FakeLocator:
    """A single-element locator double for the challenge view, canvas, prompt or submit control."""

    def __init__(  # noqa: PLR0913 - a spec-constrained double exposes each optional field
        self,
        *,
        count: int = 1,
        visible: bool = True,
        text: str | Callable[[int], str] | None = None,
        bounds: dict[str, float] | None = None,
        aria_label: str | None = None,
        on_screenshot: Callable[[int], bytes] | None = None,
        on_click: Callable[[], None] | None = None,
        on_hover: Callable[[], None] | None = None,
        events: list[object] | None = None,
    ) -> None:
        self._count = count
        self._visible = visible
        self._text = text
        self._bounds = bounds
        self._aria_label = aria_label
        self._on_screenshot = on_screenshot
        self._on_click = on_click
        self._on_hover = on_hover
        self._events = events
        self.screenshots = 0
        self.screenshot_options: list[dict[str, object]] = []
        self.clicks = 0
        self.hovers = 0
        self.text_reads = 0
        self.positions: list[object] = []
        self.hover_positions: list[object] = []

    async def count(self) -> int:
        return self._count

    async def is_visible(self) -> bool:
        return self._visible

    async def inner_text(self) -> str:
        self.text_reads += 1
        text = self._text
        if callable(text):
            return text(self.text_reads)
        return text or ""

    async def get_attribute(self, name: str) -> str | None:
        return self._aria_label if name == "aria-label" else None

    async def bounding_box(self) -> dict[str, float] | None:
        return self._bounds

    async def scroll_into_view_if_needed(self, **_kwargs: object) -> None:
        return None

    async def screenshot(self, **kwargs: object) -> bytes:
        self.screenshots += 1
        self.screenshot_options.append(kwargs)
        if self._on_screenshot is not None:
            return self._on_screenshot(self.screenshots)
        return _PNG

    async def click(self, **kwargs: object) -> None:
        self.clicks += 1
        self.positions.append(kwargs.get("position"))
        if self._events is not None:
            self._events.append(("click", kwargs.get("position")))
        if self._on_click is not None:
            self._on_click()

    async def hover(self, **kwargs: object) -> None:
        self.hovers += 1
        self.hover_positions.append(kwargs.get("position"))
        if self._events is not None:
            self._events.append(("hover", kwargs.get("position")))
        if self._on_hover is not None:
            self._on_hover()


def _default_tasks() -> list[_FakeTask]:
    """Return nine visible tiles with unique aria-label indexes 1 to 9."""
    return [_FakeTask(index) for index in range(1, 10)]


class _FakeFrame:
    """One challenge frame double exposing only URL and its selectors."""

    def __init__(  # noqa: PLR0913 - a spec-constrained double exposes each optional field
        self,
        url: str = _CHALLENGE_URL,
        *,
        view: _FakeLocator | None = None,
        tasks: list[_FakeTask] | None = None,
        canvas: _FakeLocator | None = None,
        prompt: _FakeLocator | None = None,
        submit: _FakeLocator | None = None,
        breadcrumbs: _FakeBreadcrumbs | None = None,
    ) -> None:
        self.url = url
        self.view = view if view is not None else _FakeLocator()
        self.tasks = tasks if tasks is not None else _default_tasks()
        self.canvas = canvas if canvas is not None else _FakeLocator(count=0)
        self.prompt = prompt if prompt is not None else _FakeLocator(count=0, text="")
        self.submit = (
            submit
            if submit is not None
            else _FakeLocator(bounds={"x": 150.0, "y": 250.0, "width": 50.0, "height": 25.0})
        )
        self.breadcrumbs = breadcrumbs if breadcrumbs is not None else _FakeBreadcrumbs()

    def locator(self, selector: str) -> object:
        if selector == _CHALLENGE_VIEW_SELECTOR:
            return self.view
        if selector == _TASK_SELECTOR:
            return _FakeTasks(self.tasks)
        if selector == _CANVAS_SELECTOR:
            return self.canvas
        if selector == _PROMPT_SELECTOR:
            return self.prompt
        if selector == _SUBMIT_SELECTOR:
            return self.submit
        if selector == _BREADCRUMB_SELECTOR:
            return self.breadcrumbs
        raise AssertionError(selector)


def _canvas_frame(
    *,
    canvas: _FakeLocator | None = None,
    prompt: _FakeLocator | None = None,
    submit: _FakeLocator | None = None,
    tasks: list[_FakeTask] | None = None,
    breadcrumbs: _FakeBreadcrumbs | None = None,
) -> _FakeFrame:
    """Return a challenge frame carrying a single canvas instead of the nine tiles."""
    return _FakeFrame(
        tasks=tasks if tasks is not None else [],
        canvas=canvas if canvas is not None else _FakeLocator(bounds=dict(_CANVAS_BOUNDS)),
        prompt=prompt if prompt is not None else _FakeLocator(text=_PROMPT_TEXT),
        submit=submit,
        breadcrumbs=breadcrumbs,
    )


class _FakeMouse:
    """A page mouse double recording button downs and ups, optionally in a shared event log."""

    def __init__(
        self,
        *,
        events: list[object] | None = None,
        on_down: Callable[[], None] | None = None,
    ) -> None:
        self.downs = 0
        self.ups = 0
        self._events = events
        self._on_down = on_down

    async def move(self, x: float, y: float, *, humanize: bool = False) -> None:
        if self._events is not None:
            self._events.append(("move", x, y, humanize))

    async def down(self) -> None:
        self.downs += 1
        if self._events is not None:
            self._events.append("down")
        if self._on_down is not None:
            self._on_down()

    async def up(self) -> None:
        self.ups += 1
        if self._events is not None:
            self._events.append("up")


class _FakePage:
    """A page double exposing only URL, frames, native response evaluation and the mouse."""

    def __init__(
        self,
        frames: list[_FakeFrame] | None = None,
        *,
        url: str = _PAGE_URL,
        responses: list[str] | None = None,
        mouse: _FakeMouse | None = None,
    ) -> None:
        self.url = url
        self.frames = list(frames or [])
        self.responses = list(responses or [])
        self.mouse = mouse if mouse is not None else _FakeMouse()

    async def evaluate(self, expression: str, arg: object = None) -> list[str]:
        return list(self.responses)


class _FakeVlm:
    """A VLM client double recording one grid or one canvas plan request."""

    def __init__(  # noqa: PLR0913 - a spec-constrained double exposes each optional field
        self,
        selection: GridSelection | None = None,
        *,
        selections: tuple[GridSelection, ...] | None = None,
        plan: CanvasActionPlan | None = None,
        plans: tuple[CanvasActionPlan, ...] | None = None,
        error: VisionError | None = None,
        plan_error: VisionError | None = None,
        plan_error_on: int | None = None,
        plan_hang_on: int | None = None,
        on_select: Callable[[], None] | None = None,
        hang: bool = False,
    ) -> None:
        self._selection = selection if selection is not None else GridSelection(cells=(0,))
        self._selections = tuple(selections) if selections is not None else (self._selection,)
        if plans is not None:
            self._plans = tuple(plans)
        elif plan is not None:
            self._plans = (plan,)
        else:
            self._plans = (CanvasActionPlan(actions=()),)
        self._error = error
        self._plan_error = plan_error
        self._plan_error_on = plan_error_on
        self._plan_hang_on = plan_hang_on
        self._on_select = on_select
        self._hang = hang
        self.calls: list[tuple[bytes, str, float]] = []
        self.plan_calls: list[tuple[bytes, str, float]] = []

    async def select_grid(self, image: bytes, instruction: str, seconds_remaining: float) -> GridSelection:
        self.calls.append((image, instruction, seconds_remaining))
        if self._on_select is not None:
            self._on_select()
        if self._hang:
            await asyncio.sleep(5.0)
        if self._error is not None:
            raise self._error
        return self._selections[min(len(self.calls) - 1, len(self._selections) - 1)]

    async def plan_canvas(self, image: bytes, instruction: str, seconds_remaining: float) -> CanvasActionPlan:
        self.plan_calls.append((image, instruction, seconds_remaining))
        if self._on_select is not None:
            self._on_select()
        if self._hang or len(self.plan_calls) == self._plan_hang_on:
            await asyncio.sleep(5.0)
        plan_error_due = self._plan_error_on is None or len(self.plan_calls) >= self._plan_error_on
        if self._plan_error is not None and plan_error_due:
            raise self._plan_error
        return self._plans[min(len(self.plan_calls) - 1, len(self._plans) - 1)]


async def _run_solver(
    page: _FakePage,
    vlm: _FakeVlm,
    *,
    seconds: float = 2.0,
    initial: frozenset[str] = _EMPTY_TOKENS,
    native_tab: Tab | None = None,
) -> CaptchaAttemptResult:
    return await solve_hcaptcha_challenge(
        page,
        lambda: seconds,
        vlm_client=vlm,
        initial_tokens=initial,
        native_tab=native_tab,
    )


class GateTests(IsolatedAsyncioTestCase):
    """Origin, frame and structure gates fail closed before any upload or click."""

    async def test_wrong_origin_does_not_upload(self) -> None:
        page = _FakePage([_FakeFrame()], url="file:///tmp/fixture.html")
        vlm = _FakeVlm()

        result = await _run_solver(page, vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)
        self.assertIsNone(result.token)

    async def test_exact_public_https_origin_can_upload(self) -> None:
        page = _FakePage(
            [_FakeFrame()], url="https://maximedrn.github.io/hcaptcha-solver-python-selenium/", responses=["fresh"]
        )
        vlm = _FakeVlm(GridSelection(cells=(0,)))

        result = await _run_solver(page, vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh")

    async def test_public_http_origin_can_upload(self) -> None:
        page = _FakePage([_FakeFrame()], url="http://example.com/challenge", responses=["fresh"])
        vlm = _FakeVlm()

        result = await _run_solver(page, vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(result.outcome.status, "handled")

    async def test_public_https_redirect_during_upload_cannot_click(self) -> None:
        page = _FakePage([_FakeFrame()], url="https://maximedrn.github.io/hcaptcha-solver-python-selenium/")
        frame = page.frames[0]
        vlm = _FakeVlm(GridSelection(cells=(0,)), on_select=lambda: setattr(page, "url", "https://other.example/"))

        result = await _run_solver(page, vlm)

        self.assertEqual([task.clicks for task in frame.tasks], [0] * 9)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)

    async def test_ambiguous_challenge_frames_do_not_upload(self) -> None:
        page = _FakePage([_FakeFrame(), _FakeFrame(url=_QUERY_CHALLENGE_URL)])
        vlm = _FakeVlm()

        result = await _run_solver(page, vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_missing_challenge_frame_is_unavailable(self) -> None:
        page = _FakePage([_FakeFrame(url=_CHECKBOX_URL)])
        vlm = _FakeVlm()

        result = await _run_solver(page, vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_absent_visible_view_is_unavailable(self) -> None:
        frame = _FakeFrame(view=_FakeLocator(count=0))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_fewer_than_nine_visible_tasks_is_unavailable(self) -> None:
        tasks = _default_tasks()
        tasks[8] = _FakeTask(9, visible=False)
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([_FakeFrame(tasks=tasks)]), vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_extra_visible_task_is_unavailable(self) -> None:
        tasks = [*_default_tasks(), _FakeTask(1)]
        vlm = _FakeVlm()
        result = await _run_solver(_FakePage([_FakeFrame(tasks=tasks)]), vlm)
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)
        self.assertEqual(vlm.calls, [])

    async def test_label_ten_cannot_stand_in_for_one(self) -> None:
        tasks = [_FakeTask(10), *_default_tasks()[1:]]
        vlm = _FakeVlm()
        result = await _run_solver(_FakePage([_FakeFrame(tasks=tasks)]), vlm)
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)
        self.assertEqual(vlm.calls, [])

    async def test_similar_frame_marker_is_not_a_challenge(self) -> None:
        url = _CHALLENGE_URL.replace("frame=challenge", "frame=challenge-other")
        vlm = _FakeVlm()
        result = await _run_solver(_FakePage([_FakeFrame(url=url)]), vlm)
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)
        self.assertEqual(vlm.calls, [])

    async def test_absent_submit_is_unavailable(self) -> None:
        frame = _FakeFrame(submit=_FakeLocator(count=0))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_expired_budget_is_timed_out_without_upload(self) -> None:
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([_FakeFrame()]), vlm, seconds=0.0)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(result.outcome.status, "timed_out")
        self.assertEqual(result.outcome.reason, TIMED_OUT_REASON)


class SelectionTests(IsolatedAsyncioTestCase):
    """Only the VLM-selected tiles are clicked, then exactly one submit."""

    async def test_nine_tile_index_mapping_clicks_selected_tiles(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([_FakeFrame(url=_CHECKBOX_URL), frame], responses=["fresh-token"])
        vlm = _FakeVlm(GridSelection(cells=(0, 4, 8)))

        result = await _run_solver(page, vlm)

        self.assertEqual(vlm.calls[0][1], VLM_INSTRUCTION)
        self.assertEqual([task.clicks for task in frame.tasks], [1, 0, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh-token")

    async def test_target_clicks_exactly_the_selected_tile(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame], responses=["fresh-token"])
        vlm = _FakeVlm(GridSelection(cells=(2,)))

        result = await _run_solver(page, vlm)

        self.assertEqual([task.clicks for task in frame.tasks], [0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")

    async def test_query_challenge_marker_is_recognized(self) -> None:
        frame = _FakeFrame(url=_QUERY_CHALLENGE_URL)
        page = _FakePage([frame], responses=["fresh-token"])

        result = await _run_solver(page, _FakeVlm(GridSelection(cells=(0,))))

        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(frame.submit.clicks, 1)

    async def test_empty_selection_does_not_click_or_submit(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame])
        vlm = _FakeVlm(GridSelection(cells=()))

        result = await _run_solver(page, vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual([task.clicks for task in frame.tasks], [0] * 9)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, EMPTY_SELECTION_REASON)
        self.assertIsNone(result.token)


class RevisionTests(IsolatedAsyncioTestCase):
    """A changed revision or a lost origin after the VLM never clicks or succeeds."""

    async def test_changed_screenshot_after_vlm_does_not_click(self) -> None:
        view = _FakeLocator(on_screenshot=lambda call: _PNG if call == 1 else _OTHER_PNG)
        frame = _FakeFrame(view=view)
        page = _FakePage([frame])
        vlm = _FakeVlm(GridSelection(cells=(0,)))

        result = await _run_solver(page, vlm)

        self.assertEqual([task.clicks for task in frame.tasks], [0] * 9)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_redirect_during_vlm_request_is_not_success(self) -> None:
        page = _FakePage([_FakeFrame()])
        vlm = _FakeVlm(GridSelection(cells=(0,)), on_select=lambda: setattr(page, "url", _OFF_ORIGIN_URL))
        frame = page.frames[0]

        result = await _run_solver(page, vlm)

        self.assertEqual([task.clicks for task in frame.tasks], [0] * 9)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertNotEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)

    async def test_redirect_during_click_is_not_success(self) -> None:
        page = _FakePage()
        frame = _FakeFrame()
        frame.tasks[0]._on_click = lambda: setattr(page, "url", _OFF_ORIGIN_URL)
        page.frames.append(frame)
        vlm = _FakeVlm(GridSelection(cells=(0,)))

        result = await _run_solver(page, vlm)

        self.assertEqual(frame.tasks[0].clicks, 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertNotEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)


class TokenTests(IsolatedAsyncioTestCase):
    """A fresh response token is success; a stale or absent one is not."""

    async def test_fresh_token_is_accepted(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame], responses=["old-token", "new-token"])

        result = await _run_solver(page, _FakeVlm(GridSelection(cells=(0,))), initial=frozenset({"old-token"}))

        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "new-token")
        self.assertEqual(frame.submit.clicks, 1)

    async def test_stale_token_is_not_success(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame], responses=["same-token"])
        vlm = _FakeVlm(GridSelection(cells=(0,)))

        result = await _run_solver(page, vlm, seconds=0.2, initial=frozenset({"same-token"}))

        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)


class FailureTests(IsolatedAsyncioTestCase):
    """VLM failures and cancellation map to fixed outcomes with no fabricated token."""

    async def test_malformed_response_is_error(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame])
        vlm = _FakeVlm(error=VisionError("malformed_response"))

        result = await _run_solver(page, vlm)

        self.assertEqual(result.outcome.status, "error")
        self.assertEqual(result.outcome.reason, ERROR_REASON)
        self.assertIsNone(result.token)
        self.assertEqual([task.clicks for task in frame.tasks], [0] * 9)

    async def test_vlm_timeout_maps_to_timed_out(self) -> None:
        page = _FakePage([_FakeFrame()])
        vlm = _FakeVlm(error=VisionError(REQUEST_TIMEOUT))

        result = await _run_solver(page, vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(result.outcome.status, "timed_out")
        self.assertEqual(result.outcome.reason, TIMED_OUT_REASON)
        self.assertIsNone(result.token)

    async def test_cancellation_propagates(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame])
        vlm = _FakeVlm(hang=True)

        task = asyncio.create_task(
            solve_hcaptcha_challenge(
                page,
                lambda: 5.0,
                vlm_client=vlm,
                initial_tokens=frozenset(),
            )
        )
        await asyncio.sleep(0.01)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([tile.clicks for tile in frame.tasks], [0] * 9)
        self.assertEqual(frame.submit.clicks, 0)


class ResultShapeTests(TestCase):
    """The frozen result hides its token and keeps no instance dict."""

    def test_repr_hides_the_token(self) -> None:
        value = "private-token-value"
        result = CaptchaAttemptResult(
            outcome=VisionOutcome(status="handled", reason=PAGE_ACCEPTED_REASON),
            token=value,
        )

        self.assertEqual(result.token, value)
        self.assertNotIn(value, repr(result))

    def test_result_has_slots(self) -> None:
        result = CaptchaAttemptResult(
            outcome=VisionOutcome(status="needs_human", reason=PAGE_NOT_ACCEPTED_REASON),
            token=None,
        )

        self.assertFalse(hasattr(result, "__dict__"))


class CanvasGateTests(IsolatedAsyncioTestCase):
    """A canvas frame without one visible canvas, prompt and submit never uploads or clicks."""

    async def test_hidden_challenge_view_does_not_upload(self) -> None:
        frame = _canvas_frame()
        frame.view = _FakeLocator(visible=False)
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_absent_canvas_does_not_upload(self) -> None:
        frame = _canvas_frame(canvas=_FakeLocator(count=0))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_duplicate_canvas_does_not_upload(self) -> None:
        frame = _canvas_frame(canvas=_FakeLocator(count=2))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_blank_prompt_does_not_upload(self) -> None:
        frame = _canvas_frame(prompt=_FakeLocator(text="   "))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_over_long_prompt_does_not_upload(self) -> None:
        frame = _canvas_frame(prompt=_FakeLocator(text="x" * (MAX_INSTRUCTION_CHARS + 1)))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_invalid_canvas_size_does_not_upload(self) -> None:
        bounds = {"x": 0.0, "y": 0.0, "width": 0.0, "height": 80.0}
        frame = _canvas_frame(canvas=_FakeLocator(bounds=bounds))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_absent_submit_does_not_upload(self) -> None:
        frame = _canvas_frame(submit=_FakeLocator(count=0))
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)

    async def test_ambiguous_tiles_do_not_fall_back_to_canvas(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(tasks=[_FakeTask(1), _FakeTask(1)], canvas=canvas)
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(vlm.plan_calls, [])
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(result.outcome.reason, CHALLENGE_UNAVAILABLE_REASON)


class CanvasSelectionTests(IsolatedAsyncioTestCase):
    """A canvas round uploads the canvas screenshot and performs the planned click actions."""

    async def test_planned_click_maps_to_canvas_css(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], responses=["fresh-token"])
        point = Point(0.25, 0.5)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=point),)))

        result = await _run_solver(page, vlm)

        self.assertEqual(vlm.calls, [])
        self.assertEqual(vlm.plan_calls, [(_PNG, _PROMPT_TEXT, 2.0)])
        self.assertTrue(canvas.screenshot_options)
        self.assertTrue(all(options == {"type": "png", "scale": "css"} for options in canvas.screenshot_options))
        self.assertEqual(
            canvas.positions,
            [{"x": point.x * _CANVAS_BOUNDS["width"], "y": point.y * _CANVAS_BOUNDS["height"]}],
        )
        self.assertEqual(canvas.hovers, 0)
        self.assertEqual(page.mouse.downs, 0)
        self.assertEqual(page.mouse.ups, 0)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh-token")

    async def test_screenshot_dpi_scaling_does_not_shift_clicks(self) -> None:
        point = Point(0.1, 0.5)
        bounds = {"x": 0.0, "y": 0.0, "width": 100.0, "height": 50.0}
        double_dpi = _png_with_size(200, 100)
        canvas = _FakeLocator(bounds=bounds, on_screenshot=lambda _call: double_dpi)
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=point),)))

        result = await _run_solver(_FakePage([frame], responses=["fresh-token"]), vlm)

        self.assertEqual(vlm.plan_calls[0][0], double_dpi)
        self.assertEqual(
            canvas.positions,
            [{"x": point.x * bounds["width"], "y": point.y * bounds["height"]}],
        )
        self.assertEqual(result.outcome.status, "handled")

    async def test_empty_action_plan_does_not_click_or_submit(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=()))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, EMPTY_SELECTION_REASON)
        self.assertIsNone(result.token)

    async def test_malformed_action_plan_does_not_click(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan_error=VisionError("malformed_response"))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(canvas.hovers, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "error")
        self.assertEqual(result.outcome.reason, ERROR_REASON)


class CanvasDragTests(IsolatedAsyncioTestCase):
    """A planned drag hovers the canvas endpoints and always releases the mouse button."""

    async def test_planned_drag_moves_between_canvas_points_and_releases(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], responses=["fresh-token"])
        start = Point(0.35, 0.5)
        source_center = Point(0.25, 0.5)
        target = Point(0.75, 0.5)
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(CanvasDrag(start=start, source_center=source_center, target=target, hold_ms=0),)
            )
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(
            canvas.hover_positions,
            [
                {"x": start.x * _CANVAS_BOUNDS["width"], "y": start.y * _CANVAS_BOUNDS["height"]},
                {
                    "x": (target.x + start.x - source_center.x) * _CANVAS_BOUNDS["width"],
                    "y": target.y * _CANVAS_BOUNDS["height"],
                },
            ],
        )
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(page.mouse.downs, 1)
        self.assertEqual(page.mouse.ups, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh-token")

    async def test_mixed_actions_run_in_planned_order(self) -> None:
        events: list[object] = []
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS), events=events)
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], responses=["fresh-token"], mouse=_FakeMouse(events=events))
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasClick(point=Point(0.1, 0.1)),
                    CanvasDrag(
                        start=Point(0.25, 0.5), source_center=Point(0.25, 0.5), target=Point(0.75, 0.5), hold_ms=0
                    ),
                    CanvasClick(point=Point(0.9, 0.9)),
                )
            )
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(
            events,
            [
                ("click", {"x": 12.0, "y": 8.0}),
                ("hover", {"x": 30.0, "y": 40.0}),
                "down",
                ("hover", {"x": 90.0, "y": 40.0}),
                "up",
                ("click", {"x": 108.0, "y": 72.0}),
            ],
        )
        self.assertEqual(result.outcome.status, "handled")

    async def test_changed_origin_before_press_does_not_press(self) -> None:
        mouse = _FakeMouse()
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS), on_hover=lambda: setattr(page, "url", _OFF_ORIGIN_URL))
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], mouse=mouse)
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasDrag(
                        start=Point(0.25, 0.5), source_center=Point(0.25, 0.5), target=Point(0.75, 0.5), hold_ms=1000
                    ),
                )
            )
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(mouse.downs, 0)
        self.assertEqual(mouse.ups, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)

    async def test_changed_origin_during_drag_releases_button(self) -> None:
        frame = _canvas_frame(canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS)))
        mouse = _FakeMouse(on_down=lambda: setattr(page, "url", _OFF_ORIGIN_URL))
        page = _FakePage([frame], mouse=mouse)
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasDrag(
                        start=Point(0.25, 0.5), source_center=Point(0.25, 0.5), target=Point(0.75, 0.5), hold_ms=1000
                    ),
                )
            )
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(mouse.downs, 1)
        self.assertEqual(mouse.ups, 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)

    async def test_changed_frame_during_drag_releases_button(self) -> None:
        frame = _canvas_frame(canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS)))
        mouse = _FakeMouse(on_down=lambda: setattr(frame, "url", _CHECKBOX_URL))
        page = _FakePage([frame], mouse=mouse)
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasDrag(
                        start=Point(0.25, 0.5), source_center=Point(0.25, 0.5), target=Point(0.75, 0.5), hold_ms=1000
                    ),
                )
            )
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(mouse.downs, 1)
        self.assertEqual(mouse.ups, 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_cancellation_during_hold_releases_button(self) -> None:
        frame = _canvas_frame(canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS)))
        page = _FakePage([frame])
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasDrag(
                        start=Point(0.25, 0.5), source_center=Point(0.25, 0.5), target=Point(0.75, 0.5), hold_ms=1000
                    ),
                )
            )
        )

        task = asyncio.create_task(
            solve_hcaptcha_challenge(page, lambda: 5.0, vlm_client=vlm, initial_tokens=frozenset())
        )
        await asyncio.sleep(0.05)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(page.mouse.downs, 1)
        self.assertEqual(page.mouse.ups, 1)
        self.assertEqual(frame.submit.clicks, 0)

    async def test_skip_labelled_submit_is_not_clicked(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas, submit=_FakeLocator(text="Skip"))
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)

    async def test_skip_aria_label_submit_is_not_clicked(self) -> None:
        frame = _canvas_frame(
            canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS)),
            submit=_FakeLocator(aria_label="Skip"),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)


class CanvasRevisionTests(IsolatedAsyncioTestCase):
    """A repainted canvas keeps the plan only for the same frame, prompt and click surface."""

    def _repainted(self) -> _FakeLocator:
        """Return a canvas whose screenshot changes after the baseline read."""
        return _FakeLocator(
            bounds=dict(_CANVAS_BOUNDS),
            on_screenshot=lambda call: _PNG if call == 1 else _OTHER_PNG,
        )

    async def test_repainted_canvas_after_vlm_clicks_with_fresh_token(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame], responses=["fresh-token"]), vlm)

        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-token")

    async def test_repainted_canvas_without_fresh_token_is_not_handled(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm, seconds=0.2)

        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)

    async def test_changed_frame_after_vlm_does_not_click(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(
            plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)),
            on_select=lambda: setattr(frame, "url", _OTHER_CHALLENGE_URL),
        )

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_changed_prompt_after_vlm_does_not_click(self) -> None:
        prompt = _FakeLocator(text=lambda call: _PROMPT_TEXT if call == 1 else "a different prompt")
        frame = _canvas_frame(prompt=prompt)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(frame.canvas.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_changed_canvas_size_after_vlm_does_not_click(self) -> None:
        bounds = dict(_CANVAS_BOUNDS)
        canvas = _FakeLocator(bounds=bounds, on_screenshot=lambda _call: _widen(bounds))
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_redirect_during_planning_is_not_success(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame])
        vlm = _FakeVlm(
            plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)),
            on_select=lambda: setattr(page, "url", _OFF_ORIGIN_URL),
        )

        result = await _run_solver(page, vlm)

        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertNotEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)

    async def test_absent_fresh_token_is_not_handled(self) -> None:
        canvas = _FakeLocator(bounds=dict(_CANVAS_BOUNDS))
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm, seconds=0.2)

        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)


class CanvasReplanTests(IsolatedAsyncioTestCase):
    """One bounded canvas replan after an empty plan, only for a repainted unchanged challenge."""

    def _repainted(self) -> _FakeLocator:
        """Return a canvas whose screenshot is blank on the first read and painted afterwards."""
        return _FakeLocator(
            bounds=dict(_CANVAS_BOUNDS),
            on_screenshot=lambda call: _PNG if call == 1 else _OTHER_PNG,
        )

    def _two_plans(self) -> tuple[CanvasActionPlan, CanvasActionPlan]:
        """Return an empty first plan followed by one canvas click."""
        return (CanvasActionPlan(actions=()), CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

    async def test_blank_canvas_replans_once_and_clicks_on_repaint(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=self._two_plans())

        result = await _run_solver(_FakePage([frame], responses=["fresh-token"]), vlm)

        self.assertEqual([image for image, _, _ in vlm.plan_calls], [_PNG, _OTHER_PNG])
        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh-token")

    async def test_unchanged_canvas_does_not_reupload(self) -> None:
        frame = _canvas_frame()
        vlm = _FakeVlm()

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(frame.canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "needs_human")
        self.assertEqual(result.outcome.reason, EMPTY_SELECTION_REASON)

    async def test_persistent_empty_plan_after_repaint_stops_at_two_uploads(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=(CanvasActionPlan(actions=()), CanvasActionPlan(actions=())))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 2)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, EMPTY_SELECTION_REASON)

    async def test_changed_prompt_does_not_reupload(self) -> None:
        prompt = _FakeLocator(text=lambda call: _PROMPT_TEXT if call == 1 else "a different prompt")
        frame = _canvas_frame(prompt=prompt, canvas=self._repainted())
        vlm = _FakeVlm(plans=self._two_plans())

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(frame.canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_changed_frame_does_not_reupload(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=self._two_plans(), on_select=lambda: setattr(frame, "url", _OTHER_CHALLENGE_URL))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_changed_canvas_size_does_not_reupload(self) -> None:
        bounds = dict(_CANVAS_BOUNDS)
        canvas = _FakeLocator(bounds=bounds, on_screenshot=lambda call: _PNG if call == 1 else _OTHER_PNG)
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=self._two_plans(), on_select=lambda: bounds.__setitem__("width", 240.0))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)

    async def test_repainted_canvas_after_replan_clicks_with_fresh_token(self) -> None:
        canvas = _FakeLocator(
            bounds=dict(_CANVAS_BOUNDS),
            on_screenshot=lambda call: (_PNG, _OTHER_PNG, _THIRD_PNG)[call - 1],
        )
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=self._two_plans())

        result = await _run_solver(_FakePage([frame], responses=["fresh-token"]), vlm)

        self.assertEqual(len(vlm.plan_calls), 2)
        self.assertEqual(canvas.clicks, 1)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.token, "fresh-token")

    async def test_exhausted_deadline_before_replan_is_timed_out(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        remaining = [2.0]
        vlm = _FakeVlm(plans=self._two_plans(), on_select=lambda: remaining.__setitem__(0, 0.0))

        result = await solve_hcaptcha_challenge(
            _FakePage([frame]),
            lambda: remaining[0],
            vlm_client=vlm,
            initial_tokens=frozenset(),
        )

        self.assertEqual(len(vlm.plan_calls), 1)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(result.outcome.status, "timed_out")
        self.assertEqual(result.outcome.reason, TIMED_OUT_REASON)

    async def test_cancellation_during_replan_propagates(self) -> None:
        canvas = self._repainted()
        frame = _canvas_frame(canvas=canvas)
        vlm = _FakeVlm(plans=self._two_plans(), plan_hang_on=2)

        task = asyncio.create_task(
            solve_hcaptcha_challenge(_FakePage([frame]), lambda: 5.0, vlm_client=vlm, initial_tokens=frozenset())
        )
        await asyncio.sleep(0.05)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(vlm.plan_calls), 2)
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(frame.submit.clicks, 0)


class RoundTransitionTests(IsolatedAsyncioTestCase):
    async def test_second_canvas_round_requires_second_plan_and_fresh_token(self) -> None:
        state = {"round": 0}
        page = _FakePage()

        def submit() -> None:
            state["round"] += 1
            if state["round"] == 2:
                page.responses = ["fresh-provider-token"]

        frame = _canvas_frame(
            canvas=_FakeLocator(
                bounds=dict(_CANVAS_BOUNDS),
                on_screenshot=lambda _: _PNG if state["round"] == 0 else _OTHER_PNG,
            ),
            submit=_FakeLocator(on_click=submit),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["24px", "8px"] if state["round"] == 0 else ["8px", "24px"]),
        )
        page.frames = [frame]
        plan = CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),))
        vlm = _FakeVlm(plans=(plan, plan))

        result = await _run_solver(page, vlm)

        self.assertEqual(result.outcome.status, "handled")
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-provider-token")
        self.assertEqual(frame.submit.clicks, 2)
        self.assertEqual(frame.canvas.clicks, 2)
        self.assertEqual(len(vlm.plan_calls), 2)
        self.assertNotEqual(vlm.plan_calls[0][0], vlm.plan_calls[1][0])

    async def test_repaint_without_breadcrumb_advance_cannot_upload_again(self) -> None:
        state = {"submitted": False}
        frame = _canvas_frame(
            canvas=_FakeLocator(
                bounds=dict(_CANVAS_BOUNDS),
                on_screenshot=lambda _: _OTHER_PNG if state["submitted"] else _PNG,
            ),
            submit=_FakeLocator(on_click=lambda: state.update(submitted=True)),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["24px", "8px"]),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage(frames=[frame]), vlm, seconds=0.25)

        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertEqual(frame.submit.clicks, 1)
        self.assertEqual(len(vlm.plan_calls), 1)

    async def test_skipped_breadcrumb_is_rejected_without_second_upload(self) -> None:
        state = {"submitted": False}
        frame = _canvas_frame(
            submit=_FakeLocator(on_click=lambda: state.update(submitted=True)),
            breadcrumbs=_FakeBreadcrumbs(
                lambda: ["8px", "8px", "24px"] if state["submitted"] else ["24px", "8px", "8px"]
            ),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage(frames=[frame]), vlm)

        self.assertEqual(result.outcome.reason, STALE_REVISION_REASON)
        self.assertEqual(len(vlm.plan_calls), 1)

    async def test_missing_breadcrumb_keeps_one_round_path(self) -> None:
        frame = _canvas_frame(
            canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS), on_screenshot=lambda n: _PNG if n < 3 else _OTHER_PNG),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["24px", "24px"]),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage(frames=[frame]), vlm, seconds=0.25)

        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertEqual(len(vlm.plan_calls), 1)

    async def test_token_arriving_after_breadcrumb_advance_needs_no_second_upload(self) -> None:
        class DeferredTokenPage(_FakePage):
            reads = 0

            async def evaluate(self, expression: str, arg: object = None) -> list[str]:
                self.reads += 1
                return ["fresh-provider-token"] if self.reads >= 2 else []

        state = {"submitted": False}
        frame = _canvas_frame(
            submit=_FakeLocator(on_click=lambda: state.update(submitted=True)),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["8px", "24px"] if state["submitted"] else ["24px", "8px"]),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(DeferredTokenPage(frames=[frame]), vlm)

        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-provider-token")
        self.assertEqual(len(vlm.plan_calls), 1)


class SemanticRetryTests(IsolatedAsyncioTestCase):
    """An unusable answer never causes repeated inference on an unchanged puzzle page."""

    async def test_malformed_grid_is_not_retried_on_same_page(self) -> None:
        frame = _FakeFrame()
        vlm = _FakeVlm(error=VisionError("malformed_response"))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, ERROR_REASON)
        self.assertIsNone(result.token)

    async def test_empty_grid_is_not_retried_on_same_page(self) -> None:
        frame = _FakeFrame()
        vlm = _FakeVlm(selections=(GridSelection(cells=()), GridSelection(cells=(2,))))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.reason, EMPTY_SELECTION_REASON)
        self.assertIsNone(result.token)

    async def test_request_failure_is_not_retried(self) -> None:
        vlm = _FakeVlm(error=VisionError(REQUEST_FAILED))

        result = await _run_solver(_FakePage([_FakeFrame()]), vlm)

        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(result.outcome.status, "error")


class ResetRetryTests(IsolatedAsyncioTestCase):
    """A rejected answer may consume at most two fresh provider puzzle pages."""

    _SHOTS = (_PNG, _OTHER_PNG, _THIRD_PNG)

    def _frame(self, crumbs: _ResetBreadcrumbs, submits: list[int]) -> _FakeFrame:
        canvas = _FakeLocator(
            bounds=dict(_CANVAS_BOUNDS),
            on_screenshot=lambda _call: self._SHOTS[submits[0] % len(self._SHOTS)],
        )

        def on_submit() -> None:
            submits[0] += 1
            crumbs.on_submit()

        return _canvas_frame(
            canvas=canvas,
            submit=_FakeLocator(on_click=on_submit),
            breadcrumbs=_FakeBreadcrumbs(crumbs.widths),
        )

    async def test_rejected_second_page_allows_exactly_two_extra_pages_to_finish(self) -> None:
        submits = [0]
        crumbs = _ResetBreadcrumbs(count=2, active=0, sequence=(1, 0, 1))
        frame = self._frame(crumbs, submits)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        class AcceptedAfterFour(_FakePage):
            async def evaluate(self, expression: str, arg: object = None) -> list[str]:
                return ["fresh-provider-token"] if submits[0] >= 4 else []

        result = await _run_solver(AcceptedAfterFour([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 4)
        self.assertEqual(frame.submit.clicks, 4)
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-provider-token")

    async def test_third_extra_page_is_not_uploaded(self) -> None:
        submits = [0]
        crumbs = _ResetBreadcrumbs(count=3, active=0, sequence=(1, 2, 0, 1, 2))
        frame = self._frame(crumbs, submits)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 5)
        self.assertEqual(frame.submit.clicks, 5)
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)

    async def test_second_rejection_stops_after_two_extra_pages(self) -> None:
        submits = [0]
        crumbs = _ResetBreadcrumbs(count=2, active=0, sequence=(1, 0, 1, 0))
        frame = self._frame(crumbs, submits)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 4)
        self.assertEqual(frame.submit.clicks, 4)
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)

    async def test_normal_three_page_progression_needs_no_extra_pages(self) -> None:
        submits = [0]
        crumbs = _ResetBreadcrumbs(count=3, active=0, sequence=(1, 2))
        frame = self._frame(crumbs, submits)
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        class AcceptedAfterThree(_FakePage):
            async def evaluate(self, expression: str, arg: object = None) -> list[str]:
                return ["fresh-provider-token"] if submits[0] >= 3 else []

        result = await _run_solver(AcceptedAfterThree([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 3)
        self.assertEqual(frame.submit.clicks, 3)
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-provider-token")

    async def test_changed_prompt_at_same_marker_retries_fresh_page(self) -> None:
        submits = [0]
        prompt = _FakeLocator(text=lambda _call: "Select the 3 items" if submits[0] == 0 else "Drag the piece")
        canvas = _FakeLocator(
            bounds=dict(_CANVAS_BOUNDS),
            on_screenshot=lambda _call: _PNG if submits[0] == 0 else _OTHER_PNG,
        )

        def on_submit() -> None:
            submits[0] += 1

        frame = _canvas_frame(
            prompt=prompt,
            canvas=canvas,
            submit=_FakeLocator(on_click=on_submit),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["8px", "24px"]),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        class AcceptedAfterTwo(_FakePage):
            async def evaluate(self, expression: str, arg: object = None) -> list[str]:
                return ["fresh-provider-token"] if submits[0] >= 2 else []

        result = await _run_solver(AcceptedAfterTwo([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 2)
        self.assertEqual(frame.submit.clicks, 2)
        self.assertEqual(result.outcome.reason, PAGE_ACCEPTED_REASON)
        self.assertEqual(result.token, "fresh-provider-token")

    async def test_third_same_marker_replacement_is_not_uploaded(self) -> None:
        submits = [0]
        prompts = ("Select the 3 items", "Drag the piece", "Find the animal", "Select the object")
        images = (_PNG, _OTHER_PNG, _THIRD_PNG, _PNG)
        frame = _canvas_frame(
            prompt=_FakeLocator(text=lambda _call: prompts[min(submits[0], 3)]),
            canvas=_FakeLocator(bounds=dict(_CANVAS_BOUNDS), on_screenshot=lambda _call: images[min(submits[0], 3)]),
            submit=_FakeLocator(on_click=lambda: submits.__setitem__(0, submits[0] + 1)),
            breadcrumbs=_FakeBreadcrumbs(lambda: ["8px", "24px"]),
        )
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.5, 0.5)),)))

        result = await _run_solver(_FakePage([frame]), vlm)

        self.assertEqual(len(vlm.plan_calls), 3)
        self.assertEqual(frame.submit.clicks, 3)
        self.assertEqual(result.outcome.reason, PAGE_NOT_ACCEPTED_REASON)
        self.assertIsNone(result.token)


class NativeCanvasInputTests(IsolatedAsyncioTestCase):
    """The bound PyDoll tab owns canvas input; Playwright still locates and screenshots it."""

    async def test_native_click_uses_viewport_canvas_offset_and_releases(self) -> None:
        bounds = {"x": 17.0, "y": 23.0, "width": 120.0, "height": 80.0}
        canvas = _FakeLocator(bounds=bounds)
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], responses=["fresh-token"])
        events: list[object] = []
        mouse = _FakeMouse(events=events)
        tab = cast("Tab", SimpleNamespace(mouse=mouse))
        vlm = _FakeVlm(plan=CanvasActionPlan(actions=(CanvasClick(point=Point(0.25, 0.5)),)))

        result = await _run_solver(page, vlm, native_tab=tab)

        self.assertEqual(
            events,
            [("move", 47.0, 63.0, True), "down", "up", ("move", 175.0, 262.5, True), "down", "up"],
        )
        self.assertEqual(canvas.clicks, 0)
        self.assertEqual(page.mouse.downs, 0)
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "handled")

    async def test_native_drag_keeps_pressed_move_bounded_and_releases_on_origin_loss(self) -> None:
        bounds = {"x": 17.0, "y": 23.0, "width": 120.0, "height": 80.0}
        canvas = _FakeLocator(bounds=bounds)
        frame = _canvas_frame(canvas=canvas)
        page = _FakePage([frame], responses=["fresh-token"])
        events: list[object] = []
        mouse = _FakeMouse(events=events)
        tab = cast("Tab", SimpleNamespace(mouse=mouse))
        vlm = _FakeVlm(
            plan=CanvasActionPlan(
                actions=(
                    CanvasDrag(
                        start=Point(0.35, 0.5),
                        source_center=Point(0.25, 0.5),
                        target=Point(0.75, 0.5),
                        hold_ms=0,
                    ),
                )
            )
        )

        result = await _run_solver(page, vlm, native_tab=tab)

        self.assertEqual(events[0], ("move", 59.0, 63.0, True))
        self.assertEqual(events[1], "down")
        move = cast("tuple[str, float, float, bool]", events[2])
        self.assertAlmostEqual(move[1], 119.0)
        self.assertEqual(move[2:], (63.0, False))
        self.assertEqual(events[3], "up")
        self.assertEqual(canvas.hovers, 0)
        self.assertEqual(page.mouse.downs, 0)
        self.assertEqual(result.outcome.status, "handled")

        events.clear()
        mouse = _FakeMouse(events=events, on_down=lambda: setattr(page, "url", _OFF_ORIGIN_URL))
        tab = cast("Tab", SimpleNamespace(mouse=mouse))
        page.url = _PAGE_URL
        result = await _run_solver(page, vlm, native_tab=tab)
        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)
        self.assertEqual(events[-1], "up")
        self.assertEqual(frame.submit.clicks, 0)


class NativeGridInputTests(IsolatedAsyncioTestCase):
    """DOM tile and submit clicks use the same bound native tab as canvas actions."""

    async def test_native_grid_clicks_only_selected_tiles_then_submits(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame], responses=["fresh-token"])
        events: list[object] = []
        mouse = _FakeMouse(events=events)
        tab = cast("Tab", SimpleNamespace(mouse=mouse))
        vlm = _FakeVlm(selection=GridSelection(cells=(0, 4)))

        result = await _run_solver(page, vlm, native_tab=tab)

        self.assertEqual(
            events,
            [
                ("move", 14.5, 24.5, True),
                "down",
                "up",
                ("move", 54.5, 24.5, True),
                "down",
                "up",
                ("move", 175.0, 262.5, True),
                "down",
                "up",
            ],
        )
        self.assertTrue(all(task.clicks == 0 for task in frame.tasks))
        self.assertEqual(frame.submit.clicks, 0)
        self.assertEqual(result.outcome.status, "handled")

    async def test_native_grid_button_releases_on_origin_loss(self) -> None:
        frame = _FakeFrame()
        page = _FakePage([frame])
        events: list[object] = []
        mouse = _FakeMouse(events=events, on_down=lambda: setattr(page, "url", _OFF_ORIGIN_URL))
        tab = cast("Tab", SimpleNamespace(mouse=mouse))

        result = await _run_solver(page, _FakeVlm(), native_tab=tab)

        self.assertEqual(result.outcome.reason, ORIGIN_LOST_REASON)
        self.assertEqual(events, [("move", 14.5, 24.5, True), "down", "up"])
        self.assertEqual(frame.submit.clicks, 0)
