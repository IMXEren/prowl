# pyright: reportOptionalCall=false
"""Tests for opt-in selected-page checkbox solving on ``PageHandler.get``.

The page, frames, and locators are spec-constrained doubles so provider
selection, the exactly-one contract, the opt-in default, and snapshot
propagation are verified without a live browser.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import ElementHandle, Frame, Locator, Page
from pydoll.browser.tab import Tab
from turbohtml import Element, Text

from prowl.browser.browser import TabGroup
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.browser.solvers import solve_visible_captcha, solve_visible_checkbox
from prowl.browser.solvers.vision.schemas import VisionOutcome

_RECAPTCHA_URL = "https://www.google.com/recaptcha/api2/anchor?k=site"
_HCAPTCHA_URL = "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html"
_HCAPTCHA_CHALLENGE_URL = "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html#frame=challenge"


def _checkbox(*, count: int = 1, visible: bool = True) -> Mock:
    locator = Mock(spec=Locator)
    locator.count = AsyncMock(return_value=count)
    locator.is_visible = AsyncMock(return_value=visible)
    locator.click = AsyncMock()
    return locator


def _frame(url: str, checkbox: Mock, *, frame_visible: bool = True) -> Mock:
    frame = Mock(spec=Frame)
    frame.url = url
    element = Mock(spec=ElementHandle)
    element.is_visible = AsyncMock(return_value=frame_visible)
    frame.frame_element = AsyncMock(return_value=element)
    frame.locator = Mock(return_value=checkbox)
    return frame


def _page(frames: list[Mock], evaluate: AsyncMock) -> Mock:
    page = Mock(spec=Page)
    page.frames = frames
    page.evaluate = evaluate
    page.query_selector_all = AsyncMock(return_value=[])
    return page


def _document(title_text: str) -> Element:
    html = Element("html")
    head = Element("head")
    title = Element("title")
    title.append(Text(title_text))
    head.append(title)
    html.append(head)
    return html


def _site(group: Mock) -> PageHandler:
    site = PageHandler(group)
    site.timeout = 30
    site.start = time.perf_counter()
    return site


def _group() -> tuple[Mock, Mock]:
    group = Mock(spec=TabGroup)
    tab = Mock(spec=Tab)
    tab.go_to = AsyncMock()
    tab.enable_page_events = AsyncMock()
    tab.disable_page_events = AsyncMock()
    tab.disable_fetch_events = AsyncMock()
    tab.remove_callback = AsyncMock()
    future: asyncio.Future[Tab] = asyncio.get_running_loop().create_future()
    future.set_result(tab)
    group.ptab = future
    page = Mock(spec=Page)
    page.url = "https://example.com/page"
    group.ppage = page
    return group, tab


async def _run_get(
    group: Mock,
    *,
    tree: Element,
    solve_captcha: bool,
    site: PageHandler | None = None,
) -> PageResponse:
    site = site if site is not None else _site(group)
    with (
        patch.object(PageHandler, "_add_network_listeners", AsyncMock()),
        patch.object(PageHandler, "_remove_network_listeners", AsyncMock()),
        patch.object(PageHandler, "_remove_scoped_headers", AsyncMock()),
        patch.object(PageHandler, "_check_if_loaded", AsyncMock(return_value=True)),
        patch.object(PageHandler, "_wait_page_load", AsyncMock()),
        patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=tree)),
    ):
        return await site.get("https://example.com/page", 30, solve_captcha=solve_captcha)


class VisibleCheckboxSelectionTests(IsolatedAsyncioTestCase):
    """Only one recognized provider frame is ever dispatched to a click."""

    async def test_absent_provider_returns_none_without_click(self) -> None:
        evaluate = AsyncMock(return_value=[])
        page = _page([_frame("https://example.com/plain", _checkbox())], evaluate)

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        evaluate.assert_not_awaited()

    async def test_single_recaptcha_frame_returns_fresh_token(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        evaluate = AsyncMock(side_effect=[[""], ["fresh-token"]])
        page = _page([frame], evaluate)

        self.assertEqual(await solve_visible_checkbox(page, lambda: 5.0), ("recaptcha", "fresh-token"))
        checkbox.click.assert_awaited_once()

    async def test_single_hcaptcha_frame_without_grid_returns_fresh_token(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_HCAPTCHA_URL, checkbox)
        evaluate = AsyncMock(side_effect=[[""], ["fresh-token"]])
        page = _page([frame], evaluate)

        self.assertEqual(await solve_visible_checkbox(page, lambda: 5.0), ("hcaptcha", "fresh-token"))
        checkbox.click.assert_awaited_once()

    async def test_one_visible_checkbox_among_same_provider_frames(self) -> None:
        hidden = _checkbox()
        visible = _checkbox()
        page = _page(
            [_frame(_RECAPTCHA_URL, hidden, frame_visible=False), _frame(_RECAPTCHA_URL, visible)],
            AsyncMock(side_effect=[[""], ["fresh-token"]]),
        )
        self.assertEqual(await solve_visible_checkbox(page, lambda: 5.0), ("recaptcha", "fresh-token"))
        hidden.click.assert_not_awaited()
        visible.click.assert_awaited_once()

    async def test_two_visible_checkboxes_of_same_provider_are_ambiguous(self) -> None:
        first = _checkbox()
        second = _checkbox()
        page = _page([_frame(_RECAPTCHA_URL, first), _frame(_RECAPTCHA_URL, second)], AsyncMock())
        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        first.click.assert_not_awaited()
        second.click.assert_not_awaited()

    async def test_two_provider_frames_are_ambiguous_and_never_click(self) -> None:
        recaptcha = _checkbox()
        hcaptcha = _checkbox()
        frames = [_frame(_RECAPTCHA_URL, recaptcha), _frame(_HCAPTCHA_URL, hcaptcha)]
        page = _page(frames, AsyncMock(return_value=[]))

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0))
        recaptcha.click.assert_not_awaited()
        hcaptcha.click.assert_not_awaited()


class GetCaptchaOptInTests(IsolatedAsyncioTestCase):
    """Ordinary GET stays inert; opted-in GET attaches the fresh provider pair."""

    async def test_default_get_does_not_solve_captcha(self) -> None:
        group, _ = _group()
        with patch("prowl.browser.page_handler.solve_visible_captcha", AsyncMock()) as solve:
            result = await _run_get(group, tree=_document("Ready"), solve_captcha=False)

        solve.assert_not_called()
        self.assertIsNone(result.captcha_provider)
        self.assertIsNone(result.captcha_token)

    async def test_opted_in_get_attaches_provider_and_token(self) -> None:
        group, _ = _group()
        solved = ("recaptcha", "fresh-token")
        with patch("prowl.browser.page_handler.solve_visible_captcha", AsyncMock(return_value=solved)) as solve:
            result = await _run_get(group, tree=_document("Ready"), solve_captcha=True)

        solve.assert_awaited_once()
        self.assertEqual(result.captcha_provider, "recaptcha")
        self.assertEqual(result.captcha_token, "fresh-token")


class CaptchaSnapshotPropagationTests(IsolatedAsyncioTestCase):
    """Snapshot copies a carried captcha pair unchanged."""

    async def test_snapshot_preserves_captcha_pair(self) -> None:
        group, _ = _group()
        site = _site(group)
        source = PageResponse(
            "<html></html>",
            status_code=200,
            captcha_provider="hcaptcha",
            captcha_token="h-token",  # noqa: S106
        )

        with patch.object(PageHandler, "build_dom_tree", AsyncMock(return_value=_document("Ready"))):
            result = await site.snapshot(source, wait_in_seconds=0.001)

        self.assertEqual(result.captcha_provider, "hcaptcha")
        self.assertEqual(result.captcha_token, "h-token")


class VisibleCaptchaDispatchTests(IsolatedAsyncioTestCase):
    async def test_absent_or_multiple_provider_mounts_do_not_solve(self) -> None:
        page = _page([], AsyncMock())
        with patch("prowl.browser.solvers.dispatch.solve_altcha", AsyncMock()) as solve:
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_not_awaited()
            page.query_selector_all.side_effect = [[Mock()], [Mock()], []]
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_not_awaited()

    async def test_late_friendly_widget_is_detected_within_the_same_deadline(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[], [], [], [], [Mock()], []]
        with patch(
            "prowl.browser.solvers.dispatch.solve_friendly_captcha",
            AsyncMock(return_value="fresh-payload"),
        ) as solve:
            self.assertEqual(await solve_visible_captcha(page, lambda: 1.0), ("friendly", "fresh-payload"))
            solve.assert_awaited_once()

    async def test_altcha_dispatch_returns_only_a_fresh_result(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[Mock()], [], []]
        with patch("prowl.browser.solvers.dispatch.solve_altcha", AsyncMock(return_value="fresh-payload")) as solve:
            self.assertEqual(await solve_visible_captcha(page, lambda: 5.0), ("altcha", "fresh-payload"))
            solve.assert_awaited_once()

    async def test_friendly_dispatch_does_not_report_unsolved_widget(self) -> None:
        page = _page([], AsyncMock())
        page.query_selector_all.side_effect = [[], [Mock()], []]
        with patch("prowl.browser.solvers.dispatch.solve_friendly_captcha", AsyncMock(return_value=None)) as solve:
            self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))
            solve.assert_awaited_once()

    async def test_turnstile_returns_shared_provider_pair(self) -> None:
        page = _page([_frame("https://challenges.cloudflare.com/widget", _checkbox())], AsyncMock())
        solve = AsyncMock(return_value="fresh-token")
        self.assertEqual(
            await solve_visible_captcha(page, lambda: 5.0, provider_attempts={"turnstile": solve}),
            ("turnstile", "fresh-token"),
        )
        solve.assert_awaited_once_with(frozenset())

    async def test_turnstile_without_token_or_adapter_reports_no_success(self) -> None:
        page = _page([_frame("https://challenges.cloudflare.com/widget", _checkbox())], AsyncMock())
        solve = AsyncMock(return_value=None)
        self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0, provider_attempts={"turnstile": solve}))
        self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0))

    async def test_spoofed_cloudflare_frame_never_reaches_solver(self) -> None:
        page = _page([_frame("https://challenges.cloudflare.com.attacker.invalid/widget", _checkbox())], AsyncMock())
        solve = AsyncMock()
        self.assertIsNone(await solve_visible_captcha(page, lambda: 0.01, provider_attempts={"turnstile": solve}))
        solve.assert_not_awaited()

    async def test_mixed_turnstile_and_checkbox_providers_are_refused(self) -> None:
        page = _page(
            [_frame("https://challenges.cloudflare.com/widget", _checkbox()), _frame(_RECAPTCHA_URL, _checkbox())],
            AsyncMock(),
        )
        solve = AsyncMock()
        self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0, provider_attempts={"turnstile": solve}))
        solve.assert_not_awaited()

    async def test_turnstile_timeout_returns_no_token_and_cancellation_propagates(self) -> None:
        page = _page([_frame("https://challenges.cloudflare.com/widget", _checkbox())], AsyncMock())
        solve = AsyncMock(side_effect=TimeoutError)
        self.assertIsNone(await solve_visible_captcha(page, lambda: 5.0, provider_attempts={"turnstile": solve}))
        solve.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await solve_visible_captcha(page, lambda: 5.0, provider_attempts={"turnstile": solve})


class PageHandlerVisionSeamTests(IsolatedAsyncioTestCase):
    async def test_only_provider_attempts_are_passed_to_dispatch(self) -> None:
        group, _ = _group()
        captured: dict[str, object] = {}

        async def fake_solve(*_args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        with patch("prowl.browser.page_handler.solve_visible_captcha", side_effect=fake_solve):
            await _run_get(group, tree=_document("Ready"), solve_captcha=True)

        self.assertEqual(set(captured), {"provider_attempts"})
        self.assertEqual(set(cast("dict[str, object]", captured["provider_attempts"])), {"turnstile"})

    async def test_solve_captcha_false_keeps_optional_adapters_idle(self) -> None:
        group, _ = _group()
        with patch("prowl.browser.page_handler.solve_visible_captcha", AsyncMock()) as solve:
            await _run_get(group, tree=_document("Ready"), solve_captcha=False)
        solve.assert_not_called()


class HcaptchaGridCheckboxTests(IsolatedAsyncioTestCase):
    """A wired grid attempt replaces the checkbox only when a challenge frame is present."""

    async def test_preexisting_challenge_skips_the_checkbox_and_runs_the_grid(self) -> None:
        checkbox = _checkbox()
        page = _page([_frame(_HCAPTCHA_CHALLENGE_URL, checkbox)], AsyncMock(return_value=[]))
        grid = AsyncMock(return_value="grid-token")

        self.assertEqual(
            await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid),
            ("hcaptcha", "grid-token"),
        )
        checkbox.click.assert_not_awaited()
        grid.assert_awaited_once_with(frozenset())

    async def test_preloaded_invisible_challenge_does_not_skip_the_checkbox(self) -> None:
        checkbox = _checkbox()
        view = _checkbox(visible=False)
        challenge = _frame(_HCAPTCHA_CHALLENGE_URL, view)
        frames = [challenge, _frame(_HCAPTCHA_URL, checkbox)]
        checkbox.click.side_effect = lambda **_kwargs: setattr(view, "is_visible", AsyncMock(return_value=True))
        page = _page(frames, AsyncMock(return_value=[]))
        grid = AsyncMock(return_value="grid-token")

        result = await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid)

        self.assertEqual(result, ("hcaptcha", "grid-token"))
        checkbox.click.assert_awaited_once()
        grid.assert_awaited_once_with(frozenset())

    async def test_preloaded_challenge_without_a_visible_grid_never_uploads(self) -> None:
        checkbox = _checkbox()
        frames = [_frame(_HCAPTCHA_CHALLENGE_URL, _checkbox(visible=False)), _frame(_HCAPTCHA_URL, checkbox)]
        page = _page(frames, AsyncMock(return_value=[]))
        grid = AsyncMock(return_value="grid-token")

        result = await solve_visible_checkbox(page, lambda: 0.03, challenge_attempt=grid)

        self.assertIsNone(result)
        checkbox.click.assert_awaited_once()
        grid.assert_not_awaited()

    async def test_checkbox_token_avoids_the_grid_attempt(self) -> None:
        checkbox = _checkbox()
        page = _page([_frame(_HCAPTCHA_URL, checkbox)], AsyncMock(side_effect=[[""], ["fresh-token"]]))
        grid = AsyncMock(return_value="grid-token")

        self.assertEqual(
            await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid),
            ("hcaptcha", "fresh-token"),
        )
        checkbox.click.assert_awaited_once()
        grid.assert_not_awaited()

    async def test_challenge_after_the_click_interrupts_polling_into_the_grid(self) -> None:
        checkbox = _checkbox()
        frames = [_frame(_HCAPTCHA_URL, checkbox)]
        checkbox.click.side_effect = lambda **_kwargs: frames.append(_frame(_HCAPTCHA_CHALLENGE_URL, _checkbox()))
        page = _page(frames, AsyncMock(return_value=[]))
        grid = AsyncMock(return_value="grid-token")

        self.assertEqual(
            await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid),
            ("hcaptcha", "grid-token"),
        )
        checkbox.click.assert_awaited_once()
        grid.assert_awaited_once_with(frozenset())

    async def test_grid_without_a_token_reports_no_success(self) -> None:
        page = _page([_frame(_HCAPTCHA_CHALLENGE_URL, _checkbox())], AsyncMock(return_value=[]))
        grid = AsyncMock(return_value=None)

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid))
        grid.assert_awaited_once()

    async def test_grid_reused_initial_token_reports_no_success(self) -> None:
        page = _page([_frame(_HCAPTCHA_CHALLENGE_URL, _checkbox())], AsyncMock(return_value=["stale"]))
        grid = AsyncMock(return_value="stale")

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid))

    async def test_ambiguous_preexisting_challenges_do_not_click(self) -> None:
        first, second = _checkbox(), _checkbox()
        page = _page(
            [_frame(_HCAPTCHA_CHALLENGE_URL, first), _frame(_HCAPTCHA_CHALLENGE_URL, second)],
            AsyncMock(return_value=[]),
        )
        grid = AsyncMock(return_value="grid-token")

        self.assertIsNone(await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid))
        first.click.assert_not_awaited()
        second.click.assert_not_awaited()
        grid.assert_not_awaited()

    async def test_grid_cancellation_propagates(self) -> None:
        page = _page([_frame(_HCAPTCHA_CHALLENGE_URL, _checkbox())], AsyncMock(return_value=[]))
        grid = AsyncMock(side_effect=asyncio.CancelledError)

        with self.assertRaises(asyncio.CancelledError):
            await solve_visible_checkbox(page, lambda: 5.0, challenge_attempt=grid)


class PageHandlerGridSeamTests(IsolatedAsyncioTestCase):
    """The opt-in grid seam is wired on demand, records outcomes and never alters the pair."""

    async def test_reset_clears_outcome_but_keeps_the_grid_callback(self) -> None:
        group, _ = _group()
        site = _site(group)
        grid = AsyncMock()
        site.vision_captcha_solvers["hcaptcha"] = grid
        site.vision_outcome = VisionOutcome(status="handled", reason="page_accepted")

        site._reset()

        self.assertIsNone(site.vision_outcome)
        self.assertIs(site.vision_captcha_solvers["hcaptcha"], grid)

    async def test_unset_grid_solver_binds_no_callback(self) -> None:
        group, _ = _group()
        site = _site(group)

        self.assertIsNone(site._vision_captcha_attempt(Mock(spec=Page), "hcaptcha"))

    async def test_bound_grid_wrapper_records_outcome_and_gates_the_token(self) -> None:
        group, _ = _group()
        site = _site(group)
        rejected = SimpleNamespace(
            outcome=VisionOutcome(status="needs_human", reason="page_not_accepted"),
            token="x",  # noqa: S106
        )
        site.vision_captcha_solvers["hcaptcha"] = AsyncMock(return_value=rejected)

        solve = site._vision_captcha_attempt(Mock(spec=Page), "hcaptcha")

        self.assertIsNotNone(solve)
        self.assertIsNone(await solve(frozenset()))
        self.assertEqual(site.vision_outcome, rejected.outcome)

    async def test_bound_grid_wrapper_returns_the_accepted_token(self) -> None:
        group, _ = _group()
        site = _site(group)
        accepted = SimpleNamespace(
            outcome=VisionOutcome(status="handled", reason="page_accepted"),
            token="grid-token",  # noqa: S106
        )
        site.vision_captcha_solvers["hcaptcha"] = AsyncMock(return_value=accepted)

        solve = site._vision_captcha_attempt(Mock(spec=Page), "hcaptcha")

        self.assertEqual(await solve(frozenset()), "grid-token")
        self.assertEqual(site.vision_outcome, accepted.outcome)

    async def test_wired_grid_callback_is_passed_to_the_dispatcher(self) -> None:
        group, _ = _group()
        site = _site(group)
        site.vision_captcha_solvers["hcaptcha"] = AsyncMock()
        captured: dict[str, object] = {}

        async def fake_solve(_page: object, _seconds_remaining: object, **kwargs: object) -> None:
            captured.update(kwargs)

        with patch("prowl.browser.page_handler.solve_visible_captcha", side_effect=fake_solve):
            await _run_get(group, tree=_document("Ready"), solve_captcha=True, site=site)

        self.assertEqual(set(cast("dict[str, object]", captured["provider_attempts"])), {"turnstile", "hcaptcha"})
