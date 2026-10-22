"""Fail-closed reCAPTCHA image adapter and provider selection tests."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Error as PlaywrightError

from prowl.browser.solvers.vision import recaptcha
from prowl.browser.solvers.vision.origins import _canonical_origin
from prowl.browser.solvers.vision.recaptcha import (
    _Context,
    _fresh_response,
    _Grid,
    _is_challenge_frame,
    _solve_page,
    _wait_after_verify,
    _wait_for_tiles_ready,
    solve_recaptcha_challenge,
)
from prowl.browser.solvers.vision.schemas import CaptchaAttemptResult, VisionOutcome
from prowl.vision.vlm import Point, PointSelection

if TYPE_CHECKING:
    from playwright.async_api import Frame, Locator, Page

    from prowl.vision.vlm import CloudVlmClient


def _fixture(rows: int = 4, *, image: bytes = b"image", origin: str = "https://example.test") -> tuple[Any, _Grid, Any]:
    frame_mock = Mock(url="https://www.google.com/recaptcha/enterprise/bframe")
    frame_mock.locator.return_value.count = AsyncMock(return_value=0)
    frame_mock.get_by_text.return_value.count = AsyncMock(return_value=0)
    frame = cast("Frame", frame_mock)
    page = cast("Page", SimpleNamespace(url=origin, frames=[frame]))
    tiles = tuple(cast("Locator", object()) for _ in range(rows * rows))
    grid = _Grid(
        frame,
        frame.url,
        "Select all squares with traffic lights",
        rows,
        tiles,
        cast("Locator", object()),
        cast("Locator", object()),
        image,
    )
    client = SimpleNamespace(select_points=AsyncMock())
    context = _Context(
        page, str(_canonical_origin(origin)), cast("CloudVlmClient", client), frozenset(), time.monotonic() + 5, None
    )
    return context, grid, client


class FrameTests(IsolatedAsyncioTestCase):
    def test_only_exact_google_challenge_frames_are_accepted(self) -> None:
        self.assertTrue(_is_challenge_frame("https://www.google.com/recaptcha/enterprise/bframe?x=1"))
        self.assertTrue(_is_challenge_frame("https://www.recaptcha.net/recaptcha/api2/bframe"))
        for url in (
            "http://www.google.com/recaptcha/enterprise/bframe",
            "https://www.google.com.evil.test/recaptcha/enterprise/bframe",
            "https://www.google.com/recaptcha/enterprise/anchor",
            "https://www.google.com:8443/recaptcha/enterprise/bframe",
        ):
            with self.subTest(url=url):
                self.assertFalse(_is_challenge_frame(url))

    async def test_four_by_four_points_snap_to_existing_tile_centers(self) -> None:
        context, grid, client = _fixture()
        client.select_points.return_value = PointSelection(points=(Point(0.625, 0.625),))
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock(return_value=True)) as click,
            patch("prowl.browser.solvers.vision.recaptcha._tile_signature", new=AsyncMock(return_value="same")),
            patch("prowl.browser.solvers.vision.recaptcha._wait_after_verify", new=AsyncMock(return_value=accepted)),
        ):
            result = await _solve_page(context)
        self.assertIs(result, accepted)
        self.assertIs(click.await_args_list[0].args[2], grid.tiles[10])
        self.assertIs(click.await_args_list[1].args[2], grid.verify)
        self.assertEqual(client.select_points.await_count, 1)

    async def test_same_tile_twice_is_refused_before_any_click(self) -> None:
        context, grid, client = _fixture()
        client.select_points.return_value = PointSelection(points=(Point(0.61, 0.61), Point(0.63, 0.63)))
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock()) as click,
        ):
            result = await _solve_page(context)
        self.assertIsInstance(result, CaptchaAttemptResult)
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "ambiguous_selection")
        click.assert_not_awaited()

    async def test_replacement_can_clear_last_matching_tile_before_verify(self) -> None:
        context, grid, client = _fixture(rows=3)
        grid = _Grid(
            grid.frame,
            grid.frame_url,
            "Select all images with cars Click verify once there are none left",
            3,
            grid.tiles,
            grid.target,
            grid.verify,
            b"first",
        )
        second = _Grid(grid.frame, grid.frame_url, grid.prompt, 3, grid.tiles, grid.target, grid.verify, b"second")
        client.select_points.side_effect = [PointSelection(points=(Point(0.5, 0.5),)), PointSelection(points=())]
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        calls = 0

        async def tile_source(_tile: object) -> str:
            nonlocal calls
            calls += 1
            return "old" if calls == 1 else "new"

        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(side_effect=[grid, second])),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock(return_value=True)) as click,
            patch("prowl.browser.solvers.vision.recaptcha._tile_signature", new=tile_source),
            patch("prowl.browser.solvers.vision.recaptcha._wait_replacements", new=AsyncMock(return_value=True)),
            patch("prowl.browser.solvers.vision.recaptcha._wait_after_verify", new=AsyncMock(return_value=accepted)),
        ):
            result = await _solve_page(context)
        self.assertIs(result, accepted)
        self.assertEqual(client.select_points.await_count, 2)
        self.assertEqual(click.await_count, 2)

    async def test_elimination_clicks_all_matches_then_verifies_only_after_empty_reinspection(self) -> None:
        context, grid, client = _fixture(rows=3)
        grid = _Grid(
            grid.frame,
            grid.frame_url,
            "Select all images with cars Click verify once there are none left",
            3,
            grid.tiles,
            grid.target,
            grid.verify,
            b"first",
        )
        next_grid = _Grid(grid.frame, grid.frame_url, grid.prompt, 3, grid.tiles, grid.target, grid.verify, b"second")
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        click = AsyncMock(return_value=True)

        async def inspect(*_args: object) -> PointSelection:
            if client.select_points.await_count == 1:
                return PointSelection(points=(Point(0.16, 0.16), Point(0.5, 0.5)))
            self.assertEqual(click.await_count, 2)
            self.assertNotIn(grid.verify, [call.args[2] for call in click.await_args_list])
            return PointSelection(points=())

        client.select_points.side_effect = inspect
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(side_effect=[grid, next_grid])),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=click),
            patch("prowl.browser.solvers.vision.recaptcha._tile_signature", new=AsyncMock(return_value="source")),
            patch(
                "prowl.browser.solvers.vision.recaptcha._wait_replacements", new=AsyncMock(return_value=True)
            ) as wait,
            patch("prowl.browser.solvers.vision.recaptcha._wait_after_verify", new=AsyncMock(return_value=accepted)),
        ):
            result = await _solve_page(context)
        self.assertIs(result, accepted)
        self.assertEqual(wait.await_count, 1)
        self.assertEqual(set(wait.await_args_list[-1].args[2]), {0, 4})
        self.assertEqual(click.await_count, 3)
        self.assertIs(click.await_args_list[-1].args[2], grid.verify)

    async def test_elimination_never_verifies_when_replacements_do_not_finish(self) -> None:
        context, grid, client = _fixture(rows=3)
        grid = _Grid(
            grid.frame,
            grid.frame_url,
            "Select all images with cars Click verify once there are none left",
            3,
            grid.tiles,
            grid.target,
            grid.verify,
            b"first",
        )
        client.select_points.return_value = PointSelection(points=(Point(0.5, 0.5),))
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock(return_value=True)) as click,
            patch("prowl.browser.solvers.vision.recaptcha._tile_signature", new=AsyncMock(return_value="source")),
            patch("prowl.browser.solvers.vision.recaptcha._wait_replacements", new=AsyncMock(return_value=False)),
        ):
            result = await _solve_page(context)
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "replacement_unavailable")
        self.assertEqual(click.await_count, 1)
        self.assertIs(click.await_args_list[-1].args[2], grid.tiles[4])
        self.assertEqual(client.select_points.await_count, 1)

    async def test_empty_first_selection_does_not_click_verify(self) -> None:
        context, grid, client = _fixture()
        client.select_points.return_value = PointSelection(points=())
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock()) as click,
        ):
            result = await _solve_page(context)
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "empty_selection")
        click.assert_not_awaited()

    async def test_origin_change_after_vlm_return_never_clicks(self) -> None:
        context, grid, client = _fixture()

        async def redirect(*_args: object) -> PointSelection:
            context.page.url = "https://other.test/"
            return PointSelection(points=(Point(0.5, 0.5),))

        client.select_points.side_effect = redirect
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch("prowl.browser.solvers.vision.recaptcha._click", new=AsyncMock()) as click,
        ):
            result = await _solve_page(context)
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "stale_revision")
        click.assert_not_awaited()

    async def test_cancellation_is_not_relabelled_as_an_error(self) -> None:
        page = Mock(url="https://example.test/", frames=[])
        client = Mock()
        with (
            patch(
                "prowl.browser.solvers.vision.recaptcha._solve_page", new=AsyncMock(side_effect=asyncio.CancelledError)
            ),
            self.assertRaises(asyncio.CancelledError),
        ):
            await solve_recaptcha_challenge(page, lambda: 5, vlm_client=cast("Any", client), initial_tokens=frozenset())

    async def test_normal_provider_pages_do_not_spend_wrong_answer_budget(self) -> None:
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        context, grid, _client = _fixture()
        page = context.page
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch(
                "prowl.browser.solvers.vision.recaptcha._solve_page",
                new=AsyncMock(side_effect=["next", "next", "next", accepted]),
            ) as solve,
        ):
            result = await solve_recaptcha_challenge(
                page, lambda: 5, vlm_client=cast("Any", Mock()), initial_tokens=frozenset()
            )
        self.assertIs(result, accepted)
        self.assertEqual(solve.await_count, 4)

    async def test_late_visible_rejection_counts_replacement_before_vlm_upload(self) -> None:
        context, grid, _client = _fixture()
        retry = cast("Any", grid.frame.get_by_text).return_value
        retry.count = AsyncMock(return_value=1)
        retry.nth.return_value.is_visible = AsyncMock(return_value=True)
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch(
                "prowl.browser.solvers.vision.recaptcha._solve_page",
                new=AsyncMock(side_effect=["next", "next", "next", accepted]),
            ) as solve,
        ):
            result = await solve_recaptcha_challenge(
                context.page, lambda: 5, vlm_client=cast("Any", Mock()), initial_tokens=frozenset()
            )
        self.assertEqual(result.outcome.reason, "page_limit")
        self.assertEqual(solve.await_count, 3)

    async def test_three_observed_wrong_pages_stop_before_fourth_upload(self) -> None:
        page = Mock(url="https://example.test/", frames=[])
        with patch(
            "prowl.browser.solvers.vision.recaptcha._solve_page", new=AsyncMock(return_value="rejected")
        ) as solve:
            result = await solve_recaptcha_challenge(
                page, lambda: 5, vlm_client=cast("Any", Mock()), initial_tokens=frozenset()
            )
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "page_limit")
        self.assertEqual(solve.await_count, 3)

    async def test_rejection_then_two_normal_pages_spend_both_extra_slots(self) -> None:
        context, grid, _client = _fixture()
        page = context.page
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch(
                "prowl.browser.solvers.vision.recaptcha._solve_page",
                new=AsyncMock(side_effect=["next", "rejected", "next", "next", accepted]),
            ) as solve,
        ):
            result = await solve_recaptcha_challenge(
                page, lambda: 5, vlm_client=cast("Any", Mock()), initial_tokens=frozenset()
            )
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "page_limit")
        self.assertEqual(solve.await_count, 4)

    async def test_second_extra_page_can_produce_fresh_token(self) -> None:
        context, grid, _client = _fixture()
        page = context.page
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        with (
            patch("prowl.browser.solvers.vision.recaptcha._read_grid", new=AsyncMock(return_value=grid)),
            patch(
                "prowl.browser.solvers.vision.recaptcha._solve_page",
                new=AsyncMock(side_effect=["next", "rejected", "next", accepted]),
            ) as solve,
        ):
            result = await solve_recaptcha_challenge(
                page, lambda: 5, vlm_client=cast("Any", Mock()), initial_tokens=frozenset()
            )
        self.assertIs(result, accepted)
        self.assertEqual(solve.await_count, 4)


class ReadinessAndTokenWindowTests(IsolatedAsyncioTestCase):
    """The two measured live races: late tile imagery and the short post-Verify token window."""

    async def test_tile_readiness_waits_until_every_image_has_decoded(self) -> None:
        context, _grid, _client = _fixture()
        frame = Mock()
        frame.evaluate = AsyncMock(side_effect=[{"imgs": 9, "ready": 0}, {"imgs": 9, "ready": 9}])
        with patch("prowl.browser.solvers.vision.recaptcha.asyncio.sleep", new=AsyncMock()):
            self.assertTrue(await _wait_for_tiles_ready(context, cast("Frame", frame)))
        self.assertEqual(frame.evaluate.await_count, 2)

    async def test_undecoded_tiles_never_report_ready(self) -> None:
        context, _grid, _client = _fixture()
        frame = Mock()
        frame.evaluate = AsyncMock(return_value={"imgs": 9, "ready": 0})
        with (
            patch.object(recaptcha, "_READY_SECONDS", 0.05),
            patch.object(recaptcha, "_POLL_SECONDS", 0.01),
            patch("prowl.browser.solvers.vision.recaptcha.asyncio.sleep", new=AsyncMock()),
        ):
            self.assertFalse(await _wait_for_tiles_ready(context, cast("Frame", frame)))

    async def test_layout_without_tile_images_gets_a_grace_period_only(self) -> None:
        context, _grid, _client = _fixture()
        frame = Mock()
        frame.evaluate = AsyncMock(return_value={"imgs": 0, "ready": 0})
        with patch.object(recaptcha, "_READY_GRACE", 0.0):
            self.assertTrue(await _wait_for_tiles_ready(context, cast("Frame", frame)))

    async def test_token_read_survives_the_post_acceptance_navigation(self) -> None:
        context, _grid, _client = _fixture()
        with patch(
            "prowl.browser.solvers.vision.recaptcha._read_response_values",
            new=AsyncMock(side_effect=PlaywrightError("Execution context was destroyed")),
        ):
            self.assertIsNone(await _fresh_response(context))

    async def test_token_is_polled_before_any_transition_check_runs(self) -> None:
        context, grid, _client = _fixture()
        accepted = CaptchaAttemptResult(VisionOutcome(status="handled", reason="page_accepted"), "fresh")
        order: list[str] = []

        async def fresh(_context: object) -> object:
            order.append("token")
            return None if order.count("token") == 1 else accepted

        async def next_page(*_args: object) -> bool:
            order.append("transition")
            return False

        with (
            patch("prowl.browser.solvers.vision.recaptcha._fresh_response", new=fresh),
            patch("prowl.browser.solvers.vision.recaptcha._next_page_visible", new=next_page),
        ):
            result = await _wait_after_verify(context, grid, ())
        self.assertIs(result, accepted)
        self.assertEqual(order, ["token", "token"])

    async def test_lost_context_during_verify_is_not_reported_as_an_error(self) -> None:
        context, grid, _client = _fixture()

        async def next_page(*_args: object) -> bool:
            message = "Execution context was destroyed"
            raise PlaywrightError(message)

        with (
            patch.object(recaptcha, "_TRANSITION_SECONDS", 0.0),
            patch("prowl.browser.solvers.vision.recaptcha._fresh_response", new=AsyncMock(return_value=None)),
            patch("prowl.browser.solvers.vision.recaptcha._next_page_visible", new=next_page),
        ):
            result = await _wait_after_verify(context, grid, ())
        self.assertIsInstance(result, CaptchaAttemptResult)
        self.assertEqual(cast("CaptchaAttemptResult", result).outcome.reason, "stale_revision")
