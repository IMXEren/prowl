"""Tests for the reCAPTCHA/hCaptcha checkbox solvers.

The page, frames, and locators are spec-constrained doubles so widget
detection, the single-click contract, fresh-token selection, and deadline and
cancellation behavior are verified without a live browser.
"""

from __future__ import annotations

import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from playwright.async_api import ElementHandle, Frame, Locator, Page

from prowl.browser.solvers import solve_hcaptcha_checkbox, solve_recaptcha_checkbox

_RECAPTCHA_URL = "https://www.google.com/recaptcha/api2/anchor?k=site"
_HCAPTCHA_URL = "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html"


def _checkbox(*, count: int = 1, visible: bool = True) -> Mock:
    locator = Mock(spec=Locator)
    locator.count = AsyncMock(return_value=count)
    locator.is_visible = AsyncMock(return_value=visible)
    locator.click = AsyncMock()
    return locator


def _frame(url: str, checkbox: Mock, *, visible: bool = True) -> Mock:
    frame = Mock(spec=Frame)
    frame.url = url
    element = Mock(spec=ElementHandle)
    element.is_visible = AsyncMock(return_value=visible)
    frame.frame_element = AsyncMock(return_value=element)
    frame.locator = Mock(return_value=checkbox)
    return frame


def _page(frames: list[Mock], evaluate: AsyncMock) -> Mock:
    page = Mock(spec=Page)
    page.frames = frames
    page.evaluate = evaluate
    return page


async def _hang(*_args: str) -> list[str]:
    await asyncio.sleep(10)
    return []


class AbsentWidgetTests(IsolatedAsyncioTestCase):
    """No click and no value read happen without one matching provider frame."""

    async def test_no_provider_frame_returns_none(self) -> None:
        evaluate = AsyncMock(return_value=[])
        page = _page([], evaluate)
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 5.0))
        evaluate.assert_not_awaited()

    async def test_substring_host_is_not_a_provider_frame(self) -> None:
        checkbox = _checkbox()
        frame = _frame("https://www.google.com.evil.example/recaptcha/api2/anchor", checkbox)
        page = _page([frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 5.0))
        frame.frame_element.assert_not_awaited()

    async def test_wrong_provider_path_is_rejected(self) -> None:
        checkbox = _checkbox()
        frame = _frame("https://www.google.com/recaptcha/api2/reload", checkbox)
        page = _page([frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 5.0))
        frame.frame_element.assert_not_awaited()


class AmbiguousOrHiddenWidgetTests(IsolatedAsyncioTestCase):
    """A hidden, duplicated, or multi-checkbox widget is never clicked."""

    async def test_two_matching_frames_are_ambiguous(self) -> None:
        frame = _frame(_RECAPTCHA_URL, _checkbox())
        page = _page([frame, frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 5.0))
        frame.locator.return_value.click.assert_not_awaited()
        page.evaluate.assert_not_awaited()

    async def test_hidden_iframe_is_skipped(self) -> None:
        frame = _frame(_RECAPTCHA_URL, _checkbox(), visible=False)
        page = _page([frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 0.05))

    async def test_hidden_checkbox_is_skipped(self) -> None:
        checkbox = _checkbox(visible=False)
        frame = _frame(_RECAPTCHA_URL, checkbox)
        page = _page([frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 0.05))
        checkbox.click.assert_not_awaited()

    async def test_multiple_checkboxes_are_ambiguous(self) -> None:
        checkbox = _checkbox(count=2)
        frame = _frame(_RECAPTCHA_URL, checkbox)
        page = _page([frame], AsyncMock(return_value=[]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 5.0))
        checkbox.click.assert_not_awaited()


class FreshTokenTests(IsolatedAsyncioTestCase):
    """Only a nonempty token differing from every initial value is returned."""

    async def test_duplicate_inputs_require_fresh_second_value(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        evaluate = AsyncMock(side_effect=[["stale-a", "stale-b"], ["stale-a", "stale-b", "fresh-token"]])
        page = _page([frame], evaluate)

        token = await solve_recaptcha_checkbox(page, lambda: 5.0)

        self.assertEqual(token, "fresh-token")
        checkbox.click.assert_awaited_once()

    async def test_unchanged_token_is_never_returned(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        evaluate = AsyncMock(return_value=["token"])
        page = _page([frame], evaluate)

        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 0.05))
        checkbox.click.assert_awaited_once()

    async def test_click_without_token_returns_none_after_one_click(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_HCAPTCHA_URL, checkbox)
        evaluate = AsyncMock(return_value=[""])
        page = _page([frame], evaluate)

        self.assertIsNone(await solve_hcaptcha_checkbox(page, lambda: 0.05))
        checkbox.click.assert_awaited_once()

    async def test_hcaptcha_waits_for_late_checkbox_in_existing_frame(self) -> None:
        checkbox = _checkbox()
        checkbox.count.side_effect = [0, 1]
        page = _page([_frame(_HCAPTCHA_URL, checkbox)], AsyncMock(side_effect=[[""], ["h-token"]]))

        self.assertEqual(await solve_hcaptcha_checkbox(page, lambda: 1.0), "h-token")
        checkbox.click.assert_awaited_once()

    async def test_hcaptcha_returns_fresh_token(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_HCAPTCHA_URL, checkbox)
        evaluate = AsyncMock(side_effect=[[""], ["h-token"]])
        page = _page([frame], evaluate)

        self.assertEqual(await solve_hcaptcha_checkbox(page, lambda: 5.0), "h-token")
        checkbox.click.assert_awaited_once()


class DeadlineAndCancellationTests(IsolatedAsyncioTestCase):
    """Expired deadlines return None; cancellation propagates unchanged."""

    async def test_expired_deadline_skips_the_click(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        page = _page([frame], AsyncMock(return_value=[""]))
        self.assertIsNone(await solve_recaptcha_checkbox(page, lambda: 0.0))
        checkbox.click.assert_not_awaited()

    async def test_cancellation_propagates(self) -> None:
        checkbox = _checkbox()
        checkbox.click = AsyncMock(side_effect=asyncio.CancelledError())
        frame = _frame(_RECAPTCHA_URL, checkbox)
        page = _page([frame], AsyncMock(return_value=[""]))

        with self.assertRaises(asyncio.CancelledError):
            await solve_recaptcha_checkbox(page, lambda: 5.0)

    async def test_hanging_initial_read_ends_under_deadline_without_click(self) -> None:
        checkbox = _checkbox()
        frame = _frame(_RECAPTCHA_URL, checkbox)
        page = _page([frame], AsyncMock(side_effect=_hang))

        start = time.monotonic()
        result = await solve_recaptcha_checkbox(page, lambda: 0.05)
        elapsed = time.monotonic() - start

        self.assertIsNone(result)
        self.assertLess(elapsed, 0.1)
        checkbox.click.assert_not_awaited()
