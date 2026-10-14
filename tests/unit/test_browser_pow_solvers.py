"""Tests for the local proof-of-work solvers (ALTCHA and Friendly Captcha).

The page and locators are spec-constrained doubles so mount uniqueness, the
API-first trigger, the single visible-control fallback, fresh-token selection,
and deadline and cancellation behavior are verified without a live browser.
"""

from __future__ import annotations

import asyncio
import time
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from playwright.async_api import Frame, Locator, Page

from prowl.browser.solvers.pow import solve_altcha, solve_friendly_captcha

_FRIENDLY_TOKEN = "frc-token-0123456789abcdefghijklmnopq"  # noqa: S105


def _locator(*, count: int = 1, visible: bool = True) -> Mock:
    locator = Mock(spec=Locator)
    locator.count = AsyncMock(return_value=count)
    locator.is_visible = AsyncMock(return_value=visible)
    locator.click = AsyncMock()
    return locator


def _page(mounts: int, responses: list[object], *, locator: Mock | None = None) -> Mock:
    page = Mock(spec=Page)
    page.query_selector_all = AsyncMock(return_value=[Mock() for _ in range(mounts)])
    queue = list(responses)

    def evaluate(*_args: object) -> object:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    page.evaluate = AsyncMock(side_effect=evaluate)
    page.locator = Mock(return_value=locator if locator is not None else _locator())
    page.frames = []
    return page


async def _hang(*_args: object) -> object:
    await asyncio.sleep(10)
    return {}


class AbsentOrAmbiguousMountTests(IsolatedAsyncioTestCase):
    """No value read or trigger happens without exactly one provider mount."""

    async def test_absent_mount_returns_none(self) -> None:
        page = _page(0, [{"state": "verified", "values": ["token"]}])
        self.assertIsNone(await solve_altcha(page, lambda: 5.0))
        page.evaluate.assert_not_awaited()
        page.locator.assert_not_called()

    async def test_duplicate_mount_returns_none(self) -> None:
        page = _page(2, [{"state": "verified", "values": ["token"]}])
        self.assertIsNone(await solve_friendly_captcha(page, lambda: 5.0))
        page.evaluate.assert_not_awaited()

    async def test_expired_deadline_does_nothing(self) -> None:
        page = _page(1, [{"state": "verified", "values": ["token"]}])
        self.assertIsNone(await solve_altcha(page, lambda: 0.0))
        page.query_selector_all.assert_not_awaited()


class TriggerPathTests(IsolatedAsyncioTestCase):
    """The provider API is preferred and the visible fallback is unique-only."""

    async def test_altcha_api_path_returns_fresh_token(self) -> None:
        page = _page(
            1,
            [
                {"state": "unverified", "values": []},
                True,
                {"state": "verified", "values": ["altcha-payload"]},
            ],
        )
        self.assertEqual(await solve_altcha(page, lambda: 5.0), "altcha-payload")
        page.locator.assert_not_called()

    async def test_altcha_checkbox_fallback_clicks_unique_visible_target(self) -> None:
        locator = _locator()
        page = _page(
            1,
            [
                {"state": "unverified", "values": []},
                False,
                {"state": "verified", "values": ["altcha-payload"]},
            ],
            locator=locator,
        )
        self.assertEqual(await solve_altcha(page, lambda: 5.0), "altcha-payload")
        locator.click.assert_awaited_once()

    async def test_altcha_fallback_skips_ambiguous_target(self) -> None:
        locator = _locator(count=2)
        page = _page(1, [{"state": "unverified", "values": []}, False], locator=locator)
        self.assertIsNone(await solve_altcha(page, lambda: 0.05))
        locator.click.assert_not_awaited()

    async def test_friendly_v2_clicks_only_its_unique_visible_frame_button(self) -> None:
        in_page = _locator(count=0)
        button = _locator()
        frame = Mock(spec=Frame)
        frame.url = "https://global.frcapi.com/api/v2/captcha/widget?ignored=1"
        element = _locator()
        frame.frame_element = AsyncMock(return_value=element)
        frame.locator = Mock(return_value=button)
        page = _page(
            1,
            [{"state": None, "values": []}, False, {"state": None, "values": [_FRIENDLY_TOKEN]}],
            locator=in_page,
        )
        page.frames = [frame]
        self.assertEqual(await solve_friendly_captcha(page, lambda: 5.0), _FRIENDLY_TOKEN)
        button.click.assert_awaited_once()
        in_page.click.assert_not_awaited()
        frame.locator.assert_called_once_with("button.button[aria-label]")

    async def test_friendly_v2_refuses_lookalike_frame(self) -> None:
        in_page = _locator(count=0)
        frame = Mock(spec=Frame)
        frame.url = "https://global.frcapi.com.evil.example/api/v2/captcha/widget"
        frame.frame_element = AsyncMock()
        page = _page(1, [{"state": None, "values": []}, False], locator=in_page)
        page.frames = [frame]
        self.assertIsNone(await solve_friendly_captcha(page, lambda: 0.05))
        frame.frame_element.assert_not_awaited()
        in_page.click.assert_not_awaited()

    async def test_friendly_api_path_returns_long_token(self) -> None:
        page = _page(
            1,
            [
                {"state": "init", "values": []},
                True,
                {"state": "completed", "values": [_FRIENDLY_TOKEN]},
            ],
        )
        self.assertEqual(await solve_friendly_captcha(page, lambda: 5.0), _FRIENDLY_TOKEN)


class FreshTokenTests(IsolatedAsyncioTestCase):
    """A preexisting value or a placeholder is never returned as a token."""

    async def test_prefilled_stale_token_is_never_returned(self) -> None:
        page = _page(
            1,
            [
                {"state": "unverified", "values": ["stale-token"]},
                True,
                {"state": "verified", "values": ["stale-token"]},
            ],
        )
        self.assertIsNone(await solve_altcha(page, lambda: 0.05))

    async def test_friendly_placeholder_is_rejected(self) -> None:
        page = _page(
            1,
            [
                {"state": "init", "values": []},
                True,
                {"state": "completed", "values": [".PENDING", "tooshort"]},
            ],
        )
        self.assertIsNone(await solve_friendly_captcha(page, lambda: 0.05))


class DeadlineAndCancellationTests(IsolatedAsyncioTestCase):
    """Expired deadlines return None; cancellation propagates unchanged."""

    async def test_cancellation_propagates(self) -> None:
        page = _page(1, [{}])
        page.evaluate = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await solve_altcha(page, lambda: 5.0)

    async def test_hanging_read_ends_under_deadline(self) -> None:
        page = _page(1, [{}])
        page.evaluate = AsyncMock(side_effect=_hang)
        start = time.monotonic()
        result = await solve_altcha(page, lambda: 0.05)
        self.assertIsNone(result)
        self.assertLess(time.monotonic() - start, 0.5)
