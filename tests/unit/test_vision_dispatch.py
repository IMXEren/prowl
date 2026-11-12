"""Provider-keyed CAPTCHA dispatch never guesses across multiple widgets."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.browser.solvers import solve_visible_captcha

if TYPE_CHECKING:
    from playwright.async_api import Page

_CLOUDFLARE_FRAME = "https://challenges.cloudflare.com/widget"
_HCAPTCHA_FRAME = "https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html"
_RECAPTCHA_FRAME = "https://www.google.com/recaptcha/api2/anchor?k=site"


class _FakePage:
    def __init__(self, *, frames: tuple[str, ...] = (), geetest: bool = False, altcha: bool = False) -> None:
        self.frames = [SimpleNamespace(url=url) for url in frames]
        self.geetest = geetest
        self.altcha = altcha

    async def query_selector_all(self, selector: str) -> list[object]:
        if selector == ".geetest_holder" and self.geetest:
            return [SimpleNamespace(is_visible=AsyncMock(return_value=True))]
        if selector == "altcha-widget" and self.altcha:
            return [object()]
        return []


def _page(*, frames: tuple[str, ...] = (), geetest: bool = False, altcha: bool = False) -> Page:
    return cast("Page", _FakePage(frames=frames, geetest=geetest, altcha=altcha))


class VisionDispatchTests(IsolatedAsyncioTestCase):
    async def test_lone_turnstile_returns_only_its_provider_response(self) -> None:
        solve = AsyncMock(return_value="fresh")
        result = await solve_visible_captcha(
            _page(frames=(_CLOUDFLARE_FRAME,)), lambda: 1.0, provider_attempts={"turnstile": solve}
        )
        self.assertEqual(result, ("turnstile", "fresh"))
        solve.assert_awaited_once_with(frozenset())

    async def test_mixed_providers_refuse_without_clicks(self) -> None:
        solve = AsyncMock(return_value="fresh")
        result = await solve_visible_captcha(
            _page(frames=(_CLOUDFLARE_FRAME,), altcha=True),
            lambda: 1.0,
            provider_attempts={"turnstile": solve},
        )
        self.assertIsNone(result)
        solve.assert_not_awaited()

    async def test_lone_hcaptcha_is_offered_only_its_adapter(self) -> None:
        grid = AsyncMock(return_value="fresh")
        with patch("prowl.browser.solvers.dispatch.solve_visible_checkbox", new=AsyncMock()) as checkbox:
            await solve_visible_captcha(
                _page(frames=(_HCAPTCHA_FRAME,)), lambda: 1.0, provider_attempts={"hcaptcha": grid}
            )
        assert checkbox.await_args is not None
        self.assertIs(checkbox.await_args.kwargs["challenge_attempt"], grid)

    async def test_recaptcha_never_receives_a_hcaptcha_adapter(self) -> None:
        grid = AsyncMock()
        with patch("prowl.browser.solvers.dispatch.solve_visible_checkbox", new=AsyncMock()) as checkbox:
            await solve_visible_captcha(
                _page(frames=(_RECAPTCHA_FRAME,)), lambda: 1.0, provider_attempts={"hcaptcha": grid}
            )
        assert checkbox.await_args is not None
        self.assertIsNone(checkbox.await_args.kwargs["challenge_attempt"])

    async def test_lone_geetest_uses_native_attempt(self) -> None:
        solve = AsyncMock(return_value="fresh")
        result = await solve_visible_captcha(_page(geetest=True), lambda: 1.0, provider_attempts={"geetest": solve})
        self.assertEqual(result, ("geetest", "fresh"))
        solve.assert_awaited_once_with(frozenset())

    async def test_unconfigured_geetest_returns_no_response(self) -> None:
        self.assertIsNone(await solve_visible_captcha(_page(geetest=True), lambda: 1.0))

    async def test_cancellation_propagates_from_provider(self) -> None:
        with self.assertRaises(asyncio.CancelledError):
            await solve_visible_captcha(
                _page(frames=(_CLOUDFLARE_FRAME,)),
                lambda: 1.0,
                provider_attempts={"turnstile": AsyncMock(side_effect=asyncio.CancelledError)},
            )

    async def test_no_provider_returns_none_on_expired_detection_window(self) -> None:
        self.assertIsNone(await solve_visible_captcha(_page(), Mock(side_effect=[1.0, 0.0, 0.0])))
