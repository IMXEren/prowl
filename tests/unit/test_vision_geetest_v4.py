"""GeeTest v4 response, asset, and geometry guards without provider traffic."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

from prowl.browser.solvers.vision.geetest_v4 import _asset_url, _attempt, _layout, _validation

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Page
    from pydoll.browser.tab import Tab

    from prowl.browser.solvers.vision.schemas import CaptchaAttemptResult

_BG = "https://static.geetest.com/pictures/v4_pic/test/bg/example.png"
_PIECE = "https://static.geetest.com/pictures/v4_pic/test/slide/example.png"


class GeeTestV4GuardsTests(unittest.TestCase):
    def test_paired_provider_assets_and_layout(self) -> None:
        value = {
            "bg_url": f'url("{_BG}")',
            "piece_url": f'url("{_PIECE}")',
            "bg_rect": [801, 357, 302, 201],
            "piece_rect": [801, 400, 80, 80],
            "handle_rect": [801, 567, 81, 51],
            "rail_rect": [801, 560, 302, 60],
        }
        layout = _layout(value)
        if layout is None:
            self.fail("paired v4 assets must be accepted")
        self.assertEqual(layout.bg_url, _BG)
        self.assertIsNone(_layout({**value, "piece_url": f'url("{_PIECE.replace("example", "other")}")'}))
        self.assertIsNone(_layout({**value, "bg_rect": [801, 357, float("nan"), 201]}))

    def test_asset_scope_is_https_exact_provider_image_only(self) -> None:
        self.assertEqual(_asset_url(f'url("{_BG}")', "bg"), _BG)
        self.assertIsNone(_asset_url(f'url("{_BG.replace("static.geetest.com", "static.geetest.com.evil")}")', "bg"))
        self.assertIsNone(_asset_url(f'url("{_BG.replace("https", "http")}")', "bg"))
        self.assertIsNone(_asset_url(f'url("{_BG}")', "slide"))
        self.assertIsNone(_asset_url("url(data:image/png;base64,AAAA)", "bg"))

    def test_only_success_envelope_with_real_pass_token_is_accepted(self) -> None:
        envelope = {
            "status": "success",
            "data": {"result": "success", "lot_number": "lot", "seccode": {"pass_token": "real-token"}},
        }
        body = json.dumps(envelope)
        self.assertEqual(_validation(f"geetest_cb({body});".encode()), "real-token")
        self.assertEqual(_validation(body.encode()), "real-token")
        self.assertIsNone(_validation(json.dumps({**envelope, "status": "fail"}).encode()))
        self.assertIsNone(
            _validation(json.dumps({**envelope, "data": {**envelope["data"], "result": "fail"}}).encode())
        )
        self.assertIsNone(
            _validation(json.dumps({**envelope, "data": {**envelope["data"], "lot_number": ""}}).encode())
        )
        self.assertIsNone(_validation(b"callback({}); alert(1)"))
        self.assertIsNone(_validation(body.encode() + b" " * 65536))


class GeeTestV4ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_same_frame_successful_verify_can_accept(self) -> None:
        frame = object()
        page = SimpleNamespace(url="https://2captcha.com/demo/geetest-v4", main_frame=frame)
        listeners: dict[str, Callable[[object], None]] = {}
        page.on = listeners.__setitem__
        page.remove_listener = listeners.pop
        body = json.dumps(
            {
                "status": "success",
                "data": {"result": "success", "lot_number": "lot", "seccode": {"pass_token": "real-token"}},
            }
        ).encode()

        async def drive(
            verify_body: bytes,
            *,
            same_frame: bool = True,
            path: str = "/verify",
            host: str = "gcaptcha4.geetest.com",
        ) -> CaptchaAttemptResult:
            response = SimpleNamespace(
                url=f"https://{host}{path}",
                status=200,
                request=SimpleNamespace(frame=frame if same_frame else object()),
                body=AsyncMock(return_value=verify_body),
            )

            async def prepare(*_args: object) -> None:
                listeners["response"](response)

            with patch("prowl.browser.solvers.vision.geetest_v4._prepare_drag", side_effect=prepare):
                return await _attempt(cast("Page", page), "https://2captcha.com", cast("Tab", object()), lambda: 0.3)

        accepted = await drive(body)
        self.assertEqual(accepted.outcome.status, "handled")
        self.assertEqual(accepted.token, "real-token")
        for rejected in (
            await drive(body, same_frame=False),
            await drive(body, path="/load"),
            await drive(body, host="gcaptcha4.geetest.com.evil"),
            await drive(body.replace(b'"result": "success"', b'"result": "fail"')),
        ):
            self.assertEqual(rejected.outcome.status, "needs_human")
            self.assertIsNone(rejected.token)
        self.assertEqual(listeners, {})


if __name__ == "__main__":
    unittest.main()
