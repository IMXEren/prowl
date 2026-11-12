"""Turnstile uses solveCaptcha and the shared provider/token response."""

# Pyright does not narrow command unions through unittest assertions.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchResult
from prowl.service.protocol import FetchCommand, ProtocolError, parse_request

_GET = {"cmd": "request.get", "url": "https://example.com/"}
VERIFICATION_VALUE = "solved-widget"


def _result(*, token: str | None = None, screenshot: str | None = None) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={},
        response="body",
        cookies=[{"name": "cf_clearance", "value": "fixture"}],
        user_agent="Test",
        screenshot=screenshot,
        captcha_provider="turnstile" if token else None,
        captcha_token=token,
    )


def _backend(result: FetchResult) -> Mock:
    backend = Mock(spec=Backend)
    backend.fetch = AsyncMock(return_value=result)
    backend.close_session = AsyncMock()
    backend.aclose = AsyncMock()
    return backend


class VerificationOptionParsingTests(TestCase):
    def test_removed_field_is_rejected_even_when_zero_or_disabled(self) -> None:
        for value in (0, 1, False, None, "1"):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                parse_request({**_GET, "tabs_till_verify": value})

    def test_shared_flag_enables_browser_or_auto_get(self) -> None:
        for mode in ("browser", "auto"):
            command = parse_request({**_GET, "mode": mode, "solveCaptcha": True})
            self.assertIsInstance(command, FetchCommand)
            self.assertTrue(command.solve_captcha)


class VerificationOptionDispatchTests(IsolatedAsyncioTestCase):
    async def test_only_requested_actual_provider_pair_is_projected(self) -> None:
        backend = _backend(_result(token=VERIFICATION_VALUE))
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)
        _, body = await service.handle({**_GET, "solveCaptcha": True})
        self.assertTrue(backend.fetch.await_args.args[1].solve_captcha)
        self.assertEqual(body["solution"]["captcha_provider"], "turnstile")
        self.assertEqual(body["solution"]["captcha_token"], VERIFICATION_VALUE)
        self.assertNotIn("turnstile_token", body["solution"])
        _, body = await service.handle(dict(_GET))
        self.assertFalse(backend.fetch.await_args.args[1].solve_captcha)
        self.assertNotIn("captcha_token", body["solution"])
        backend.fetch = AsyncMock(return_value=_result())
        _, body = await service.handle({**_GET, "solveCaptcha": True})
        self.assertNotIn("captcha_token", body["solution"])
        self.assertNotIn("captcha_provider", body["solution"])

    async def test_cookie_only_capture_keeps_provider_pair_and_screenshot(self) -> None:
        result = _result(token=VERIFICATION_VALUE, screenshot="PNG")
        backend = _backend(result)
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)
        _, body = await service.handle(
            {**_GET, "returnOnlyCookies": True, "waitInSeconds": 5, "returnScreenshot": True, "solveCaptcha": True}
        )
        request = backend.fetch.await_args.args[1]
        self.assertTrue(request.solve_captcha)
        self.assertEqual(request.wait_in_seconds, 0.0)
        self.assertEqual(body["solution"]["response"], "")
        self.assertEqual(body["solution"]["cookies"], result.cookies)
        self.assertEqual(body["solution"]["captcha_token"], VERIFICATION_VALUE)
        self.assertEqual(body["solution"]["screenshot"], "PNG")

    async def test_typed_fetch_preserves_result_identity(self) -> None:
        result = _result(token=VERIFICATION_VALUE)
        backend = _backend(result)
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)
        command = parse_request({**_GET, "solveCaptcha": True})
        self.assertIsInstance(command, FetchCommand)
        self.assertIs(await service.fetch(command), result)

    async def test_invalid_or_removed_options_never_reach_backend(self) -> None:
        backend = _backend(_result())
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)
        for options in ({"tabs_till_verify": 0}, {"solveCaptcha": "true"}, {"solveCaptcha": True, "mode": "http"}):
            _, body = await service.handle({**_GET, **options})
            self.assertEqual(body["status"], "error")
        backend.fetch.assert_not_awaited()
