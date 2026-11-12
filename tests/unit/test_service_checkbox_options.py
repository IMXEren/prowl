"""Strict opt-in checkbox CAPTCHA request and service response projection."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchResult
from prowl.service.protocol import FetchCommand, ProtocolError, parse_request, solution_payload

_GET = {"cmd": "request.get", "url": "https://example.com/"}


def _result(provider: str | None = None, response_value: str | None = None) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={},
        response="ready",
        cookies=[],
        user_agent="test",
        captcha_provider=provider,
        captcha_token=response_value,
    )


def _backend(result: FetchResult) -> Mock:
    backend = Mock(spec=Backend)
    backend.fetch = AsyncMock(return_value=result)
    backend.close_session = AsyncMock()
    backend.aclose = AsyncMock()
    return backend


class CaptchaParsingTests(TestCase):
    def test_opt_in_defaults_off_and_accepts_strict_booleans(self) -> None:
        for field, expected in (({}, False), ({"solveCaptcha": True}, True), ({"solveCaptcha": False}, False)):
            command = parse_request({**_GET, **field})
            assert isinstance(command, FetchCommand)
            self.assertIs(command.solve_captcha, expected)
        command = parse_request({**_GET, "mode": "auto", "solveCaptcha": True})
        assert isinstance(command, FetchCommand)
        self.assertTrue(command.solve_captcha)

    def test_non_boolean_and_wrong_mode_fail_without_backend_work(self) -> None:
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(ProtocolError) as error:
                parse_request({**_GET, "solveCaptcha": value})
            self.assertEqual(error.exception.message, "solveCaptcha must be a boolean")
        for payload in (
            {**_GET, "mode": "http", "solveCaptcha": True},
            {"cmd": "request.post", "url": _GET["url"], "solveCaptcha": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError) as error:
                parse_request(payload)
            self.assertEqual(error.exception.message, "captcha solving requires browser or auto GET")
        for cmd in ("browser.open", "sessions.create", "cookies.list"):
            with self.subTest(cmd=cmd), self.assertRaises(ProtocolError):
                parse_request({"cmd": cmd, "solveCaptcha": True})

    def test_incomplete_provider_token_pair_is_never_projected(self) -> None:
        fields = {
            "url": _GET["url"],
            "status_code": 200,
            "headers": {},
            "response": "ok",
            "cookies": [],
            "user_agent": "test",
        }
        response_value = "value"
        self.assertNotIn("captcha_token", solution_payload(**fields, captcha_token=response_value))
        self.assertNotIn("captcha_provider", solution_payload(**fields, captcha_provider="hcaptcha"))


class CaptchaDispatchTests(IsolatedAsyncioTestCase):
    async def test_requested_fresh_result_projects_pair_and_default_leaks_neither(self) -> None:
        backend = _backend(_result("hcaptcha", "fresh-value"))
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)

        status, body = await service.handle({**_GET, "solveCaptcha": True})
        self.assertEqual(status, 200)
        self.assertTrue(backend.fetch.await_args.args[1].solve_captcha)
        self.assertEqual(body["solution"]["captcha_provider"], "hcaptcha")
        self.assertEqual(body["solution"]["captcha_token"], "fresh-value")

        backend.fetch = AsyncMock(return_value=_result("hcaptcha", "old-value"))
        status, body = await service.handle(dict(_GET))
        self.assertEqual(status, 200)
        self.assertFalse(backend.fetch.await_args.args[1].solve_captcha)
        self.assertNotIn("captcha_provider", body["solution"])
        self.assertNotIn("captcha_token", body["solution"])

    async def test_requested_unsolved_result_has_no_success_fields(self) -> None:
        backend = _backend(_result())
        service = Service(ServiceConfig(proxy_url=None, egresses={}), backend)
        self.addAsyncCleanup(service.aclose)

        status, body = await service.handle({**_GET, "solveCaptcha": True, "mode": "auto"})
        self.assertEqual(status, 200)
        self.assertTrue(backend.fetch.await_args.args[1].solve_captcha)
        self.assertNotIn("captcha_token", body["solution"])
        self.assertNotIn("captcha_provider", body["solution"])
