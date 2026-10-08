"""Service wire for the opt-in explicit-verification option.

An upstream client asks a GET to run the native keyboard verification by sending
``tabs_till_verify``. The option is browser/auto GET only and its resulting token is
projected as ``solution.turnstile_token`` only when it was requested. These tests use the real
``FetchCommand``/``FetchResult`` shapes against a complete ``Backend`` double; no browser is
launched and nothing external is contacted.
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchResult
from prowl.service.protocol import FetchCommand, ProtocolError, parse_request

_GET = {"cmd": "request.get", "url": "https://example.com/"}
_TYPE_ERROR = "tabs_till_verify must be a nonnegative integer"
_MODE_ERROR = "explicit verification requires browser or auto GET"
#: An innocuous fixture value for the verified result, not a credential.
VERIFICATION_VALUE = "solved-widget"


def _config(*, max_concurrency: int = 4) -> ServiceConfig:
    return ServiceConfig(proxy_url=None, egresses={}, max_concurrency=max_concurrency)


def _result(
    *,
    turnstile_token: str | None = None,
    screenshot: str | None = None,
    response: str = "hello",
    cookies: list[dict[str, object]] | None = None,
) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={"content-type": "text/html"},
        response=response,
        cookies=[] if cookies is None else cookies,
        user_agent="Mozilla/5.0 (Test)",
        screenshot=screenshot,
        turnstile_token=turnstile_token,
    )


def _backend(result: FetchResult) -> Mock:
    """A complete protocol backend double whose ``fetch`` yields *result*."""
    backend = Mock(spec=Backend)
    backend.fetch = AsyncMock(return_value=result)
    backend.close_session = AsyncMock()
    backend.aclose = AsyncMock()
    return backend


def _command(**fields: object) -> FetchCommand:
    command = parse_request({**_GET, **fields})
    assert isinstance(command, FetchCommand)
    return command


class VerificationOptionParsingTests(TestCase):
    """Strict validation of the ``tabs_till_verify`` fetch field."""

    def test_omitted_field_disables_and_presence_enables_on_a_browser_or_auto_get(self) -> None:
        self.assertIsNone(_command().tabs_till_verify)
        self.assertEqual(_command(tabs_till_verify=0).tabs_till_verify, 0)
        self.assertEqual(_command(tabs_till_verify=2).tabs_till_verify, 2)
        self.assertEqual(_command(tabs_till_verify=1, mode="auto").tabs_till_verify, 1)

    def test_invalid_values_are_rejected_without_echoing_them(self) -> None:
        for value in (True, None, -1, 1.0, "1", []):
            with self.assertRaises(ProtocolError) as caught:
                _command(tabs_till_verify=value)
            self.assertEqual(caught.exception.message, _TYPE_ERROR, value)
            self.assertEqual(caught.exception.http_status, 400)
            self.assertNotIn(str(value), caught.exception.message, value)

    def test_http_post_or_another_command_rejects_the_field(self) -> None:
        with self.assertRaises(ProtocolError) as caught:
            _command(tabs_till_verify=1, mode="http")
        self.assertEqual(caught.exception.message, _MODE_ERROR)
        self.assertEqual(caught.exception.http_status, 400)

        with self.assertRaises(ProtocolError) as caught:
            parse_request(
                {"cmd": "request.post", "url": "https://example.com/", "postData": "a=1", "tabs_till_verify": 1}
            )
        self.assertEqual(caught.exception.message, _MODE_ERROR)

        for payload in (
            {"cmd": "browser.open", "url": "https://example.com/", "tabs_till_verify": 1},
            {"cmd": "sessions.create", "tabs_till_verify": 0},
            {"cmd": "cookies.list", "tabs_till_verify": 1},
        ):
            with self.assertRaises(ProtocolError, msg=payload):
                parse_request(payload)


class VerificationOptionDispatchTests(IsolatedAsyncioTestCase):
    """The typed fetch forwards the option and the reply projects the requested token."""

    async def test_handle_forwards_the_option_and_projects_only_an_actual_requested_token(self) -> None:
        backend = _backend(_result(turnstile_token=VERIFICATION_VALUE))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({**_GET, "tabs_till_verify": 2})

        backend.fetch.assert_awaited_once()
        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.tabs_till_verify, 2)
        self.assertEqual(body["solution"]["turnstile_token"], VERIFICATION_VALUE)

        # An unrequested result token (a fake or legacy backend) must never surface.
        backend.fetch = AsyncMock(return_value=_result(turnstile_token=VERIFICATION_VALUE))
        _status, body = await service.handle(dict(_GET))
        self.assertIsNone(backend.fetch.await_args.args[1].tabs_till_verify)
        self.assertNotIn("turnstile_token", body["solution"])

        # A requested verification whose result carried no token reports nothing.
        backend.fetch = AsyncMock(return_value=_result())
        _status, body = await service.handle({**_GET, "tabs_till_verify": 0})
        self.assertEqual(backend.fetch.await_args.args[1].tabs_till_verify, 0)
        self.assertNotIn("turnstile_token", body["solution"])

    async def test_return_only_cookies_keeps_token_and_cookies_with_body_cleared(self) -> None:
        cookie: dict[str, object] = {"name": "cf_clearance", "value": "tok"}
        backend = _backend(
            _result(turnstile_token=VERIFICATION_VALUE, response="body", screenshot="PNG", cookies=[cookie])
        )
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle(
            {**_GET, "returnOnlyCookies": True, "waitInSeconds": 5, "returnScreenshot": True, "tabs_till_verify": 0}
        )

        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.wait_in_seconds, 0.0)
        self.assertEqual(request.tabs_till_verify, 0)
        self.assertEqual(body["solution"]["response"], "")
        self.assertEqual(body["solution"]["cookies"], [cookie])
        self.assertEqual(body["solution"]["turnstile_token"], VERIFICATION_VALUE)
        self.assertEqual(body["solution"]["screenshot"], "PNG")

    async def test_typed_fetch_preserves_the_result_identity(self) -> None:
        result = _result(turnstile_token=VERIFICATION_VALUE)
        backend = _backend(result)
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        returned = await service.fetch(_command(tabs_till_verify=3))

        self.assertIs(returned, result)
        self.assertEqual(backend.fetch.await_args.args[1].tabs_till_verify, 3)
        self.assertEqual(returned.turnstile_token, VERIFICATION_VALUE)

    async def test_rejected_options_never_reach_the_backend(self) -> None:
        backend = _backend(_result())
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        for payload in (
            {**_GET, "tabs_till_verify": -1},
            {**_GET, "tabs_till_verify": "1"},
            {**_GET, "tabs_till_verify": 1, "mode": "http"},
        ):
            _status, body = await service.handle(payload)
            self.assertEqual(body["status"], "error", payload)
        backend.fetch.assert_not_awaited()
