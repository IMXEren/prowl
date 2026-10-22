"""Browser response options preserve strict validation and legacy payload defaults."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchRequest, FetchResult
from prowl.service.protocol import FetchCommand, ProtocolError, parse_request

_GET = {"cmd": "request.get", "url": "https://example.com/"}
_WAIT_ERROR = "waitInSeconds must be a finite number of seconds within maxTimeout"
_SCREENSHOT_ERROR = "returnScreenshot must be a boolean"


def _config(*, max_concurrency: int = 4) -> ServiceConfig:
    return ServiceConfig(proxy_url=None, egresses={}, max_concurrency=max_concurrency)


def _result(
    *,
    screenshot: str | None = None,
    response: str = "hello",
    body_bytes: bytes | None = None,
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
        body_bytes=body_bytes,
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


class SnapshotOptionParsingTests(TestCase):
    """Strict validation of ``waitInSeconds`` and ``returnScreenshot`` on a fetch."""

    def test_fractional_wait_and_boolean_screenshot_are_accepted(self) -> None:
        command = _command(maxTimeout=60000, waitInSeconds=0.25, returnScreenshot=True)
        self.assertEqual(command.wait_in_seconds, 0.25)
        self.assertTrue(command.return_screenshot)

    def test_post_accepts_the_same_options(self) -> None:
        command = parse_request(
            {
                "cmd": "request.post",
                "url": "https://example.com/",
                "postData": "a=1",
                "waitInSeconds": 1.5,
                "returnScreenshot": True,
            }
        )
        assert isinstance(command, FetchCommand)
        self.assertEqual(command.wait_in_seconds, 1.5)
        self.assertTrue(command.return_screenshot)

    def test_omitted_fields_keep_their_defaults(self) -> None:
        command = _command()
        self.assertEqual(command.wait_in_seconds, 0.0)
        self.assertFalse(command.return_screenshot)

    def test_the_wait_bound_tracks_the_unrounded_timeout(self) -> None:
        # 1500 ms is 1.5 s: 1.5 fits, 1.6 does not, even though timeout_seconds rounds up to 2.
        command = _command(maxTimeout=1500, waitInSeconds=1.5)
        self.assertEqual(command.wait_in_seconds, 1.5)
        self.assertEqual(command.timeout_seconds, 2)
        with self.assertRaises(ProtocolError):
            _command(maxTimeout=1500, waitInSeconds=1.6)

    def test_invalid_wait_values_are_rejected_without_echoing_them(self) -> None:
        for value in (True, None, float("nan"), float("inf"), -1, 60.001, 10**400, "1.0"):
            with self.assertRaises(ProtocolError) as caught:
                _command(waitInSeconds=value)
            self.assertEqual(caught.exception.message, _WAIT_ERROR, value)
            self.assertNotIn(str(value), caught.exception.message, value)

    def test_invalid_screenshot_values_are_rejected(self) -> None:
        for value in (1, 0, None, "true", 1.0, []):
            with self.assertRaises(ProtocolError) as caught:
                _command(returnScreenshot=value)
            self.assertEqual(caught.exception.message, _SCREENSHOT_ERROR, value)

    def test_unknown_options_stay_rejected_on_other_commands(self) -> None:
        for payload in (
            {"cmd": "sessions.create", "waitInSeconds": 1},
            {"cmd": "browser.open", "url": "https://example.com/", "returnScreenshot": True},
            {"cmd": "sessions.list", "waitInSeconds": 1},
        ):
            with self.assertRaises(ProtocolError, msg=payload):
                parse_request(payload)


class SnapshotOptionDispatchTests(IsolatedAsyncioTestCase):
    """The typed fetch forwards the options and the JSON reply projects them."""

    async def test_requested_screenshot_reaches_the_backend_and_the_reply(self) -> None:
        backend = _backend(_result(screenshot="UE5H"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({**_GET, "waitInSeconds": 0.25, "returnScreenshot": True})

        backend.fetch.assert_awaited_once()
        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.wait_in_seconds, 0.25)
        self.assertTrue(request.return_screenshot)
        self.assertEqual(body["solution"]["screenshot"], "UE5H")

    async def test_default_reply_keeps_its_exact_shape(self) -> None:
        backend = _backend(_result(screenshot="UE5H"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle(dict(_GET))

        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.wait_in_seconds, 0.0)
        self.assertFalse(request.return_screenshot)
        self.assertEqual(
            set(body["solution"]),
            {"url", "status", "headers", "response", "cookies", "userAgent"},
        )

    async def test_a_requested_screenshot_the_result_lacks_is_not_reported(self) -> None:
        backend = _backend(_result(screenshot=None))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({**_GET, "returnScreenshot": True})
        self.assertNotIn("screenshot", body["solution"])

    async def test_return_only_cookies_skips_the_wait_but_keeps_screenshot_and_cookies(self) -> None:
        cookie: dict[str, object] = {"name": "cf_clearance", "value": "tok"}
        backend = _backend(_result(response="body", body_bytes=b"body", cookies=[cookie], screenshot="PNG"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle(
            {**_GET, "returnOnlyCookies": True, "waitInSeconds": 5, "returnScreenshot": True}
        )

        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.wait_in_seconds, 0.0)
        self.assertTrue(request.return_screenshot)
        self.assertEqual(body["solution"]["response"], "")
        self.assertEqual(body["solution"]["screenshot"], "PNG")
        self.assertEqual(body["solution"]["cookies"], [cookie])

        returned = await service.fetch(_command(waitInSeconds=5))
        self.assertEqual(returned.response, "body")
        self.assertEqual(returned.body_bytes, b"body")
        self.assertEqual(backend.fetch.await_args.args[1].wait_in_seconds, 5.0)


class SnapshotOptionSafetyTests(IsolatedAsyncioTestCase):
    """Invalid options never reach the backend, and cancellation frees the lease."""

    async def test_parse_errors_never_reach_the_backend(self) -> None:
        backend = _backend(_result())
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({**_GET, "waitInSeconds": -1})
        self.assertEqual(body["status"], "error")
        _status, body = await service.handle({**_GET, "returnScreenshot": 1})
        self.assertEqual(body["status"], "error")
        backend.fetch.assert_not_awaited()

    async def test_cancellation_while_waiting_frees_the_admission(self) -> None:
        entered = asyncio.Event()
        never = asyncio.Event()
        result = _result()
        backend = _backend(result)

        async def blocking_fetch(_session_id: str | None, _request: FetchRequest) -> FetchResult:
            entered.set()
            await never.wait()
            return result

        backend.fetch = AsyncMock(side_effect=blocking_fetch)
        service = Service(_config(max_concurrency=1), backend)
        self.addAsyncCleanup(service.aclose)

        task = asyncio.create_task(service.fetch(_command(waitInSeconds=5, returnScreenshot=True)))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        backend.fetch = AsyncMock(return_value=result)
        self.assertIs(await asyncio.wait_for(service.fetch(_command()), timeout=1), result)
