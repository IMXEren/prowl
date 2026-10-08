"""Execution-mode request and result contract tests for a fetch.

A fetch's mode travels from the wire into the request the backend receives inside its session
lease, and the mode a fetch reports travels back into the reply's additive diagnostics. Nothing
routes on a mode yet: these tests cover parsing, forwarding and serialization only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase, TestCase

from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import FetchRequest, FetchResult
from prowl.service.classification import SUCCESS, Classification
from prowl.service.protocol import (
    AUTO_MODE,
    BROWSER_MODE,
    EXECUTION_MODES,
    HTTP_MODE,
    FetchCommand,
    ProtocolError,
    execution_payload,
    parse_request,
    solution_payload,
)
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE

if TYPE_CHECKING:
    from prowl.service.backend import CookieQuery, InteractiveOpenResult, InteractiveRequest, InteractiveTab

GET = {"cmd": "request.get", "url": "https://example.com/"}

SOLUTION_FIELDS = {"url", "status", "headers", "response", "cookies", "userAgent"}


class RecordingBackend:
    """In-memory backend double recording every request it receives."""

    def __init__(self, *, result: FetchResult | None = None) -> None:
        self.closed_sessions: list[str] = []
        self.requests: list[tuple[str | None, FetchRequest]] = []
        self._result = result

    def result(self) -> FetchResult:
        if self._result is not None:
            return self._result
        return FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            response="<html><body>ok</body></html>",
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def start(self) -> None:
        self.started = True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        self.requests.append((session_id, request))
        return self.result()

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        message = "unexpected interactive open"
        raise AssertionError(message)

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        return []

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        return []

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        return []

    async def aclose(self) -> None:
        self.closed = True


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {"max_concurrency": 2}
    values.update(overrides)
    return ServiceConfig(**values)


def _solution(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "url": "https://example.com/",
        "status_code": 200,
        "headers": {},
        "response": "<html></html>",
        "cookies": [],
        "user_agent": "Mozilla/5.0 (Test)",
    }
    values.update(overrides)
    return solution_payload(**values)


class FetchModeParsingTests(TestCase):
    """The wire accepts exactly the three fetch modes and defaults to the browser."""

    def test_the_wire_accepts_only_the_three_modes(self) -> None:
        self.assertEqual(EXECUTION_MODES, {"browser", "http", "auto"})

    def test_omitted_mode_is_the_browser(self) -> None:
        command = parse_request(GET)
        assert isinstance(command, FetchCommand)
        self.assertEqual(command.mode, BROWSER_MODE)

    def test_every_mode_is_accepted(self) -> None:
        for mode in ("browser", "http", "auto"):
            with self.subTest(mode=mode):
                command = parse_request({**GET, "mode": mode})
                assert isinstance(command, FetchCommand)
                self.assertEqual(command.mode, mode)

    def test_request_post_accepts_a_mode(self) -> None:
        command = parse_request({"cmd": "request.post", "url": "https://example.com/api", "mode": AUTO_MODE})
        assert isinstance(command, FetchCommand)
        self.assertEqual(command.mode, AUTO_MODE)

    def test_invalid_modes_are_rejected(self) -> None:
        for value in ("screenshot", "", "HTTP", True, 1, 2.5, ["http"], {"url": "http"}, None):
            with self.subTest(value=value):
                with self.assertRaises(ProtocolError) as raised:
                    parse_request({**GET, "mode": value})
                self.assertEqual(raised.exception.http_status, 400)
                self.assertEqual(raised.exception.message, "mode must be 'browser', 'http' or 'auto'")

    def test_only_a_fetch_accepts_a_mode(self) -> None:
        """An interactive or session command rejects the field outright rather than ignoring it."""
        payloads = (
            {"cmd": "browser.open", "url": "https://example.com/"},
            {"cmd": "browser.close"},
            {"cmd": "browser.list"},
            {"cmd": "cookies.list"},
            {"cmd": "sessions.create"},
            {"cmd": "sessions.list"},
            {"cmd": "sessions.destroy", "session": "example"},
        )
        for payload in payloads:
            with self.subTest(cmd=payload["cmd"]):
                with self.assertRaises(ProtocolError) as raised:
                    parse_request({**payload, "mode": "http"})
                self.assertEqual(raised.exception.message, "unknown field(s): mode")


class FetchModeForwardingTests(IsolatedAsyncioTestCase):
    """The command's mode reaches the backend request inside the request's own lease."""

    async def test_default_mode_reaches_the_backend_as_the_browser(self) -> None:
        backend = RecordingBackend()
        await Service(_config(), backend).handle(GET)
        ((session, request),) = backend.requests
        self.assertIsNone(session)
        self.assertEqual(request.mode, BROWSER_MODE)
        self.assertEqual(request.session_mode, SHARED_MODE)
        self.assertEqual(request.egress, DEFAULT_EGRESS_NAME)

    async def test_explicit_modes_are_forwarded_unchanged(self) -> None:
        for mode in ("http", "auto", "browser"):
            with self.subTest(mode=mode):
                backend = RecordingBackend()
                await Service(_config(), backend).handle({**GET, "mode": mode})
                self.assertEqual(backend.requests[0][1].mode, mode)

    async def test_mode_is_forwarded_with_an_anonymous_proxied_fetch(self) -> None:
        backend = RecordingBackend()
        config = _config(proxy_url="socks5://127.0.0.1:1080")
        await Service(config, backend).handle(
            {**GET, "mode": HTTP_MODE, "proxy": {"url": "socks5://127.0.0.1:1080"}},
        )
        ((session, request),) = backend.requests
        self.assertIsNone(session)
        self.assertEqual(request.mode, HTTP_MODE)
        self.assertEqual(request.egress, DEFAULT_EGRESS_NAME)
        self.assertEqual(request.session_mode, SHARED_MODE)

    async def test_shared_session_keeps_mode_apart_from_its_session_mode(self) -> None:
        backend = RecordingBackend()
        service = Service(_config(), backend)
        await service.handle({**GET, "session": "example", "sessionMode": SHARED_MODE, "mode": AUTO_MODE})
        (session, request) = backend.requests[0]
        self.assertEqual(session, "example")
        self.assertEqual(request.session_id, "example")
        self.assertEqual(request.session_mode, SHARED_MODE)
        self.assertEqual(request.mode, AUTO_MODE)

    async def test_isolated_session_keeps_mode_apart_from_session_mode_and_egress(self) -> None:
        backend = RecordingBackend()
        config = _config(egresses={"eu": "socks5://127.0.0.1:1080"})
        service = Service(config, backend)
        await service.handle(
            {**GET, "session": "iso", "sessionMode": "isolated", "proxy": {"name": "eu"}, "mode": HTTP_MODE},
        )
        (session, request) = backend.requests[0]
        self.assertEqual(session, "iso")
        self.assertEqual(request.session_mode, ISOLATED_MODE)
        self.assertEqual(request.egress, "eu")
        self.assertEqual(request.mode, HTTP_MODE)


class ExecutionDiagnosticsSerializationTests(TestCase):
    """Execution diagnostics are additive: nothing reported keeps the legacy shape."""

    def test_nothing_reported_builds_no_execution_object(self) -> None:
        self.assertIsNone(execution_payload(None, None))
        self.assertEqual(set(_solution()), SOLUTION_FIELDS)
        self.assertEqual(set(_solution(execution=None)), SOLUTION_FIELDS)

    def test_reported_mode_and_classification_are_serialized(self) -> None:
        classification = Classification(SUCCESS, "HTTP 200 carries no challenge signature", browser_required=False)
        self.assertEqual(
            execution_payload(HTTP_MODE, classification),
            {
                "mode": "http",
                "category": "SUCCESS",
                "reason": "HTTP 200 carries no challenge signature",
            },
        )

    def test_reported_mode_alone_is_serialized(self) -> None:
        self.assertEqual(execution_payload(BROWSER_MODE), {"mode": "browser"})

    def test_reported_classification_alone_is_serialized(self) -> None:
        classification = Classification(SUCCESS, "HTTP 200 carries no challenge signature", browser_required=False)
        self.assertEqual(
            execution_payload(None, classification),
            {"category": "SUCCESS", "reason": "HTTP 200 carries no challenge signature"},
        )

    def test_solution_carries_the_execution_object_and_every_legacy_field(self) -> None:
        solution = _solution(execution=execution_payload(BROWSER_MODE))
        self.assertEqual(set(solution) - {"execution"}, SOLUTION_FIELDS)
        self.assertEqual(solution["execution"], {"mode": "browser"})


class FetchDiagnosticsResponseTests(IsolatedAsyncioTestCase):
    """The reply reports what the fetch reported, and nothing when it reported nothing."""

    async def test_legacy_browser_reply_has_no_execution_object(self) -> None:
        _, body = await Service(_config(), RecordingBackend()).handle(GET)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(set(body["solution"]), SOLUTION_FIELDS)

    async def test_reported_diagnostics_reach_the_reply(self) -> None:
        classification = Classification(SUCCESS, "HTTP 200 carries no challenge signature", browser_required=False)
        backend = RecordingBackend(
            result=FetchResult(
                url="https://example.com/",
                status_code=200,
                headers={"content-type": "text/html"},
                response="<html><body>ok</body></html>",
                cookies=[],
                user_agent="Mozilla/5.0 (Test)",
                mode=HTTP_MODE,
                classification=classification,
            ),
        )
        _, body = await Service(_config(), backend).handle(GET)
        self.assertEqual(
            body["solution"]["execution"],
            {"mode": "http", "category": SUCCESS, "reason": "HTTP 200 carries no challenge signature"},
        )

    async def test_a_reported_mode_alone_reaches_the_reply(self) -> None:
        backend = RecordingBackend(
            result=FetchResult(
                url="https://example.com/",
                status_code=200,
                headers={},
                response="<html></html>",
                cookies=[],
                user_agent=None,
                mode=AUTO_MODE,
            ),
        )
        _, body = await Service(_config(), backend).handle(GET)
        self.assertEqual(body["solution"]["execution"], {"mode": "auto"})


class LegacyConstructorTests(TestCase):
    """Positional construction and browser-only defaults stay compatible."""

    def test_fetch_command_positional_fields_still_construct(self) -> None:
        command = FetchCommand("request.get", "https://example.com/", 60_000)
        self.assertEqual(command.mode, BROWSER_MODE)
        self.assertIsNone(command.proxy)

    def test_fetch_request_defaults_to_the_browser(self) -> None:
        self.assertEqual(FetchRequest("https://example.com/").mode, BROWSER_MODE)
        self.assertEqual(FetchRequest("https://example.com/", "POST").mode, BROWSER_MODE)

    def test_fetch_result_positional_fields_still_construct(self) -> None:
        result = FetchResult("https://example.com/", 200, {}, "<html></html>", [], "Mozilla/5.0 (Test)")
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)
