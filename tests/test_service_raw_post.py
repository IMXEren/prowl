"""Internal raw POST body seam: an explicit HTTP POST may carry exact caller bytes.

The new ``FetchRequest.body_bytes`` field stays internal (it never enters the JSON wire or
``FetchCommand``): ``Service.fetch`` carries it through the existing lease, ``_http_stage`` hands
the same object to the HTTP client, and ``BrowserBackend.fetch`` refuses raw bytes unless the
request is an explicit HTTP POST, before any tab steal, owner, context or client acquisition.
Everything here runs against the loopback HTTP/1.1 server, stand-in contexts and stub browsers from
``test_service_http_transport`` / ``test_service_routing_backend``; no browser is launched and
nothing external is contacted.
"""

from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import AsyncMock, Mock

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchRequest, FetchResult
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import (
    AUTO_MODE,
    BROWSER_MODE,
    CMD_REQUEST_POST,
    HTTP_MODE,
    FetchCommand,
)
from test_service_http_transport import _Reply, _routes, _TransportTestCase
from test_service_routing_backend import _http_result, _request, _RoutingFixture, _StubHttpClient


class RawBodyTransportTests(_TransportTestCase):
    """The real HTTP client sends exact bytes for an ordinary POST, including empty ones."""

    async def test_binary_body_with_nul_and_non_utf8_arrives_exactly(self) -> None:
        payload = b"\x00\xff\xfe\x01binary\r\n\x80"
        server = await self.start(_routes({"/upload": (200, [], b"ok")}))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/upload", method="POST", content=payload, deadline_seconds=10)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(server.received[0].method, "POST")
        self.assertEqual(server.received[0].body, payload)
        # An ordinary POST reaches the server exactly once: no retry, no duplicated effect.
        self.assertEqual(len(server.received), 1)

    async def test_json_body_arrives_exactly(self) -> None:
        payload = json.dumps({"name": "prowl", "n": None, "u": "\u00e9"}).encode("utf-8")
        server = await self.start(_routes({"/json": (200, [], b"ok")}))
        client = self.make_client()

        await client.fetch(
            server.base_url + "/json",
            method="POST",
            headers={"Content-Type": "application/json"},
            content=payload,
            deadline_seconds=10,
        )

        self.assertEqual(server.received[0].body, payload)
        self.assertEqual(server.received[0].headers["content-type"], "application/json")
        decoded = json.loads(server.received[0].body.decode("utf-8"))
        self.assertEqual(decoded, {"name": "prowl", "n": None, "u": "\u00e9"})

    async def test_empty_bytes_post_is_sent_as_empty_not_a_get(self) -> None:
        server = await self.start(_routes({"/empty": (200, [], b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/empty", method="POST", content=b"", deadline_seconds=10)

        self.assertEqual(server.received[0].method, "POST")
        self.assertEqual(server.received[0].body, b"")


class RawBodyRedirectTests(_TransportTestCase):
    """The existing redirect loop carries bytes unchanged, or drops them, as the status decides."""

    async def test_307_keeps_the_bytes_and_303_turns_the_post_into_a_bodyless_get(self) -> None:
        routes: dict[str, _Reply] = {
            "/keep": (307, [("Location", "/kept")], b""),
            "/kept": (200, [], b"done"),
            "/see": (303, [("Location", "/seen")], b""),
            "/seen": (200, [], b"ok"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()
        payload = b"\x00\xffkeep"

        await client.fetch(server.base_url + "/keep", method="POST", content=payload, deadline_seconds=10)
        await client.fetch(server.base_url + "/see", method="POST", content=payload, deadline_seconds=10)

        self.assertEqual(server.received[0].body, payload)
        self.assertEqual(server.received[1].method, "POST")
        self.assertEqual(server.received[1].body, payload)
        self.assertEqual(server.received[3].method, "GET")
        self.assertEqual(server.received[3].body, b"")


class HttpStageBodyPropagationTests(_RoutingFixture):
    """``_http_stage`` hands the raw bytes object straight to the HTTP client."""

    async def test_raw_bytes_are_passed_unchanged_when_present(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, recorder, _clients, _site = self._backend(http_client=client)
        payload = b"\x00\xff\x01binary"

        result = await backend.fetch(
            None,
            _request(mode=HTTP_MODE, method="POST", post_data="wrong", body_bytes=payload),
        )

        self.assertIs(client.calls[0][1]["content"], payload)
        self.assertEqual(recorder.groups, [])
        self.assertEqual(result.mode, HTTP_MODE)
        await backend.aclose()

    async def test_empty_bytes_win_over_legacy_post_data(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, _clients, _site = self._backend(http_client=client)

        await backend.fetch(None, _request(mode=HTTP_MODE, method="POST", post_data="wrong", body_bytes=b""))

        self.assertEqual(client.calls[0][1]["content"], b"")
        self.assertNotEqual(client.calls[0][1]["content"], "wrong")
        await backend.aclose()

    async def test_legacy_post_data_still_uses_the_string_path(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, _clients, _site = self._backend(http_client=client)

        await backend.fetch(None, _request(mode=HTTP_MODE, method="POST", post_data="message=hello"))

        self.assertEqual(client.calls[0][1]["content"], "message=hello")
        await backend.aclose()


class RawBodyPreflightGuardTests(_RoutingFixture):
    """Raw bytes are refused before any tab steal, owner, context or client acquisition."""

    async def test_browser_mode_post_with_bytes_is_refused_before_acquisition(self) -> None:
        backend, _recorder, _clients, _site = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        pool_acquire = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for
        backend._pool.acquire = pool_acquire

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, _request(mode=BROWSER_MODE, method="POST", body_bytes=b"x"))

        self.assertEqual(str(caught.exception), "raw request bodies require explicit HTTP POST")
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()
        pool_acquire.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)
        await backend.aclose()

    async def test_auto_mode_bytes_are_refused_before_acquisition(self) -> None:
        client = _StubHttpClient(result=_http_result())
        backend, _recorder, clients, _site = self._backend(http_client=client)
        steal = AsyncMock()
        backend._steal_least_recent = steal

        with self.assertRaises(CallerSafeError):
            await backend.fetch(None, _request(mode=AUTO_MODE, method="POST", body_bytes=b"x"))

        self.assertEqual(clients.client_calls, [])
        steal.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)
        await backend.aclose()

    async def test_get_with_bytes_is_refused_before_acquisition(self) -> None:
        backend, _recorder, clients, _site = self._backend(http_client=_StubHttpClient(result=_http_result()))

        with self.assertRaises(CallerSafeError):
            await backend.fetch(None, _request(mode=HTTP_MODE, method="GET", body_bytes=b"x"))

        self.assertEqual(clients.client_calls, [])
        self.assertEqual(backend.metrics.requests_active, 0)
        await backend.aclose()


class ServiceRawBodyTests(_RoutingFixture):
    """``Service.fetch`` carries the optional bytes into the backend request unchanged."""

    def _service(self, result: FetchResult) -> tuple[Service, list[FetchRequest]]:
        captured: list[FetchRequest] = []
        backend = Mock(spec=Backend)

        async def _fetch(_session_id: str | None, request: FetchRequest) -> FetchResult:
            captured.append(request)
            return result

        backend.fetch = _fetch
        return Service(ServiceConfig(), backend), captured

    async def test_service_carries_both_the_bytes_and_the_binding_into_the_request(self) -> None:
        result = FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={},
            response="ok",
            cookies=[],
            user_agent="ua",
        )
        service, captured = self._service(result)
        command = FetchCommand(cmd=CMD_REQUEST_POST, url="https://example.com/", timeout_ms=60000, mode=HTTP_MODE)
        payload = b"\x00\xffraw"

        returned = await service.fetch(command, body_bytes=payload)

        self.assertIs(returned, result)
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0].body_bytes, payload)
        self.assertEqual(captured[0].headers, {})
        self.assertIsNone(captured[0].session_id)

    async def test_service_legacy_post_keeps_the_string_path(self) -> None:
        result = FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={},
            response="ok",
            cookies=[],
            user_agent="ua",
        )
        service, captured = self._service(result)
        command = FetchCommand(
            cmd=CMD_REQUEST_POST,
            url="https://example.com/",
            timeout_ms=60000,
            post_data='{"a":1}',
            mode=HTTP_MODE,
        )

        await service.fetch(command)

        self.assertEqual(captured[0].post_data, '{"a":1}')
        self.assertIsNone(captured[0].body_bytes)


class FetchRequestShapeTests(TestCase):
    """The appended field keeps the legacy positional and default shape."""

    def test_body_bytes_position_and_default_are_preserved(self) -> None:
        request = FetchRequest("https://example.com/")

        self.assertIsNone(request.body_bytes)
        self.assertEqual(request.mode, BROWSER_MODE)
        fields = list(FetchRequest.__dataclass_fields__)
        self.assertEqual(fields[fields.index("body_bytes") + 1], "cookie_header")
        self.assertEqual(
            fields[fields.index("wait_in_seconds") : fields.index("solve_captcha") + 1],
            ["wait_in_seconds", "return_screenshot", "disable_media", "tabs_till_verify", "solve_captcha"],
        )
