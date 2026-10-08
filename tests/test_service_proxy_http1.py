"""Pure asyncio tests for the ordinary HTTP/1 request/response boundary.

Every test drives a real :class:`asyncio.StreamReader` with ``feed_data``/``feed_eof`` and a
``StreamWriter`` mock whose ``drain`` is an ``AsyncMock``. Nothing here launches a browser, opens a
socket, or contacts the network; the only peer is h11 itself, used to check the serialized wire.
"""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import h11

from prowl.browser.proxy import http1 as proxy_http1
from prowl.browser.proxy.http1 import (
    MAX_BODY_BYTES,
    ProxyProtocolError,
    read_request,
    write_response,
)
from prowl.browser.proxy.response import ProxyResponse

_GET = b"GET / HTTP/1.1\r\nHost: h\r\n\r\n"


def _reader(*chunks: bytes, limit: int = 2**16) -> asyncio.StreamReader:
    reader = asyncio.StreamReader(limit=limit)
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


def _writer() -> Mock:
    writer = Mock(spec=asyncio.StreamWriter)
    writer.drain = AsyncMock()
    return writer


def _client_after_get() -> h11.Connection:
    """Return an h11 client that has sent a request and may receive one response."""
    client = h11.Connection(h11.CLIENT)
    client.send(h11.Request(method="GET", target="/", headers=[("Host", "h")]))
    return client


def _read_message(client: h11.Connection) -> tuple[h11.Response, bytes]:
    """Read one response and its body from ``client``."""
    response = client.next_event()
    assert isinstance(response, h11.Response)
    body = bytearray()
    while True:
        event = client.next_event()
        if isinstance(event, h11.Data):
            body.extend(event.data)
            continue
        if isinstance(event, h11.EndOfMessage):
            return response, bytes(body)
        msg = f"unexpected event: {event!r}"
        raise AssertionError(msg)


class RequestReadTests(IsolatedAsyncioTestCase):
    """Parsed ordinary requests keep their exact target, fields, and body bytes."""

    async def test_get_absolute_uri_and_repeated_fields_parse(self) -> None:
        raw = b"GET https://example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\nX-Repeat: 1\r\nX-Repeat: 2\r\n\r\n"

        exchange = await read_request(_reader(raw))
        assert exchange is not None
        request, connection = exchange

        assert request is not None
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.target, "https://example.com/a?b=1")
        self.assertEqual(
            request.headers,
            (("Host", "example.com"), ("X-Repeat", "1"), ("X-Repeat", "2")),
        )
        self.assertEqual(request.body, b"")
        self.assertIsInstance(connection, h11.Connection)

    async def test_post_exact_binary_body(self) -> None:
        payload = b"\x00\xff\xfe\x01ok"
        raw = b"POST /submit HTTP/1.1\r\nHost: h\r\nContent-Length: 6\r\n\r\n" + payload

        exchange = await read_request(_reader(raw))
        assert exchange is not None
        request, _ = exchange

        assert request is not None
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.body, payload)

    async def test_chunked_body_is_decoded_to_exact_bytes(self) -> None:
        raw = (
            b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"4\r\n\x00\x01\x02\x03\r\n3\r\nabc\r\n0\r\n\r\n"
        )

        exchange = await read_request(_reader(raw))
        assert exchange is not None
        request, _ = exchange

        assert request is not None
        self.assertEqual(request.body, b"\x00\x01\x02\x03abc")

    async def test_connect_is_parsed_without_tunnel_claim(self) -> None:
        raw = b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n"

        exchange = await read_request(_reader(raw))
        assert exchange is not None
        request, _ = exchange

        assert request is not None
        self.assertEqual(request.method, "CONNECT")
        self.assertEqual(request.target, "example.com:443")
        self.assertEqual(request.body, b"")


class RequestRejectionTests(IsolatedAsyncioTestCase):
    """Malformed or unsupported requests fail with a fixed, content-free status."""

    async def _status_for(self, raw: bytes, *, limit: int = 2**16) -> int:
        with self.assertRaises(ProxyProtocolError) as caught:
            await read_request(_reader(raw, limit=limit))
        return caught.exception.status

    async def test_malformed_requests_are_rejected(self) -> None:
        cases = {
            "conflicting-content-length": (
                b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabc"
            ),
            "transfer-encoding-and-length": (
                b"POST / HTTP/1.1\r\nHost: h\r\n"
                b"Transfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n3\r\nabc\r\n0\r\n\r\n"
            ),
            "control-byte": b"GET / HTTP/1.1\r\nHost: h\r\nX: a\x01b\r\n\r\n",
            "folded-header": b"GET / HTTP/1.1\r\nHost: h\r\nX: a\r\n  continued\r\n\r\n",
            "bare-line-ending": b"GET / HTTP/1.1\nHost: h\r\n\r\n",
            "malformed-header-line": b"GET / HTTP/1.1\r\nno-colon-here\r\n\r\n",
        }

        for name, raw in cases.items():
            with self.subTest(name=name):
                self.assertEqual(await self._status_for(raw), 400)

    async def test_oversized_declared_body_and_header_bound(self) -> None:
        declared = b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: %d\r\n\r\n" % (MAX_BODY_BYTES + 1)
        self.assertEqual(await self._status_for(declared), 413)

        oversized_head = b"GET / HTTP/1.1\r\nHost: h\r\nX: " + b"a" * 128 + b"\r\n\r\n"
        with patch.object(proxy_http1, "MAX_HEADER_BYTES", 64):
            self.assertEqual(await self._status_for(oversized_head), 431)

        unterminated = b"GET / HTTP/1.1\r\nHost: h\r\n"
        self.assertEqual(await self._status_for(unterminated, limit=16), 431)

    async def test_unsupported_http_version_is_rejected(self) -> None:
        self.assertEqual(await self._status_for(b"GET / HTTP/2.0\r\nHost: h\r\n\r\n"), 505)

    async def test_unsupported_method_upgrade_and_expect(self) -> None:
        self.assertEqual(await self._status_for(b"PUT / HTTP/1.1\r\nHost: h\r\n\r\n"), 405)

        upgrade = b"GET / HTTP/1.1\r\nHost: h\r\nConnection: upgrade\r\nUpgrade: websocket\r\n\r\n"
        self.assertEqual(await self._status_for(upgrade), 501)

        expect = b"POST / HTTP/1.1\r\nHost: h\r\nExpect: 100-continue\r\nContent-Length: 3\r\n\r\nabc"
        self.assertEqual(await self._status_for(expect), 417)

    async def test_empty_eof_incomplete_head_and_body_hazards(self) -> None:
        self.assertIsNone(await read_request(_reader()))

        self.assertEqual(await self._status_for(b"GET / HTTP/1.1\r\nHost: h\r\n"), 400)

        premature = b"POST / HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n\r\nab"
        self.assertEqual(await self._status_for(premature), 400)

        get_with_body = b"GET / HTTP/1.1\r\nHost: h\r\nContent-Length: 2\r\n\r\nhi"
        self.assertEqual(await self._status_for(get_with_body), 400)

        trailers = b"POST / HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nX-T: 1\r\n\r\n"
        self.assertEqual(await self._status_for(trailers), 400)

    async def test_read_timeout_maps_to_408(self) -> None:
        reader = asyncio.StreamReader()
        with (
            patch.object(proxy_http1, "REQUEST_READ_SECONDS", 0),
            self.assertRaises(ProxyProtocolError) as caught,
        ):
            await read_request(reader)
        self.assertEqual(caught.exception.status, 408)

    async def test_native_cancellation_propagates(self) -> None:
        reader = asyncio.StreamReader()
        entered = asyncio.Event()
        original = reader.readuntil

        async def read_head(separator: bytes = b"\n") -> bytes:
            entered.set()
            return await original(separator)

        with patch.object(reader, "readuntil", side_effect=read_head):
            task = asyncio.create_task(read_request(reader))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class ResponseWriteTests(IsolatedAsyncioTestCase):
    """Serialized responses round-trip through an h11 client as ordinary wire bytes."""

    async def _serialize(self, response: ProxyResponse) -> bytes:
        exchange = await read_request(_reader(_GET))
        assert exchange is not None
        _, connection = exchange
        writer = _writer()
        await write_response(writer, connection, response)
        return b"".join(call.args[0] for call in writer.write.call_args_list)

    async def test_serialized_response_matches_ordinary_wire(self) -> None:
        body = b"\x00\xff\xfeok"
        response = ProxyResponse(
            status=200,
            headers=(
                ("Content-Type", "application/octet-stream"),
                ("Set-Cookie", "a=1; Path=/"),
                ("Set-Cookie", "b=2; Path=/"),
                ("Content-Length", str(len(body))),
            ),
            body=body,
        )

        payload = await self._serialize(response)

        client = _client_after_get()
        client.receive_data(payload)
        event, received_body = _read_message(client)

        self.assertEqual(event.status_code, 200)
        self.assertEqual(received_body, body)
        self.assertEqual(
            [value for name, value in event.headers.raw_items() if name.lower() == b"set-cookie"],
            [b"a=1; Path=/", b"b=2; Path=/"],
        )
        self.assertIn((b"Connection", b"close"), event.headers.raw_items())

    async def test_bodyless_statuses_round_trip_on_wire(self) -> None:
        cases = (
            (204, ()),
            (205, (("Content-Length", "0"),)),
            (304, ()),
        )

        for status, headers in cases:
            with self.subTest(status=status):
                payload = await self._serialize(ProxyResponse(status=status, headers=headers, body=b""))
                client = _client_after_get()
                client.receive_data(payload)
                event, received_body = _read_message(client)
                self.assertEqual(event.status_code, status)
                self.assertEqual(received_body, b"")
