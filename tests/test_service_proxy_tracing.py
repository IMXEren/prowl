"""Request-correlation tracing for the plain-HTTP proxy listener.

Every admitted connection mints one opaque id, binds it for the whole exchange, stamps it as
``X-Request-ID`` on the single response, and logs one finish line under it. An inbound
``X-Request-ID`` is ignored outright and never forwarded to the origin.

Each test binds a real :class:`ProxyServer` on ``127.0.0.1`` and drives it over a real asyncio
socket. The service is a complete ``Mock(spec=Service)`` whose ``fetch`` is an ``AsyncMock``
returning real :class:`FetchResult` DTOs. Nothing here starts the real service or backend, launches
a browser, opens a TLS tunnel, or leaves loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import h11
from loguru import logger
from tests.test_service_proxy_server import _close_writer, _http_result, _parse, _read_all, _values

from prowl.browser.proxy.dispatch import dispatch_request
from prowl.browser.proxy.http1 import ProxyRequest
from prowl.browser.proxy.server import ProxyServer
from prowl.service.app import Service
from prowl.service.protocol import FetchCommand

if TYPE_CHECKING:
    from loguru import Message

    from prowl.service.backend import FetchResult

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_FINISH = "Proxy request"
_BACKEND_STAGE = "backend stage"
_PATH_MARKER = "SUPERSECRETPATH"


def _http(*, header_items: tuple[tuple[str, str], ...] = (("Content-Type", "text/plain"),)) -> FetchResult:
    return _http_result(header_items=header_items)


def _service(result: FetchResult) -> Mock:
    """A complete ``Service`` double whose ``fetch`` returns *result* without logging."""
    service = Mock(spec=Service)
    service.fetch = AsyncMock(return_value=result)
    return service


def _logging_service(result: FetchResult) -> Mock:
    """A ``Service`` double whose ``fetch`` logs one fixed backend-stage line, then returns *result*."""
    service = Mock(spec=Service)

    async def fetch(*_args: object, **_kwargs: object) -> FetchResult:
        logger.info(_BACKEND_STAGE)
        return result

    service.fetch = AsyncMock(side_effect=fetch)
    return service


def _capture_records(case: IsolatedAsyncioTestCase) -> list[tuple[str, dict[str, object]]]:
    """Collect loguru records (formatted message plus ``extra``) for *case*, dropped on cleanup."""
    records: list[tuple[str, dict[str, object]]] = []

    def sink(message: Message) -> None:
        records.append((message.record["message"], dict(message.record["extra"])))

    handler_id = logger.add(sink, level="DEBUG")
    case.addCleanup(logger.remove, handler_id)
    return records


def _ids(records: list[tuple[str, dict[str, object]]], prefix: str) -> list[str]:
    ids: list[str] = []
    for message, extra in records:
        if message.startswith(prefix):
            value = extra.get("request_id")
            if not isinstance(value, str):
                msg = "correlation id must be a string"
                raise AssertionError(msg)
            ids.append(value)
    return ids


def _request_id(raw: bytes) -> str:
    """The single ``X-Request-ID`` value on a raw response, asserted to appear exactly once."""
    response, _ = _parse(raw)
    values = _values(response, "x-request-id")
    assert len(values) == 1, values
    return values[0].decode("ascii")


async def _open(server: ProxyServer, request_bytes: bytes) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(request_bytes)
    await writer.drain()
    return reader, writer


def _get(target: str, *extra: bytes) -> bytes:
    head = b"GET " + target.encode() + b" HTTP/1.1\r\nHost: example.com\r\n"
    for line in extra:
        head += line + b"\r\n"
    return head + b"\r\n"


def _parse_connect_head(raw: bytes) -> h11.Response:
    """Parse a CONNECT response head written by the listener with an h11 client connection."""
    client = h11.Connection(h11.CLIENT)
    client.send(h11.Request(method="CONNECT", target="example.test:443", headers=[("Host", "example.test:443")]))
    client.receive_data(raw)
    response = client.next_event()
    assert isinstance(response, h11.Response)
    return response


class OrdinaryResponseTracingTests(IsolatedAsyncioTestCase):
    """One response carries exactly one minted id that correlates the whole connection."""

    async def test_response_carries_one_id_and_rejects_caller_and_origin_ids(self) -> None:
        records = _capture_records(self)
        spoof = "caller-supplied-id"
        service = _logging_service(_http(header_items=(("X-Request-ID", "origin-one"), ("X-Request-ID", "origin-two"))))
        server = ProxyServer(service)
        await server.start()
        try:
            reader, writer = await _open(
                server,
                _get(f"https://example.com/{_PATH_MARKER}", b"X-Request-ID: " + spoof.encode()),
            )
            try:
                raw = await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        response, body = _parse(raw, target=f"https://example.com/{_PATH_MARKER}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, b"body")

        ids = _values(response, "x-request-id")
        self.assertEqual(len(ids), 1)
        returned = ids[0].decode("ascii")
        self.assertRegex(returned, _HEX32)
        self.assertNotEqual(returned, spoof)
        self.assertNotIn(spoof.encode(), raw)
        self.assertNotIn(b"origin-one", raw)
        self.assertNotIn(b"origin-two", raw)

        command = service.fetch.await_args.args[0]
        self.assertIsInstance(command, FetchCommand)
        self.assertNotIn("x-request-id", command.headers)

        self.assertEqual(_ids(records, _BACKEND_STAGE), [returned])
        self.assertEqual(_ids(records, _FINISH), [returned])
        self.assertNotIn(_PATH_MARKER, repr(records))


class AdapterIdStateTests(IsolatedAsyncioTestCase):
    """The pure adapter drops only the inbound id and never mutates the caller's DTO."""

    async def test_adapter_drops_inbound_id_and_leaves_request_unchanged(self) -> None:
        service = _service(_http())
        request = ProxyRequest(
            method="GET",
            target="https://example.com/",
            headers=(("Host", "example.com"), ("X-Request-ID", "caller-id"), ("X-Trace", "keep")),
            body=b"",
        )

        await dispatch_request(service, request)

        command = service.fetch.await_args.args[0]
        self.assertEqual(command.headers, {"x-trace": "keep"})
        self.assertEqual([name for name, _ in request.headers], ["Host", "X-Request-ID", "X-Trace"])
        self.assertEqual(request.headers[1], ("X-Request-ID", "caller-id"))


class LocalErrorTracingTests(IsolatedAsyncioTestCase):
    """A reject or a failed dispatch still answers under one id and never echoes the failure."""

    async def test_malformed_head_gets_a_400_with_the_id(self) -> None:
        records = _capture_records(self)
        service = _service(_http())
        server = ProxyServer(service)
        await server.start()
        try:
            reader, writer = await _open(
                server, b"GET https://example.com/ HTTP/1.1\r\nHost: example.com\r\nX-S: SECRET\r\nno-colon\r\n\r\n"
            )
            try:
                raw = await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        response, body = _parse(raw)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(body, b"proxy request failed")
        self.assertNotIn(b"SECRET", raw)
        service.fetch.assert_not_awaited()

        returned = _request_id(raw)
        self.assertEqual(_ids(records, _FINISH), [returned])

    async def test_failed_dispatch_gets_a_502_with_the_id_and_no_exception_echo(self) -> None:
        records = _capture_records(self)
        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=RuntimeError("SECRET-EXCEPTION"))
        server = ProxyServer(service)
        await server.start()
        try:
            reader, writer = await _open(server, _get("https://example.com/"))
            try:
                raw = await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        response, body = _parse(raw)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(body, b"proxy request failed")
        self.assertNotIn(b"SECRET-EXCEPTION", raw)

        returned = _request_id(raw)
        self.assertEqual(_ids(records, _FINISH), [returned])


class ConcurrentTracingTests(IsolatedAsyncioTestCase):
    """Concurrent connections keep distinct ids; each backend stage sees its own."""

    async def test_concurrent_requests_keep_distinct_ids(self) -> None:
        records = _capture_records(self)
        entered = asyncio.Event()
        release = asyncio.Event()
        arrived = 0

        async def fetch(command: FetchCommand, **_kwargs: object) -> FetchResult:
            nonlocal arrived
            with logger.contextualize(stage=command.url.rsplit("/", 1)[-1]):
                logger.info(_BACKEND_STAGE)
            arrived += 1
            if arrived == 2:
                entered.set()
            await release.wait()
            return _http()

        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=fetch)
        server = ProxyServer(service)
        await server.start()
        connections: list[asyncio.StreamWriter] = []
        try:
            one = await _open(server, _get("https://example.com/one"))
            two = await _open(server, _get("https://example.com/two"))
            connections = [one[1], two[1]]
            await asyncio.wait_for(entered.wait(), 2)
            release.set()
            raw_one = await asyncio.wait_for(_read_all(one[0]), 2)
            raw_two = await asyncio.wait_for(_read_all(two[0]), 2)
        finally:
            release.set()
            for writer in connections:
                await _close_writer(writer)
            await server.aclose()

        one_id = _request_id(raw_one)
        two_id = _request_id(raw_two)
        self.assertRegex(one_id, _HEX32)
        self.assertRegex(two_id, _HEX32)
        self.assertNotEqual(one_id, two_id)

        stage = {extra["stage"]: extra["request_id"] for message, extra in records if message == _BACKEND_STAGE}
        self.assertEqual(set(stage), {"one", "two"})
        self.assertEqual(stage["one"], one_id)
        self.assertEqual(stage["two"], two_id)
        self.assertEqual(sorted(_ids(records, _FINISH)), sorted([one_id, two_id]))


class CancellationTracingTests(IsolatedAsyncioTestCase):
    """A cancelled connection logs its own id and never leaks it into the next request."""

    async def test_cancelled_request_id_does_not_bleed_into_the_next(self) -> None:
        records = _capture_records(self)
        entered = asyncio.Event()

        async def hang(*_args: object, **_kwargs: object) -> FetchResult:
            entered.set()
            await asyncio.Event().wait()
            return _http()

        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=hang)
        server = ProxyServer(service)
        await server.start()
        writers: list[asyncio.StreamWriter] = []
        try:
            _first_reader, first_writer = await _open(server, _get("https://example.com/one"))
            writers.append(first_writer)
            await asyncio.wait_for(entered.wait(), 2)
            task = next(iter(server._connections))
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            service.fetch.side_effect = None
            service.fetch.return_value = _http()
            reader, second_writer = await _open(server, _get("https://example.com/two"))
            writers.append(second_writer)
            raw = await asyncio.wait_for(_read_all(reader), 2)
        finally:
            for writer in writers:
                await _close_writer(writer)
            await server.aclose()

        cancelled_id = _ids(records, _FINISH)[0]
        self.assertRegex(cancelled_id, _HEX32)
        next_id = _request_id(raw)
        self.assertRegex(next_id, _HEX32)
        self.assertNotEqual(next_id, cancelled_id)
        self.assertEqual(_ids(records, _FINISH), [cancelled_id, next_id])


class HandshakeTracingTests(IsolatedAsyncioTestCase):
    """The 200 CONNECT head carries the connection id; this is a helper check, not live TLS."""

    async def test_200_handshake_head_carries_the_connection_id(self) -> None:
        writer = Mock(spec=asyncio.StreamWriter)
        writer.drain = AsyncMock()
        connection = h11.Connection(h11.SERVER)
        connection.receive_data(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
        connection.next_event()
        request_id = "0123456789abcdef0123456789abcdef"

        await ProxyServer(Mock(spec=Service))._handshake(writer, connection, request_id)

        written = b"".join(call.args[0] for call in writer.write.call_args_list)
        response = _parse_connect_head(written)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(_values(response, "x-request-id"), [request_id.encode("ascii")])
        self.assertNotIn(b"Content-Length", written)
