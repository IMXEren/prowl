"""Loopback socket tests for the plain-HTTP proxy listener.

Each test binds a real :class:`ProxyServer` on ``127.0.0.1`` and drives it over a real asyncio
socket. The service is a complete ``Mock(spec=Service)`` whose ``fetch`` is an ``AsyncMock``
returning real :class:`FetchResult` DTOs. Nothing here starts the real service or backend, launches
a browser, or leaves loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import h11

from prowl.browser.proxy import server as proxy_server
from prowl.browser.proxy.server import ProxyServer
from prowl.service.app import Service
from prowl.service.backend import FetchResult
from prowl.service.protocol import (
    AUTO_MODE,
    CMD_REQUEST_GET,
    CMD_REQUEST_POST,
    HTTP_MODE,
    FetchCommand,
    ProxySelection,
)
from prowl.service.sessions import ISOLATED_MODE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_GET_A = b"GET https://example.com/a HTTP/1.1\r\nHost: example.com\r\n\r\n"
_GET_B = b"GET https://example.com/b HTTP/1.1\r\nHost: example.com\r\n\r\n"


def _http_result(
    *,
    body_bytes: bytes = b"body",
    header_items: tuple[tuple[str, str], ...] = (("Content-Type", "text/plain"),),
) -> FetchResult:
    """A real HTTP-mode fetch result carrying exact origin bytes and repeated fields."""
    return FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={},
        response="",
        cookies=[],
        user_agent="ua",
        mode=HTTP_MODE,
        body_bytes=body_bytes,
        header_items=header_items,
    )


def _service(result: FetchResult | None = None) -> Mock:
    """A complete ``Service`` double whose ``fetch`` returns *result*."""
    service = Mock(spec=Service)
    service.fetch = AsyncMock(return_value=_http_result() if result is None else result)
    return service


def _post(target: str, payload: bytes, extra: bytes = b"") -> bytes:
    """An absolute-form POST with an exact Content-Length and extra fields."""
    head = b"POST " + target.encode() + b" HTTP/1.1\r\nHost: example.com\r\n"
    head += b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
    return head + extra + b"\r\n" + payload


async def _read_all(reader: asyncio.StreamReader, timeout: float = 5.0) -> bytes:
    """Read a connection to EOF under a bound, proving the server closed it."""
    chunks: list[bytes] = []
    async with asyncio.timeout(timeout):
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _parse(raw: bytes, target: str = "https://example.com/a") -> tuple[h11.Response, bytes]:
    """Parse one response and its body from raw wire bytes with an h11 client."""
    client = h11.Connection(h11.CLIENT)
    client.send(h11.Request(method="GET", target=target, headers=[("Host", "example.com")]))
    client.receive_data(raw)
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


def _values(response: h11.Response, name: str) -> list[bytes]:
    """Every value of one response field, preserving order and duplicates."""
    return [value for field, value in response.headers.raw_items() if field.lower() == name.lower().encode("ascii")]


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    """Close a client transport, consuming an expected reset."""
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()


class GetWireTests(IsolatedAsyncioTestCase):
    """A GET returns the exact origin bytes under the auto routing the dispatch seam chose."""

    async def test_get_returns_exact_binary_body_and_auto_command(self) -> None:
        payload = b"\x00\xff\xfeok"
        items = (("Set-Cookie", "a=1; Path=/"), ("Set-Cookie", "b=2; Path=/"))
        service = _service(_http_result(body_bytes=payload, header_items=items))
        server = ProxyServer(service)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            try:
                writer.write(b"GET https://example.com/secret?q=1 HTTP/1.1\r\nHost: example.com\r\n\r\n")
                await writer.drain()
                raw = await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        response, body = _parse(raw, target="https://example.com/secret?q=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, payload)
        self.assertEqual(_values(response, "set-cookie"), [b"a=1; Path=/", b"b=2; Path=/"])
        self.assertEqual(_values(response, "x-prowl-representation"), [b"origin"])
        self.assertEqual(_values(response, "connection"), [b"close"])

        command = service.fetch.await_args.args[0]
        self.assertIsInstance(command, FetchCommand)
        self.assertEqual(command.cmd, CMD_REQUEST_GET)
        self.assertEqual(command.mode, AUTO_MODE)
        self.assertEqual(command.url, "https://example.com/secret?q=1")
        self.assertIsNone(service.fetch.await_args.kwargs["body_bytes"])


class PostWireTests(IsolatedAsyncioTestCase):
    """A POST forwards its exact raw bytes down the explicit HTTP path, once."""

    async def test_post_forwards_exact_non_utf8_bytes_and_uses_http_mode(self) -> None:
        payload = b"\x00\xff body\x1b\x80"

        for name, body in (("non-utf8", payload), ("empty", b"")):
            with self.subTest(name=name):
                service = _service(_http_result(body_bytes=b"ok"))
                server = ProxyServer(service)
                await server.start()
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                    try:
                        writer.write(_post("https://example.com/submit", body))
                        await writer.drain()
                        raw = await _read_all(reader)
                    finally:
                        await _close_writer(writer)
                finally:
                    await server.aclose()

                response, response_body = _parse(raw, target="https://example.com/submit")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response_body, b"ok")

                service.fetch.assert_awaited_once()
                command = service.fetch.await_args.args[0]
                self.assertEqual(command.cmd, CMD_REQUEST_POST)
                self.assertEqual(command.mode, HTTP_MODE)
                self.assertEqual(service.fetch.await_args.kwargs["body_bytes"], body)


class RejectedRequestTests(IsolatedAsyncioTestCase):
    """Local failures answer with a fixed generic body and never reach the service."""

    async def test_connect_malformed_and_encoded_bodies_are_generic(self) -> None:
        cases = {
            "connect": (b"CONNECT secret.example:443 HTTP/1.1\r\nHost: secret.example:443\r\nX-S: SECRET\r\n\r\n", 501),
            "malformed": (b"GET https://example.com/ HTTP/1.1\r\nX-S: SECRET\r\nno-colon-here\r\n\r\n", 400),
            "encoded": (_post("https://example.com/", b"abc", extra=b"X-S: SECRET\r\nContent-Encoding: gzip\r\n"), 415),
        }

        for name, (request_bytes, status) in cases.items():
            with self.subTest(name=name):
                service = _service()
                server = ProxyServer(service)
                await server.start()
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
                    try:
                        writer.write(request_bytes)
                        await writer.drain()
                        raw = await _read_all(reader)
                    finally:
                        await _close_writer(writer)
                finally:
                    await server.aclose()

                response, body = _parse(raw)
                self.assertEqual(response.status_code, status)
                self.assertEqual(body, b"proxy request failed")
                self.assertNotIn(b"SECRET", raw)
                self.assertEqual(_values(response, "content-type"), [b"text/plain; charset=utf-8"])
                service.fetch.assert_not_awaited()


class PipeliningTests(IsolatedAsyncioTestCase):
    """Only the first request on a connection is served; the connection is then closed."""

    async def test_pipelined_second_request_is_never_executed(self) -> None:
        service = _service()
        server = ProxyServer(service)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            try:
                writer.write(_GET_A + _GET_B)
                await writer.drain()
                raw = await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        response, _ = _parse(raw)
        self.assertEqual(response.status_code, 200)
        service.fetch.assert_awaited_once()
        self.assertEqual(service.fetch.await_args.args[0].url, "https://example.com/a")


class TrustedBindingTests(IsolatedAsyncioTestCase):
    """The listener's constructor bindings reach the dispatch verbatim."""

    async def test_constructor_bindings_reach_the_command(self) -> None:
        service = _service()
        selection = ProxySelection(name="warp")
        server = ProxyServer(service, session="s", session_mode=ISOLATED_MODE, proxy=selection)
        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
            try:
                writer.write(_GET_A)
                await writer.drain()
                await _read_all(reader)
            finally:
                await _close_writer(writer)
        finally:
            await server.aclose()

        command = service.fetch.await_args.args[0]
        self.assertEqual(command.session, "s")
        self.assertIs(command.session_mode, ISOLATED_MODE)
        self.assertIs(command.proxy, selection)


class CloseTests(IsolatedAsyncioTestCase):
    """Closing cancels in-flight work, drains connections, and is idempotent."""

    async def test_close_cancels_inflight_fetch_and_drains_everything(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fetch(*_args: object, **_kwargs: object) -> FetchResult:
            entered.set()
            await release.wait()
            return _http_result()

        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=fetch)
        server = ProxyServer(service)
        await server.start()
        port = server.port
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_GET_A)
            await writer.drain()
            await asyncio.wait_for(entered.wait(), 2)

            await server.aclose()

            self.assertEqual(await asyncio.wait_for(_read_all(reader), 2), b"")
            self.assertEqual(len(server._connections), 0)
            with self.assertRaises(OSError):
                await asyncio.open_connection("127.0.0.1", port)
        finally:
            release.set()
            await _close_writer(writer)

    async def test_start_and_close_are_idempotent_and_fenced(self) -> None:
        server = ProxyServer(_service())
        await server.start()
        port = server.port
        self.assertGreater(port, 0)

        await server.start()
        self.assertEqual(server.port, port)

        await asyncio.gather(server.aclose(), server.aclose())

        with self.assertRaises(OSError):
            await asyncio.open_connection("127.0.0.1", port)
        with self.assertRaises(RuntimeError):
            await server.start()

    async def test_cancelled_close_waiter_joins_the_owned_cleanup(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fetch(*_args: object, **_kwargs: object) -> FetchResult:
            entered.set()
            await release.wait()
            return _http_result()

        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=fetch)
        server = ProxyServer(service)
        await server.start()
        port = server.port
        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            writer.write(_GET_A)
            await writer.drain()
            await asyncio.wait_for(entered.wait(), 2)

            closing = asyncio.Event()
            original_close = server._close

            async def controlled_close() -> None:
                closing.set()
                await original_close()

            with patch.object(server, "_close", controlled_close):
                waiter = asyncio.create_task(server.aclose())
                await asyncio.wait_for(closing.wait(), 2)
            owned = server._close_task
            self.assertIsNotNone(owned)
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter

            # A later caller joins the cancelled waiter's owned cleanup and finishes it.
            await asyncio.wait_for(server.aclose(), 15)

            self.assertIs(server._close_task, owned)
            self.assertEqual(len(server._connections), 0)
            with self.assertRaises(OSError):
                await asyncio.open_connection("127.0.0.1", port)
        finally:
            release.set()
            await _close_writer(writer)

    async def test_close_while_binding_does_not_leak_the_socket(self) -> None:
        real_start_server = asyncio.start_server
        binding_started = asyncio.Event()
        release_binding = asyncio.Event()
        bound_port = -1

        async def controlled(
            callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None] | None],
            host: str,
            port: int,
            *,
            limit: int,
        ) -> asyncio.Server:
            nonlocal bound_port
            server = await real_start_server(callback, host, port, limit=limit)
            bound_port = server.sockets[0].getsockname()[1]
            binding_started.set()
            await release_binding.wait()
            return server

        server = ProxyServer(_service())
        with patch.object(asyncio, "start_server", controlled):
            start_task = asyncio.create_task(server.start())
            await asyncio.wait_for(binding_started.wait(), 2)
            close_task = asyncio.create_task(server.aclose())
            release_binding.set()
            await asyncio.wait_for(start_task, 2)
            await asyncio.wait_for(close_task, 2)

        with self.assertRaises(OSError):
            await asyncio.open_connection("127.0.0.1", bound_port)


class AdmissionCapTests(IsolatedAsyncioTestCase):
    """A connection above the cap is closed immediately without dispatching a fetch."""

    async def test_extra_connection_is_closed_without_fetch(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fetch(*_args: object, **_kwargs: object) -> FetchResult:
            entered.set()
            await release.wait()
            return _http_result()

        service = Mock(spec=Service)
        service.fetch = AsyncMock(side_effect=fetch)
        server = ProxyServer(service)
        with patch.object(proxy_server, "MAX_PROXY_CONNECTIONS", 1):
            await server.start()
            try:
                first_reader, first_writer = await asyncio.open_connection("127.0.0.1", server.port)
                try:
                    first_writer.write(_GET_A)
                    await first_writer.drain()
                    await asyncio.wait_for(entered.wait(), 2)

                    second_reader, second_writer = await asyncio.open_connection("127.0.0.1", server.port)
                    try:
                        second_writer.write(_GET_B)
                        await second_writer.drain()
                        self.assertEqual(await asyncio.wait_for(_read_all(second_reader), 2), b"")
                    finally:
                        await _close_writer(second_writer)

                    service.fetch.assert_awaited_once()

                    release.set()
                    await asyncio.wait_for(_read_all(first_reader), 2)
                    self.assertEqual(len(server._connections), 0)
                finally:
                    release.set()
                    await _close_writer(first_writer)
            finally:
                await server.aclose()


class StartupCancellationTests(IsolatedAsyncioTestCase):
    async def test_cancelled_start_waiter_does_not_abandon_binding(self) -> None:
        real_start = asyncio.start_server
        entered = asyncio.Event()
        release = asyncio.Event()
        bound_port = 0

        async def controlled(
            callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None] | None],
            host: str,
            port: int,
            *,
            limit: int,
        ) -> asyncio.Server:
            nonlocal bound_port
            native = await real_start(callback, host, port, limit=limit)
            bound_port = native.sockets[0].getsockname()[1]
            entered.set()
            await release.wait()
            return native

        server = ProxyServer(_service())
        with patch.object(asyncio, "start_server", controlled):
            waiter = asyncio.create_task(server.start())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
            finally:
                release.set()
                await asyncio.wait_for(server.aclose(), 2)

        with self.assertRaises(OSError):
            await asyncio.open_connection("127.0.0.1", bound_port)
