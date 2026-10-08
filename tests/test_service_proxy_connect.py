"""CONNECT wire tests with verified TLS and a mocked service."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import h11
from tests.test_service_proxy_certificates import _generate_authority
from tests.test_service_proxy_server import _http_result, _parse, _read_all, _values

from prowl.browser.proxy.certificates import ProxyCertificates
from prowl.browser.proxy.server import ProxyServer
from prowl.service.app import Service
from prowl.service.protocol import HTTP_MODE, ProxySelection


class ConnectTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ca_cert, self.ca_key = _generate_authority(Path(self.directory.name))
        self.service = Mock(spec=Service)
        self.service.fetch = AsyncMock(
            return_value=_http_result(
                body_bytes=b"\x00\xffbody", header_items=(("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"))
            )
        )
        self.selection = ProxySelection(name="test-egress")
        self.server = ProxyServer(
            self.service,
            certificates=ProxyCertificates(self.ca_cert, self.ca_key),
            session="test-session",
            proxy=self.selection,
        )
        await self.server.start()
        self.addAsyncCleanup(self.server.aclose)

    async def _plain(
        self, authority: str = "example.test:8443"
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
        reader, writer = await asyncio.open_connection("127.0.0.1", self.server.port)
        self.addAsyncCleanup(self._close, writer)
        writer.write(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode("ascii"))
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 3)
        client = h11.Connection(h11.CLIENT)
        client.send(h11.Request(method="CONNECT", target=authority, headers=[("Host", authority)]))
        client.receive_data(head)
        response = client.next_event()
        self.assertIsInstance(response, h11.Response)
        return reader, writer, head

    async def _tls(
        self, hostname: str = "example.test", ca_file: Path | None = None
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer, head = await self._plain()
        self.assertTrue(head.startswith(b"HTTP/1.1 200 "))
        self.assertNotIn(b"Content-Length", head)
        self.assertNotIn(b"Connection: close", head)
        context = ssl.create_default_context(cafile=str(self.ca_cert if ca_file is None else ca_file))
        context.set_alpn_protocols(["http/1.1"])
        await writer.start_tls(context, server_hostname=hostname, ssl_handshake_timeout=3)
        return reader, writer

    async def _close(self, writer: asyncio.StreamWriter) -> None:
        writer.close()
        with contextlib.suppress(OSError, TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), 3)

    async def test_verified_get_preserves_connected_authority_and_origin_bytes(self) -> None:
        reader, writer = await self._tls()
        writer.write(b"GET /path?q=%2F HTTP/1.1\r\nHost: example.test:8443\r\n\r\n")
        await writer.drain()
        response, body = _parse(await _read_all(reader), target="https://example.test:8443/path?q=%2F")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, b"\x00\xffbody")
        self.assertEqual(_values(response, "set-cookie"), [b"a=1", b"b=2"])
        self.service.fetch.assert_awaited_once()
        command = self.service.fetch.await_args.args[0]
        self.assertEqual(command.url, "https://example.test:8443/path?q=%2F")
        self.assertEqual(command.session, "test-session")
        self.assertIs(command.proxy, self.selection)

    async def test_encrypted_post_keeps_binary_and_empty_bodies(self) -> None:
        for body in (b"\x00\xff\x80", b""):
            with self.subTest(body=body):
                self.service.fetch.reset_mock()
                reader, writer = await self._tls()
                writer.write(
                    b"POST /submit HTTP/1.1\r\nHost: example.test:8443\r\nContent-Length: "
                    + str(len(body)).encode("ascii")
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
                response, _ = _parse(await _read_all(reader))
                self.assertEqual(response.status_code, 200)
                self.service.fetch.assert_awaited_once()
                self.assertEqual(self.service.fetch.await_args.args[0].mode, HTTP_MODE)
                self.assertEqual(self.service.fetch.await_args.kwargs["body_bytes"], body)

    async def test_encrypted_authority_violations_and_nested_connect_never_fetch(self) -> None:
        for request, status in (
            (b"GET https://foreign.test/ HTTP/1.1\r\nHost: foreign.test\r\n\r\n", 400),
            (b"GET https://example.test/ HTTP/1.1\r\nHost: example.test\r\n\r\n", 400),
            (b"GET http://example.test:8443/ HTTP/1.1\r\nHost: example.test:8443\r\n\r\n", 400),
            (b"CONNECT foreign.test:443 HTTP/1.1\r\nHost: foreign.test:443\r\n\r\n", 501),
            (b"GET / HTTP/1.1\r\ninvalid-header\r\n\r\n", 400),
        ):
            with self.subTest(request=request):
                reader, writer = await self._tls()
                writer.write(request)
                await writer.drain()
                response, body = _parse(await _read_all(reader))
                self.assertEqual(response.status_code, status)
                self.assertEqual(body, b"proxy request failed")
        self.service.fetch.assert_not_awaited()

    async def test_foreign_ca_and_wrong_hostname_fail_verified_handshake(self) -> None:
        foreign_ca, _ = _generate_authority(Path(self.directory.name), name="foreign")
        for hostname, ca in (("wrong.test", self.ca_cert), ("example.test", foreign_ca)):
            with self.subTest(hostname=hostname), self.assertRaises(ssl.SSLCertVerificationError):
                await self._tls(hostname, ca)
        self.service.fetch.assert_not_awaited()

    async def test_encrypted_pipeline_only_executes_first_request(self) -> None:
        reader, writer = await self._tls()
        request = b"GET /first HTTP/1.1\r\nHost: example.test:8443\r\n\r\n"
        writer.write(request + request.replace(b"/first", b"/second"))
        await writer.drain()
        response, _ = _parse(await _read_all(reader))
        self.assertEqual(response.status_code, 200)
        self.service.fetch.assert_awaited_once()
        self.assertTrue(self.service.fetch.await_args.args[0].url.endswith("/first"))

    async def test_shutdown_drains_pending_tls_handshake(self) -> None:
        reader, _writer, head = await self._plain()
        self.assertTrue(head.startswith(b"HTTP/1.1 200 "))
        await asyncio.wait_for(self.server.aclose(), 3)
        self.assertEqual(await asyncio.wait_for(reader.read(), 3), b"")
        self.assertFalse(self.server._connections)
        self.service.fetch.assert_not_awaited()

    async def test_shutdown_cancels_active_encrypted_fetch(self) -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def fetch(*_args: object, **_kwargs: object) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.service.fetch.side_effect = fetch
        reader, writer = await self._tls()
        writer.write(b"GET / HTTP/1.1\r\nHost: example.test:8443\r\n\r\n")
        await writer.drain()
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.wait_for(self.server.aclose(), 3)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(await asyncio.wait_for(reader.read(), 3), b"")
        self.assertFalse(self.server._connections)

    async def test_invalid_authority_and_certificate_failure_reject_before_200(self) -> None:
        reader, _writer, head = await self._plain("example.test")
        self.assertTrue(head.startswith(b"HTTP/1.1 400 "))
        await asyncio.wait_for(reader.read(), 3)
        certificates = Mock(spec=ProxyCertificates)
        certificates.context_for.side_effect = RuntimeError("SECRET")
        self.server._certificates = certificates
        reader, _writer, head = await self._plain()
        self.assertTrue(head.startswith(b"HTTP/1.1 502 "))
        self.assertNotIn(b"SECRET", head + await asyncio.wait_for(reader.read(), 3))
        self.service.fetch.assert_not_awaited()
