"""Focused loopback tests for the authenticated upstream proxy bridge.

Every test runs against an in-process fake upstream on 127.0.0.1; no real network, credentials,
installs, or commits are involved.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ssl
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from tests.test_service_proxy_certificates import _generate_authority

from prowl.browser.proxy.bridge import BrowserProxyBridge, BrowserProxyBridgeError
from prowl.browser.proxy.certificates import ProxyCertificates

_PLAIN_AUTH = "Basic " + base64.b64encode(b"user:pass").decode("ascii")


def _listen_port(bridge: BrowserProxyBridge) -> int:
    """Return the bound local port from a started bridge's credential-free URL."""
    assert bridge.listen_url is not None
    return int(bridge.listen_url.rsplit(":", 1)[1])


class _Upstream:
    """A tiny loopback stand-in for an authenticated forward proxy."""

    def __init__(self) -> None:
        self.heads: list[bytes] = []
        self.bodies: list[bytes] = []
        self.connect_status = 200
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=ssl_context)
        assert self.server.sockets
        self.port = int(self.server.sockets[0].getsockname()[1])

    async def aclose(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            writer.close()
            return
        self.heads.append(head)
        try:
            if head.startswith(b"CONNECT "):
                if self.connect_status != 200:
                    writer.write(b"HTTP/1.1 407 Forbidden\r\nContent-Length: 12\r\n\r\nprivate-body")
                    await writer.drain()
                    return
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
                while True:
                    data = await reader.read(65536)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            else:
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        self.bodies.append(await reader.readexactly(int(line.split(b":", 1)[1].strip())))
                body = b"upstream-body"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(body)).encode("ascii")
                    + b"\r\nConnection: close\r\n\r\n"
                    + body
                )
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()


class BridgeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.upstream = _Upstream()
        await self.upstream.start()
        self.addAsyncCleanup(self.upstream.aclose)
        self.bridge: BrowserProxyBridge | None = None

    async def _start_bridge(
        self, proxy_url: str | None = None, *, ssl_context: ssl.SSLContext | None = None
    ) -> BrowserProxyBridge:
        url = proxy_url or f"http://user:pass@127.0.0.1:{self.upstream.port}"
        bridge = BrowserProxyBridge(url, ssl_context=ssl_context)
        await bridge.start()
        self.addAsyncCleanup(bridge.aclose)
        self.bridge = bridge
        return bridge

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        assert self.bridge is not None
        return await asyncio.open_connection("127.0.0.1", _listen_port(self.bridge))

    async def _read_all(self, reader: asyncio.StreamReader) -> bytes:
        return await asyncio.wait_for(reader.read(), 5)

    async def test_absolute_http_get_is_forwarded_with_upstream_auth(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"GET http://example.test/path?q=1 HTTP/1.1\r\nHost: example.test\r\n\r\n")
        await writer.drain()
        raw = await self._read_all(reader)
        writer.close()
        self.assertTrue(raw.startswith(b"HTTP/1.1 200 "), raw[:40])
        self.assertIn(b"upstream-body", raw)
        head = self.upstream.heads[0]
        self.assertIn(b"GET http://example.test/path?q=1 HTTP/1.1", head)
        self.assertIn(_PLAIN_AUTH.encode("ascii"), head)
        self.assertIn(b"Connection: close", head)

    async def test_url_decoded_credentials_reach_upstream(self) -> None:
        await self._start_bridge(f"http://us%65r:p%3Dass@127.0.0.1:{self.upstream.port}")
        reader, writer = await self._connect()
        writer.write(b"GET http://example.test/ HTTP/1.1\r\nHost: example.test\r\n\r\n")
        await writer.drain()
        await self._read_all(reader)
        writer.close()
        expected = b"Basic " + base64.b64encode(b"user:p=ass")
        self.assertIn(expected, self.upstream.heads[0])

    async def test_connect_tunnel_echoes_bytes(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
        self.assertTrue(head.startswith(b"HTTP/1.1 200 "), head[:40])
        writer.write(b"ping")
        await writer.drain()
        self.assertEqual(await asyncio.wait_for(reader.readexactly(4), 5), b"ping")
        writer.close()

    async def test_post_body_is_forwarded_exactly(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        payload = b"\x00\xffraw\x00"
        writer.write(
            b"POST http://example.test/upload HTTP/1.1\r\nHost: example.test\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        self.assertTrue((await self._read_all(reader)).startswith(b"HTTP/1.1 200"))
        self.assertEqual(self.upstream.bodies, [payload])
        writer.close()

    async def test_upstream_connect_denial_does_not_leak_its_body(self) -> None:
        self.upstream.connect_status = 407
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
        await writer.drain()
        raw = await self._read_all(reader)
        self.assertTrue(raw.startswith(b"HTTP/1.1 502"))
        self.assertNotIn(b"private-body", raw)
        writer.close()

    async def test_close_cancels_an_active_tunnel(self) -> None:
        bridge = await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
        await writer.drain()
        self.assertTrue((await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200"))
        await asyncio.wait_for(bridge.aclose(), 5)
        self.assertIsNone(bridge.listen_url)
        self.assertEqual(await asyncio.wait_for(reader.read(), 5), b"")
        writer.close()

    async def test_client_proxy_authorization_is_rejected_and_close_cleans_up(self) -> None:
        await self._start_bridge()
        assert self.bridge is not None
        port = _listen_port(self.bridge)
        reader, writer = await self._connect()
        writer.write(
            b"GET http://example.test/ HTTP/1.1\r\nHost: example.test\r\nProxy-Authorization: Basic zzz\r\n\r\n"
        )
        await writer.drain()
        raw = await self._read_all(reader)
        writer.close()
        self.assertTrue(raw.startswith(b"HTTP/1.1 407 "), raw[:40])
        self.assertEqual(self.upstream.heads, [])
        await self.bridge.aclose()
        with self.assertRaises(OSError):
            await asyncio.open_connection("127.0.0.1", port)

    async def test_oversized_request_head_is_rejected(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"GET http://example.test/ HTTP/1.1\r\nX-Large: " + b"a" * 65536 + b"\r\n\r\n")
        await writer.drain()
        raw = await self._read_all(reader)
        self.assertTrue(raw.startswith(b"HTTP/1.1 431"), raw[:40])
        self.assertEqual(self.upstream.heads, [])
        writer.close()

    async def test_invalid_header_name_is_rejected(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(b"GET http://example.test/ HTTP/1.1\r\nBad@Name: x\r\n\r\n")
        await writer.drain()
        raw = await self._read_all(reader)
        writer.close()
        self.assertTrue(raw.startswith(b"HTTP/1.1 400 "), raw[:40])
        self.assertEqual(self.upstream.heads, [])

    async def test_duplicate_content_length_is_rejected(self) -> None:
        await self._start_bridge()
        reader, writer = await self._connect()
        writer.write(
            b"POST http://example.test/ HTTP/1.1\r\nHost: example.test\r\n"
            b"Content-Length: 1\r\nContent-Length: 1\r\n\r\n"
        )
        await writer.drain()
        raw = await self._read_all(reader)
        writer.close()
        self.assertTrue(raw.startswith(b"HTTP/1.1 400 "), raw[:40])
        self.assertEqual(self.upstream.heads, [])

    async def test_proxy_url_with_path_query_or_fragment_is_rejected(self) -> None:
        for url in (
            "http://user:pass@127.0.0.1:8080/path",
            "http://user:pass@127.0.0.1:8080/?q=1",
            "http://user:pass@127.0.0.1:8080#frag",
            "http://user:pass@127.0.0.1:8080/",
        ):
            with self.subTest(url=url), self.assertRaises(BrowserProxyBridgeError):
                BrowserProxyBridge(url)

    async def test_https_upstream_requires_verified_tls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ca_cert, ca_key = _generate_authority(Path(temp))
            upstream = _Upstream()
            await upstream.start(ssl_context=ProxyCertificates(ca_cert, ca_key).context_for("127.0.0.1"))
            self.addAsyncCleanup(upstream.aclose)
            upstream_url = f"https://user:pass@127.0.0.1:{upstream.port}"
            trusted = await self._start_bridge(
                upstream_url, ssl_context=ssl.create_default_context(cafile=str(ca_cert))
            )
            reader, writer = await self._connect()
            writer.write(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
            await writer.drain()
            self.assertTrue((await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)).startswith(b"HTTP/1.1 200"))
            writer.write(b"test")
            await writer.drain()
            self.assertEqual(await asyncio.wait_for(reader.readexactly(4), 5), b"test")
            writer.close()
            self.assertIn(_PLAIN_AUTH.encode(), upstream.heads[0])
            await trusted.aclose()

            untrusted = BrowserProxyBridge(upstream_url)
            await untrusted.start()
            self.addAsyncCleanup(untrusted.aclose)
            self.bridge = untrusted
            reader, writer = await self._connect()
            writer.write(b"CONNECT example.test:443 HTTP/1.1\r\nHost: example.test:443\r\n\r\n")
            await writer.drain()
            self.assertTrue((await self._read_all(reader)).startswith(b"HTTP/1.1 502"))
            writer.close()
            self.assertEqual(len(upstream.heads), 1)

    async def test_repr_and_listen_url_never_expose_credentials(self) -> None:
        bridge = await self._start_bridge()
        self.assertNotIn("user", repr(bridge))
        self.assertNotIn("pass", repr(bridge))
        self.assertEqual(bridge.listen_url, f"http://127.0.0.1:{_listen_port(bridge)}")
