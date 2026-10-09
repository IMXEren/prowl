"""Pure no-browser tests for the forward-proxy probe.

These tests never launch a browser or the service: they cover the ownership fence, the raw
HTTP/1 client helper against a loopback server, the verified CONNECT helper against a real proxy
server, and the startup/fence/cleanup ordering in the probe bootstrap. Real-browser verification
runs separately through the probe itself.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import ssl
import tempfile
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

import h11
from scripts import verify_forward_proxy as probe

from prowl.browser import Browser
from prowl.browser.proxy.certificates import ProxyCertificates
from prowl.browser.proxy.http1 import ProxyRequest
from prowl.browser.proxy.server import ProxyServer
from prowl.service.app import Service
from prowl.service.backend import BrowserBackend, FetchResult
from prowl.service.metrics import Metrics
from prowl.service.protocol import HTTP_MODE


class ProfileFenceTests(TestCase):
    """The fence must reject a foreign profile without touching native browser state."""

    def test_matching_profile_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile"
            probe._assert_profile_owned(profile, profile)

    def test_foreign_profile_is_rejected_without_native_work(self) -> None:
        with (
            tempfile.TemporaryDirectory() as declared,
            tempfile.TemporaryDirectory() as other,
            self.assertRaises(probe._ProfileFenceError),
        ):
            probe._assert_profile_owned(Path(other) / "webdata", Path(declared))
        self.assertFalse(Browser.is_running())


def _response_bytes(connection: h11.Connection, body: bytes) -> bytes:
    return (
        connection.send(
            h11.Response(
                status_code=200,
                headers=[
                    (b"Content-Type", b"text/plain; charset=utf-8"),
                    (b"X-Multi", b"one"),
                    (b"X-Multi", b"two"),
                    (b"Content-Length", str(len(body)).encode("ascii")),
                ],
            )
        )
        + connection.send(h11.Data(data=body))
        + connection.send(h11.EndOfMessage())
    )


class RequestHelperTests(IsolatedAsyncioTestCase):
    """The client helper must parse an exact body and preserve repeated headers."""

    async def test_request_returns_exact_body_and_repeated_headers(self) -> None:
        body = b"forward-proxy-probe-body"

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            connection = h11.Connection(h11.SERVER)
            try:
                while True:
                    data = await reader.read(4096)
                    if not data:
                        return
                    connection.receive_data(data)
                    event = connection.next_event()
                    if isinstance(event, h11.Request):
                        writer.write(_response_bytes(connection, body))
                        await writer.drain()
                        return
            finally:
                writer.close()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            response = await probe._request(port, "https://proxy-target.invalid/probe")
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, body)
        multi = [value for name, value in response.headers if name.lower() == "x-multi"]
        self.assertEqual(multi, ["one", "two"])


def _fetch_result(body: bytes) -> FetchResult:
    """A real HTTP-mode fetch result carrying exact origin bytes."""
    return FetchResult(
        url="https://origin.invalid/",
        status_code=200,
        headers={},
        response="",
        cookies=[],
        user_agent="ua",
        mode=HTTP_MODE,
        body_bytes=body,
        header_items=(("Content-Type", "text/html; charset=utf-8"),),
    )


class ConnectHelperTests(IsolatedAsyncioTestCase):
    """``_request`` must verify the tunnel CA, match the handshake id, and reject other authorities."""

    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ca_cert, self.ca_key = probe._write_proxy_ca(Path(self.directory.name))
        self.service = Mock(spec=Service)
        self.service.fetch = AsyncMock(return_value=_fetch_result(b"<html>origin</html>"))
        self.server = ProxyServer(self.service, certificates=ProxyCertificates(self.ca_cert, self.ca_key))
        await self.server.start()
        self.addAsyncCleanup(self.server.aclose)

    def _url(self) -> str:
        return f"https://127.0.0.1:{self.server.port}/"

    async def test_verified_tunnel_returns_origin_bytes_under_a_matching_id(self) -> None:
        response = await probe._request(
            self.server.port, self._url(), ca_cert=self.ca_cert, cookie_header="sid=from-tunnel"
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, b"<html>origin</html>")
        self.assertRegex(probe._header(response, "X-Request-ID") or "", r"^[0-9a-f]{32}$")
        self.service.fetch.assert_awaited_once()
        self.assertEqual(self.service.fetch.await_args.kwargs["cookie_header"], "sid=from-tunnel")

    async def test_wrong_authority_target_is_rejected_without_fetch(self) -> None:
        target = f"https://127.0.0.2:{self.server.port}/"
        response = await probe._request(
            self.server.port, self._url(), ca_cert=self.ca_cert, request=ProxyRequest("GET", target, (), b"")
        )
        self.assertEqual(response.status, 400)
        self.service.fetch.assert_not_awaited()

    async def test_binary_and_empty_post_preserve_bytes_through_verified_tls(self) -> None:
        for payload in (b"\x00\xff\x80raw\x00", b""):
            with self.subTest(payload=payload):
                self.service.fetch.reset_mock()
                response = await probe._request(
                    self.server.port, self._url(), ca_cert=self.ca_cert, request=ProxyRequest("POST", "/", (), payload)
                )
                self.assertEqual(response.status, 200)
                self.service.fetch.assert_awaited_once()
                command = self.service.fetch.await_args.args[0]
                self.assertEqual(command.mode, HTTP_MODE)
                self.assertEqual(command.method, "POST")
                self.assertEqual(command.headers["content-type"], "application/octet-stream")
                self.assertEqual(self.service.fetch.await_args.kwargs["body_bytes"], payload)

    async def test_foreign_ca_fails_the_verified_handshake(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            foreign_ca, _ = probe._write_proxy_ca(Path(other))
            with self.assertRaises(ssl.SSLCertVerificationError):
                await probe._request(self.server.port, self._url(), ca_cert=foreign_ca)
        self.service.fetch.assert_not_awaited()


class BootstrapOrderTests(TestCase):
    """The bootstrap must patch, start, fence, then request, and clean up inside the patches."""

    def test_startup_and_fence_run_before_the_first_request(self) -> None:
        source = inspect.getsource(probe._run_probe)
        self.assertLess(source.index("_offline_launch_patches"), source.index("_on_startup"))
        self.assertLess(source.index("_pinned_tls_patches"), source.index("_on_startup"))
        self.assertLess(source.index("_on_startup"), source.index("_assert_profile_owned"))
        self.assertLess(source.index("_assert_profile_owned"), source.index("await _request("))
        self.assertLess(source.index("_offline_launch_patches"), source.index("_on_cleanup"))
        self.assertLess(source.index("_pinned_tls_patches"), source.index("_on_cleanup"))


class BinaryOriginTests(IsolatedAsyncioTestCase):
    async def test_origin_records_binary_and_empty_post_once_without_decoding(self) -> None:
        observed: list[bytes] = []
        handler = probe._binary_handler(probe._handler_class(probe._SiteState()), observed)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def send(payload: bytes) -> bytes:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                connection.request("POST", "/post", body=payload)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                return response.read()
            finally:
                connection.close()

        try:
            for payload in (b"\x00\xff\x80raw\x00", b""):
                self.assertEqual(await asyncio.to_thread(send, payload), probe._ProbePage.INDEX.encode())
            self.assertEqual(observed, [b"\x00\xff\x80raw\x00", b""])
        finally:
            self.assertTrue(await asyncio.to_thread(probe._cleanup_servers, server))


class MediaOriginTests(IsolatedAsyncioTestCase):
    async def test_each_asset_request_is_observed_with_exact_type_and_body(self) -> None:
        observed: list[str] = []
        handler = probe._media_handler(probe._handler_class(probe._SiteState()), observed)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]

        def request_asset(path: str, expected_type: str) -> bytes:
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                connection.request("GET", path + "?case=fixture")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Content-Type"), expected_type)
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                return response.read()
            finally:
                connection.close()

        try:
            for path, (content_type, body) in probe._MEDIA_ASSETS.items():
                self.assertEqual(await asyncio.to_thread(request_asset, path, content_type), body)
            self.assertEqual(observed, [path + "?case=fixture" for path in probe._MEDIA_ASSETS])
        finally:
            self.assertTrue(await asyncio.to_thread(probe._cleanup_servers, server))


class VerificationProbeTests(IsolatedAsyncioTestCase):
    async def test_widget_checks_validate_the_full_payload_and_zero_tab_projection(self) -> None:
        observed: list[str] = []
        service = Mock(spec=Service)
        backend = Mock(spec=BrowserBackend)
        backend.metrics = Metrics()
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5y0AAAAASUVORK5CYII="
        )
        verification_value = "offline-widget-value"

        async def reply(command: dict[str, object]) -> tuple[int, dict[str, object]]:
            url = str(command["url"])
            solution: dict[str, object] = {"url": url}
            if "/widget" in url:
                observed.append(url.removeprefix("https://fixture.invalid"))
                solution.update(
                    turnstile_token=verification_value,
                    screenshot=base64.b64encode(png).decode("ascii"),
                    cookies=[{"name": "widgetdone", "value": "1"}],
                    response="" if command["returnOnlyCookies"] else "Verified locally",
                )
            return 200, {"status": "ok", "solution": solution}

        service.handle = AsyncMock(side_effect=reply)
        self.assertTrue(
            await probe._check_verification(service, backend, "https://fixture.invalid", probe._SiteState(), observed)
        )
        self.assertEqual(service.handle.await_count, 3)
        commands = [call.args[0] for call in service.handle.await_args_list]
        self.assertEqual([command["tabs_till_verify"] for command in commands], [1, 0, 0])
        self.assertTrue(commands[1]["returnOnlyCookies"])
