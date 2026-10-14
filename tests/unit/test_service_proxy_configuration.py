"""App wiring for the optional integrated forward proxy listener.

These tests cover the ``forward_proxy_port``/``proxy_ca_cert``/``proxy_ca_key`` config seam, the
disabled startup path that touches no certificate or listener, the enabled path that binds a real
loopback listener behind ``create_app``, the ordering that loads the operator authority before the
backend starts, and the one owned cleanup that closes the proxy before the service. No browser
process, no real ``BrowserBackend``, and no native facade is ever involved.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

import h11
from aiohttp.test_utils import TestClient, TestServer

from prowl.browser.proxy.certificates import ProxyCertificateError, ProxyCertificates
from prowl.browser.proxy.server import ProxyServer
from prowl.service import app as app_module
from prowl.service.app import ServiceConfig, create_app
from prowl.service.backend import Backend, FetchResult
from prowl.service.protocol import HTTP_MODE

if TYPE_CHECKING:
    from aiohttp import web

_DISABLED: dict[str, str] = {
    app_module.FORWARD_PROXY_PORT_ENV: "",
    app_module.PROXY_CA_CERT_ENV: "",
    app_module.PROXY_CA_KEY_ENV: "",
}

_FETCH_BODY = b"\x00\xffok"
_FETCH_ITEMS = (("Content-Type", "text/plain"), ("Set-Cookie", "a=1; Path=/"))
_GET = b"GET https://example.com/secret?q=1 HTTP/1.1\r\nHost: example.com\r\n\r\n"
_CONNECT = b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n"


class _ProxyCleanupError(Exception):
    """A fixed failure from the listener's close."""


class _BackendCleanupError(Exception):
    """A fixed failure from the service's close."""


def _http_result() -> FetchResult:
    """A real HTTP-mode fetch result carrying exact origin bytes and repeated fields."""
    return FetchResult(
        url="https://example.com/secret?q=1",
        status_code=200,
        headers={},
        response="",
        cookies=[],
        user_agent="ua",
        mode=HTTP_MODE,
        body_bytes=_FETCH_BODY,
        header_items=_FETCH_ITEMS,
    )


def _backend() -> Mock:
    """A complete ``Backend`` double whose native work is entirely mocked."""
    backend = Mock(spec=Backend)
    backend.start = AsyncMock()
    backend.fetch = AsyncMock(return_value=_http_result())
    backend.aclose = AsyncMock()
    return backend


def _proxy_double() -> Mock:
    """A faithful ``ProxyServer`` double with controllable start and close."""
    proxy = Mock(spec=ProxyServer)
    proxy.start = AsyncMock()
    proxy.aclose = AsyncMock()
    return proxy


def _enabled_config() -> ServiceConfig:
    """The minimal enabled configuration: loopback host and an ephemeral port."""
    return ServiceConfig(max_concurrency=1, host="127.0.0.1", forward_proxy_port=0)


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


def _parse(raw: bytes, target: str = "https://example.com/secret?q=1") -> tuple[h11.Response, bytes]:
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


async def _proxy_exchange(port: int, request: bytes) -> tuple[h11.Response, bytes]:
    """Send *request* to the raw loopback proxy socket and parse its one response."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(request)
        await writer.drain()
        raw = await _read_all(reader)
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
    return _parse(raw)


class ForwardProxyConfigTests(TestCase):
    """The three new fields are appended, default to disabled, and reject bad combinations."""

    def test_defaults_and_appended_field_order(self) -> None:
        names = [f.name for f in dataclasses.fields(ServiceConfig)]
        self.assertEqual(names[-3:], ["forward_proxy_port", "proxy_ca_cert", "proxy_ca_key"])
        config = ServiceConfig()
        self.assertIsNone(config.forward_proxy_port)
        self.assertIsNone(config.proxy_ca_cert)
        self.assertIsNone(config.proxy_ca_key)

    def test_env_values_are_parsed_and_blank_values_stay_disabled(self) -> None:
        with patch.dict(os.environ, {**_DISABLED, app_module.FORWARD_PROXY_PORT_ENV: "   "}):
            disabled = ServiceConfig.from_env()
        self.assertIsNone(disabled.forward_proxy_port)
        self.assertIsNone(disabled.proxy_ca_cert)
        self.assertIsNone(disabled.proxy_ca_key)

        env = {
            **_DISABLED,
            app_module.FORWARD_PROXY_PORT_ENV: " 0 ",
            app_module.PROXY_CA_CERT_ENV: " ca.pem ",
            app_module.PROXY_CA_KEY_ENV: " ca.key ",
        }
        with patch.dict(os.environ, env):
            enabled = ServiceConfig.from_env()
        self.assertEqual(enabled.forward_proxy_port, 0)
        self.assertEqual(enabled.proxy_ca_cert, "ca.pem")
        self.assertEqual(enabled.proxy_ca_key, "ca.key")

    def test_invalid_values_are_rejected_without_echoing_the_input(self) -> None:
        for raw in ("abc", "-1", "65536"):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {**_DISABLED, app_module.FORWARD_PROXY_PORT_ENV: raw}),
                self.assertRaises(ValueError),
            ):
                ServiceConfig.from_env()

        cases: tuple[tuple[int | None, str | None, str | None], ...] = (
            (-1, None, None),
            (65536, None, None),
            (0, "ca.pem", None),
            (0, None, "ca.key"),
            (None, "ca.pem", "ca.key"),
        )
        for port, certificate, key in cases:
            with self.subTest(port=port, certificate=certificate, key=key), self.assertRaises(ValueError):
                ServiceConfig(forward_proxy_port=port, proxy_ca_cert=certificate, proxy_ca_key=key)

        with self.assertRaises(ValueError) as caught:
            ServiceConfig(proxy_ca_cert="/secret/ca.pem", proxy_ca_key="/secret/ca.key")
        self.assertNotIn("/secret", str(caught.exception))


class DisabledProxyStartupTests(IsolatedAsyncioTestCase):
    """A disabled proxy leaves the legacy startup and cleanup contracts untouched."""

    async def test_disabled_startup_touches_no_certificate_or_listener(self) -> None:
        backend = _backend()
        app = create_app(ServiceConfig(host="127.0.0.1"), backend)
        with (
            patch.object(app_module, "ProxyCertificates") as certificates,
            patch.object(app_module, "ProxyServer") as proxy_server,
        ):
            await app_module._on_startup(app)
            self.assertTrue(app[app_module._STATE_KEY].ready)
            self.assertIsNone(app.get(app_module._PROXY_KEY))
            backend.start.assert_awaited_once()
            certificates.assert_not_called()
            proxy_server.assert_not_called()
            await app_module._on_cleanup(app)

        self.assertFalse(app[app_module._STATE_KEY].ready)
        backend.aclose.assert_awaited_once()


class EnabledProxyStartupTests(IsolatedAsyncioTestCase):
    """An enabled proxy binds a real loopback listener that lives and dies with the app."""

    async def test_real_listener_forwards_one_request_and_closes_with_the_app(self) -> None:
        backend = _backend()
        app = create_app(_enabled_config(), backend)
        client = TestClient(TestServer(app))
        with patch.object(app_module, "ProxyCertificates") as certificates:
            await client.start_server()
            proxy = app[app_module._PROXY_KEY]
            port = proxy.port
            self.assertGreater(port, 0)
            try:
                response, body = await _proxy_exchange(port, _GET)
                connect, connect_body = await _proxy_exchange(port, _CONNECT)
            finally:
                await client.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, _FETCH_BODY)
        self.assertEqual(connect.status_code, 501)
        self.assertEqual(connect_body, b"proxy request failed")
        certificates.assert_not_called()
        backend.fetch.assert_awaited_once()
        self.assertIsNone(backend.fetch.await_args.args[0])
        self.assertEqual(backend.fetch.await_args.args[1].url, "https://example.com/secret?q=1")
        backend.aclose.assert_awaited_once()
        with self.assertRaises(OSError):
            await asyncio.open_connection("127.0.0.1", port)


class AuthorityStartupOrderTests(IsolatedAsyncioTestCase):
    """The operator authority is validated off the loop before any browser starts."""

    async def test_authority_is_loaded_before_the_backend_starts(self) -> None:
        order: list[str] = []
        calls: list[tuple[Path, Path]] = []
        certificates = Mock(spec=ProxyCertificates)

        def factory(ca_cert: Path, ca_key: Path) -> Mock:
            order.append("certificates")
            calls.append((ca_cert, ca_key))
            return certificates

        async def start() -> None:
            order.append("backend.start")

        backend = _backend()
        backend.start = AsyncMock(side_effect=start)
        config = ServiceConfig(
            host="127.0.0.1",
            forward_proxy_port=0,
            proxy_ca_cert="ca.pem",
            proxy_ca_key="ca.key",
        )
        app = create_app(config, backend)
        with patch.object(app_module, "ProxyCertificates", new=factory):
            await app_module._on_startup(app)
            try:
                self.assertEqual(order, ["certificates", "backend.start"])
                self.assertEqual(calls, [(Path("ca.pem"), Path("ca.key"))])
                self.assertGreater(app[app_module._PROXY_KEY].port, 0)
            finally:
                await app_module._on_cleanup(app)

    async def test_an_unusable_authority_starts_no_backend_and_no_listener(self) -> None:
        backend = _backend()
        boom = ProxyCertificateError("the proxy certificate authority is not usable")
        config = ServiceConfig(
            host="127.0.0.1",
            forward_proxy_port=0,
            proxy_ca_cert="ca.pem",
            proxy_ca_key="ca.key",
        )
        app = create_app(config, backend)
        with (
            patch.object(app_module, "ProxyCertificates", side_effect=boom),
            self.assertRaises(ProxyCertificateError) as caught,
        ):
            await app_module._on_startup(app)

        self.assertIs(caught.exception, boom)
        backend.start.assert_not_awaited()
        backend.aclose.assert_not_awaited()
        self.assertIsNone(app.get(app_module._PROXY_KEY))


class ProxyStartupFailureTests(IsolatedAsyncioTestCase):
    """A startup failure after ownership closes both owners and keeps the original error."""

    async def test_bind_failure_closes_both_owners_and_keeps_the_original_error(self) -> None:
        backend = _backend()
        proxy = _proxy_double()
        boom = RuntimeError("listener bind failed")
        proxy.start = AsyncMock(side_effect=boom)
        proxy.aclose = AsyncMock(side_effect=_ProxyCleanupError())
        app = create_app(_enabled_config(), backend)
        with (
            patch.object(app_module, "ProxyServer", return_value=proxy),
            self.assertRaises(RuntimeError) as caught,
        ):
            await app_module._on_startup(app)

        self.assertIs(caught.exception, boom)
        self.assertIs(app[app_module._PROXY_KEY], proxy)
        proxy.start.assert_awaited_once()
        proxy.aclose.assert_awaited_once()
        backend.aclose.assert_awaited_once()
        self.assertFalse(app[app_module._STATE_KEY].ready)


class ProxyCleanupTests(IsolatedAsyncioTestCase):
    """One owned cleanup drains the proxy first, then the service, for every waiter."""

    async def _started(self, proxy: Mock) -> tuple[web.Application, Mock]:
        backend = _backend()
        app = create_app(_enabled_config(), backend)
        with patch.object(app_module, "ProxyServer", return_value=proxy):
            await app_module._on_startup(app)
        return app, backend

    async def test_cleanup_drains_the_proxy_before_the_service_and_joins_waiters(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def close() -> None:
            entered.set()
            await release.wait()

        proxy = _proxy_double()
        proxy.aclose = AsyncMock(side_effect=close)
        app, backend = await self._started(proxy)

        first = asyncio.create_task(app_module._on_cleanup(app))
        await asyncio.wait_for(entered.wait(), 2)
        owned = app[app_module._STATE_KEY].cleanup_task
        self.assertIsNotNone(owned)
        self.assertEqual(backend.aclose.await_count, 0)

        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(app_module._on_cleanup(app))
        release.set()
        await asyncio.wait_for(second, 2)

        self.assertFalse(app[app_module._STATE_KEY].ready)
        self.assertIs(app[app_module._STATE_KEY].cleanup_task, owned)
        proxy.aclose.assert_awaited_once()
        self.assertEqual(backend.aclose.await_count, 1)

    async def test_a_cleanup_failure_still_closes_the_service_and_surfaces_first(self) -> None:
        proxy = _proxy_double()
        proxy.aclose = AsyncMock(side_effect=_ProxyCleanupError())
        app, backend = await self._started(proxy)
        backend.aclose = AsyncMock(side_effect=_BackendCleanupError())

        with self.assertRaises(_ProxyCleanupError):
            await app_module._on_cleanup(app)

        proxy.aclose.assert_awaited_once()
        backend.aclose.assert_awaited_once()
