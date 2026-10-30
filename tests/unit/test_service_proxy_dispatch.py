"""Adapter tests: a parsed proxy request runs through the one service fetch path and projects back.

Every test builds real :class:`ProxyRequest` and :class:`FetchResult` DTOs and drives the adapter
against a complete ``Mock(spec=Service)`` whose ``fetch`` is an ``AsyncMock``. Nothing here launches
a browser, opens a socket, or touches native cookie state; the mock only proves the adapter reuses
the one fetch seam and the exact binding, headers, and body it forwards. It is not evidence that a
live fetch admits a lease or applies a native cookie.
"""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from prowl.browser.proxy.cookies import CookieHeaderError
from prowl.browser.proxy.dispatch import dispatch_request
from prowl.browser.proxy.http1 import ProxyProtocolError, ProxyRequest
from prowl.service.app import Service
from prowl.service.backend import FetchResult
from prowl.service.protocol import (
    AUTO_MODE,
    BROWSER_MODE,
    CMD_REQUEST_GET,
    CMD_REQUEST_POST,
    DEFAULT_TIMEOUT_MS,
    HTTP_MODE,
    FetchCommand,
    ProxySelection,
)
from prowl.service.sessions import ISOLATED_MODE


def _request(
    method: str = "GET",
    target: str = "https://example.com/",
    headers: tuple[tuple[str, str], ...] = (("Host", "example.com"),),
    body: bytes = b"",
) -> ProxyRequest:
    return ProxyRequest(method=method, target=target, headers=headers, body=body)


def _http_result(
    *,
    status_code: int = 200,
    body_bytes: bytes = b"body",
    header_items: tuple[tuple[str, str], ...] = (("Content-Type", "text/plain"),),
) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=status_code,
        headers={},
        response="",
        cookies=[],
        user_agent="ua",
        mode=HTTP_MODE,
        body_bytes=body_bytes,
        header_items=header_items,
    )


def _rendered_result(
    *,
    response: str = "<html>ok</html>",
    headers: dict[str, str] | None = None,
) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=200,
        headers={} if headers is None else headers,
        response=response,
        cookies=[],
        user_agent="ua",
        mode=BROWSER_MODE,
    )


def _service(result: FetchResult) -> Mock:
    """A complete ``Service`` double whose ``fetch`` yields *result*."""
    service = Mock(spec=Service)
    service.fetch = AsyncMock(return_value=result)
    return service


def _dispatched(service: Mock) -> FetchCommand:
    """Return the single command the adapter handed to ``service.fetch``."""
    service.fetch.assert_awaited_once()
    command = service.fetch.await_args.args[0]
    assert isinstance(command, FetchCommand)
    return command


def _kwargs(service: Mock) -> dict[str, object]:
    return service.fetch.await_args.kwargs


class GetDispatchTests(IsolatedAsyncioTestCase):
    """A GET runs the auto mode, origin-scoped, with no raw body."""

    async def test_get_is_auto_origin_scoped_and_bodyless(self) -> None:
        service = _service(_http_result())
        request = _request(
            headers=(
                ("Host", "example.com"),
                ("User-Agent", "native"),
                ("Authorization", "Bearer sekret"),
                ("X-Trace", "abc"),
            )
        )

        response = await dispatch_request(service, request)

        command = _dispatched(service)
        self.assertEqual(command.cmd, CMD_REQUEST_GET)
        self.assertEqual(command.url, "https://example.com/")
        self.assertEqual(command.timeout_ms, DEFAULT_TIMEOUT_MS)
        self.assertEqual(command.mode, AUTO_MODE)
        self.assertEqual(command.header_scope, "origin")
        self.assertEqual(command.headers, {"authorization": "Bearer sekret", "x-trace": "abc"})
        self.assertIsNone(_kwargs(service)["body_bytes"])
        self.assertIsNone(_kwargs(service)["cookie_header"])
        self.assertEqual(response.status, 200)

    async def test_raw_cookie_travels_separately_and_duplicate_names_survive(self) -> None:
        service = _service(_http_result())
        request = _request(
            headers=(
                ("Host", "example.com"),
                ("Cookie", "a=1"),
                ("Cookie", "a=2; b=3"),
                ("Authorization", "Bearer sekret"),
            )
        )

        await dispatch_request(service, request)

        command = _dispatched(service)
        self.assertNotIn("cookie", command.headers)
        self.assertEqual(command.headers["authorization"], "Bearer sekret")
        self.assertEqual(_kwargs(service)["cookie_header"], "a=1; a=2; b=3")


class PostDispatchTests(IsolatedAsyncioTestCase):
    """A POST runs the explicit HTTP path and forwards its exact raw body bytes."""

    async def test_post_forwards_exact_non_utf8_body_and_http_mode(self) -> None:
        body = b"\x00\xff raw\x1b\x80"
        service = _service(_http_result(body_bytes=b"ok"))
        request = _request(
            method="POST",
            target="https://example.com/post",
            headers=(("Host", "example.com"), ("Content-Type", "application/octet-stream")),
            body=body,
        )

        await dispatch_request(service, request)

        command = _dispatched(service)
        self.assertEqual(command.cmd, CMD_REQUEST_POST)
        self.assertEqual(command.mode, HTTP_MODE)
        self.assertEqual(command.method, "POST")
        self.assertIs(_kwargs(service)["body_bytes"], body)

    async def test_empty_post_body_is_still_sent_as_bytes(self) -> None:
        service = _service(_http_result(body_bytes=b""))
        request = _request(
            method="POST",
            target="https://example.com/post",
            headers=(("Host", "example.com"),),
            body=b"",
        )

        await dispatch_request(service, request)

        _dispatched(service)
        self.assertEqual(_kwargs(service)["body_bytes"], b"")
        self.assertIsNotNone(_kwargs(service)["body_bytes"])


class ProjectionTests(IsolatedAsyncioTestCase):
    """The projected response keeps origin bytes or marks a rendered fallback."""

    async def test_http_result_bytes_and_repeated_set_cookie_are_projected(self) -> None:
        payload = b"\x00\xff\xfeok"
        items = (("Set-Cookie", "a=1; Path=/"), ("Set-Cookie", "b=2; Path=/"))
        service = _service(_http_result(body_bytes=payload, header_items=items))

        response = await dispatch_request(service, _request())

        self.assertIs(response.body, payload)
        self.assertEqual([v for n, v in response.headers if n.lower() == "set-cookie"], ["a=1; Path=/", "b=2; Path=/"])
        self.assertEqual([v for n, v in response.headers if n.lower() == "x-prowl-representation"], ["origin"])

    async def test_rendered_fallback_is_marked_rendered_utf8_html(self) -> None:
        service = _service(_rendered_result(response="<html>caf\u00e9</html>"))

        response = await dispatch_request(service, _request())

        self.assertEqual(response.body, "<html>caf\u00e9</html>".encode())
        self.assertEqual([v for n, v in response.headers if n.lower() == "x-prowl-representation"], ["rendered"])
        self.assertEqual([v for n, v in response.headers if n.lower() == "content-type"], ["text/html; charset=utf-8"])


class BindingTests(IsolatedAsyncioTestCase):
    """The listener's trusted bindings are carried verbatim; caller headers never choose them."""

    async def test_trusted_operator_bindings_are_carried_exactly(self) -> None:
        service = _service(_http_result())
        selection = ProxySelection(name="warp")

        await dispatch_request(
            service,
            _request(),
            session="s",
            session_mode=ISOLATED_MODE,
            proxy=selection,
        )

        command = _dispatched(service)
        self.assertEqual(command.session, "s")
        self.assertIs(command.session_mode, ISOLATED_MODE)
        self.assertIs(command.proxy, selection)

    async def test_omitted_bindings_stay_unset(self) -> None:
        service = _service(_http_result())

        await dispatch_request(service, _request())

        command = _dispatched(service)
        self.assertIsNone(command.session)
        self.assertIsNone(command.session_mode)
        self.assertIsNone(command.proxy)

    async def test_untrusted_headers_cannot_choose_the_binding(self) -> None:
        service = _service(_http_result())
        request = _request(
            headers=(
                ("Host", "example.com"),
                ("X-Prowl-Session", "attacker"),
                ("X-Prowl-Mode", "isolated"),
                ("Proxy", "http://attacker:8080"),
            )
        )

        await dispatch_request(service, request)

        command = _dispatched(service)
        self.assertIsNone(command.session)
        self.assertIsNone(command.session_mode)
        self.assertIsNone(command.proxy)
        self.assertEqual(command.headers["x-prowl-session"], "attacker")


class TunnelTests(IsolatedAsyncioTestCase):
    """A tunneled target is bound to the connected authority before the service runs."""

    async def test_tunneled_origin_form_binds_to_the_connected_authority(self) -> None:
        service = _service(_http_result())
        request = _request(target="/path?q=1", headers=(("Host", "tunnel.example"),))

        await dispatch_request(service, request, tunnel_authority="tunnel.example:443")

        self.assertEqual(_dispatched(service).url, "https://tunnel.example:443/path?q=1")

    async def test_cross_host_target_is_rejected_before_the_service_call(self) -> None:
        service = _service(_http_result())
        request = _request(target="https://other.example/", headers=(("Host", "tunnel.example"),))

        with self.assertRaises(ProxyProtocolError) as ctx:
            await dispatch_request(service, request, tunnel_authority="tunnel.example:443")

        self.assertEqual(ctx.exception.status, 400)
        service.fetch.assert_not_awaited()


class RejectionTests(IsolatedAsyncioTestCase):
    """Every preparation failure is raised before the service is ever called."""

    async def test_malformed_target_is_rejected_before_the_service_call(self) -> None:
        service = _service(_http_result())

        with self.assertRaises(ProxyProtocolError) as ctx:
            await dispatch_request(service, _request(target="https://example.com/#frag"))

        self.assertEqual(ctx.exception.status, 400)
        service.fetch.assert_not_awaited()

    async def test_connect_and_encoded_body_are_rejected_before_the_service_call(self) -> None:
        service = _service(_http_result())

        with self.assertRaises(ProxyProtocolError) as connect_ctx:
            await dispatch_request(service, _request(method="CONNECT", target="example.com:443", headers=()))
        self.assertEqual(connect_ctx.exception.status, 501)

        with self.assertRaises(ProxyProtocolError) as encoded_ctx:
            await dispatch_request(
                service,
                _request(
                    method="POST",
                    headers=(("Host", "example.com"), ("Content-Encoding", "gzip")),
                    body=b"x",
                ),
            )
        self.assertEqual(encoded_ctx.exception.status, 415)
        service.fetch.assert_not_awaited()

    async def test_direct_get_with_a_body_is_rejected(self) -> None:
        service = _service(_http_result())

        with self.assertRaises(ProxyProtocolError) as ctx:
            await dispatch_request(service, _request(method="GET", body=b"smuggled"))

        self.assertEqual(ctx.exception.status, 400)
        service.fetch.assert_not_awaited()


class PropagationTests(IsolatedAsyncioTestCase):
    """The adapter catches nothing, so fetch errors and cancellation surface unchanged."""

    async def test_fetch_errors_propagate_unchanged(self) -> None:
        service = _service(_http_result())
        for error in (CookieHeaderError(), TimeoutError(), asyncio.CancelledError()):
            service.fetch = AsyncMock(side_effect=error)
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                await dispatch_request(service, _request())
