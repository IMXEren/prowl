"""Bounded HTTP listener with owned startup, shutdown and connection tasks.

Ordinary HTTP requests are forwarded through the service. When a certificate authority is
configured, a CONNECT request is answered with a tunnel that terminates TLS locally and serves one
ordinary request inside it.
"""

from __future__ import annotations

import asyncio
import contextlib
from time import perf_counter
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import h11
from loguru import logger

from prowl.browser.proxy.cookies import CookieHeaderError
from prowl.browser.proxy.dispatch import dispatch_request
from prowl.browser.proxy.http1 import (
    MAX_HEADER_BYTES,
    ProxyProtocolError,
    ProxyRequest,
    read_request,
    write_response,
)
from prowl.browser.proxy.request import connect_authority
from prowl.browser.proxy.response import ProxyResponse

if TYPE_CHECKING:
    from prowl.browser.proxy.certificates import ProxyCertificates
    from prowl.service.app import Service
    from prowl.service.protocol import ProxySelection
    from prowl.service.sessions import SessionMode

__all__ = ["MAX_PROXY_CONNECTIONS", "ProxyServer"]

#: Largest number of connections served at once. Fixed so admission can never grow an unbounded
#: backlog of rejection work; a connection above the cap is closed immediately.
MAX_PROXY_CONNECTIONS: Final[int] = 64

#: Deadline for writing one response to a client.
WRITE_SECONDS: Final[float] = 30.0

#: Deadline for closing one transport or waiting for a listener to stop.
CLOSE_SECONDS: Final[float] = 5.0

#: Deadline for the TLS handshake that completes a CONNECT tunnel.
TLS_HANDSHAKE_SECONDS: Final[float] = 30.0

#: The one generic error body. It never carries a target, body, header, credential, or exception.
_ERROR_BODY: Final[bytes] = b"proxy request failed"

#: The correlation header echoed on every response, matching the HTTP service's own field.
_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

#: The fixed projection for every local error, independent of the failure's origin.
_ERROR_HEADERS: Final[tuple[tuple[str, str], ...]] = (
    ("Content-Type", "text/plain; charset=utf-8"),
    ("Content-Length", str(len(_ERROR_BODY))),
)


class ProxyServer:
    """One loopback listener that forwards ordinary HTTP requests through ``service``.

    The listener owns its listening socket and its connection tasks, serves exactly one request per
    connection, and always advertises ``Connection: close``. When ``certificates`` is configured, a
    CONNECT request is answered with a locally terminated TLS tunnel that carries one ordinary
    request. ``start`` is idempotent; ``aclose`` is final and idempotent and fences any later
    ``start``.
    """

    def __init__(  # noqa: PLR0913 - the caller names the listener's trusted bindings explicitly
        self,
        service: Service,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        session: str | None = None,
        session_mode: SessionMode | None = None,
        proxy: ProxySelection | None = None,
        certificates: ProxyCertificates | None = None,
    ) -> None:
        self._service = service
        self._host = host
        self._session = session
        self._session_mode: SessionMode | None = session_mode
        self._proxy = proxy
        self._certificates = certificates
        self._port = port
        self._server: asyncio.Server | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def port(self) -> int:
        """Return the port the listener is actually bound to, or the requested one before start."""
        return self._port

    async def start(self) -> None:
        """Bind the listening socket and begin accepting connections.

        Binding runs inside one owned task, so a cancelled waiter cannot orphan a bound server.
        A later call joins that same task and is a no-op once it has succeeded.

        :raises RuntimeError: when the listener has already been closed.
        """
        if self._closing:
            msg = "proxy server is closed"
            raise RuntimeError(msg)
        task = self._start_task
        if task is None:
            task = asyncio.create_task(self._start())
            task.add_done_callback(_consume_exception)
            self._start_task = task
        await asyncio.shield(task)

    async def aclose(self) -> None:
        """Close the listener and drain every owned connection, exactly once.

        The first call fences admission permanently and starts the one owned cleanup task; later
        calls join it. Cleanup is shielded, so a cancelled waiter cannot abandon a half-closed
        listener or a live connection.
        """
        self._closing = True
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close())
            task.add_done_callback(_consume_exception)
            self._close_task = task
        await asyncio.shield(task)

    async def _start(self) -> None:
        """Bind the socket, publishing it only while the listener is still open."""
        server = await asyncio.start_server(self._accept, self._host, self._port, limit=MAX_HEADER_BYTES + 4)
        self._server = server
        self._port = _listening_port(server)
        if self._closing:
            server.close()

    async def _close(self) -> None:
        """Join startup, stop the listener, then cancel and drain every owned connection."""
        await _silent(self._start_task)
        server = self._server
        if server is not None:
            server.close()
        await self._drain_connections()
        await self._close_listener()

    async def _close_listener(self) -> None:
        """Stop accepting and wait for the listening socket to close, bounded."""
        server = self._server
        if server is not None:
            await _stop_server(server)
            self._server = None

    async def _drain_connections(self) -> None:
        """Cancel and await every owned connection task."""
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Admit one connection, or close it at once when the listener is closed or full."""
        if self._closing or len(self._connections) >= MAX_PROXY_CONNECTIONS:
            writer.close()
            return
        task = asyncio.create_task(self._serve(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connection_done)

    def _connection_done(self, task: asyncio.Task[None]) -> None:
        """Drop a finished connection and consume its exception so it is never unobserved."""
        self._connections.discard(task)
        if not task.cancelled():
            task.exception()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Correlate the whole connection, including CONNECT and cancellation."""
        request_id = uuid4().hex
        started = perf_counter()
        with logger.contextualize(request_id=request_id):
            try:
                await self._serve_request(reader, writer, request_id)
            finally:
                logger.info(
                    "Proxy request {} finished in {}ms",
                    request_id,
                    round((perf_counter() - started) * 1000),
                )

    async def _serve_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        request_id: str,
    ) -> None:
        """Serve one request and one response, then always close the connection."""
        completed = False
        try:
            connection, response, tunnel = await self._response(reader)
            if tunnel is not None:
                request, _ = tunnel
                await self._connect(reader, writer, connection, request, request_id)
            elif response is not None:
                await self._write(writer, connection, response, request_id)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a slow or broken client is a connection-local failure
            # The client is gone or the write failed, so no second response is attempted.
            return
        finally:
            if completed:
                await _close_writer(writer)
            else:
                writer.transport.abort()

    async def _response(
        self,
        reader: asyncio.StreamReader,
        *,
        tunnel_authority: str | None = None,
    ) -> tuple[h11.Connection, ProxyResponse | None, tuple[ProxyRequest, ProxyCertificates] | None]:
        """Read one request and produce its response, or the CONNECT request to tunnel.

        Returns ``(connection, response, tunnel)``. ``tunnel`` carries the parsed CONNECT request
        and the configured authority when TLS termination is enabled, so ``_connect`` answers the
        handshake; the ordinary path leaves it ``None`` and carries the response to write.
        """
        try:
            exchange = await read_request(reader)
        except ProxyProtocolError as error:
            return _server_connection(), _error_response(error.status), None
        except Exception:  # noqa: BLE001 - an unexpected parse failure is still a client error
            return _server_connection(), _error_response(502), None
        if exchange is None:
            return _server_connection(), None, None
        request, connection = exchange
        certificates = self._certificates
        if request.method == "CONNECT" and certificates is not None and tunnel_authority is None:
            return connection, None, (request, certificates)
        return connection, await self._dispatch(request, tunnel_authority=tunnel_authority), None

    async def _connect(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        connection: h11.Connection,
        request: ProxyRequest,
        request_id: str,
    ) -> None:
        """Upgrade CONNECT and serve one authority-bound request under the same id."""
        try:
            hostname, _port = connect_authority(request.target)
        except ProxyProtocolError as error:
            await self._write(writer, connection, _error_response(error.status), request_id)
            return
        certificates = self._certificates
        if certificates is None:  # only a tunneled CONNECT, which requires a certificate authority
            await self._write(writer, connection, _error_response(502), request_id)
            return
        try:
            context = await asyncio.to_thread(certificates.context_for, hostname)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - no certificate failure may reach the wire
            await self._write(writer, connection, _error_response(502), request_id)
            return
        await self._handshake(writer, connection, request_id)
        await writer.start_tls(context, ssl_handshake_timeout=TLS_HANDSHAKE_SECONDS)
        inner_connection, inner_response, _tunnel = await self._response(reader, tunnel_authority=request.target)
        if inner_response is not None:
            await self._write(writer, inner_connection, inner_response, request_id)

    async def _handshake(
        self,
        writer: asyncio.StreamWriter,
        connection: h11.Connection,
        request_id: str,
    ) -> None:
        """Write the one 200 handshake that turns the connection into a tunnel.

        Only the response head is written: a 2xx CONNECT reply switches the protocol, so no body,
        ``Content-Length``, ``Connection`` field, or end-of-message event follows it. The head
        carries the connection's correlation id like every other response.
        """
        headers = [(_REQUEST_ID_HEADER, request_id)]
        handshake = connection.send(h11.Response(status_code=200, reason="Connection established", headers=headers))
        async with asyncio.timeout(WRITE_SECONDS):
            writer.write(handshake)
            await writer.drain()

    async def _dispatch(
        self,
        request: ProxyRequest,
        *,
        tunnel_authority: str | None = None,
    ) -> ProxyResponse:
        """Dispatch one parsed request with the listener's trusted bindings."""
        try:
            response = await dispatch_request(
                self._service,
                request,
                tunnel_authority=tunnel_authority,
                session=self._session,
                session_mode=self._session_mode,
                proxy=self._proxy,
            )
        except CookieHeaderError:
            return _error_response(400)
        except ProxyProtocolError as error:
            return _error_response(error.status)
        except TimeoutError:
            return _error_response(504)
        except Exception:  # noqa: BLE001 - no native failure may reach the wire
            return _error_response(502)
        return response

    async def _write(
        self,
        writer: asyncio.StreamWriter,
        connection: h11.Connection,
        response: ProxyResponse,
        request_id: str,
    ) -> None:
        """Serialize and write one response under the write deadline, stamped with its id."""
        async with asyncio.timeout(WRITE_SECONDS):
            await write_response(writer, connection, _with_request_id(response, request_id))


def _server_connection() -> h11.Connection:
    """Return a fresh h11 server connection for sending one response."""
    return h11.Connection(h11.SERVER)


def _error_response(status: int) -> ProxyResponse:
    """Return the fixed generic error projection for ``status``."""
    return ProxyResponse(status=status, headers=_ERROR_HEADERS, body=_ERROR_BODY)


def _with_request_id(response: ProxyResponse, request_id: str) -> ProxyResponse:
    """Return ``response`` with exactly one trailing ``X-Request-ID`` of ``request_id``.

    Any field already carried under that name is dropped first, so a response can never expose two
    ids or echo an origin-supplied one. Status and body are unchanged, and the origin DTO is not
    mutated.
    """
    retained = tuple((name, value) for name, value in response.headers if name.lower() != _REQUEST_ID_HEADER.lower())
    headers = (*retained, (_REQUEST_ID_HEADER, request_id))
    return ProxyResponse(status=response.status, headers=headers, body=response.body)


async def _stop_server(server: asyncio.Server) -> None:
    """Stop accepting and wait for the listening socket to close, bounded."""
    server.close()
    async with asyncio.timeout(CLOSE_SECONDS):
        await server.wait_closed()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    """Close a connection's transport and wait for it under the close deadline.

    A reset or a client that never finishes closing is an expected connection-local outcome, not a
    cleanup failure, so both are consumed here.
    """
    writer.close()
    try:
        async with asyncio.timeout(CLOSE_SECONDS):
            await writer.wait_closed()
    except TimeoutError:
        writer.transport.abort()
    except OSError:
        pass


async def _silent(task: asyncio.Task[None] | None) -> None:
    """Await ``task`` if present, consuming a failure cleanup cannot act on."""
    if task is None:
        return
    with contextlib.suppress(Exception):
        await asyncio.shield(task)


def _consume_exception(task: asyncio.Task[None]) -> None:
    """Retrieve a finished task's exception so a cancelled waiter cannot leave it unobserved."""
    if not task.cancelled():
        task.exception()


def _listening_port(server: asyncio.Server) -> int:
    """Return the actual port of a started server.

    :raises RuntimeError: when the server has no listening socket.
    """
    sockets = server.sockets
    if not sockets:
        msg = "proxy listener has no socket"
        raise RuntimeError(msg)
    return int(sockets[0].getsockname()[1])
