"""Loopback relay that carries client traffic through an authenticated upstream HTTP(S) proxy.

The bridge binds ``127.0.0.1:0`` and offers a *credential-free* local endpoint: an upstream URL
carrying a username and password is stored only in memory, and every outbound request attaches the
derived ``Proxy-Authorization`` itself. Neither the upstream host, its port, nor its credentials are
ever placed in the local URL, ``repr``, or an error message.

One request is served per client connection. A ``CONNECT`` request opens an upstream tunnel and then
copies bytes in both directions; an ordinary absolute-form HTTP request is forwarded to the upstream
proxy and its response is streamed back without buffering the body. Any framing this relay cannot
reproduce safely (a chunked request body, a client-supplied ``Proxy-Authorization``) is rejected
rather than guessed at.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ssl
from dataclasses import dataclass
from typing import Final
from urllib.parse import SplitResult, unquote, urlsplit

from prowl.browser.proxy.authority import connect_authority
from prowl.browser.proxy.http1 import ProxyProtocolError
from prowl.service.errors import CallerSafeError

__all__ = [
    "MAX_BRIDGE_CONNECTIONS",
    "MAX_BRIDGE_HEADER_BYTES",
    "BrowserProxyBridge",
    "BrowserProxyBridgeError",
]

#: Largest accepted request head, including its terminating CRLFCRLF.
MAX_BRIDGE_HEADER_BYTES: Final[int] = 64 * 1024

#: Largest accepted ordinary request body, whether streamed through or rejected.
MAX_BRIDGE_BODY_BYTES: Final[int] = 8 * 1024 * 1024

#: Largest number of client connections served at once; a connection above the cap is closed at once.
MAX_BRIDGE_CONNECTIONS: Final[int] = 32

#: Deadline for reading a client head, opening the upstream, and reading the upstream reply head.
HANDSHAKE_SECONDS: Final[float] = 30.0

#: Whole-relay deadline for streaming one response or tunnel in either direction.
RELAY_SECONDS: Final[float] = 300.0

#: Deadline for closing one transport.
CLOSE_SECONDS: Final[float] = 5.0

#: Bytes copied per relay read.
_CHUNK: Final[int] = 64 * 1024

_HEAD_END: Final[bytes] = b"\r\n\r\n"
_CR: Final[int] = 0x0D
_LF: Final[int] = 0x0A
_HTAB: Final[int] = 0x09
_SPACE: Final[int] = 0x20
_DELETE: Final[int] = 0x7F

#: Bytes that need no control-byte rejection because HTTP/1 uses them structurally.
_ALLOWED_CONTROLS: Final[frozenset[int]] = frozenset({_CR, _LF, _HTAB})

#: Bytes that begin an obsolete folded header line.
_FOLD_PREFIXES: Final[tuple[bytes, bytes]] = (b" ", b"\t")

_HTTP_VERSIONS: Final[frozenset[str]] = frozenset({"HTTP/1.0", "HTTP/1.1"})

#: Default port per upstream scheme.
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}

#: RFC 7230 token bytes, the only characters a method or header name may contain.
_TOKEN_CHARS: Final[frozenset[int]] = frozenset(
    b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)

#: Lowest and highest accepted ports.
_MIN_PORT: Final[int] = 1
_MAX_PORT: Final[int] = 65535

#: Named wire constants, so no bare protocol number appears in a comparison.
_HTTP_OK: Final[int] = 200
_PROXY_AUTH_REQUIRED: Final[int] = 407
_DEFAULT_HTTP_PORT: Final[int] = 80
_REQUEST_LINE_FIELDS: Final[int] = 3
_STATUS_LINE_FIELDS: Final[int] = 2
_ASCII_LIMIT: Final[int] = 0x80

#: Fixed, credential-free construction-failure messages.
_INVALID_PROXY_URL: Final[str] = "invalid upstream proxy URL"
_UNSUPPORTED_PROXY_SCHEME: Final[str] = "upstream proxy URL must use http or https"
_MISSING_PROXY_CREDENTIALS: Final[str] = "upstream proxy URL requires a username and password"
_PROXY_URL_HAS_PATH: Final[str] = "upstream proxy URL must not carry a path, query, or fragment"

#: Request fields never forwarded upstream: the caller's own proxy credentials, hop-by-hop framing,
#: and the fields this relay owns. ``Host`` is re-synthesized from the request target.
_DROPPED_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "host",
        "proxy-authorization",
        "proxy-connection",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
    }
)

#: Fixed generic reason phrases, one per status this relay can answer with.
_REASON: Final[dict[int, bytes]] = {
    400: b"Bad Request",
    407: b"Proxy Authentication Required",
    408: b"Request Timeout",
    413: b"Payload Too Large",
    431: b"Request Header Fields Too Large",
    501: b"Not Implemented",
    502: b"Bad Gateway",
}

#: The one generic error body. It never carries a target, body, header, credential, or exception.
_ERROR_BODY: Final[bytes] = b"proxy bridge request failed"


class BrowserProxyBridgeError(CallerSafeError):
    """Raised when an upstream proxy URL cannot be used.

    The message is a fixed string chosen at construction and never echoes the upstream host, port,
    or credentials.
    """


@dataclass(frozen=True, slots=True)
class _Head:
    """One parsed request head: method, target, HTTP version, and ordered ``(lower_name, value)``."""

    method: str
    target: str
    version: str
    headers: tuple[tuple[str, str], ...]


class _RejectedError(Exception):
    """A connection-local failure carrying the status this relay should answer with."""

    def __init__(self, status: int) -> None:
        self.status = status


class BrowserProxyBridge:
    """One loopback listener that forwards client traffic through an authenticated upstream proxy.

    Constructed from an ``http://`` or ``https://`` upstream URL whose userinfo is URL-decoded before
    it becomes Basic credentials. For an ``https`` upstream the upstream TLS session is verified with
    the default context unless an explicit ``ssl_context`` is injected; verification is never
    disabled. ``start`` binds ``host:port`` (``0`` selects a free local port, exposed through
    :attr:`listen_url`) and ``aclose`` stops the listener and cancels every owned connection.
    """

    def __init__(
        self,
        proxy_url: str,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        scheme, username, password, upstream_host, upstream_port = _parse_proxy_url(proxy_url)
        self._scheme = scheme
        self._auth = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        self._upstream_host = upstream_host
        self._upstream_port = upstream_port
        self._ssl_context = ssl_context
        self._host = "127.0.0.1"
        self._port = 0
        self._max_connections = MAX_BRIDGE_CONNECTIONS
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._closing = False

    def __repr__(self) -> str:
        """Return a representation that never names the upstream host, port, or credentials."""
        return f"BrowserProxyBridge(local={self._host}:{self._port})"

    @property
    def listen_url(self) -> str | None:
        """Return the credential-free local URL once started, or ``None`` before start."""
        if self._server is None:
            return None
        return f"http://{self._host}:{self._port}"

    async def start(self) -> None:
        """Bind the listening socket and begin accepting connections.

        :raises RuntimeError: when the bridge has already been closed.
        """
        if self._closing:
            msg = "proxy bridge is closed"
            raise RuntimeError(msg)
        if self._server is not None:
            return
        server = await asyncio.start_server(self._accept, self._host, 0, limit=MAX_BRIDGE_HEADER_BYTES + 4)
        sockets = server.sockets
        if self._closing or not sockets:
            server.close()
            msg = "proxy bridge listener is unavailable"
            raise RuntimeError(msg)
        self._server = server
        self._port = int(sockets[0].getsockname()[1])

    async def aclose(self) -> None:
        """Stop accepting, close the listener, and cancel every owned connection. Idempotent."""
        self._closing = True
        server = self._server
        if server is not None:
            server.close()
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if server is not None:
            async with asyncio.timeout(CLOSE_SECONDS):
                await server.wait_closed()
        self._server = None
        self._connections.clear()

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Admit one connection, or close it at once when the bridge is closed or full."""
        if self._closing or len(self._connections) >= self._max_connections:
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

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        """Serve one request and one response, then always close the connection."""
        try:
            try:
                head = await _read_head(client_reader)
                parsed = _parse_head(head)
                if parsed.method == "CONNECT":
                    await self._connect(client_reader, client_writer, parsed)
                else:
                    await self._http(client_reader, client_writer, parsed)
            except _RejectedError as error:
                await _send_error(client_writer, error.status)
        finally:
            await _close_writer(client_writer)

    async def _connect(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter, head: _Head
    ) -> None:
        """Open an upstream tunnel for ``head`` and blind-copy bytes in both directions."""
        try:
            host, port = connect_authority(head.target)
        except ProxyProtocolError:
            raise _RejectedError(400) from None
        self._reject_client_auth(head.headers)
        upstream_reader, upstream_writer = await self._open_upstream()
        try:
            connect_head = (
                f"CONNECT {host}:{port} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Proxy-Authorization: Basic {self._auth}\r\n"
                "\r\n"
            )
            try:
                async with asyncio.timeout(HANDSHAKE_SECONDS):
                    upstream_writer.write(connect_head.encode("ascii"))
                    await upstream_writer.drain()
            except TimeoutError:
                raise _RejectedError(502) from None
            if await _read_upstream_status(upstream_reader) != _HTTP_OK:
                raise _RejectedError(502)
            client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await client_writer.drain()
            await _pump_both(client_reader, client_writer, upstream_reader, upstream_writer)
        finally:
            await _close_writer(upstream_writer)

    async def _http(
        self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter, head: _Head
    ) -> None:
        """Forward one ordinary absolute-form HTTP request and stream the upstream response back."""
        parts = _absolute_http_target(head.target)
        self._reject_client_auth(head.headers)
        length = _body_length(head.headers)
        if length is not None and length > MAX_BRIDGE_BODY_BYTES:
            raise _RejectedError(413)
        upstream_reader, upstream_writer = await self._open_upstream()
        try:
            try:
                async with asyncio.timeout(HANDSHAKE_SECONDS):
                    upstream_writer.write(_upstream_request_head(head, parts, self._auth))
                    await upstream_writer.drain()
            except TimeoutError:
                raise _RejectedError(502) from None
            if length:
                await _stream_body(client_reader, upstream_writer, length)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(RELAY_SECONDS):
                    await _relay(upstream_reader, client_writer)
        finally:
            await _close_writer(upstream_writer)

    def _reject_client_auth(self, headers: tuple[tuple[str, str], ...]) -> None:
        """Reject a caller that presents its own proxy credentials.

        :raises _RejectedError: 407 when a ``Proxy-Authorization`` field is present.
        """
        if any(name == "proxy-authorization" for name, _ in headers):
            raise _RejectedError(_PROXY_AUTH_REQUIRED)

    async def _open_upstream(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open the verified upstream connection, honoring the configured scheme.

        :raises _RejectedError: 502 when the upstream cannot be reached or its TLS session is not trusted.
        """
        context: ssl.SSLContext | None = None
        server_hostname: str | None = None
        if self._scheme == "https":
            context = self._ssl_context if self._ssl_context is not None else ssl.create_default_context()
            server_hostname = self._upstream_host
        try:
            async with asyncio.timeout(HANDSHAKE_SECONDS):
                return await asyncio.open_connection(
                    self._upstream_host,
                    self._upstream_port,
                    ssl=context,
                    server_hostname=server_hostname,
                )
        except (OSError, TimeoutError, ssl.SSLError):
            raise _RejectedError(502) from None


def _parse_proxy_url(proxy_url: str) -> tuple[str, str, str, str, int]:
    """Split an authenticated upstream proxy URL into scheme, credentials, host, and port.

    :raises BrowserProxyBridgeError: for a URL that is not an ``http``/``https`` proxy with a
        username and password; the offending URL is never echoed.
    """
    try:
        parts = urlsplit(proxy_url)
        port = parts.port
    except ValueError:
        raise BrowserProxyBridgeError(_INVALID_PROXY_URL) from None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise BrowserProxyBridgeError(_UNSUPPORTED_PROXY_SCHEME)
    if parts.hostname is None:
        raise BrowserProxyBridgeError(_INVALID_PROXY_URL)
    if parts.username is None or parts.password is None or not parts.username:
        raise BrowserProxyBridgeError(_MISSING_PROXY_CREDENTIALS)
    if port is not None and not _MIN_PORT <= port <= _MAX_PORT:
        raise BrowserProxyBridgeError(_INVALID_PROXY_URL)
    if parts.path or parts.query or parts.fragment:
        raise BrowserProxyBridgeError(_PROXY_URL_HAS_PATH)
    return (
        scheme,
        unquote(parts.username),
        unquote(parts.password),
        parts.hostname,
        _DEFAULT_PORTS[scheme] if port is None else port,
    )


async def _read_head(reader: asyncio.StreamReader) -> bytes:
    """Read one bounded CRLFCRLF-terminated head from ``reader``.

    :raises _RejectedError: 431 when the head exceeds the bound, 408 when the read times out, 400 when the
        peer sent a partial head and then closed.
    """
    try:
        async with asyncio.timeout(HANDSHAKE_SECONDS):
            head = await reader.readuntil(_HEAD_END)
    except asyncio.LimitOverrunError:
        raise _RejectedError(431) from None
    except asyncio.IncompleteReadError:
        raise _RejectedError(400) from None
    except TimeoutError:
        raise _RejectedError(408) from None
    if len(head) > MAX_BRIDGE_HEADER_BYTES:
        raise _RejectedError(431)
    _validate_head(head)
    return head


def _validate_head(head: bytes) -> None:
    """Reject bare line endings, control bytes, and folded header lines.

    :raises _RejectedError: 400 for a head that is not strictly CRLF-delimited and fold-free.
    """
    previous = 0
    for byte in head:
        if previous == _CR and byte != _LF:
            raise _RejectedError(400)
        if byte == _LF and previous != _CR:
            raise _RejectedError(400)
        if byte < _SPACE and byte not in _ALLOWED_CONTROLS:
            raise _RejectedError(400)
        if byte == _DELETE:
            raise _RejectedError(400)
        previous = byte
    if previous != _LF:
        raise _RejectedError(400)
    for line in head.split(b"\r\n"):
        if line[:1] in _FOLD_PREFIXES:
            raise _RejectedError(400)


def _parse_head(head: bytes) -> _Head:
    """Parse a validated raw head into method, target, version, and ordered lowercase-name fields.

    :raises _RejectedError: 400 for a malformed request line, an unsupported version, or a malformed field.
    """
    lines = head[:-4].split(b"\r\n") if head.endswith(_HEAD_END) else []
    if not lines:
        raise _RejectedError(400)
    request_line = lines[0].decode("ascii", "strict") if _is_ascii(lines[0]) else None
    if request_line is None:
        raise _RejectedError(400)
    fields = request_line.split(" ")
    if len(fields) != _REQUEST_LINE_FIELDS:
        raise _RejectedError(400)
    method, target, version = fields
    if not _is_token(method.encode("ascii")) or not target:
        raise _RejectedError(400)
    if version not in _HTTP_VERSIONS:
        raise _RejectedError(400)
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if not separator or not _is_token(name):
            raise _RejectedError(400)
        decoded_value = value.strip().decode("ascii") if _is_ascii(value.strip()) else None
        if decoded_value is None:
            raise _RejectedError(400)
        headers.append((name.decode("ascii").lower(), decoded_value))
    return _Head(method=method, target=target, version=version, headers=tuple(headers))


def _is_ascii(raw: bytes) -> bool:
    """Return whether ``raw`` is pure ASCII without raising on a decode failure."""
    return all(byte < _ASCII_LIMIT for byte in raw)


def _is_token(raw: bytes) -> bool:
    """Return whether ``raw`` is a non-empty RFC 7230 token."""
    return bool(raw) and all(byte in _TOKEN_CHARS for byte in raw)


def _absolute_http_target(target: str) -> SplitResult:
    """Validate an ordinary ``http://`` absolute-form target and return its parsed parts.

    :raises _RejectedError: 501 for a non-``http`` scheme (HTTPS requires a CONNECT tunnel), 400 for a
        target without a hostname or with userinfo.
    """
    try:
        parts = urlsplit(target)
        _ = parts.port
    except ValueError:
        raise _RejectedError(400) from None
    if parts.scheme.lower() != "http":
        raise _RejectedError(501)
    if parts.hostname is None or parts.username is not None or parts.password is not None:
        raise _RejectedError(400)
    return parts


def _body_length(headers: tuple[tuple[str, str], ...]) -> int | None:
    """Return the declared ``Content-Length``, or ``None`` when the request carries no body.

    :raises _RejectedError: 501 for a ``Transfer-Encoding`` body this relay cannot reproduce, 400 for a
        duplicate, conflicting, or non-integer length.
    """
    lengths = [value.strip() for name, value in headers if name == "transfer-encoding"]
    if lengths:
        raise _RejectedError(501)
    raw = [value.strip() for name, value in headers if name == "content-length"]
    if not raw:
        return None
    if len(raw) > 1:
        raise _RejectedError(400)
    if not (raw[0].isascii() and raw[0].isdigit()):
        raise _RejectedError(400)
    return int(raw[0])


def _upstream_request_head(head: _Head, parts: SplitResult, auth: str) -> bytes:
    """Serialize the forwarded request head: original line, safe fields, and this relay's auth."""
    port = parts.port
    host = parts.hostname or ""
    host_value = f"{host}:{port}" if port is not None and port != _DEFAULT_HTTP_PORT else host
    lines = [
        f"{head.method} {head.target} HTTP/1.1",
        f"Host: {host_value}",
        f"Proxy-Authorization: Basic {auth}",
        "Connection: close",
    ]
    for name, value in head.headers:
        if name in _DROPPED_HEADERS:
            continue
        lines.append(f"{name}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


async def _read_upstream_status(reader: asyncio.StreamReader) -> int:
    """Read an upstream reply head and return its status code without consuming a body.

    :raises _RejectedError: 502 when the head is missing, oversized, timed out, or malformed.
    """
    try:
        async with asyncio.timeout(HANDSHAKE_SECONDS):
            head = await reader.readuntil(_HEAD_END)
    except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, TimeoutError):
        raise _RejectedError(502) from None
    if len(head) > MAX_BRIDGE_HEADER_BYTES:
        raise _RejectedError(502)
    parts = head.split(b"\r\n", 1)[0].split(b" ", 2)
    if len(parts) < _STATUS_LINE_FIELDS or not parts[0].startswith(b"HTTP/"):
        raise _RejectedError(502)
    try:
        return int(parts[1])
    except ValueError:
        raise _RejectedError(502) from None


async def _stream_body(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, length: int) -> None:
    """Copy exactly ``length`` bytes from ``reader`` to ``writer``, bounded and without buffering.

    :raises _RejectedError: 400 when the peer closes before delivering the declared body.
    """
    remaining = length
    try:
        async with asyncio.timeout(RELAY_SECONDS):
            while remaining > 0:
                chunk = await reader.read(min(_CHUNK, remaining))
                if not chunk:
                    raise _RejectedError(400)
                writer.write(chunk)
                await writer.drain()
                remaining -= len(chunk)
    except TimeoutError:
        raise _RejectedError(408) from None


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Stream every byte from ``reader`` to ``writer`` until EOF, one bounded chunk at a time.

    EOF is propagated as a half-close when the transport supports it, so a peer that waits for the
    end of the stream before closing cannot deadlock the reverse direction.
    """
    while True:
        data = await reader.read(_CHUNK)
        if not data:
            break
        writer.write(data)
        await writer.drain()
    with contextlib.suppress(OSError, NotImplementedError):
        if writer.can_write_eof():
            writer.write_eof()
            await writer.drain()


async def _pump_both(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    """Copy a tunnel in both directions until both sides reach EOF, bounded by ``RELAY_SECONDS``."""
    tasks = [
        asyncio.create_task(_relay(client_reader, upstream_writer)),
        asyncio.create_task(_relay(upstream_reader, client_writer)),
    ]
    try:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(RELAY_SECONDS):
                await asyncio.gather(*tasks)
    except (OSError, ssl.SSLError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_error(writer: asyncio.StreamWriter, status: int) -> None:
    """Write one fixed generic error response, then leave the caller to close the connection."""
    reason = _REASON.get(status, b"Bad Gateway")
    lines = [
        f"HTTP/1.1 {status} {reason.decode('ascii')}",
        f"Content-Length: {len(_ERROR_BODY)}",
        "Content-Type: text/plain; charset=utf-8",
        "Connection: close",
    ]
    if status == _PROXY_AUTH_REQUIRED:
        lines.append('Proxy-Authenticate: Basic realm="prowl"')
    payload = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + _ERROR_BODY
    with contextlib.suppress(TimeoutError, OSError):
        async with asyncio.timeout(CLOSE_SECONDS):
            writer.write(payload)
            await writer.drain()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    """Close a transport and wait for it, bounded; a reset or stuck peer is an expected outcome."""
    writer.close()
    with contextlib.suppress(TimeoutError, OSError):
        async with asyncio.timeout(CLOSE_SECONDS):
            await writer.wait_closed()
