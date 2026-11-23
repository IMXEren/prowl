"""Bounded single-request HTTP/1 parsing and response serialization using h11."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import h11

from prowl.service.errors import CallerSafeError

if TYPE_CHECKING:
    from prowl.browser.proxy.response import ProxyResponse

__all__ = ["ProxyProtocolError", "ProxyRequest", "read_request", "write_response"]

#: Largest accepted request head, including its terminating CRLFCRLF.
MAX_HEADER_BYTES: Final[int] = 64 * 1024

#: Largest accepted request body, whether declared by Content-Length or decoded from chunked framing.
MAX_BODY_BYTES: Final[int] = 8 * 1024 * 1024

#: Whole-request read deadline.
REQUEST_READ_SECONDS: Final[float] = 30.0

#: Bytes pulled from the reader per body read.
_READ_CHUNK: Final[int] = 64 * 1024

_HEAD_TERMINATOR: Final[bytes] = b"\r\n\r\n"
_LINE_TERMINATOR: Final[bytes] = b"\r\n"

_CR: Final[int] = 0x0D
_LF: Final[int] = 0x0A
_HTAB: Final[int] = 0x09
_SPACE: Final[int] = 0x20
_DELETE: Final[int] = 0x7F

#: Bytes that begin an obsolete folded header line.
_FOLD_PREFIXES: Final[tuple[bytes, bytes]] = (b" ", b"\t")

#: Bytes that need no control-byte rejection because HTTP/1 uses them structurally.
_ALLOWED_CONTROLS: Final[frozenset[int]] = frozenset({_CR, _LF, _HTAB})

#: Exact methods this boundary serves; any other method is rejected rather than echoed back.
_ALLOWED_METHODS: Final[frozenset[str]] = frozenset({"GET", "POST", "CONNECT"})

#: Methods that must not carry a request body.
_BODYLESS_METHODS: Final[frozenset[str]] = frozenset({"GET", "CONNECT"})
_HTTP_VERSIONS: Final[frozenset[bytes]] = frozenset({b"1.0", b"1.1"})

#: Fixed generic messages, one per status. A message never echoes a target, body, or header.
_MESSAGE_BY_STATUS: Final[dict[int, str]] = {
    400: "malformed HTTP request",
    405: "unsupported request method",
    408: "request read timed out",
    413: "request too large",
    415: "request content encoding not supported",
    417: "request expectation not supported",
    431: "request headers too large",
    501: "request upgrade not supported",
    502: "invalid proxy response",
    505: "unsupported HTTP version",
}
_DEFAULT_MESSAGE: Final[str] = "invalid proxy request"


@dataclass(frozen=True, slots=True)
class ProxyRequest:
    """One parsed ordinary request: method, request target, ordered fields, and exact body bytes."""

    method: str
    target: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


class ProxyProtocolError(CallerSafeError):
    """A request or response that cannot be represented safely on this boundary.

    ``status`` is the HTTP status the listener should answer with; it defaults to 400. The message
    is a fixed generic string chosen by status, so it can never echo a request target, a body, a
    header value, or a credential.
    """

    def __init__(self, status: int = 400) -> None:
        self.status = status
        super().__init__(_MESSAGE_BY_STATUS.get(status, _DEFAULT_MESSAGE))


async def read_request(reader: asyncio.StreamReader) -> tuple[ProxyRequest, h11.Connection] | None:
    """Read one complete ordinary HTTP/1 request, or ``None`` when the peer sent nothing.

    Returns the parsed request and the h11 server connection, left ready to send one final response.
    The whole read is bounded by :data:`REQUEST_READ_SECONDS`. A trailing or pipelined request is
    never executed; the future listener closes the connection after one response.

    :raises ProxyProtocolError: for malformed, oversized, unsupported, or timed-out requests.
    """
    try:
        async with asyncio.timeout(REQUEST_READ_SECONDS):
            return await _read_exchange(reader)
    except TimeoutError:
        raise ProxyProtocolError(408) from None


async def write_response(
    writer: asyncio.StreamWriter,
    connection: h11.Connection,
    response: ProxyResponse,
) -> None:
    """Serialize one final proxy response onto ``writer`` without closing the transport.

    The exact body bytes and repeated fields are preserved; h11 adds framing where the response
    carries none. ``Connection: close`` is always advertised, because one request is served per
    connection. A local serialization failure becomes a generic :class:`ProxyProtocolError` so no
    response field can leak through an exception.

    :raises ProxyProtocolError: when the response cannot be represented on the wire.
    """
    try:
        events = _response_events(response)
        chunks = [chunk for event in events if (chunk := connection.send(event))]
    except (h11.LocalProtocolError, UnicodeError):
        raise ProxyProtocolError(502) from None
    for chunk in chunks:
        writer.write(chunk)
    await writer.drain()


async def _read_exchange(reader: asyncio.StreamReader) -> tuple[ProxyRequest, h11.Connection] | None:
    """Read the head, validate it, and collect the bounded body of a single request."""
    head = await _read_head(reader)
    if head is None:
        return None
    _validate_head(head)
    connection = h11.Connection(h11.SERVER, max_incomplete_event_size=MAX_HEADER_BYTES)
    event = _receive_request(connection, head)
    if event.http_version not in _HTTP_VERSIONS:
        raise ProxyProtocolError(505)
    method, target = _decode_line(event)
    if method not in _ALLOWED_METHODS:
        raise ProxyProtocolError(405)
    headers = tuple((name.decode("latin-1"), value.decode("latin-1")) for name, value in event.headers.raw_items())
    _check_supported(tuple((name.lower(), value) for name, value in headers))
    body = await _read_body(reader, connection, method)
    return ProxyRequest(method=method, target=target, headers=headers, body=body), connection


async def _read_head(reader: asyncio.StreamReader) -> bytes | None:
    """Read up to and including the first CRLFCRLF, so the body is not consumed eagerly.

    :raises ProxyProtocolError: 431 when the head exceeds the bound, 400 when the peer sent a
        partial head and then closed.
    """
    try:
        head = await reader.readuntil(_HEAD_TERMINATOR)
    except asyncio.LimitOverrunError:
        raise ProxyProtocolError(431) from None
    except asyncio.IncompleteReadError as exc:
        if exc.partial:
            raise ProxyProtocolError(400) from None
        return None
    if len(head) > MAX_HEADER_BYTES:
        raise ProxyProtocolError(431)
    return head


def _validate_head(head: bytes) -> None:
    """Reject bare line endings, control bytes, and obsolete folded header lines.

    h11 tolerates all of these; a proxy must not, because a folded field or a bare LF is a request
    smuggling primitive once a second hop re-frames the message.

    :raises ProxyProtocolError: 400 for any head that is not strictly CRLF-delimited and fold-free.
    """
    previous = 0
    for byte in head:
        if previous == _CR and byte != _LF:
            raise ProxyProtocolError(400)
        if byte == _LF and previous != _CR:
            raise ProxyProtocolError(400)
        if byte < _SPACE and byte not in _ALLOWED_CONTROLS:
            raise ProxyProtocolError(400)
        if byte == _DELETE:
            raise ProxyProtocolError(400)
        previous = byte
    if previous != _LF:
        raise ProxyProtocolError(400)
    for line in head.split(_LINE_TERMINATOR):
        if line[:1] in _FOLD_PREFIXES:
            raise ProxyProtocolError(400)


def _receive_request(connection: h11.Connection, head: bytes) -> h11.Request:
    """Feed the head to h11 and require that it parsed as a request.

    :raises ProxyProtocolError: 400 for malformed syntax or any non-request first event.
    """
    try:
        connection.receive_data(head)
        event = connection.next_event()
    except (h11.RemoteProtocolError, UnicodeError):
        raise ProxyProtocolError(400) from None
    if not isinstance(event, h11.Request):
        raise ProxyProtocolError(400)
    return event


def _decode_line(event: h11.Request) -> tuple[str, str]:
    """Decode the method and target as ASCII.

    :raises ProxyProtocolError: 400 when either field is not ASCII.
    """
    try:
        return event.method.decode("ascii"), event.target.decode("ascii")
    except UnicodeDecodeError:
        raise ProxyProtocolError(400) from None


def _check_supported(headers: tuple[tuple[str, str], ...]) -> None:
    """Reject request features this ordinary-response boundary does not implement.

    ``headers`` carries lowercase names. Upgrades and expectations are rejected before any body is
    read, so a ``Expect: 100-continue`` request is never left waiting on an interim response.

    :raises ProxyProtocolError: 501 for upgrades, 417 for expectations, 400 for a simultaneous
        Transfer-Encoding and Content-Length, 413 for a declared body above the bound.
    """
    if any(name == "upgrade" for name, _ in headers):
        raise ProxyProtocolError(501)
    if any(name == "expect" for name, _ in headers):
        raise ProxyProtocolError(417)
    has_te = any(name == "transfer-encoding" for name, _ in headers)
    length = _declared_length(headers)
    if has_te and length is not None:
        raise ProxyProtocolError(400)
    if length is not None and length > MAX_BODY_BYTES:
        raise ProxyProtocolError(413)


def _declared_length(headers: tuple[tuple[str, str], ...]) -> int | None:
    """Return the single Content-Length value, or ``None`` when the field is absent.

    h11 has already validated duplicate and conflicting Content-Length fields, so at most one
    normalized value reaches here.

    :raises ProxyProtocolError: 400 when the value is not an integer.
    """
    for name, value in headers:
        if name != "content-length":
            continue
        try:
            return int(value.strip())
        except ValueError:
            raise ProxyProtocolError(400) from None
    return None


async def _read_body(reader: asyncio.StreamReader, connection: h11.Connection, method: str) -> bytes:
    """Collect the bounded body until h11 reports the message end.

    :raises ProxyProtocolError: 413 when the decoded body exceeds the bound, 400 for premature EOF,
        request trailers, an unexpected event, or a body on a bodyless method.
    """
    body = bytearray()
    while True:
        event = _next_event(connection)
        if event is h11.NEED_DATA:
            connection.receive_data(await reader.read(_READ_CHUNK))
            continue
        if isinstance(event, h11.Data):
            if len(body) + len(event.data) > MAX_BODY_BYTES:
                raise ProxyProtocolError(413)
            body.extend(event.data)
            continue
        if isinstance(event, h11.EndOfMessage):
            if any(event.headers.raw_items()):
                raise ProxyProtocolError(400)
            break
        raise ProxyProtocolError(400)
    if body and method in _BODYLESS_METHODS:
        raise ProxyProtocolError(400)
    return bytes(body)


def _next_event(connection: h11.Connection) -> h11.Event | type[h11.NEED_DATA | h11.PAUSED]:
    """Advance h11 by one event, mapping its framing errors to a generic 400.

    :raises ProxyProtocolError: 400 for malformed framing or an incomplete body at EOF.
    """
    try:
        return connection.next_event()
    except (h11.RemoteProtocolError, UnicodeError):
        raise ProxyProtocolError(400) from None


def _response_events(response: ProxyResponse) -> tuple[h11.Event, ...]:
    """Build the h11 events for one final response, preserving order, duplicates, and the body."""
    headers: list[tuple[bytes, bytes]] = [
        (name.encode("ascii"), value.encode("latin-1")) for name, value in response.headers
    ]
    headers.append((b"Connection", b"close"))
    events: list[h11.Event] = [h11.Response(status_code=response.status, headers=headers, reason=b"")]
    if response.body:
        events.append(h11.Data(data=response.body))
    events.append(h11.EndOfMessage())
    return tuple(events)
