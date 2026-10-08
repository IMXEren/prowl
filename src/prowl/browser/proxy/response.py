"""Project decoded HTTP bytes or rendered HTML into safe proxy response fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from prowl.service.errors import CallerSafeError
from prowl.service.protocol import HTTP_MODE

if TYPE_CHECKING:
    from prowl.service.backend import FetchResult

__all__ = ["ProxyResponse", "ProxyResponseError", "connection_header_names", "project_response"]


@dataclass(frozen=True, slots=True)
class ProxyResponse:
    """A framed HTTP/1 response: a final status, ordered repeated header fields, and a body."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class ProxyResponseError(CallerSafeError):
    """Raised when a fetch result cannot be projected into a safe proxy response."""


_REPRESENTATION_HEADER: Final[str] = "X-Prowl-Representation"
_RENDERED: Final[str] = "rendered"
_ORIGIN: Final[str] = "origin"
_RENDERED_CONTENT_TYPE: Final[str] = "text/html; charset=utf-8"

#: Final statuses a proxy reply can carry: no informational or absent status is representable.
_MIN_FINAL_STATUS: Final[int] = 200
_MAX_FINAL_STATUS: Final[int] = 599

#: Statuses with their own body/framing rules.
_NO_CONTENT_STATUS: Final[int] = 204
_RESET_CONTENT_STATUS: Final[int] = 205
_NOT_MODIFIED_STATUS: Final[int] = 304

#: Byte boundaries for a writable field value: HTAB and obs-text are allowed, other controls and
#: DEL are not.
_HTAB: Final[int] = 0x09
_SPACE: Final[int] = 0x20
_DELETE: Final[int] = 0x7F
_LATIN1_MAX: Final[int] = 0xFF

#: Hop-by-hop fields the listener's transport owns rather than the origin.
_HOP_BY_HOP: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    }
)

#: Fields stripped from every projection: framing the listener owns, hop-by-hop fields, and any
#: representation marker an earlier hop may have added.
_ALWAYS_DROPPED: Final[frozenset[str]] = _HOP_BY_HOP | frozenset(
    {"content-length", "content-encoding", "x-prowl-representation"}
)

#: Fields that only describe the origin bytes and cannot be carried onto rendered text.
_RENDERED_DROPPED: Final[frozenset[str]] = frozenset(
    {
        "content-type",
        "content-disposition",
        "content-range",
        "accept-ranges",
        "etag",
        "last-modified",
        "content-md5",
        "digest",
        "content-digest",
        "repr-digest",
        "set-cookie",
    }
)

#: Validators, range metadata, and digests that describe the encoded bytes rather than the decoded
#: entity, so they are stale once curl has decoded a compressed body.
_ENCODED_ONLY: Final[frozenset[str]] = _RENDERED_DROPPED - {
    "content-type",
    "content-disposition",
    "set-cookie",
}

_TOKEN_CHARS: Final[frozenset[str]] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'*+-.^_`|~"
)


def project_response(result: FetchResult) -> ProxyResponse:
    """Project ``result`` into a serializable :class:`ProxyResponse`.

    Only final, ordinary statuses are projectable; a missing or informational status has no
    representable proxy reply. Raises :class:`ProxyResponseError` when the result cannot be
    projected safely.
    """
    status = result.status_code
    if status is None or not _MIN_FINAL_STATUS <= status <= _MAX_FINAL_STATUS:
        msg = "fetch result has no projectable final status"
        raise ProxyResponseError(msg)

    if result.mode == HTTP_MODE:
        return _project_http(result, status)
    return _project_rendered(result, status)


def _project_http(result: FetchResult, status: int) -> ProxyResponse:
    """Project curl's decoded entity bytes and original final header pairs."""
    if result.body_bytes is None or result.header_items is None:
        msg = "HTTP fetch result is missing its origin representation"
        raise ProxyResponseError(msg)

    source_items = result.header_items
    dropped = set(_ALWAYS_DROPPED) | connection_header_names(source_items)
    if _has_nonidentity_encoding(source_items):
        dropped |= _ENCODED_ONLY

    retained = _retain(source_items, dropped)
    return _assemble(status, retained, result.body_bytes, rendered=False)


def _project_rendered(result: FetchResult, status: int) -> ProxyResponse:
    """Project rendered text as UTF-8 and mark it as a non-origin representation."""
    source_items = tuple(result.headers.items())
    dropped = set(_ALWAYS_DROPPED) | _RENDERED_DROPPED | connection_header_names(source_items)
    retained = _retain(source_items, dropped)
    body = result.response.encode("utf-8")
    return _assemble(status, retained, body, rendered=True)


def _assemble(
    status: int,
    retained: tuple[tuple[str, str], ...],
    source_body: bytes,
    *,
    rendered: bool,
) -> ProxyResponse:
    """Apply the body and framing rules for ``status`` and append the projection's own fields."""
    if status in (_NO_CONTENT_STATUS, _NOT_MODIFIED_STATUS):
        body = b""
        emit_length = False
    elif status == _RESET_CONTENT_STATUS:
        body = b""
        emit_length = True
    else:
        body = source_body
        emit_length = True

    headers = list(retained)
    headers.append((_REPRESENTATION_HEADER, _RENDERED if rendered else _ORIGIN))
    if rendered:
        headers.append(("Content-Type", _RENDERED_CONTENT_TYPE))
    if emit_length:
        headers.append(("Content-Length", str(len(body))))

    _validate(headers)
    return ProxyResponse(status=status, headers=tuple(headers), body=body)


def connection_header_names(items: tuple[tuple[str, str], ...]) -> set[str]:
    """Read each ``Connection`` field's comma tokens as lowercase names to drop as well."""
    nominated: set[str] = set()
    for name, value in items:
        if name.lower() != "connection":
            continue
        for token in value.split(","):
            stripped = token.strip().lower()
            if stripped:
                nominated.add(stripped)
    return nominated


def _has_nonidentity_encoding(items: tuple[tuple[str, str], ...]) -> bool:
    """Whether any ``Content-Encoding`` field names a coding other than ``identity``."""
    for name, value in items:
        if name.lower() != "content-encoding":
            continue
        for coding in value.split(","):
            stripped = coding.strip().lower()
            if stripped and stripped != "identity":
                return True
    return False


def _retain(items: tuple[tuple[str, str], ...], dropped: set[str]) -> tuple[tuple[str, str], ...]:
    """Keep every field whose lowercase name is not dropped, preserving order and duplicates."""
    return tuple((name, value) for name, value in items if name.lower() not in dropped)


def _validate(headers: list[tuple[str, str]]) -> None:
    """Reject retained fields that cannot be written as HTTP/1 fields, without echoing them."""
    for name, value in headers:
        if not _is_token(name) or not _is_safe_value(value):
            msg = "fetch result has an unsafe header field"
            raise ProxyResponseError(msg)


def _is_token(name: str) -> bool:
    return bool(name) and all(char in _TOKEN_CHARS for char in name)


def _is_safe_value(value: str) -> bool:
    try:
        encoded = value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return all(byte == _HTAB or (_SPACE <= byte <= _LATIN1_MAX and byte != _DELETE) for byte in encoded)
