"""Prepare proxy targets and caller headers without native or network IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import SplitResult, urlsplit

from prowl.browser.headers import browser_owns_header, normalize_custom_headers
from prowl.browser.proxy.authority import (
    connect_authority,
)
from prowl.browser.proxy.authority import (
    host_authority as _host_authority,
)
from prowl.browser.proxy.authority import (
    reject_unsafe_chars as _reject_unsafe_chars,
)
from prowl.browser.proxy.http1 import ProxyProtocolError
from prowl.browser.proxy.response import connection_header_names

if TYPE_CHECKING:
    from prowl.browser.proxy.http1 import ProxyRequest

__all__ = ["PreparedRequest", "connect_authority", "prepare_request"]

#: Default port per absolute scheme, used to normalize an omitted port.
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}

#: Scheme every tunneled request is bound to once TLS is terminated.
_TUNNEL_SCHEME: Final[str] = "https"

#: Methods this boundary forwards; CONNECT is handled before TLS, any other method is unsupported.
_FORWARDED_METHODS: Final[frozenset[str]] = frozenset({"GET", "POST"})

#: Lowest and highest ports accepted in an authority, in either URI or Host form.
_MIN_PORT: Final[int] = 1
_MAX_PORT: Final[int] = 65535


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One request resolved for forwarding: URL, surviving header map, raw cookie, exact body."""

    method: str
    url: str
    headers: dict[str, str]
    cookie_header: str | None
    body: bytes


def prepare_request(
    request: ProxyRequest,
    *,
    tunnel_authority: str | None = None,
) -> PreparedRequest:
    """Resolve ``request`` into a :class:`PreparedRequest`.

    Without ``tunnel_authority`` the target must be an absolute ``http``/``https`` URL; its text,
    query, and percent escapes are preserved unchanged. With ``tunnel_authority`` the target is
    either origin-form (bound literally to the connected authority) or an absolute HTTPS URL whose
    normalized authority matches the connection.

    :raises ProxyProtocolError: 501 for CONNECT, 405 for any other non-GET/POST method, and 400 for
        a target, Host field, or header set that cannot be forwarded safely.
    """
    _check_method(request.method)
    _check_content_encoding(request.headers)
    url, authority, default_port = _resolve_target(request.target, tunnel_authority)
    _check_host(request.headers, authority, default_port)
    nominated = connection_header_names(request.headers)
    cookie_header = _raw_cookie(request.headers, nominated)
    headers = _custom_headers(request.headers, nominated)
    return PreparedRequest(
        method=request.method,
        url=url,
        headers=headers,
        cookie_header=cookie_header,
        body=request.body,
    )


def _check_content_encoding(items: tuple[tuple[str, str], ...]) -> None:
    for name, value in items:
        if name.lower() == "content-encoding" and any(
            coding.strip().lower() != "identity" for coding in value.split(",")
        ):
            raise ProxyProtocolError(415)


def _check_method(method: str) -> None:
    """Reject CONNECT and any method this boundary does not forward."""
    if method == "CONNECT":
        raise ProxyProtocolError(501)
    if method not in _FORWARDED_METHODS:
        raise ProxyProtocolError(405)


def _resolve_target(
    target: str,
    tunnel_authority: str | None,
) -> tuple[str, tuple[str, int], int]:
    """Return the outbound URL, its normalized authority, and the scheme's default port."""
    if tunnel_authority is None:
        parts = _absolute_url(target)
        return target, _url_authority(parts), _DEFAULT_PORTS[parts.scheme]

    connected = connect_authority(tunnel_authority)
    default_port = _DEFAULT_PORTS[_TUNNEL_SCHEME]
    if target.startswith("/"):
        url = _TUNNEL_SCHEME + "://" + tunnel_authority + target
        _absolute_url(url)
        return url, connected, default_port
    parts = _absolute_url(target)
    if parts.scheme != _TUNNEL_SCHEME or _url_authority(parts) != connected:
        raise ProxyProtocolError(400)
    return target, connected, default_port


def _absolute_url(target: str) -> SplitResult:
    """Validate an absolute proxy target, returning its parsed parts unchanged.

    :raises ProxyProtocolError: 400 for an unsafe character, a non-http(s) scheme, a missing
        hostname, userinfo, a fragment, or an out-of-range port.
    """
    _reject_unsafe_chars(target)
    try:
        parts = urlsplit(target)
        port = parts.port
    except ValueError:
        raise ProxyProtocolError(400) from None
    if parts.scheme not in _DEFAULT_PORTS or parts.hostname is None:
        raise ProxyProtocolError(400)
    if parts.username is not None or parts.password is not None or "#" in target or parts.netloc.endswith(":"):
        raise ProxyProtocolError(400)
    if port is not None and not _MIN_PORT <= port <= _MAX_PORT:
        raise ProxyProtocolError(400)
    return parts


def _url_authority(parts: SplitResult) -> tuple[str, int]:
    """Return the lowercase host and explicit-or-default port named by ``parts``."""
    host = parts.hostname
    if host is None:
        raise ProxyProtocolError(400)
    port = parts.port
    return host.lower(), _DEFAULT_PORTS[parts.scheme] if port is None else port


def _check_host(
    items: tuple[tuple[str, str], ...],
    authority: tuple[str, int],
    default_port: int,
) -> None:
    """Require at most one Host field and, when present, that it matches ``authority``.

    :raises ProxyProtocolError: 400 for a duplicate Host or one that names another authority.
    """
    values = [value for name, value in items if name.lower() == "host"]
    if not values:
        return
    if len(values) > 1:
        raise ProxyProtocolError(400)
    if _host_authority(values[0], default_port) != authority:
        raise ProxyProtocolError(400)


def _raw_cookie(items: tuple[tuple[str, str], ...], nominated: set[str]) -> str | None:
    """Return the raw ``Cookie`` value joined in wire order, or ``None``.

    Each ``Cookie`` field is kept byte-for-byte and multiple fields are joined with ``"; "``. A
    ``Cookie`` nominated by ``Connection`` is dropped entirely rather than extracted.
    """
    if "cookie" in nominated:
        return None
    values = [value for name, value in items if name.lower() == "cookie"]
    return "; ".join(values) if values else None


def _custom_headers(items: tuple[tuple[str, str], ...], nominated: set[str]) -> dict[str, str]:
    """Drop browser-owned and ``Connection``-nominated fields, then normalize the custom remainder.

    :raises ProxyProtocolError: 400 for a duplicate retained field name or one the shared header
        policy rejects; the offending credential is never echoed.
    """
    custom: dict[str, str] = {}
    for name, value in items:
        lowered = name.lower()
        if browser_owns_header(lowered) or lowered in nominated:
            continue
        if lowered in custom:
            raise ProxyProtocolError(400)
        custom[lowered] = value
    try:
        return normalize_custom_headers(custom)
    except ValueError:
        raise ProxyProtocolError(400) from None
