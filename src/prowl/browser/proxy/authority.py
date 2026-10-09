"""Shared strict authority parsing for CONNECT and Host validation."""

from __future__ import annotations

from urllib.parse import urlsplit

from prowl.browser.proxy.http1 import ProxyProtocolError

_MIN_PORT = 1
_MAX_PORT = 65535
_SPACE = 0x20
_DELETE = 0x7F


def connect_authority(target: str) -> tuple[str, int]:
    """Parse a CONNECT authority with an explicit port."""
    return host_authority(target, None)


def host_authority(value: str, default_port: int | None) -> tuple[str, int]:
    """Parse a bare authority, rejecting credentials, controls and ambiguous ports."""
    reject_unsafe_chars(value)
    try:
        parts = urlsplit("//" + value)
        port = parts.port
    except ValueError:
        raise ProxyProtocolError(400) from None
    if parts.hostname is None or parts.username is not None or parts.password is not None:
        raise ProxyProtocolError(400)
    if parts.path or "?" in value or "#" in value or value.endswith(":"):
        raise ProxyProtocolError(400)
    if port is not None and not _MIN_PORT <= port <= _MAX_PORT:
        raise ProxyProtocolError(400)
    port = default_port if port is None else port
    if port is None:
        raise ProxyProtocolError(400)
    return parts.hostname.lower(), port


def reject_unsafe_chars(text: str) -> None:
    """Reject control bytes, whitespace, DEL and backslash before urlsplit."""
    for char in text:
        code = ord(char)
        if code <= _SPACE or code == _DELETE or char == "\\":
            raise ProxyProtocolError(400)
