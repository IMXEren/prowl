"""Exact origin pinning shared by the live CAPTCHA adapters."""

from __future__ import annotations

import re
from ipaddress import ip_address
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from playwright.async_api import Page

_ORIGIN_PORTS: Final = {"http": 80, "https": 443}
_HOST_LABEL: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MAX_PORT: Final = 65536


def _canonical_origin(url: object) -> str | None:
    """Return an exact HTTP or HTTPS origin, or ``None``."""
    if not isinstance(url, str) or not url:
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme not in _ORIGIN_PORTS or parts.username is not None or parts.password is not None:
            return None
        host = parts.hostname
        if host is None:
            return None
        if ":" in host:
            host = str(ip_address(host))
        elif not all(_HOST_LABEL.fullmatch(label) for label in host.lower().split(".")):
            return None
        port = parts.port
    except ValueError:
        return None
    if port is None:
        port = _ORIGIN_PORTS[parts.scheme]
    if not 0 < port < _MAX_PORT:
        return None
    hostname = f"[{host}]" if ":" in host else host.lower()
    return f"{parts.scheme}://{hostname}:{port}"


def _page_origin_allowed(page: Page, allowed: frozenset[str]) -> bool:
    """Return whether the page remains on one pinned starting origin."""
    origin = _canonical_origin(page.url)
    return origin is not None and origin in allowed
