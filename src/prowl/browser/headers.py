"""Policies for adding caller headers without corrupting browser requests."""

import re
from collections.abc import Mapping
from typing import Final

HEADER_SCOPE_DOCUMENT: Final[str] = "document"
HEADER_SCOPE_ORIGIN: Final[str] = "origin"
HEADER_SCOPES: Final[frozenset[str]] = frozenset({HEADER_SCOPE_DOCUMENT, HEADER_SCOPE_ORIGIN})

_HTTP_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

BROWSER_OWNED_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "accept",
        "accept-charset",
        "accept-encoding",
        "accept-language",
        "connection",
        "content-encoding",
        "content-length",
        "cookie",
        "cookie2",
        "date",
        "dnt",
        "expect",
        "host",
        "keep-alive",
        "origin",
        "permissions-policy",
        "priority",
        "proxy-authorization",
        "referer",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "upgrade-insecure-requests",
        "user-agent",
        "via",
        "x-http-method",
        "x-http-method-override",
        "x-method-override",
    },
)


def browser_owns_header(name: str) -> bool:
    """Return whether Chromium must generate *name* for each request."""
    lowered = name.lower()
    return lowered in BROWSER_OWNED_HEADERS or lowered.startswith(
        ("access-control-request-", "proxy-", "sec-"),
    )


def normalize_custom_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Validate custom headers and return a lowercase-name copy.

    :raises ValueError: when a name or value is unsafe, or Chromium owns the
        header as part of its transport behavior or browser fingerprint.
    """
    for name, value in headers.items():
        if not _HTTP_TOKEN_RE.match(name):
            msg = f"invalid header name: {name!r}"
            raise ValueError(msg)
        if "\r" in value or "\n" in value:
            msg = f"header {name!r} value must not contain CR or LF"
            raise ValueError(msg)

    lowered = {name.lower(): value for name, value in headers.items()}
    forbidden = sorted(name for name in lowered if browser_owns_header(name))
    if forbidden:
        msg = f"browser-controlled header(s) cannot be set or overridden: {', '.join(forbidden)}"
        raise ValueError(msg)
    return lowered
