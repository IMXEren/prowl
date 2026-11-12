"""Resolve a caller-supplied ``Cookie`` request header against native context cookies.

A caller's raw ``Cookie`` header cannot be forwarded as a custom header: the owned native
browser context keeps its own cookie store and is the authority on cookie scope. This module
turns the header into ``SetCookieParam`` updates that can be applied to the already
URL-filtered native cookies of the selected context, preserving every attribute the native
store accepted. It performs no native IO and makes no host, site, or expiry guesses: a request
that cannot be applied without guessing fails closed with a generic, credential-free error.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, NoReturn

from prowl.service.errors import CallerSafeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from playwright._impl._api_structures import Cookie, SetCookieParam

#: The single generic message every rejected header returns; it echoes no names or values.
_COOKIE_HEADER_UNUSABLE = "request cookie header cannot be applied safely"

_OCTET = r"\x21\x23-\x2b\x2d-\x3a\x3c-\x5b\x5d-\x7e"
#: An RFC 6265 cookie-name token.
_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
#: An RFC 6265 cookie-value: cookie-octets, optionally surrounded by preserved DQUOTEs.
_VALUE = re.compile(rf'"[{_OCTET}]*"|[{_OCTET}]*')


class CookieHeaderError(CallerSafeError):
    """A cookie header cannot be applied without guessing scope."""

    def __init__(self) -> None:
        super().__init__(_COOKIE_HEADER_UNUSABLE)


def request_cookie_updates(header: str, current: Sequence[Cookie], url: str) -> list[SetCookieParam]:
    """Resolve the request *header* against the URL-filtered native *current* cookies.

    Returns the ``SetCookieParam`` updates to apply to the selected context, or an empty list
    when the header is empty or already matches the native cookies. The *url* is the already
    validated request URL and is used only to scope a brand-new cookie. This function performs
    no native IO; the caller applies the result while holding the existing context lease. A
    header that cannot be applied without guessing cookie scope raises :class:`CookieHeaderError`.
    """
    pairs = _parse_pairs(header)
    if not pairs:
        return []
    incoming: dict[str, list[str]] = {}
    for name, value in pairs:
        incoming.setdefault(name, []).append(value)
    native: dict[str, list[Cookie]] = {}
    for cookie in current:
        name = cookie.get("name")
        if name is None:
            _reject()
        native.setdefault(name, []).append(cookie)
    updates: list[SetCookieParam] = []
    for name, values in incoming.items():
        updates.extend(_resolve_name(name, values, native.get(name, []), url))
    return updates


def _parse_pairs(header: str) -> list[tuple[str, str]]:
    """Return the ordered ``(name, value)`` pairs of an RFC 6265 request cookie header."""
    if not header.strip():
        return []
    pairs: list[tuple[str, str]] = []
    for segment in header.split(";"):
        name, separator, value = segment.strip().partition("=")
        if not separator or _NAME.fullmatch(name) is None or _VALUE.fullmatch(value) is None:
            _reject()
        pairs.append((name, value))
    return pairs


def _resolve_name(name: str, values: list[str], existing: list[Cookie], url: str) -> list[SetCookieParam]:
    """Return the updates that make the native *existing* cookies match *values*."""
    if not existing:
        if len(values) != 1:
            _reject()
        update: SetCookieParam = {"name": name, "value": values[0], "url": url}
        return [update]
    if Counter(values) == Counter(cookie.get("value") for cookie in existing):
        return []
    if len(values) == 1 and len(existing) == 1:
        return [_changed_value(name, existing[0], values[0])]
    raise CookieHeaderError from None


def _changed_value(name: str, cookie: Cookie, value: str) -> SetCookieParam:
    """Return *cookie* with only its value replaced, keeping its accepted identity verbatim."""
    if cookie.get("partitionKey") is not None:
        # A partitioned cookie's scope cannot be reconstructed from a bare request value.
        _reject()
    update: SetCookieParam = {"name": name, "value": value}
    if "domain" in cookie:
        update["domain"] = cookie["domain"]
    if "path" in cookie:
        update["path"] = cookie["path"]
    if "expires" in cookie:
        update["expires"] = cookie["expires"]
    if "httpOnly" in cookie:
        update["httpOnly"] = cookie["httpOnly"]
    if "secure" in cookie:
        update["secure"] = cookie["secure"]
    if "sameSite" in cookie:
        update["sameSite"] = cookie["sameSite"]
    return update


def _reject() -> NoReturn:
    """Raise the generic cookie-header error without echoing names, values, URLs, or causes."""
    raise CookieHeaderError from None


__all__ = ["CookieHeaderError", "request_cookie_updates"]
