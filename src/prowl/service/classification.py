"""Bounded response classification for the HTTP fast path.

A classification names why a response is not an ordinary success so the router can decide whether
a browser is worth starting. It stays deliberately conservative: a plain 403, an unrecognized
status, or HTML that merely contains scripts is never read as a challenge.

Two rules keep browser work from being spent for nothing. A Cloudflare, CAPTCHA, address or region
signature is only trusted on a block status, because an ordinary served page can legitimately
carry such words or a CAPTCHA widget. A ``<noscript>`` shell that states outright that JavaScript
has to run the page is trusted at any status, because rendering it is exactly what a browser can
do that the fast path cannot. That shell rule reads raw HTTP entity bytes, so it is skipped for an
already-rendered browser DOM, which :func:`classify` is told about with ``rendered=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The response is an ordinary success.
SUCCESS: Final[str] = "SUCCESS"
#: The response is a shell that only a browser can turn into content.
JAVASCRIPT_REQUIRED: Final[str] = "JAVASCRIPT_REQUIRED"
#: The response is a Cloudflare challenge interstitial.
CLOUDFLARE_CHALLENGE: Final[str] = "CLOUDFLARE_CHALLENGE"
#: The response carries a CAPTCHA widget the caller has to solve.
CAPTCHA: Final[str] = "CAPTCHA"
#: The response refuses the request until it slows down.
RATE_LIMITED: Final[str] = "RATE_LIMITED"
#: The response denies the request because of the client's network address.
IP_BLOCKED: Final[str] = "IP_BLOCKED"
#: The response requires credentials or a session.
AUTH_REQUIRED: Final[str] = "AUTH_REQUIRED"
#: The response refuses the request because of the client's region.
GEO_BLOCKED: Final[str] = "GEO_BLOCKED"
#: No response arrived at all.
NETWORK_ERROR: Final[str] = "NETWORK_ERROR"
#: The response is blocked in a way this classifier cannot name.
UNKNOWN_BLOCK: Final[str] = "UNKNOWN_BLOCK"

#: The only categories a browser is worth starting for. Everything else is answered as it is.
BROWSER_REQUIRED_CATEGORIES: Final[frozenset[str]] = frozenset(
    {JAVASCRIPT_REQUIRED, CLOUDFLARE_CHALLENGE, CAPTCHA},
)

#: Statuses that ask the caller to authenticate. 407 is the proxy's own challenge.
_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 407})
_RATE_LIMIT_STATUS: Final[int] = 429
_GEO_BLOCK_STATUS: Final[int] = 451
#: Statuses below this are read as ordinary successes when no signature says otherwise.
_SUCCESS_STATUS_FLOOR: Final[int] = 200
#: Statuses at or above this are blocks; a body signature is only trusted inside them.
_BLOCK_STATUS_FLOOR: Final[int] = 400

_CF_MITIGATED_HEADER: Final[str] = "cf-mitigated"
_CF_MITIGATED_CHALLENGE: Final[str] = "challenge"
_AUTHENTICATE_HEADER: Final[str] = "www-authenticate"

#: Cloudflare challenge pages carry these; an ordinary fronted response carries none of them.
_CLOUDFLARE_CHALLENGE_MARKERS: Final[tuple[str, ...]] = (
    "/cdn-cgi/challenge-platform/",
    "__cf_chl_",
    "cf_chl_opt",
    "cf-please-wait",
    "challenge-form",
    "jschl_vc",
)

#: Widget names a CAPTCHA interstitial has to mount.
_CAPTCHA_MARKERS: Final[tuple[str, ...]] = (
    "g-recaptcha",
    "grecaptcha",
    "hcaptcha",
    "h-captcha",
    "cf-turnstile",
    "recaptcha/api.js",
    "data-sitekey",
)

#: Phrases a refusal page uses when it refuses the network address itself.
_IP_BLOCK_MARKERS: Final[tuple[str, ...]] = (
    "your ip has been banned",
    "your ip address has been banned",
    "your ip address has been blocked",
    "ip address has been banned",
    "banned your ip",
    "error 1020",
)

#: Phrases a refusal page uses when it refuses the client's region.
_GEO_BLOCK_MARKERS: Final[tuple[str, ...]] = (
    "not available in your country",
    "not available in your region",
    "not available in your location",
    "unavailable in your country",
    "unavailable in your region",
    "blocked in your country",
    "is not available in your area",
)

#: A ``<noscript>`` shell that states outright that JavaScript has to run the page.
_JS_REQUIRED_RE: Final[re.Pattern[str]] = re.compile(
    r"<noscript[^>]*>[^<]{0,800}?"
    r"(enable javascript|javascript is (disabled|not enabled|required)|requires javascript|turn on javascript)",
    re.IGNORECASE | re.DOTALL,
)

#: Body signatures that are only trusted on a block status, strongest first.
_BODY_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (CLOUDFLARE_CHALLENGE, _CLOUDFLARE_CHALLENGE_MARKERS),
    (CAPTCHA, _CAPTCHA_MARKERS),
    (IP_BLOCKED, _IP_BLOCK_MARKERS),
    (GEO_BLOCKED, _GEO_BLOCK_MARKERS),
)


@dataclass(frozen=True, slots=True)
class Classification:
    """What a response turned out to be, and why.

    ``browser_required`` is derived from the category by :func:`_classify`, so a category can
    never disagree with the routing decision it implies.
    """

    category: str
    reason: str
    browser_required: bool


def classify(
    status_code: int,
    headers: Mapping[str, Any],
    body: str,
    *,
    rendered: bool = False,
) -> Classification:
    """Classify one completed exchange.

    ``body`` only has to carry the leading bytes: every signature here is a short literal. Header
    names are matched case-insensitively and only the signatures this module names are trusted, so
    an ordinary fronted response that happens to carry ``cf-ray``, ``server`` or a ``<script>`` tag
    stays an ordinary success.

    Both stages inspect at most 64 KiB. ``rendered=True`` skips the HTTP-only JavaScript-shell
    routing heuristic, while preserving header and block-status challenge detection.
    """
    body = body[:65536]
    lowered = {str(name).lower(): str(value).lower() for name, value in headers.items()}
    if lowered.get(_CF_MITIGATED_HEADER, "").strip() == _CF_MITIGATED_CHALLENGE:
        return _classify(CLOUDFLARE_CHALLENGE, "cf-mitigated: challenge")
    if status_code in _AUTH_STATUSES or _AUTHENTICATE_HEADER in lowered:
        return _classify(AUTH_REQUIRED, f"HTTP {status_code} asks the caller to authenticate")
    if status_code == _RATE_LIMIT_STATUS:
        return _classify(RATE_LIMITED, f"HTTP {_RATE_LIMIT_STATUS} rate limited the request")
    if status_code == _GEO_BLOCK_STATUS:
        return _classify(GEO_BLOCKED, f"HTTP {_GEO_BLOCK_STATUS} refused the region")
    signature = _body_signature(body) if status_code >= _BLOCK_STATUS_FLOOR else None
    if signature is not None:
        return signature
    if not rendered:
        shell = _javascript_shell(body)
        if shell is not None:
            return shell
    if _SUCCESS_STATUS_FLOOR <= status_code < _BLOCK_STATUS_FLOOR:
        return _classify(SUCCESS, f"HTTP {status_code} carries no challenge signature")
    return _classify(UNKNOWN_BLOCK, f"HTTP {status_code} carries no recognized challenge signature")


def classify_network_error(error: BaseException) -> Classification:
    """Classify a request that never produced a response.

    Only the exception's type is reported, so no URL, proxy or credential can reach a caller
    through the reason.
    """
    return _classify(NETWORK_ERROR, f"no response arrived: {type(error).__name__}")


def _classify(category: str, reason: str) -> Classification:
    """Build a classification whose routing decision follows from its category."""
    return Classification(category, reason, category in BROWSER_REQUIRED_CATEGORIES)


def _body_signature(body: str) -> Classification | None:
    """Return the block a body signature names, or ``None`` when no signature matches."""
    lowered = body.lower()
    for category, markers in _BODY_MARKERS:
        for marker in markers:
            if marker in lowered:
                return _classify(category, f"{marker!r} appears in the response body")
    return None


class _VisibleContent(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._hidden = 0
        self.has_text = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"head", "script", "style", "noscript", "template"}:
            self._hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"head", "script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self._hidden and data.strip():
            self.has_text = True


def _javascript_shell(body: str) -> Classification | None:
    """Return the shell a ``<noscript>`` statement names, or ``None`` when the page is not one."""
    match = _JS_REQUIRED_RE.search(body)
    if match is None:
        return None
    visible = _VisibleContent()
    visible.feed(body)
    if visible.has_text:
        return None
    return _classify(JAVASCRIPT_REQUIRED, f"a <noscript> shell asks to {match.group(1).lower()}")


__all__ = [
    "AUTH_REQUIRED",
    "BROWSER_REQUIRED_CATEGORIES",
    "CAPTCHA",
    "CLOUDFLARE_CHALLENGE",
    "GEO_BLOCKED",
    "IP_BLOCKED",
    "JAVASCRIPT_REQUIRED",
    "NETWORK_ERROR",
    "RATE_LIMITED",
    "SUCCESS",
    "UNKNOWN_BLOCK",
    "Classification",
    "classify",
    "classify_network_error",
]
