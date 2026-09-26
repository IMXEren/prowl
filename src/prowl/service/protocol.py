"""FlareSolverr-compatible request parsing and response shaping.

Only the FlareSolverr v1 subset needed by clients is implemented. Unknown fields,
unsupported command names, and browser-controlled headers are rejected with a
deterministic error rather than being silently ignored.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any, Final
from urllib.parse import urlsplit

from prowl.browser.headers import HEADER_SCOPES, normalize_custom_headers

CMD_REQUEST_GET: Final[str] = "request.get"
CMD_REQUEST_POST: Final[str] = "request.post"
CMD_SESSIONS_CREATE: Final[str] = "sessions.create"
CMD_SESSIONS_LIST: Final[str] = "sessions.list"
CMD_SESSIONS_DESTROY: Final[str] = "sessions.destroy"
CMD_BROWSER_OPEN: Final[str] = "browser.open"
CMD_BROWSER_CLOSE: Final[str] = "browser.close"
CMD_BROWSER_LIST: Final[str] = "browser.list"
CMD_COOKIES_LIST: Final[str] = "cookies.list"

SUPPORTED_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        CMD_REQUEST_GET,
        CMD_REQUEST_POST,
        CMD_SESSIONS_CREATE,
        CMD_SESSIONS_LIST,
        CMD_SESSIONS_DESTROY,
        CMD_BROWSER_OPEN,
        CMD_BROWSER_CLOSE,
        CMD_BROWSER_LIST,
        CMD_COOKIES_LIST,
    },
)

ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
ALLOWED_PROXY_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "socks4", "socks5"})

DEFAULT_TIMEOUT_MS: Final[int] = 60_000
MIN_TIMEOUT_MS: Final[int] = 1_000
MAX_TIMEOUT_MS: Final[int] = 300_000

#: Upper bound for a logical session TTL (one week).
MAX_TTL_MINUTES: Final[int] = 60 * 24 * 7

#: Conservative bound on caller-supplied logical session names.
MAX_SESSION_ID_LENGTH: Final[int] = 128
_SESSION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._:-]+$")

#: Conservative bound on an interactive tab id, which callers echo back to close one.
MAX_TAB_ID_LENGTH: Final[int] = 64

#: Fields each command accepts. Anything else is rejected as unknown.
_FETCH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "cmd",
        "url",
        "maxTimeout",
        "session",
        "session_ttl_minutes",
        "cookies",
        "returnOnlyCookies",
        "headers",
        "headerScope",
        "proxy",
        "postData",
    },
)
_SESSIONS_CREATE_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "session", "session_ttl_minutes"})
_SESSIONS_LIST_FIELDS: Final[frozenset[str]] = frozenset({"cmd"})
_SESSIONS_DESTROY_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "session"})

#: ``browser.open`` takes the fetch's url, timeout and egress selector, the session that
#: binds the tab to an egress, the cookies to seed the browser with before it navigates, and
#: whether to open a second tab for a url that is already open. The tab stays open after the
#: command returns, so there is no field for a method or a body.
_BROWSER_OPEN_FIELDS: Final[frozenset[str]] = frozenset(
    {"cmd", "url", "maxTimeout", "session", "cookies", "newTab", "proxy"},
)
_BROWSER_CLOSE_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "tab"})
_BROWSER_LIST_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "tab"})

#: ``cookies.list`` reads the cookies held by an egress's browser profile. ``url`` scopes the
#: answer to the cookies that would be sent there, and ``proxy`` selects which profile is read.
_COOKIES_LIST_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "url", "proxy"})

#: Fields the ``proxy`` object may carry: a configured url, or a configured name.
_PROXY_FIELDS: Final[frozenset[str]] = frozenset({"url", "name"})

#: The only POST header a caller may set. It carries no browser identity and
#: is needed to send a JSON body from the page's own ``fetch``.
POST_CALLER_HEADERS: Final[frozenset[str]] = frozenset({"content-type"})


def _resolve_version() -> str:
    """Derive the protocol version from the installed package version."""
    try:
        return _distribution_version("prowl")
    except PackageNotFoundError:
        return "0.0.0"


VERSION: Final[str] = f"prowl/{_resolve_version()}"


class ProtocolError(Exception):
    """A caller-visible protocol violation.

    ``http_status`` is the transport status; FlareSolverr clients read
    ``status`` from the JSON body, so validated command errors use 200.
    """

    def __init__(self, message: str, *, http_status: int = 200) -> None:
        super().__init__(message)
        self.message = message
        self.http_status = http_status


@dataclass(frozen=True, slots=True)
class ProxySelection:
    """A caller's egress request: a configured url, or a configured egress name.

    A url keeps its original meaning, accepted only when it matches a configured
    egress. A name selects one configured egress directly. Exactly one of the two
    is set, and neither may embed credentials.
    """

    url: str | None = None
    name: str | None = None


@dataclass(slots=True)
class FetchCommand:
    """A ``request.get`` or ``request.post`` command."""

    cmd: str
    url: str
    timeout_ms: int
    session: str | None = None
    session_ttl_minutes: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    header_scope: str | None = None
    cookies: list[dict[str, Any]] = field(default_factory=list)
    post_data: str | None = None
    return_only_cookies: bool = False
    proxy: ProxySelection | None = None

    @property
    def method(self) -> str:
        """Return the HTTP method implied by the command."""
        return "POST" if self.cmd == CMD_REQUEST_POST else "GET"

    @property
    def timeout_seconds(self) -> int:
        """Return the command timeout rounded up to whole seconds."""
        return max(1, (self.timeout_ms + 999) // 1000)


@dataclass(slots=True)
class SessionsCreateCommand:
    """A ``sessions.create`` command."""

    session: str | None = None
    session_ttl_minutes: int | None = None


@dataclass(slots=True)
class SessionsListCommand:
    """A ``sessions.list`` command."""


@dataclass(slots=True)
class SessionsDestroyCommand:
    """A ``sessions.destroy`` command."""

    session: str


@dataclass(slots=True)
class BrowserOpenCommand:
    """A ``browser.open`` command, which leaves the tab open when it returns."""

    url: str
    timeout_ms: int
    session: str | None = None
    cookies: list[dict[str, Any]] = field(default_factory=list)
    new_tab: bool = False
    proxy: ProxySelection | None = None

    @property
    def timeout_seconds(self) -> int:
        """Return the command timeout rounded up to whole seconds."""
        return max(1, (self.timeout_ms + 999) // 1000)


@dataclass(slots=True)
class CookiesListCommand:
    """A ``cookies.list`` command. Without a url it reports every cookie in the profile."""

    url: str | None = None
    proxy: ProxySelection | None = None


@dataclass(slots=True)
class BrowserCloseCommand:
    """A ``browser.close`` command. Without a tab it closes every interactive tab."""

    tab: str | None = None


@dataclass(slots=True)
class BrowserListCommand:
    """A ``browser.list`` command.

    Without a tab it only reports the open tabs. A named tab is the one the caller is
    displaying, and only that tab's idle countdown is refreshed.
    """

    tab: str | None = None


Command = (
    FetchCommand
    | SessionsCreateCommand
    | SessionsListCommand
    | SessionsDestroyCommand
    | BrowserOpenCommand
    | BrowserCloseCommand
    | BrowserListCommand
    | CookiesListCommand
)


def now_ms() -> int:
    """Return the current epoch time in milliseconds."""
    return int(time.time() * 1000)


def parse_request(payload: Any) -> Command:
    """Validate *payload* and return a typed command.

    :raises ProtocolError: for malformed commands, URLs, unknown fields, or
        browser-controlled headers.
    """
    if not isinstance(payload, dict):
        msg = "request body must be a JSON object"
        raise ProtocolError(msg, http_status=400)

    cmd = payload.get("cmd")
    if not isinstance(cmd, str) or not cmd:
        msg = "missing required field: cmd"
        raise ProtocolError(msg, http_status=400)
    if cmd not in SUPPORTED_COMMANDS:
        msg = f"unsupported cmd: {cmd}"
        raise ProtocolError(msg, http_status=400)

    if cmd in (CMD_REQUEST_GET, CMD_REQUEST_POST):
        _reject_unknown_fields(payload, _FETCH_FIELDS)
        return _parse_fetch(payload, cmd)
    return _parse_command(payload, cmd)


def _parse_command(payload: dict[str, Any], cmd: str) -> Command:
    """Parse a command other than a fetch, which owns its own field set.

    :raises ProtocolError: for unknown fields or malformed values.
    """
    if cmd == CMD_SESSIONS_CREATE:
        _reject_unknown_fields(payload, _SESSIONS_CREATE_FIELDS)
        return SessionsCreateCommand(
            session=_optional_session(payload.get("session")),
            session_ttl_minutes=_parse_ttl(payload.get("session_ttl_minutes")),
        )
    if cmd == CMD_SESSIONS_LIST:
        _reject_unknown_fields(payload, _SESSIONS_LIST_FIELDS)
        return SessionsListCommand()
    if cmd == CMD_BROWSER_OPEN:
        _reject_unknown_fields(payload, _BROWSER_OPEN_FIELDS)
        return BrowserOpenCommand(
            url=_parse_url(payload.get("url")),
            timeout_ms=_parse_timeout(payload.get("maxTimeout")),
            session=_optional_session(payload.get("session")),
            cookies=_parse_cookies(payload.get("cookies")),
            new_tab=_parse_bool(payload.get("newTab"), "newTab"),
            proxy=_parse_proxy(payload.get("proxy")),
        )
    if cmd == CMD_BROWSER_CLOSE:
        _reject_unknown_fields(payload, _BROWSER_CLOSE_FIELDS)
        return BrowserCloseCommand(tab=_optional_tab(payload.get("tab")))
    if cmd == CMD_BROWSER_LIST:
        _reject_unknown_fields(payload, _BROWSER_LIST_FIELDS)
        return BrowserListCommand(tab=_optional_tab(payload.get("tab")))
    if cmd == CMD_COOKIES_LIST:
        _reject_unknown_fields(payload, _COOKIES_LIST_FIELDS)
        return CookiesListCommand(
            url=_optional_url(payload.get("url")),
            proxy=_parse_proxy(payload.get("proxy")),
        )
    _reject_unknown_fields(payload, _SESSIONS_DESTROY_FIELDS)
    return SessionsDestroyCommand(session=_required_session(payload.get("session")))


def _reject_unknown_fields(payload: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        msg = f"unknown field(s): {', '.join(unknown)}"
        raise ProtocolError(msg, http_status=400)


def _parse_fetch(payload: dict[str, Any], cmd: str) -> FetchCommand:
    url = _parse_url(payload.get("url"))

    header_scope = _parse_header_scope(payload.get("headerScope"), cmd)
    headers = _parse_headers(payload.get("headers"), cmd, header_scope)
    cookies = _parse_cookies(payload.get("cookies"))
    post_data = _parse_post_data(payload.get("postData"), cmd)
    session = _optional_session(payload.get("session"))
    ttl = _parse_ttl(payload.get("session_ttl_minutes"))
    if ttl is not None and session is None:
        msg = "session_ttl_minutes requires a session"
        raise ProtocolError(msg, http_status=400)

    return FetchCommand(
        cmd=cmd,
        url=url,
        timeout_ms=_parse_timeout(payload.get("maxTimeout")),
        session=session,
        session_ttl_minutes=ttl,
        headers=headers,
        header_scope=header_scope,
        cookies=cookies,
        post_data=post_data,
        return_only_cookies=_parse_bool(payload.get("returnOnlyCookies"), "returnOnlyCookies"),
        proxy=_parse_proxy(payload.get("proxy")),
    )


def _parse_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = "missing required field: url"
        raise ProtocolError(msg, http_status=400)
    url = value.strip()
    try:
        parts = urlsplit(url)
        host = parts.hostname
        userinfo = parts.username is not None or parts.password is not None
    except ValueError as exc:
        msg = "url is not a valid URL"
        raise ProtocolError(msg, http_status=400) from exc
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        msg = f"unsupported url scheme: {scheme or '(none)'}; only http and https are allowed"
        raise ProtocolError(msg, http_status=400)
    if not host:
        msg = "url is missing a host"
        raise ProtocolError(msg, http_status=400)
    if userinfo:
        msg = "url must not embed credentials"
        raise ProtocolError(msg, http_status=400)
    return url


def _optional_url(value: Any) -> str | None:
    """Validate a url that scopes a reply rather than addressing one.

    :raises ProtocolError: when a value is given but is not an http or https url.
    """
    if value is None:
        return None
    return _parse_url(value)


def _parse_bool(value: Any, field_name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        msg = f"{field_name} must be a boolean"
        raise ProtocolError(msg, http_status=400)
    return value


def _parse_header_scope(value: Any, cmd: str) -> str | None:
    if value is None:
        return None
    if cmd != CMD_REQUEST_GET:
        msg = "headerScope is supported only for request.get"
        raise ProtocolError(msg, http_status=400)
    if not isinstance(value, str) or value not in HEADER_SCOPES:
        msg = f"headerScope must be one of: {', '.join(sorted(HEADER_SCOPES))}"
        raise ProtocolError(msg, http_status=400)
    return value


def _parse_header_object(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        msg = "headers must be an object of string values"
        raise ProtocolError(msg, http_status=400)
    try:
        return normalize_custom_headers(value)
    except ValueError as error:
        raise ProtocolError(str(error), http_status=400) from error


def _parse_headers(value: Any, cmd: str, header_scope: str | None) -> dict[str, str]:
    lowered = _parse_header_object(value)
    if not lowered:
        if header_scope is not None:
            msg = "headerScope requires at least one request header"
            raise ProtocolError(msg, http_status=400)
        return lowered
    if cmd == CMD_REQUEST_GET:
        if header_scope is None:
            msg = "request.get headers require headerScope=document or headerScope=origin"
            raise ProtocolError(msg, http_status=400)
        return lowered
    unsupported = sorted(name for name in lowered if name not in POST_CALLER_HEADERS)
    if unsupported:
        msg = f"header(s) are not accepted: {', '.join(unsupported)}"
        raise ProtocolError(msg, http_status=400)
    return lowered


def _parse_post_data(value: Any, cmd: str) -> str | None:
    if cmd == CMD_REQUEST_POST:
        if isinstance(value, dict):
            return json.dumps(value)
        if value is None:
            return ""
        if not isinstance(value, str):
            msg = "postData must be a string or object"
            raise ProtocolError(msg, http_status=400)
        return value
    if value is not None:
        msg = "postData is only valid for request.post"
        raise ProtocolError(msg, http_status=400)
    return None


def _parse_proxy(value: Any) -> ProxySelection | None:
    """Parse the ``proxy`` field into a configured url or a configured name.

    :raises ProtocolError: when the field is malformed, names both a url and a
        name, or carries a proxy URL that policy rejects.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "proxy must be an object with a url or a name"
        raise ProtocolError(msg, http_status=400)
    unknown = sorted(key for key in value if key not in _PROXY_FIELDS)
    if unknown:
        msg = "proxy only supports the url and name fields"
        raise ProtocolError(msg, http_status=400)
    name = value.get("name")
    url = value.get("url")
    if name is not None and url is not None:
        msg = "proxy must name either a url or an egress name, not both"
        raise ProtocolError(msg, http_status=400)
    if name is not None:
        if not isinstance(name, str) or not name.strip():
            msg = "proxy.name must be a non-empty string"
            raise ProtocolError(msg, http_status=400)
        return ProxySelection(name=name.strip())
    if url is None:
        msg = "proxy must name either a url or an egress name"
        raise ProtocolError(msg, http_status=400)
    if not isinstance(url, str) or not url.strip():
        msg = "proxy.url must be a non-empty string"
        raise ProtocolError(msg, http_status=400)
    clean_url = url.strip()
    error = _classify_proxy_url(clean_url)
    if error is not None:
        raise ProtocolError(error, http_status=400)
    return ProxySelection(url=clean_url)


def _classify_proxy_url(url: str) -> str | None:
    """Return a caller-safe error for *url*, or ``None`` when it is a valid proxy URL."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        userinfo = parts.username is not None or parts.password is not None
    except ValueError:
        return "proxy url is not valid"
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_PROXY_SCHEMES:
        return f"unsupported proxy scheme: {scheme or '(none)'}; supported: http, https, socks4, socks5"
    if not host:
        return "proxy url must include a host"
    if userinfo:
        return "proxy url must not embed credentials; inject credentials at a local hop"
    return None


def validate_proxy_url(url: str) -> str:
    """Return *url* when it is a valid credential-free proxy URL.

    :raises ValueError: when the proxy URL is malformed or unsupported.
    """
    error = _classify_proxy_url(url)
    if error is not None:
        raise ValueError(error)
    return url


def _parse_timeout(value: Any) -> int:
    if value is None:
        return DEFAULT_TIMEOUT_MS
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = "maxTimeout must be a finite number of milliseconds"
        raise ProtocolError(msg, http_status=400)
    return int(min(MAX_TIMEOUT_MS, max(MIN_TIMEOUT_MS, value)))


def _parse_ttl(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        msg = "session_ttl_minutes must be a finite number of minutes"
        raise ProtocolError(msg, http_status=400)
    ttl = int(value)
    if ttl <= 0:
        msg = "session_ttl_minutes must be positive"
        raise ProtocolError(msg, http_status=400)
    if ttl > MAX_TTL_MINUTES:
        msg = f"session_ttl_minutes must be at most {MAX_TTL_MINUTES}"
        raise ProtocolError(msg, http_status=400)
    return ttl


def _parse_cookies(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        msg = "cookies must be a list"
        raise ProtocolError(msg, http_status=400)
    parsed: list[dict[str, Any]] = []
    for index, cookie in enumerate(value):
        if not isinstance(cookie, dict):
            msg = f"cookie[{index}] must be an object"
            raise ProtocolError(msg, http_status=400)
        name = cookie.get("name")
        cookie_value = cookie.get("value")
        if not isinstance(name, str) or not isinstance(cookie_value, str):
            msg = f"cookie[{index}] requires string name and value"
            raise ProtocolError(msg, http_status=400)
        entry: dict[str, Any] = {"name": name, "value": cookie_value}
        for key in ("domain", "path"):
            if isinstance(cookie.get(key), str):
                entry[key] = cookie[key]
        parsed.append(entry)
    return parsed


def _optional_session(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = "session must be a non-empty string"
        raise ProtocolError(msg, http_status=400)
    session = value.strip()
    if len(session) > MAX_SESSION_ID_LENGTH:
        msg = f"session must be at most {MAX_SESSION_ID_LENGTH} characters"
        raise ProtocolError(msg, http_status=400)
    if not _SESSION_ID_RE.match(session):
        msg = "session may only contain letters, digits, '.', '_', ':', or '-'"
        raise ProtocolError(msg, http_status=400)
    return session


def _required_session(value: Any) -> str:
    session = _optional_session(value)
    if session is None:
        msg = "missing required field: session"
        raise ProtocolError(msg, http_status=400)
    return session


def _optional_tab(value: Any) -> str | None:
    """Validate a tab id, which is generated by the service and echoed back by the caller."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = "tab must be a non-empty string"
        raise ProtocolError(msg, http_status=400)
    tab = value.strip()
    if len(tab) > MAX_TAB_ID_LENGTH:
        msg = f"tab must be at most {MAX_TAB_ID_LENGTH} characters"
        raise ProtocolError(msg, http_status=400)
    if not _SESSION_ID_RE.match(tab):
        msg = "tab may only contain letters, digits, '.', '_', ':', or '-'"
        raise ProtocolError(msg, http_status=400)
    return tab


def tab_payload(tab: Any) -> dict[str, Any]:
    """Build the caller-visible description of an interactive tab.

    Only the identity and location of the tab are reported. The egress it leaves through is
    deliberately omitted, so a response can never disclose a proxy url or its credentials.
    """
    return {
        "id": tab.tab_id,
        "url": tab.url,
        "title": tab.title,
        "status": tab.status_code if isinstance(tab.status_code, int) else 0,
    }


def cookies_payload(cookies: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the ``cookies.list`` reply fragment in the shape the fetch replies use.

    Each cookie keeps the exact object a fetch reports in ``solution.cookies``, so a client
    reads one cookie shape whether it fetched a page or asked for the profile's cookies.
    """
    return {"cookies": [dict(cookie) for cookie in cookies]}


def solution_payload(  # noqa: PLR0913
    *,
    url: str,
    status_code: int,
    headers: dict[str, str],
    response: str,
    cookies: list[dict[str, Any]],
    user_agent: str,
) -> dict[str, Any]:
    """Build the FlareSolverr ``solution`` object."""
    return {
        "url": url,
        "status": status_code,
        "headers": headers,
        "response": response,
        "cookies": cookies,
        "userAgent": user_agent,
    }


def ok_response(start: int, *, solution: dict[str, Any] | None = None, message: str = "") -> dict[str, Any]:
    """Build a successful FlareSolverr response envelope."""
    return {
        "status": "ok",
        "message": message,
        "startTimestamp": start,
        "endTimestamp": now_ms(),
        "version": VERSION,
        "solution": solution if solution is not None else {},
    }


def error_response(start: int, message: str) -> dict[str, Any]:
    """Build a failed FlareSolverr response envelope.

    The message is intentionally caller-safe: it never embeds tracebacks,
    credentials, proxy passwords, profile paths, or browser internals.
    """
    return {
        "status": "error",
        "message": message,
        "startTimestamp": start,
        "endTimestamp": now_ms(),
        "version": VERSION,
        "solution": {},
    }
