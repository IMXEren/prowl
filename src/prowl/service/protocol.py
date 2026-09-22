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

CMD_REQUEST_GET: Final[str] = "request.get"
CMD_REQUEST_POST: Final[str] = "request.post"
CMD_SESSIONS_CREATE: Final[str] = "sessions.create"
CMD_SESSIONS_LIST: Final[str] = "sessions.list"
CMD_SESSIONS_DESTROY: Final[str] = "sessions.destroy"

SUPPORTED_COMMANDS: Final[frozenset[str]] = frozenset(
    {CMD_REQUEST_GET, CMD_REQUEST_POST, CMD_SESSIONS_CREATE, CMD_SESSIONS_LIST, CMD_SESSIONS_DESTROY},
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
_HTTP_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")

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
        "proxy",
        "postData",
    },
)
_SESSIONS_CREATE_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "session", "session_ttl_minutes"})
_SESSIONS_LIST_FIELDS: Final[frozenset[str]] = frozenset({"cmd"})
_SESSIONS_DESTROY_FIELDS: Final[frozenset[str]] = frozenset({"cmd", "session"})

#: Request headers the browser owns. A caller cannot override them without
#: breaking the shared profile's trust or the browser's own framing.
FORBIDDEN_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "content-encoding",
        "content-length",
        "cookie",
        "expect",
        "host",
        "keep-alive",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    },
)


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


@dataclass(slots=True)
class FetchCommand:
    """A ``request.get`` or ``request.post`` command."""

    cmd: str
    url: str
    timeout_ms: int
    session: str | None = None
    session_ttl_minutes: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    cookies: list[dict[str, Any]] = field(default_factory=list)
    post_data: str | None = None
    return_only_cookies: bool = False
    proxy_url: str | None = None

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


Command = FetchCommand | SessionsCreateCommand | SessionsListCommand | SessionsDestroyCommand


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
    if cmd == CMD_SESSIONS_CREATE:
        _reject_unknown_fields(payload, _SESSIONS_CREATE_FIELDS)
        return SessionsCreateCommand(
            session=_optional_session(payload.get("session")),
            session_ttl_minutes=_parse_ttl(payload.get("session_ttl_minutes")),
        )
    if cmd == CMD_SESSIONS_LIST:
        _reject_unknown_fields(payload, _SESSIONS_LIST_FIELDS)
        return SessionsListCommand()
    _reject_unknown_fields(payload, _SESSIONS_DESTROY_FIELDS)
    return SessionsDestroyCommand(session=_required_session(payload.get("session")))


def _reject_unknown_fields(payload: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(key for key in payload if key not in allowed)
    if unknown:
        msg = f"unknown field(s): {', '.join(unknown)}"
        raise ProtocolError(msg, http_status=400)


def _parse_fetch(payload: dict[str, Any], cmd: str) -> FetchCommand:
    url = _parse_url(payload.get("url"))

    headers = _parse_headers(payload.get("headers"))
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
        cookies=cookies,
        post_data=post_data,
        return_only_cookies=_parse_bool(payload.get("returnOnlyCookies"), "returnOnlyCookies"),
        proxy_url=_parse_proxy(payload.get("proxy")),
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


def _parse_bool(value: Any, field_name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        msg = f"{field_name} must be a boolean"
        raise ProtocolError(msg, http_status=400)
    return value


def _parse_headers(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        msg = "headers must be an object of string values"
        raise ProtocolError(msg, http_status=400)
    for name, val in value.items():
        if not _HTTP_TOKEN_RE.match(name):
            msg = f"invalid header name: {name!r}"
            raise ProtocolError(msg, http_status=400)
        if "\r" in val or "\n" in val:
            msg = f"header {name!r} value must not contain CR or LF"
            raise ProtocolError(msg, http_status=400)
    lowered = {key.lower(): val for key, val in value.items()}
    forbidden = sorted(name for name in lowered if name in FORBIDDEN_HEADERS)
    if forbidden:
        msg = f"header(s) cannot be set by callers: {', '.join(forbidden)}"
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


def _parse_proxy(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "proxy must be an object with a url"
        raise ProtocolError(msg, http_status=400)
    unknown = sorted(key for key in value if key != "url")
    if unknown:
        msg = "proxy only supports the url field"
        raise ProtocolError(msg, http_status=400)
    url = value.get("url")
    if not isinstance(url, str) or not url.strip():
        msg = "proxy.url must be a non-empty string"
        raise ProtocolError(msg, http_status=400)
    url = url.strip()
    error = _classify_proxy_url(url)
    if error is not None:
        raise ProtocolError(error, http_status=400)
    return url


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
