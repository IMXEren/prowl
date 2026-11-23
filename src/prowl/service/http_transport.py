"""Identity-matched HTTP fast path for one live browser context.

A client belongs to exactly one live browser context. Before every request it reads that
context's cookies, sends them behind the same client identity the running browser reports, and
mirrors back only the cookies the exchange actually created, changed or deleted. The live context
stays the single owner of session state: this client keeps no cookie of its own beyond the request
in flight, never replaces the context's jar, and never touches a cookie the exchange did not
touch.

The identity match is exact. A browser persona with no explicit impersonation profile, an
unsupported platform, or client-hint brands that disagree with its own user agent is refused
rather than approximated, so the fast path can never present a version or platform the browser
would not.

Known boundaries:

- Partitioned cookies cannot be represented in a ``CookieJar``, so a context that holds one is
  refused before any request is sent.
- A caller cookie that the site does not itself set stays request-scoped; unlike the browser path,
  it is not written into the context. Only what the site accepts is persisted.
- The accepted jar decides a cookie's scope, so a ``Set-Cookie`` the site was not allowed to set can
  neither move an existing cookie nor make one look deleted. ``Domain`` is never re-applied from an
  unvalidated header.
- ``SameSite`` for a changed cookie comes from the ``Set-Cookie`` that declared it - searched through
  the redirect chain and the final response, oldest first, keyed by the cookie's own name, domain
  and path. A declaration that omits it resets the cookie to the browser's own default, so the flag
  is never inherited from an older declaration. Untouched cookies are not written at all, so their
  existing metadata is preserved.
- ``HttpOnly`` is read back from the accepted jar, which is where libcurl records the ``#HttpOnly_``
  marker, so a re-set cookie both gains and loses the flag truthfully.
- The result reports the live context's own cookies after the write-back, never a reconstruction
  from the transport's jar.
- Redirects are followed by this transport rather than by libcurl, one hop at a time, so a
  response's cookie changes are validated and mirrored before its ``Location`` is used. Only a
  hop's supported cookies are written, and a hop that declares a ``Partitioned``, ``SameParty`` or
  ``SameSite=None`` without ``Secure`` cookie is refused before anything from it is written or
  followed.
- A redirect to another host or scheme is refused when a cookie matching the target is ``Strict``,
  or when a redirect that keeps its body would send a ``Lax`` cookie, because deciding that needs a
  schemeful-site calculation: no public suffix list is consulted and no eTLD+1 is guessed. Cookies
  the target does not match never block a redirect.
- A cancellation during the cookie write-back can leave part of that exchange's cookie changes
  applied. The lock and the session stay usable and no work is left running in the background.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
from dataclasses import dataclass
from email.message import Message
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from http.cookies import CookieError
from typing import TYPE_CHECKING, Any, Final, Protocol
from urllib.parse import urljoin, urlsplit
from urllib.request import Request

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError
from loguru import logger

from prowl.browser.headers import (
    HEADER_SCOPE_DOCUMENT,
    HEADER_SCOPE_ORIGIN,
    HEADER_SCOPES,
    browser_owns_header,
    normalize_custom_headers,
)
from prowl.service.classification import Classification, classify
from prowl.service.errors import CallerSafeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    from playwright._impl._api_structures import SetCookieParam

#: How much of a response body is scanned for a challenge signature.
CLASSIFICATION_BODY_BYTES: Final[int] = 65536

#: Chrome majors this transport has an explicit, desktop Chrome impersonation profile for.
SUPPORTED_CHROME_MAJORS: Final[frozenset[int]] = frozenset({142, 145, 146, 150})

#: Client-hint platform values the supported profiles describe.
SUPPORTED_PLATFORMS: Final[frozenset[str]] = frozenset({"Windows", "macOS", "Linux"})
SUPPORTED_ARCHITECTURES: Final[frozenset[str]] = frozenset({"x86", "arm"})
SUPPORTED_BITNESSES: Final[frozenset[str]] = frozenset({"32", "64"})

#: A browser reports a session cookie as an expiry of -1.
_SESSION_COOKIE_EXPIRY: Final[int] = -1

_SET_COOKIE: Final[str] = "set-cookie"
_COOKIE_POLICY = DefaultCookiePolicy()

_CHROME_MAJOR_RE: Final[re.Pattern[str]] = re.compile(r"Chrome/(\d+)")

#: A cookie's identity as the accepted jar and a live browser context both key it: name, domain
#: without its leading dot, path, and whether it is a domain cookie rather than host-only.
_CookieKey = tuple[str, str, str, bool]

#: The part of a cookie's identity a ``Set-Cookie`` declaration chooses: name, domain, path, and
#: whether the declaration makes it a domain cookie rather than a host-only one.
_DeclarationKey = tuple[str, str, str, bool]

#: Statuses a redirect can use, how many one exchange may follow, and the schemes it may reach.
_REDIRECT_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})
_POST_TO_GET_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303})
_MAX_REDIRECTS: Final[int] = 20
_LOCATION: Final[str] = "location"
_HTTP_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}


class HttpTransportError(CallerSafeError):
    """Base class for HTTP fast-path failures whose message is safe to return."""


class UnsupportedIdentityError(HttpTransportError):
    """Raised when a browser persona has no matching HTTP impersonation profile."""


class UnsupportedCookieError(HttpTransportError):
    """Raised when cookie state cannot be represented or mirrored without changing it."""


class IdentityHeaderConflictError(HttpTransportError):
    """Raised when a caller tries to set a header the client identity owns."""


class HttpNetworkError(HttpTransportError):
    """Raised when a request produced no response; the cause carries the transport failure."""


@dataclass(frozen=True, slots=True)
class ObservedBrowserIdentity:
    """What a running browser reports about itself, read from its own client-hint metadata.

    ``brands`` is the ``sec-ch-ua`` list (major versions) and ``full_version_list`` the
    ``sec-ch-ua-full-version-list`` list (full versions), each in the order the browser reports.
    """

    user_agent: str
    platform: str
    brands: tuple[tuple[str, str], ...] = ()
    full_version_list: tuple[tuple[str, str], ...] = ()
    architecture: str = ""
    bitness: str = ""
    platform_version: str = ""
    mobile: bool = False


@dataclass(frozen=True, slots=True)
class HttpIdentity:
    """The client identity matched to one live browser identity."""

    user_agent: str
    platform: str
    chrome_major: int
    impersonate: str
    headers: tuple[tuple[str, str], ...]

    def request_headers(self) -> dict[str, str]:
        """Return the headers every request from this identity has to send."""
        return dict(self.headers)


class CookieContext(Protocol):
    """The live browser context a client reads and mirrors cookies through.

    ``playwright.async_api.BrowserContext`` satisfies this protocol as it is; tests use a
    stand-in. Only these three operations are used, so a client can never reach past the state the
    context owns into another context's.
    """

    async def cookies(self) -> Sequence[Mapping[str, Any]]:
        """Return every cookie the context currently holds."""
        ...

    async def add_cookies(self, cookies: Sequence[SetCookieParam]) -> None:
        """Add or replace cookies in the context."""
        ...

    async def clear_cookies(
        self,
        *,
        name: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> None:
        """Remove the cookies matching the exact filters given."""
        ...


@dataclass(slots=True)
class HttpResult:
    """One completed HTTP exchange.

    ``headers`` is lowercase-keyed and joins repeated list-valued names, except ``set-cookie``:
    duplicates of that header are separate cookies, so the first one stands in the mapping and
    every one of them is kept in ``set_cookie_headers``. ``cookies`` is the live context's own
    cookie list as it stood after the write-back, in the shape the context reports it.

    ``body_bytes`` is curl's decoded entity content, not the original compressed or framed
    transfer bytes; ``header_items`` keeps the final response headers as repeated name/value pairs
    in the order curl reported them. Both are the final response's alone and ``None`` when not
    available, while ``b""`` is a genuinely empty body.
    """

    url: str
    status_code: int
    headers: dict[str, str]
    body: str
    cookies: list[dict[str, Any]]
    set_cookie_headers: tuple[str, ...]
    body_bytes: bytes | None = None
    header_items: tuple[tuple[str, str], ...] | None = None

    def classify(self) -> Classification:
        """Classify this response over a bounded prefix of its body."""
        return classify(self.status_code, self.headers, self.body[:CLASSIFICATION_BODY_BYTES])


def chrome_major_from_user_agent(user_agent: str) -> int | None:
    """Return the Chrome major version *user_agent* claims, or ``None`` when it names none."""
    match = _CHROME_MAJOR_RE.search(user_agent)
    return int(match.group(1)) if match is not None else None


def resolve_http_identity(observed: ObservedBrowserIdentity) -> HttpIdentity:
    """Match a running browser's own persona to an explicit impersonation profile.

    The match is exact: the profile's Chrome major has to be the major the browser reports. A
    persona that cannot be matched exactly is refused instead of approximated, so nothing here
    changes the browser it was read from.

    :raises UnsupportedIdentityError: when no supported profile fits the observed persona.
    """
    major = chrome_major_from_user_agent(observed.user_agent)
    if major is None:
        msg = "the reported user agent does not name a Chrome major version"
        raise UnsupportedIdentityError(msg)
    if major not in SUPPORTED_CHROME_MAJORS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_CHROME_MAJORS))
        msg = f"no HTTP impersonation profile for Chrome {major}; supported majors: {supported}"
        raise UnsupportedIdentityError(msg)
    if observed.mobile:
        msg = "a mobile persona has no supported desktop HTTP impersonation profile"
        raise UnsupportedIdentityError(msg)
    if observed.platform not in SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(SUPPORTED_PLATFORMS))
        msg = f"unsupported client-hint platform {observed.platform!r}; supported platforms: {supported}"
        raise UnsupportedIdentityError(msg)
    if observed.architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        msg = f"unsupported client-hint architecture {observed.architecture!r}; supported: {supported}"
        raise UnsupportedIdentityError(msg)
    if observed.bitness not in SUPPORTED_BITNESSES:
        supported = ", ".join(sorted(SUPPORTED_BITNESSES))
        msg = f"unsupported client-hint bitness {observed.bitness!r}; supported: {supported}"
        raise UnsupportedIdentityError(msg)
    full_version_list = _brand_list(observed.full_version_list, major, "sec-ch-ua-full-version-list")
    headers = (
        ("user-agent", observed.user_agent),
        ("sec-ch-ua", _brand_list(observed.brands, major, "sec-ch-ua")),
        ("sec-ch-ua-arch", _quoted(observed.architecture)),
        ("sec-ch-ua-bitness", _quoted(observed.bitness)),
        ("sec-ch-ua-full-version-list", full_version_list),
        ("sec-ch-ua-mobile", "?0"),
        ("sec-ch-ua-platform", _quoted(observed.platform)),
        ("sec-ch-ua-platform-version", _quoted(observed.platform_version)),
    )
    return HttpIdentity(
        user_agent=observed.user_agent,
        platform=observed.platform,
        chrome_major=major,
        impersonate=f"chrome{major}",
        headers=headers,
    )


class HttpClient:
    """The HTTP fast path for one live browser context.

    One client, one lock and one connection pool belong to one actual context. The caller owns the
    context's lifetime and closes the client when the context goes away.
    """

    def __init__(
        self,
        *,
        context: CookieContext,
        identity: HttpIdentity,
        proxy: str | None = None,
    ) -> None:
        """Bind a client to *context*, sending as *identity* behind *proxy*.

        ``proxy`` is the egress the same browser runs behind. The proxy is always set explicitly -
        to that egress, or to an empty one that turns proxying off - because libcurl reads the
        proxy environment variables itself when no proxy is set. A client can therefore never leave
        through a different egress than its browser did, and TLS verification is never disabled.
        """
        self._context = context
        self._identity = identity
        self._lock = asyncio.Lock()
        self._closed = False
        options: dict[str, Any] = {
            "impersonate": identity.impersonate,
            "proxies": {"all": proxy or ""},
            "trust_env": False,
            "verify": True,
        }
        self._session = AsyncSession(**options)

    @property
    def identity(self) -> HttpIdentity:
        """The client identity every request from this client sends."""
        return self._identity

    async def fetch(  # noqa: PLR0913 - the fields one validated request already carries
        self,
        url: str,
        *,
        method: Literal["GET", "POST"] = "GET",
        headers: Mapping[str, str] | None = None,
        cookies: Sequence[Mapping[str, Any]] = (),
        content: str | bytes | None = None,
        deadline_seconds: float,
        header_scope: str = HEADER_SCOPE_DOCUMENT,
    ) -> HttpResult:
        """Execute one request, following redirects itself, and mirror the cookies it changed.

        ``header_scope`` says where the caller's own headers may go: ``document`` sends them on the
        initial request only, and ``origin`` sends them whenever the request is still for the
        original scheme, host and effective port, which is what the browser path does for the page's
        own requests.

        One budget covers the whole call: waiting for this client's lock, reading the context's
        cookies, the transfer with its redirects, and the write-back. Cancelling the call releases
        the lock and leaves the client usable.

        :raises IdentityHeaderConflictError: when *headers* carries a header the identity owns.
        :raises UnsupportedCookieError: when the context or *cookies* holds state that cannot be
            mirrored without changing it.
        :raises HttpNetworkError: when the request times out or fails before a response arrives.
        """
        if header_scope not in HEADER_SCOPES:
            msg = f"headerScope must be one of: {', '.join(sorted(HEADER_SCOPES))}"
            raise HttpTransportError(msg)
        if self._closed:
            msg = "the HTTP client is closed"
            raise HttpTransportError(msg)
        budget = max(0.0, float(deadline_seconds))
        request_cookies = _request_cookie_jar(cookies) if cookies else None
        try:
            async with asyncio.timeout(budget), self._lock:
                # A caller that queued behind this client's lock can only run once the lock is free,
                # which may be after the client was closed.
                if self._closed:
                    msg = "the HTTP client is closed"
                    raise HttpTransportError(msg)
                return await self._follow(
                    url,
                    method,
                    self._identity.request_headers(),
                    _caller_headers(headers),
                    request_cookies,
                    content,
                    budget,
                    header_scope,
                )
        except TimeoutError as error:
            msg = f"HTTP {method} exceeded its {budget:g}s budget"
            raise HttpNetworkError(msg) from error

    async def aclose(self) -> None:
        """Close the HTTP session, leaving the browser context exactly as it is."""
        async with self._lock:
            if self._closed:
                return
            await self._session.close()
            self._closed = True

    async def _follow(  # noqa: PLR0913, PLR0917 - one validated request, passed on as received
        self,
        url: str,
        method: Literal["GET", "POST"],
        identity_headers: Mapping[str, str],
        caller_headers: Mapping[str, str],
        caller_cookies: CookieJar | None,
        content: str | bytes | None,
        budget: float,
        header_scope: str,
    ) -> HttpResult:
        """Run one exchange hop by hop, mirroring each hop's cookies before the next is sent.

        Redirects are followed here rather than by libcurl for two reasons: a response's cookie
        semantics have to be refused before its ``Location`` is followed, and the context has to be
        told about a hop's cookies before the next hop goes out, so a later network failure cannot
        lose browsing state that already arrived. One budget covers every hop, and the session's jar
        is refreshed from the context between hops, which is the authority on what to send.
        """
        deadline = asyncio.get_running_loop().time() + budget
        origin = _origin(url)
        current = url
        hop_method = method
        hop_content = content
        hops = 0
        cross_site = False
        set_cookie_headers: list[str] = []
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                msg = f"HTTP {method} exceeded its {budget:g}s budget"
                raise HttpNetworkError(msg)
            headers = dict(identity_headers)
            if (hops == 0 and header_scope == HEADER_SCOPE_DOCUMENT) or (
                header_scope == HEADER_SCOPE_ORIGIN and _origin(current) == origin
            ):
                headers.update(caller_headers)
            previous, loaded, jar = await self._context_jar()
            self._session.cookies = jar
            try:
                response = await self._session.request(
                    hop_method,
                    current,
                    headers=headers,
                    cookies=caller_cookies if hops == 0 else None,
                    data=hop_content if hop_method == "POST" else None,
                    timeout=remaining,
                    allow_redirects=False,
                )
            except RequestsError as error:
                msg = f"HTTP {hop_method} failed before a response arrived: {type(error).__name__}"
                raise HttpNetworkError(msg) from error
            sources = _declaration_sources(response)
            hop_set_cookies = tuple(header for _, hop_headers in sources for header in hop_headers)
            _reject_unsupported_cookies(hop_set_cookies, response.status_code)
            await self._mirror_cookies(previous, loaded, jar, sources)
            set_cookie_headers.extend(hop_set_cookies)
            location = response.headers.get(_LOCATION)
            if response.status_code not in _REDIRECT_STATUSES or not location:
                return HttpResult(
                    url=response.url,
                    status_code=response.status_code,
                    headers=_response_headers(response),
                    body=response.text,
                    cookies=[dict(entry) for entry in await self._context.cookies()],
                    set_cookie_headers=tuple(set_cookie_headers),
                    body_bytes=response.content,
                    header_items=tuple(response.headers.multi_items()),
                )
            if hops >= _MAX_REDIRECTS:
                msg = f"HTTP {method} exceeded {_MAX_REDIRECTS} redirects"
                raise HttpNetworkError(msg)
            next_url = urljoin(current, location.strip())
            if urlsplit(next_url).scheme.lower() not in _HTTP_SCHEMES:
                msg = "redirect left HTTP(S), which this transport cannot follow"
                raise HttpNetworkError(msg)
            next_method, next_content = _redirected_request(hop_method, hop_content, response.status_code)
            cross_site = cross_site or _origin(current)[:2] != _origin(next_url)[:2]
            unsafe = _unsafe_redirect(
                await self._context.cookies(),
                url,
                next_url,
                next_method,
                cross_site=cross_site,
            )
            if unsafe is not None:
                raise UnsupportedCookieError(unsafe)
            hop_method, hop_content = next_method, next_content
            current = next_url
            hops += 1

    async def _context_jar(self) -> tuple[dict[_CookieKey, dict[str, Any]], dict[_CookieKey, Cookie], CookieJar]:
        """Read the live context into a fresh jar, refusing state a jar cannot hold.

        A partitioned cookie is refused here, before anything is sent: flattening it would send it
        to sites it was never scoped to. The context's own cookie objects and the jar cookies they
        were loaded as are both kept, so an untouched cookie can be told from a changed one without
        a lossy round trip.
        """
        previous: dict[_CookieKey, dict[str, Any]] = {}
        loaded: dict[_CookieKey, Cookie] = {}
        jar = CookieJar()
        scopes: dict[tuple[str, str, str], bool] = {}
        for entry in await self._context.cookies():
            key, cookie = _jar_cookie(entry, source="browser context")
            if key[:3] in scopes and scopes[key[:3]] != key[3]:
                msg = "host-only and domain cookies overlap; use browser mode"
                raise UnsupportedCookieError(msg)
            scopes[key[:3]] = key[3]
            previous[key] = dict(entry)
            loaded[key] = cookie
            try:
                jar.set_cookie(cookie)
            except (CookieError, ValueError) as error:
                msg = f"browser context cookie {cookie.name!r} cannot be loaded into a jar"
                raise UnsupportedCookieError(msg) from error
        return previous, loaded, jar

    async def _mirror_cookies(
        self,
        previous: Mapping[_CookieKey, Mapping[str, Any]],
        loaded: Mapping[_CookieKey, Cookie],
        jar: CookieJar,
        sources: Sequence[tuple[str, Sequence[str]]],
    ) -> None:
        """Write back only the cookies this exchange changed, and only at their own keys.

        A cookie this exchange declared is written as its declaration says, including the metadata
        the declaration omits, which resets it to the browser's own default. A cookie no declaration
        named is written only if the transfer really changed it. The accepted jar is the authority
        on scope: a ``Set-Cookie`` the site was not allowed to set never reaches it, so it can
        neither move an existing cookie nor make one look deleted.
        """
        declarations = _declarations(sources)
        stored = {_cookie_key(cookie): cookie for cookie in jar}
        scopes = {key[:3]: key[3] for key in previous}
        for key in stored:
            if key[:3] in scopes and scopes[key[:3]] != key[3]:
                msg = "response cookie scope overlaps an existing native cookie; use browser mode"
                raise UnsupportedCookieError(msg)
        writes: list[SetCookieParam] = []
        for key, cookie in stored.items():
            before = previous.get(key)
            if key in declarations:
                entry = _browser_cookie(key, cookie, declarations)
                if before is None or _browser_shape(before) != _browser_shape(entry):
                    writes.append(entry)
                continue
            original = loaded.get(key)
            if original is None or _content_differs(original, cookie):
                writes.append(_browser_cookie(key, cookie, declarations))
        removals = [key for key in previous if key not in stored and key in declarations]
        if writes:
            await self._context.add_cookies(writes)
        for name, domain, path, domain_cookie in removals:
            await self._context.clear_cookies(
                name=name,
                domain=f".{domain}" if domain_cookie else domain,
                path=path,
            )
        if writes or removals:
            logger.debug(f"HTTP path mirrored {len(writes)} cookie writes and {len(removals)} deletions.")


def _origin(url: str) -> tuple[str, str, int]:
    """Return the scheme, host and effective port that make up *url*'s origin."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return (scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme, 0))


def _redirected_request(
    method: Literal["GET", "POST"],
    content: str | bytes | None,
    status: int,
) -> tuple[Literal["GET", "POST"], str | bytes | None]:
    """Return the method and body a redirect keeps: 301, 302 and 303 turn a POST into a GET."""
    if method == "POST" and status in _POST_TO_GET_STATUSES:
        return "GET", None
    return method, content


def _unsafe_redirect(
    cookies: Sequence[Mapping[str, Any]],
    current: str,
    next_url: str,
    method: str,
    *,
    cross_site: bool = False,
) -> str | None:
    """Return why a redirect cannot be followed safely, or ``None`` when it can.

    A native browser decides whether a Lax or Strict cookie goes to a redirect target by comparing
    schemeful sites, which needs a public-suffix calculation this transport deliberately does not
    do. A target on another host or scheme is therefore refused when a cookie that matches the
    target is Strict, or when a request that keeps its body is about to send a Lax cookie, because a
    browser would leave that cookie behind. A cookie the target does not match never blocks a
    redirect, so ordinary browsing keeps working.
    """
    if not cross_site and _origin(current)[:2] == _origin(next_url)[:2]:
        return None
    for entry in cookies:
        if not _target_matches(entry, next_url):
            continue
        same_site = str(entry.get("sameSite") or "").capitalize()
        if same_site == "Strict":
            return (
                f"a Strict cookie {entry.get('name')!r} for the redirect target would be withheld "
                "by a browser, which needs a schemeful-site comparison the HTTP path does not do"
            )
        if same_site == "Lax" and method == "POST":
            return (
                f"a Lax cookie {entry.get('name')!r} would be withheld by a browser from this "
                "cross-site POST redirect, which the HTTP path cannot decide"
            )
    return None


def _target_matches(entry: Mapping[str, Any], url: str) -> bool:
    """Return whether a request to *url* would carry the context cookie *entry*."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    domain = str(entry.get("domain") or "").lower()
    if scheme not in _HTTP_SCHEMES or not host or not domain:
        return False
    if domain.startswith("."):
        if host != domain[1:] and not host.endswith(domain):
            return False
    elif host != domain:
        return False
    if entry.get("secure") is True and scheme != "https":
        return False
    path = str(entry.get("path") or "/")
    request_path = parts.path or "/"
    return request_path == path or request_path.startswith(path if path.endswith("/") else f"{path}/")


def _reject_unsupported_cookies(set_cookies: Sequence[str], status: int) -> None:
    """Refuse a response whose cookies cannot be represented, before they are written or followed.

    :raises UnsupportedCookieError: when a declaration is partitioned, uses ``SameParty``, or asks
        for ``SameSite=None`` without ``Secure``.
    """
    for header in set_cookies:
        reason = _unsupported_flag(header)
        if reason is not None:
            msg = f"HTTP {status} declared a {reason} cookie, which the HTTP path cannot represent"
            raise UnsupportedCookieError(msg)


def _unsupported_flag(set_cookie: str) -> str | None:
    """Return the unrepresentable attribute one ``Set-Cookie`` line uses, or ``None``.

    Names and values are read from the attribute pairs, so a quoted value that contains a semicolon
    cannot be mistaken for another attribute.
    """
    attributes = dict(_attributes(set_cookie))
    flags = attributes.keys()
    if "partitioned" in flags:
        return "Partitioned"
    if "sameparty" in flags:
        return "SameParty"
    if attributes.get("samesite", "").strip().lower() == "none" and "secure" not in flags:
        return "SameSite=None without Secure"
    return None


def _attributes(set_cookie: str) -> list[tuple[str, str]]:
    """Split one ``Set-Cookie`` line into its attribute pairs, keeping quoted values whole."""
    message = Message()
    message["Content-Type"] = set_cookie
    return [(name.lower(), value) for name, value in (message.get_params() or [])[1:] if isinstance(value, str)]


def _brand_list(brands: tuple[tuple[str, str], ...], major: int, header: str) -> str:
    """Return a client-hint brand list that names the browser's own Chrome major.

    :raises UnsupportedIdentityError: when the browser reported no usable brand list, or a list
        that does not carry the Chrome version its user agent claims.
    """
    if not brands:
        msg = f"the browser reported no {header} brands to match"
        raise UnsupportedIdentityError(msg)
    if not any(brand.lower() == "chromium" and version.split(".")[0] == str(major) for brand, version in brands):
        msg = f"the browser's {header} brands do not carry its reported Chrome {major} version"
        raise UnsupportedIdentityError(msg)
    return ", ".join(f'"{brand}";v="{version}"' for brand, version in brands)


def _quoted(value: str) -> str:
    """Return a client-hint string value."""
    return f'"{value}"'


def _caller_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Return the caller's headers, refusing any that would give the request a second identity.

    :raises IdentityHeaderConflictError: when a header belongs to the client identity.
    """
    if not headers:
        return {}
    lowered = {name.lower(): value for name, value in headers.items()}
    conflicts = sorted(name for name in lowered if browser_owns_header(name))
    if conflicts:
        msg = f"these headers are owned by the browser identity: {', '.join(conflicts)}"
        raise IdentityHeaderConflictError(msg)
    try:
        return normalize_custom_headers(lowered)
    except ValueError as error:
        msg = f"invalid request header: {error}"
        raise HttpTransportError(msg) from error


def _request_cookie_jar(cookies: Sequence[Mapping[str, Any]]) -> CookieJar:
    """Return the caller's cookies as a jar of their own, leaving the session state untouched.

    These cookies are sent with the request but are not written into the context. What the site
    accepts or changes comes back through the exchange's own cookie changes and is mirrored then.

    :raises UnsupportedCookieError: when a caller cookie carries unsupported state.
    """
    jar = CookieJar()
    for entry in cookies:
        _, cookie = _jar_cookie(entry, source="caller")
        jar.set_cookie(cookie)
    return jar


def _jar_cookie(entry: Mapping[str, Any], *, source: str) -> tuple[_CookieKey, Cookie]:
    """Build a jar cookie from a browser-shaped cookie, keeping every attribute it can carry.

    :raises UnsupportedCookieError: when the cookie carries state a jar cannot express.
    """
    name = str(entry.get("name") or "")
    if not name:
        msg = f"a {source} cookie has no name"
        raise UnsupportedCookieError(msg)
    if entry.get("partitionKey"):
        msg = f"{source} cookie {name!r} is partitioned, which the HTTP path cannot represent"
        raise UnsupportedCookieError(msg)
    domain = str(entry.get("domain") or "").lower()
    if not domain:
        msg = f"{source} cookie {name!r} names no domain"
        raise UnsupportedCookieError(msg)
    path = str(entry.get("path") or "/")
    expires = entry.get("expires")
    expiry: int | None = None
    if isinstance(expires, int | float) and not isinstance(expires, bool) and expires > 0:
        expiry = int(expires)
    same_site = str(entry.get("sameSite") or "")
    rest: dict[str, str] = {"HttpOnly": "True"} if entry.get("httpOnly") is True else {}
    if same_site:
        rest["SameSite"] = same_site
    cookie = Cookie(
        version=0,
        name=name,
        value=str(entry.get("value") or ""),
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=domain.startswith("."),
        domain_initial_dot=domain.startswith("."),
        path=path,
        path_specified=True,
        secure=entry.get("secure") is True,
        expires=expiry,
        discard=expiry is None,
        comment=None,
        comment_url=None,
        rest=rest,
        rfc2109=False,
    )
    return _cookie_key(cookie), cookie


def _cookie_key(cookie: Cookie) -> _CookieKey:
    """Return a cookie's identity as the accepted jar keys it."""
    domain_cookie = bool(cookie.domain_specified or cookie.domain_initial_dot)
    return (cookie.name, cookie.domain.lstrip(".").lower(), cookie.path or "/", domain_cookie)


def _browser_cookie(
    key: _CookieKey,
    cookie: Cookie,
    declarations: Mapping[_DeclarationKey, str],
) -> SetCookieParam:
    """Return *cookie* in the shape a browser context reports and accepts cookies.

    ``SameSite`` comes from the declaration that owns this cookie, and is left unset when that
    declaration names none, so the browser applies its own default instead of an older declaration's
    value. ``HttpOnly`` is read back from the jar, where libcurl records the marker.
    """
    _, domain, path, domain_cookie = key
    expires = cookie.expires
    same_site = declarations.get(key, "")
    entry: SetCookieParam = {
        "name": cookie.name,
        "value": cookie.value or "",
        "domain": f".{domain}" if domain_cookie else domain,
        "path": path,
        "secure": cookie.secure,
        "httpOnly": _http_only(cookie),
        "expires": _SESSION_COOKIE_EXPIRY if expires is None else expires,
    }
    if same_site == "Lax":
        entry["sameSite"] = "Lax"
    elif same_site == "Strict":
        entry["sameSite"] = "Strict"
    elif same_site == "None":
        entry["sameSite"] = "None"
    return entry


def _content_differs(before: Cookie, after: Cookie) -> bool:
    """Return whether a transfer changed any field curl's jar carries for a cookie."""
    return not (
        before.value == after.value
        and before.secure == after.secure
        and before.expires == after.expires
        and _http_only(before) == _http_only(after)
    )


def _browser_shape(entry: Mapping[str, Any]) -> tuple[str, str, str, str, bool, bool, int, str]:
    """Return the fields a context reports a cookie with, in a form two cookies can be compared in."""
    expires = entry.get("expires")
    expiry = _SESSION_COOKIE_EXPIRY
    if isinstance(expires, int | float) and not isinstance(expires, bool) and expires > 0:
        expiry = int(expires)
    return (
        str(entry.get("name") or ""),
        str(entry.get("value") or ""),
        str(entry.get("domain") or "").lower(),
        str(entry.get("path") or "/"),
        entry.get("secure") is True,
        entry.get("httpOnly") is True,
        expiry,
        str(entry.get("sameSite") or ""),
    )


def _http_only(cookie: Cookie) -> bool:
    """Return whether *cookie* is marked HttpOnly, in either spelling a jar carries it in."""
    value = cookie.get_nonstandard_attr("HttpOnly", cookie.get_nonstandard_attr("http_only", ""))
    return str(value).lower() == "true"


def _declaration_sources(response: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Return each response's url and ``Set-Cookie`` lines, oldest first, ending with the final one."""
    return [(item.url, tuple(item.headers.get_list(_SET_COOKIE))) for item in (*response.history, response)]


def _declarations(sources: Sequence[tuple[str, Sequence[str]]]) -> dict[_DeclarationKey, str]:
    """Read the ``SameSite`` every response declaration asked for, keyed by the cookie it declares.

    Every source's declarations are read in order, so a cookie created by an earlier response keeps
    the metadata that response gave it. Only the declaration's own name, domain and path choose the
    key; the accepted jar remains the authority on the scope the cookie actually ended up with.
    """
    declared: dict[_DeclarationKey, str] = {}
    for url, headers in sources:
        for header in headers:
            name, separator, _ = header.partition("=")
            if not separator:
                continue
            attributes = dict(_attributes(header))
            key = _declaration_key(url, name.strip(), attributes)
            if key is not None:
                declared[key] = attributes.get("samesite", "").capitalize()
    return declared


def _declaration_key(url: str, name: str, morsel: Any) -> _DeclarationKey | None:
    """Ignore declarations that cannot belong to their response origin.

    The key is the complete accepted identity: a host-only cookie and a domain cookie with the same
    name, domain and path are different cookies, and only the one a declaration actually names takes
    its metadata. A declared domain that is an IP literal stays host-only, which is the scope
    libcurl and a browser both accept for it.
    """
    host = (urlsplit(url).hostname or "").lower()
    declared = str(morsel.get("domain") or "").strip().lower().lstrip(".")
    if declared and not _COOKIE_POLICY.domain_return_ok(declared, Request(url)):
        return None
    domain = declared or host
    path = str(morsel.get("path") or "")
    if not path.startswith("/"):
        path = _default_path(url)
    return (name, domain, path, bool(declared) and not _is_address(declared))


def _is_address(domain: str) -> bool:
    """Return whether a cookie domain is an IP literal rather than a host name."""
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        return False
    return True


def _default_path(url: str) -> str:
    """Return the path a cookie gets when its declaration names none, per RFC 6265 5.1.4."""
    path = urlsplit(url).path or "/"
    if not path.startswith("/"):
        return "/"
    index = path.rfind("/")
    return "/" if index <= 0 else path[:index]


def _response_headers(response: Any) -> dict[str, str]:
    """Return the response's headers as one lowercase-keyed mapping.

    Repeated list-valued names are joined as the HTTP syntax allows. ``set-cookie`` is not a
    list-valued header, so only its first value stands in the mapping; every one of them is kept in
    order in ``HttpResult.set_cookie_headers``.
    """
    headers: dict[str, str] = {}
    for name, value in response.headers.multi_items():
        key = name.lower()
        existing = headers.get(key)
        if existing is None:
            headers[key] = value
        elif key != _SET_COOKIE:
            headers[key] = f"{existing}, {value}"
    return headers


__all__ = [
    "CLASSIFICATION_BODY_BYTES",
    "SUPPORTED_ARCHITECTURES",
    "SUPPORTED_BITNESSES",
    "SUPPORTED_CHROME_MAJORS",
    "SUPPORTED_PLATFORMS",
    "CookieContext",
    "HttpClient",
    "HttpIdentity",
    "HttpNetworkError",
    "HttpResult",
    "HttpTransportError",
    "IdentityHeaderConflictError",
    "ObservedBrowserIdentity",
    "UnsupportedCookieError",
    "UnsupportedIdentityError",
    "chrome_major_from_user_agent",
    "resolve_http_identity",
]
