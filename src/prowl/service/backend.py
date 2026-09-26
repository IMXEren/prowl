"""Browser backend that executes fetch commands against the reusable core.

Every fetch runs in its own tab group inside the browser process that serves the
request's egress and closes that group on every path, including errors and
cancellation. The egress named :data:`~prowl.browser.egress.DEFAULT_EGRESS_NAME`
uses the process-wide browser; every other egress owns a browser created on first
use and shut down once idle. The backend owns no per-session state; logical-session
serialization is owned by the session registry.

An interactive tab is the one case where a group outlives its command: it is opened on
the same browser, and therefore the same profile directory and profile archive, as the
egress's fetches, because a Cloudflare clearance earned by a person clicking through a
challenge has to be the clearance the fetches then use. While a tab is open the backend
holds the egress claim, so the pool cannot idle that browser out from under it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable
from urllib.parse import urlsplit

from loguru import logger

from prowl.browser import Browser, BrowserConfig
from prowl.browser.egress import (
    DEFAULT_EGRESS_IDLE_SECONDS,
    DEFAULT_EGRESS_NAME,
    EgressPool,
)
from prowl.browser.site import resolve_site

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Cookie fields FlareSolverr clients expect in ``solution.cookies``.
_COOKIE_FIELDS = ("name", "value", "domain", "path", "expires", "secure", "httpOnly", "sameSite")

#: Environment variable holding how long an interactive tab may sit untouched.
INTERACTIVE_IDLE_SECONDS_ENV: Final[str] = "PROWL_INTERACTIVE_IDLE_SECONDS"

#: Whether a caller may take over the least recently used interactive tab. On by default, so a
#: caller that loses track of its tabs cannot starve the fetch path the app's own challenge solving
#: depends on. Set it to 0 to keep every tab until its idle timeout instead.
STEAL_LEAST_RECENT_ENV: Final[str] = "PROWL_STEAL_LEAST_RECENT"
DEFAULT_STEAL_LEAST_RECENT: Final[bool] = True


#: Fewest open tabs for a takeover to be worth doing: with one tab there is nothing to pick that
#: would not be the tab the caller is most likely looking at.
_MIN_TABS_TO_STEAL: Final[int] = 2


def _default_steal_least_recent() -> bool:
    """Whether a caller may take over the least recently used interactive tab."""
    raw = os.environ.get(STEAL_LEAST_RECENT_ENV, "").strip().lower()
    if not raw:
        return DEFAULT_STEAL_LEAST_RECENT
    return raw in {"1", "true", "yes", "on"}


#: Interactive tabs are kept until they are closed. A positive value closes a forgotten one
#: after that many idle seconds; anything else leaves it open indefinitely.
DEFAULT_INTERACTIVE_IDLE_SECONDS: Final[float] = 0.0

_TITLE_RE: Final[re.Pattern[str]] = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class FetchRequest:
    """A validated fetch to execute in the browser."""

    url: str
    method: str = "GET"
    timeout_seconds: int = 60
    headers: dict[str, str] = field(default_factory=dict)
    header_scope: str | None = None
    cookies: list[dict[str, Any]] = field(default_factory=list)
    post_data: str = ""
    egress: str = DEFAULT_EGRESS_NAME


@dataclass(slots=True)
class FetchResult:
    """The outcome of a browser fetch."""

    url: str
    status_code: int | None
    headers: dict[str, str]
    response: str
    cookies: list[dict[str, Any]]
    user_agent: str | None


@dataclass(slots=True)
class InteractiveRequest:
    """A validated request to open a tab and leave it open."""

    url: str
    timeout_seconds: int = 60
    cookies: list[dict[str, Any]] = field(default_factory=list)
    new_tab: bool = False
    egress: str = DEFAULT_EGRESS_NAME


@dataclass(slots=True)
class CookieQuery:
    """A validated request to read the cookies an egress's browser profile holds.

    A url scopes the answer to the cookies the browser would send there. Without one, every
    cookie in the profile is reported.
    """

    url: str | None = None
    egress: str = DEFAULT_EGRESS_NAME


@dataclass(slots=True)
class InteractiveTab:
    """An open interactive tab, as reported to a caller.

    ``url`` is the location the tab was navigated to, refreshed from the live page when it can
    be read. ``title`` is the title the page carried when it was opened. ``requested_url`` is
    the url the tab was opened with, kept so a repeat request for it can be recognised without
    reading the live page. The egress is carried for the backend's own bookkeeping and is never
    part of a response.
    """

    tab_id: str
    url: str
    title: str
    status_code: int | None
    requested_url: str = ""
    egress: str = DEFAULT_EGRESS_NAME


@dataclass(slots=True)
class InteractiveOpenResult:
    """The outcome of an ``browser.open``.

    ``reused`` is true when the url was already open on the egress, in which case the existing
    tab is handed back and nothing was created or navigated.
    """

    tab: InteractiveTab
    reused: bool = False


@dataclass(slots=True)
class _LiveTab:
    """An open tab with the group that owns it and its pending idle teardown."""

    tab: InteractiveTab
    group: Any
    idle_task: asyncio.Task[None] | None = None
    last_used: float = 0.0


@runtime_checkable
class Backend(Protocol):
    """The browser operations the service depends on."""

    async def start(self) -> None:
        """Start the shared browser process."""
        ...

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        """Execute *request*, serialized within *session_id* when supplied."""
        ...

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        """Open a tab, navigate it to the request's url, and leave it open."""
        ...

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        """Close one interactive tab, or every one when *tab_id* is None."""
        ...

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        """Return the open interactive tabs, refreshing *tab_id*'s idle countdown when named."""
        ...

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        """Return the cookies held by the browser profile of the query's egress."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Release any backend state for *session_id*; idempotent no-op here."""
        ...

    async def aclose(self) -> None:
        """Release all browser resources."""
        ...


class BrowserBackend:
    """Concrete backend owning the default browser and every named egress one.

    The default browser is started once and kept up. A named egress browser is
    created on first use and shut down after ``egress_idle_seconds`` without work,
    so an egress that nothing is using costs no memory. The backend holds no
    per-session locks; the service's session registry serializes per-session work.
    """

    def __init__(  # noqa: PLR0913
        self,
        browser_config: BrowserConfig | None = None,
        *,
        egresses: Mapping[str, str] | None = None,
        egress_idle_seconds: float = DEFAULT_EGRESS_IDLE_SECONDS,
        interactive_idle_seconds: float = DEFAULT_INTERACTIVE_IDLE_SECONDS,
        steal_least_recent: bool | None = None,
        max_concurrency: int = 0,
    ) -> None:
        self._browser_config = browser_config
        self._pool = EgressPool(
            browser_config or BrowserConfig(),
            egresses or {},
            idle_seconds=egress_idle_seconds,
        )
        self._interactive_idle_seconds = max(0.0, float(interactive_idle_seconds))
        if steal_least_recent is None:
            self._steal_least_recent_enabled = _default_steal_least_recent()
        else:
            self._steal_least_recent_enabled = bool(steal_least_recent)
        #: How many operations may be in flight before a caller is allowed to take a tab over. Zero
        #: means never, which is what the default and every test that does not care about pressure
        #: get. The service passes its configured concurrency, so a tab is only ever taken when
        #: every slot is already busy, which is the situation the takeover exists for.
        self._max_concurrency = max(0, int(max_concurrency))
        self._in_flight = 0
        self._tabs: dict[str, _LiveTab] = {}
        self._tabs_lock = asyncio.Lock()
        self._tab_counter = 0

    async def start(self) -> None:
        """Apply launch configuration and start the default browser process."""
        if self._browser_config is not None:
            Browser.configure(self._browser_config)
        await Browser.start()

    async def close_session(self, session_id: str) -> None:
        """No-op: the backend retains no per-session state."""

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        """Execute *request* in a fresh tab group of its egress and close it on every path."""
        # Free the oldest interactive tab when every slot is busy, so an abandoned tab cannot starve
        # the callers this path serves. A fetch is what the app's own challenge solving uses, and it
        # was the tab a lost view left behind that made those calls time out.
        await self._steal_least_recent()
        self._in_flight += 1
        try:
            return await self._fetch(request)
        finally:
            self._in_flight -= 1

    async def _fetch(self, request: FetchRequest) -> FetchResult:
        owner = await self._owner_for(request.egress)
        group = None
        try:
            await owner.start()
            group = await owner.create()
            return await self._fetch_in_group(group, request)
        finally:
            # A start or create failure leaves no group to close, but the egress claim it took
            # is still owed back.
            if group is not None:
                with contextlib.suppress(Exception):
                    await group.quit()
            await self._release(request.egress)

    async def _owner_for(self, egress: str) -> type[Browser]:
        """Return the browser class that serves *egress*.

        :raises EgressError: when *egress* is not a configured named egress.
        """
        if egress == DEFAULT_EGRESS_NAME:
            return Browser
        return await self._pool.acquire(egress)

    async def _release(self, egress: str) -> None:
        """Return a named egress to the pool so it can be idled out later."""
        if egress == DEFAULT_EGRESS_NAME:
            return
        await self._pool.release(egress)

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        """Open a tab on the egress's browser and leave it there for a person to drive.

        The tab shares the egress's browser, and so its profile directory and profile archive,
        with every fetch to that egress. The egress claim is held until the tab is closed, which
        is what stops the idle pool from shutting down the browser the tab is displayed in.

        A url that is already open on that egress is reused instead of opened a second time.
        That is what keeps a caller which never learned the first tab's id, because its own
        request timed out, from leaving a tab nobody can close. The reused tab is left exactly
        as it is, and only its idle countdown is refreshed, so a person reading it is not
        interrupted. ``new_tab`` forces a second tab for the same url.

        Cookies supplied with the request are installed before the navigation, so a caller can
        hand the browser a session it already holds.
        """
        if not request.new_tab:
            existing = await self._find_reusable_tab(request)
            if existing is not None:
                self._touch(existing)
                logger.debug(f"Interactive {existing.tab.tab_id} reused for {request.url}.")
                return InteractiveOpenResult(tab=existing.tab, reused=True)

        # Nothing to reuse, so a new tab is about to be opened. When every slot is busy the oldest
        # tab is taken over instead, so callers cannot grow the set without bound.
        await self._steal_least_recent()

        owner = await self._owner_for(request.egress)
        group = None
        try:
            await owner.start()
            group = await owner.create()
            if request.cookies:
                tab = await group.ptab
                await tab.set_cookies(request.cookies)
            site = resolve_site(group, request.url)
            source = await site.get(request.url, request.timeout_seconds)
        except BaseException:
            # The egress claim has to come back even when the browser never produced a group.
            if group is not None:
                with contextlib.suppress(Exception):
                    await group.quit()
            await self._release(request.egress)
            raise

        tab = InteractiveTab(
            tab_id="",
            url=source.url or request.url,
            title=_extract_title(source.text),
            status_code=source.status_code if isinstance(source.status_code, int) else None,
            requested_url=request.url,
            egress=request.egress,
        )
        async with self._tabs_lock:
            self._tab_counter += 1
            tab.tab_id = f"tab-{self._tab_counter}"
            entry = _LiveTab(tab=tab, group=group)
            self._tabs[tab.tab_id] = entry
        self._touch(entry)
        logger.debug(f"Interactive {tab.tab_id} opened, {len(self._tabs)} open.")
        return InteractiveOpenResult(tab=tab)

    async def _find_reusable_tab(self, request: InteractiveRequest) -> _LiveTab | None:
        """Return an open tab on the request's egress that already shows its url, if any.

        The tab was requested with this url, or it has navigated to it since, and both count:
        a caller that passes the url a redirect landed on is asking for the page it is already
        showing.
        """
        async with self._tabs_lock:
            for entry in self._tabs.values():
                if entry.tab.egress != request.egress:
                    continue
                if request.url in (entry.tab.requested_url, entry.tab.url):
                    return entry
        return None

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        """Close one interactive tab, or every one when *tab_id* is None.

        A tab that is not open is ignored, so closing is idempotent.
        """
        async with self._tabs_lock:
            if tab_id is None:
                entries = list(self._tabs.items())
            else:
                entry = self._tabs.get(tab_id)
                entries = [] if entry is None else [(tab_id, entry)]
            for key, entry in entries:
                del self._tabs[key]
                idle_task = entry.idle_task
                entry.idle_task = None
                # An idle teardown reaches this method from its own task, so cancelling here
                # would abort the close that is in progress and leak the egress claim.
                if idle_task is not None and idle_task is not asyncio.current_task():
                    idle_task.cancel()

        closed: list[str] = []
        for key, entry in entries:
            with contextlib.suppress(Exception):
                await entry.group.quit()
            await self._release(entry.tab.egress)
            closed.append(key)
        return closed

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        """List tabs; refresh only the named tab's idle countdown when it exists."""
        async with self._tabs_lock:
            entries = list(self._tabs.values())
            if tab_id is not None:
                entry = self._tabs.get(tab_id)
                if entry is not None:
                    self._touch(entry)
        tabs: list[InteractiveTab] = []
        for entry in entries:
            entry.tab.url = _live_url(entry.group) or entry.tab.url
            tabs.append(entry.tab)
        return tabs

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        """Return the cookies held by the browser profile of the query's egress.

        The browser is read, never driven, so an open interactive tab is left exactly as it is.
        A caller that harvests these into its own cookie jar carries the session a person
        established in that tab into its own requests. When the query names a url, only the
        cookies the browser would send to it are reported.
        """
        owner = await self._owner_for(query.egress)
        try:
            await owner.start()
            cookies = _normalize_cookies(await _collect_cookies(owner))
        finally:
            await self._release(query.egress)
        if query.url is None:
            return cookies
        return _cookies_for_url(cookies, query.url)

    async def _steal_least_recent(self) -> str | None:
        """Under concurrency pressure, close the oldest tab but never the only tab."""
        if not self._steal_least_recent_enabled:
            return None
        # Snapshot the tabs and the pressure together, so the tab that is picked is one the
        # pressure check actually saw.
        async with self._tabs_lock:
            # Only under pressure: with a slot free there is no reason to disturb anybody's tab.
            if self._max_concurrency <= 0 or len(self._tabs) + self._in_flight < self._max_concurrency:
                return None
            # A lone tab is both the oldest and the one being looked at, and an empty set is not
            # a candidate at all: neither may be taken.
            if len(self._tabs) < _MIN_TABS_TO_STEAL:
                return None
            tab_id = min(self._tabs.values(), key=lambda entry: entry.last_used).tab.tab_id
        closed = await self.close_interactive(tab_id)
        if closed:
            logger.debug(f"Interactive {tab_id} taken over by a newer caller.")
        return closed[0] if closed else None

    def _touch(self, entry: _LiveTab) -> None:
        """Record use and restart an entry's idle countdown; a no-op when idling is disabled."""
        entry.last_used = time.monotonic()
        if self._interactive_idle_seconds <= 0:
            return
        if entry.idle_task is not None:
            entry.idle_task.cancel()
        entry.idle_task = asyncio.ensure_future(self._close_when_idle(entry))

    async def _close_when_idle(self, entry: _LiveTab) -> None:
        """Close *entry* once it has been untouched for the configured delay.

        Only the interactive tab is closed here. Requests keep their own tab groups, so idling a
        tab out cannot disturb one that is in flight.
        """
        await asyncio.sleep(self._interactive_idle_seconds)
        async with self._tabs_lock:
            if self._tabs.get(entry.tab.tab_id) is not entry:
                return
        logger.debug(f"Interactive {entry.tab.tab_id} idle, closing.")
        await self.close_interactive(entry.tab.tab_id)

    @staticmethod
    async def _fetch_in_group(group: Any, request: FetchRequest) -> FetchResult:
        tab = await group.ptab
        if request.cookies:
            await tab.set_cookies(request.cookies)

        site = resolve_site(group, request.url)
        if request.method == "POST":
            source = await site.post(
                request.url,
                request.timeout_seconds,
                post_data=request.post_data,
                headers=request.headers,
            )
        elif request.headers:
            source = await site.get(
                request.url,
                request.timeout_seconds,
                headers=request.headers,
                header_scope=request.header_scope,
            )
        else:
            source = await site.get(request.url, request.timeout_seconds)

        cookies = _normalize_cookies(await _collect_cookies(group))
        return FetchResult(
            url=source.url or request.url,
            status_code=source.status_code if isinstance(source.status_code, int) else None,
            headers=dict(source.headers),
            response=source.text,
            cookies=cookies,
            user_agent=source.user_agent,
        )

    async def aclose(self) -> None:
        """Close every interactive tab, then shut every browser down."""
        with contextlib.suppress(Exception):
            await self.close_interactive(None)
        try:
            await Browser.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to shut the browser down: {type(exc).__name__}")
        await self._pool.aclose()


def _extract_title(html: str) -> str:
    """Return the document title carried in *html*, or an empty string."""
    match = _TITLE_RE.search(html or "")
    return " ".join(match.group(1).split()) if match else ""


def _live_url(group: Any) -> str | None:
    """Return the page's current location when it can be read, else ``None``."""
    try:
        url = group.ppage.url
    except Exception:  # noqa: BLE001
        return None
    return url if isinstance(url, str) and url else None


async def _collect_cookies(owner: Any) -> list[dict[str, Any]]:
    """Return a browser profile's cookies in FlareSolverr shape.

    ``owner`` is the browser class or one of its tab groups. Both expose the shared pydoll
    connection, which reads the profile rather than any single page in it.
    """
    try:
        raw = await owner.pd().get_cookies()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read browser cookies: {type(exc).__name__}")
        return []
    return [cookie for cookie in raw if isinstance(cookie, dict)]


def _cookies_for_url(cookies: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    """Return the cookies from *cookies* that the browser would send to *url*."""
    return [cookie for cookie in cookies if _cookie_applies_to(cookie, url)]


def _cookie_applies_to(cookie: dict[str, Any], url: str) -> bool:
    """Return whether *cookie* is one the browser would send to *url*.

    These are the request-side rules of standard cookie matching: the cookie's domain has to
    cover the host, its path has to cover the request path on a path boundary, a secure cookie
    is only sent over https, and an expired cookie is not sent at all.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return False
    domain = str(cookie.get("domain") or "").lower().lstrip(".")
    if not domain:
        # A cookie that names no domain cannot be attributed to the requested host.
        return False
    if host != domain and not host.endswith("." + domain):
        return False
    if cookie.get("secure") is True and parts.scheme.lower() != "https":
        return False
    expiry = _cookie_expiry(cookie.get("expires"))
    if expiry is not None and expiry <= time.time():
        return False
    return _cookie_path_covers(str(cookie.get("path") or "/"), parts.path or "/")


def _cookie_path_covers(cookie_path: str, request_path: str) -> bool:
    """Return whether *cookie_path* covers *request_path*, on a path boundary."""
    if request_path == cookie_path:
        return True
    if not request_path.startswith(cookie_path):
        return False
    return cookie_path.endswith("/") or request_path[len(cookie_path) :].startswith("/")


def _cookie_expiry(value: Any) -> float | None:
    """Return a cookie's expiry in unix seconds, or ``None`` for a session cookie.

    The browser reports a session cookie as a non-positive expiry, which must not be read as an
    expiry in the past: that would filter out exactly the login cookies this command exists for.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value > 0 else None


def _normalize_cookies(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce pydoll cookie dicts to the FlareSolverr cookie shape."""
    normalized: list[dict[str, Any]] = []
    for cookie in raw:
        entry = {key: cookie[key] for key in _COOKIE_FIELDS if key in cookie and cookie[key] is not None}
        if "name" in entry and "value" in entry:
            normalized.append(entry)
    return normalized


__all__ = [
    "DEFAULT_INTERACTIVE_IDLE_SECONDS",
    "DEFAULT_STEAL_LEAST_RECENT",
    "INTERACTIVE_IDLE_SECONDS_ENV",
    "STEAL_LEAST_RECENT_ENV",
    "Backend",
    "BrowserBackend",
    "CookieQuery",
    "FetchRequest",
    "FetchResult",
    "InteractiveOpenResult",
    "InteractiveRequest",
    "InteractiveTab",
]
