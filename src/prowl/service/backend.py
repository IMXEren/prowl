"""Browser backend that executes fetch commands against the reusable core.

Every fetch runs in its own tab group inside the browser process that serves the
request's egress and closes that group on every path, including errors and
cancellation. The egress named :data:`~prowl.browser.proxy.egress.DEFAULT_EGRESS_NAME`
uses the process-wide browser; every other egress owns a browser created on first
use and shut down once idle. One context - and, for a named egress, one held pool
claim - is owned per live isolated session; a shared session keeps no backend state,
and logical-session serialization is owned by the session registry.

An interactive tab is the one case where a group outlives its command: it is opened on
 the same browser, and therefore the same profile directory and profile archive, as the
 egress's fetches, because a Cloudflare clearance earned by a person clicking through a
 challenge has to be the clearance the fetches then use. While a tab is open the backend
 holds the egress claim, so the pool cannot idle that browser out from under it.

A fetch's execution mode decides how it reaches the network. The default ``browser`` mode
 keeps driving the browser exactly as it always has. ``http`` answers from the
 identity-matched HTTP fast path and never escalates. ``auto`` tries HTTP first and escalates
to the browser only for a response that genuinely needs one, reusing the same context and
held egress claim for both stages.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit

from loguru import logger
from playwright.async_api import Error as PlaywrightError

from prowl.browser import Browser, BrowserConfig, BrowserContextHandle, TabGroup
from prowl.browser.exceptions import BrowserContextError, BrowserTabError
from prowl.browser.headers import HEADER_SCOPE_DOCUMENT
from prowl.browser.page_handler import PageHandler, resolve_page_handler
from prowl.browser.proxy.cookies import CookieHeaderError, request_cookie_updates
from prowl.browser.proxy.egress import (
    DEFAULT_EGRESS_IDLE_SECONDS,
    DEFAULT_EGRESS_NAME,
    EgressPool,
)
from prowl.service.browser_http import BrowserHttpClients
from prowl.service.classification import CAPTCHA, CLOUDFLARE_CHALLENGE, SUCCESS, classify
from prowl.service.context_state import ContextStateError, ContextStateStore
from prowl.service.errors import CallerSafeError, SessionError
from prowl.service.http_transport import (
    UnsupportedCookieError,
    UnsupportedIdentityError,
)
from prowl.service.metrics import Metrics
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE, ExecutionMode
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE, SessionInfo, SessionMode

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from playwright._impl._api_structures import SetCookieParam
    from pydoll.protocol.network.types import CookieParam

    from prowl.service.classification import Classification

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

#: The HTTP methods the fast path can send.
type _FetchMethod = Literal["GET", "POST"]

#: Cookie fields the HTTP fast path can seed a native context with. Anything else, including a
#: partitioned cookie's ``partitionKey``, is refused instead of being flattened into a plain one.
_SEED_COOKIE_FIELDS: Final[frozenset[str]] = frozenset(
    {"name", "value", "domain", "path", "expires", "secure", "httpOnly", "sameSite"},
)

#: A cookie the context accepts is keyed by name, domain, path, secure, httpOnly and expiry.
_SEED_COOKIE_SAME_SITE: Final[frozenset[str]] = frozenset({"Lax", "Strict", "None"})

#: The classifications that count as a detected challenge rather than a plain refusal.
_CHALLENGE_CATEGORIES: Final[frozenset[str]] = frozenset({CLOUDFLARE_CHALLENGE, CAPTCHA})


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
    session_id: str | None = None
    session_mode: SessionMode = SHARED_MODE
    mode: ExecutionMode = BROWSER_MODE
    body_bytes: bytes | None = None
    cookie_header: str | None = None
    wait_in_seconds: float = 0.0
    return_screenshot: bool = False
    disable_media: bool = False
    tabs_till_verify: int | None = None
    solve_captcha: bool = False


@dataclass(slots=True)
class FetchResult:
    """The outcome of a browser fetch.

    ``mode`` is the mode the fetch actually ran in and ``classification`` says why its response was
    answered as it was. Both are execution diagnostics: a fetch that reports neither keeps the
    reply's original shape, which is what the browser fetch this backend runs today does.

    ``body_bytes`` is the HTTP-decoded entity content, not the original compressed or framed
    transfer bytes; ``header_items`` keeps the final response headers as repeated name/value pairs
    in the order curl reported them. Both are ``None`` when not available (a browser-rendered or
    legacy result), while ``b""`` is a genuinely empty body.
    """

    url: str
    status_code: int | None
    headers: dict[str, str]
    response: str
    cookies: list[dict[str, Any]]
    user_agent: str | None
    mode: ExecutionMode | None = None
    classification: Classification | None = None
    body_bytes: bytes | None = None
    header_items: tuple[tuple[str, str], ...] | None = None
    screenshot: str | None = None
    turnstile_token: str | None = None
    captcha_provider: str | None = None
    captcha_token: str | None = None


@dataclass(slots=True)
class InteractiveRequest:
    """A validated request to open a tab and leave it open."""

    url: str
    timeout_seconds: int = 60
    cookies: list[dict[str, Any]] = field(default_factory=list)
    new_tab: bool = False
    egress: str = DEFAULT_EGRESS_NAME
    session_id: str | None = None
    session_mode: SessionMode = SHARED_MODE


@dataclass(slots=True)
class CookieQuery:
    """A validated request to read the cookies an egress's browser profile holds.

    A url scopes the answer to the cookies the browser would send there. Without one, every
    cookie in the profile is reported.
    """

    url: str | None = None
    egress: str = DEFAULT_EGRESS_NAME
    session_id: str | None = None
    session_mode: SessionMode = SHARED_MODE


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
    """An open tab with the group that owns it and its pending idle teardown.

    ``session_id`` is the isolated session the tab belongs to, or ``None`` for a shared tab.
    ``owns_claim`` is whether closing this tab has to release its egress claim: an isolated
    tab shares the lifetime claim its session holds, so it never releases one itself.
    """

    tab: InteractiveTab
    group: Any
    session_id: str | None = None
    owns_claim: bool = True
    idle_task: asyncio.Task[None] | None = None
    last_used: float = 0.0


@dataclass(slots=True)
class _IsolatedSession:
    """One live isolated session: its initialization task, context, and held egress claim.

    ``task`` is the single initialization shared by every first-use caller, so concurrent
    callers cannot create two contexts and a failure is observed by all of them. ``claim_held``
    records whether this session holds a named egress's pool claim for its whole lifetime; the
    default egress is a process singleton and needs none.
    """

    session_id: str
    egress: str
    task: asyncio.Task[None] | None = None
    owner: type[Browser] | None = None
    context: BrowserContextHandle | None = None
    claim_held: bool = False
    ready: bool = False

    def require_context(self) -> BrowserContextHandle:
        """Return this session's context once initialization produced one."""
        if self.context is None:
            msg = f"isolated session {self.session_id} has no context"
            raise BrowserContextError(msg)
        return self.context

    def live(self) -> tuple[type[Browser], BrowserContextHandle]:
        """Return this session's owner and context once both exist."""
        if self.owner is None:
            msg = f"isolated session {self.session_id} has no owner"
            raise BrowserContextError(msg)
        return self.owner, self.require_context()


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
        """Close any isolated context and interactive tabs owned by *session_id*."""
        ...

    async def aclose(self) -> None:
        """Release all browser resources."""
        ...


class BrowserBackend:
    """Concrete backend owning the default browser and every named egress one.

    The default browser is started once and kept up. A named egress browser is
    created on first use and shut down after ``egress_idle_seconds`` without work,
    so an egress that nothing is using costs no memory. A live isolated session owns
    one context beside the persistent one and holds its egress's pool claim until that
    context closes; the service's session registry serializes per-session work.
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
        metrics: Metrics | None = None,
        state_store: ContextStateStore | None = None,
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
        #: Live request counts per egress, so one identity's busy work never drives the
        #: pressure that decides whether another identity's tab may be taken over. Keys are
        #: removed as their count returns to zero; this is scalar state, not a registry.
        self._in_flight_by_egress: dict[str, int] = {}
        self._tabs: dict[str, _LiveTab] = {}
        self._tabs_lock = asyncio.Lock()
        self._tab_counter = 0
        self._isolated: dict[str, _IsolatedSession] = {}
        self._isolated_lock = asyncio.Lock()
        #: Identity-matched HTTP clients for the live native contexts of the ``http`` and
        #: ``auto`` fetch modes. The manager owns only the HTTP sessions; the backend owns the
        #: native contexts and closures this class already owns.
        self._http = BrowserHttpClients()
        #: Fixed-shape counters and timings this backend updates as requests run. A caller may
        #: supply its own instance so several backends can report into one process's registry.
        self.metrics = metrics if metrics is not None else Metrics()
        #: Optional on-disk store for isolated session storage state. When set, a newly created
        #: isolated context is seeded from its binding's snapshot and a session close persists or
        #: forgets that snapshot according to the cleanup reason it was given.
        self._state_store = state_store

    @property
    def state_persistence_enabled(self) -> bool:
        """Whether this backend restores and persists isolated session state on a store."""
        return self._state_store is not None

    async def start(self) -> None:
        """Apply launch configuration and start the default browser process."""
        if self._browser_config is not None:
            Browser.configure(self._browser_config)
        await Browser.start()

    def render_metrics(self) -> str:
        """Combine default and named-owner resource snapshots with existing request metrics."""
        native = Browser.resource_metrics()
        pooled = self._pool.resource_metrics()
        self.metrics.context_count = native["context_count"] + pooled["context_count"]
        self.metrics.tabgroups_active = native["tabgroups_active"] + pooled["tabgroups_active"]
        self.metrics.context_created_total = native["context_created_total"] + pooled["context_created_total"]
        self.metrics.context_evicted_total = native["context_evicted_total"] + pooled["context_evicted_total"]
        self.metrics.browser_restart_total = native["browser_restart_total"] + pooled["browser_restart_total"]
        return self.metrics.render()

    async def close_session(
        self,
        session_id: str,
        *,
        evicted: bool = False,
        cleanup: SessionInfo | None = None,
    ) -> None:
        """Close an isolated session's interactive tabs and context, releasing its egress claim.

        A shared session keeps no backend state, so this is a no-op for it. Initialization is
        waited out rather than skipped, so a context created concurrently is closed instead of
        leaked. The session's HTTP client is closed before its native context, so no HTTP
        session outlives the context it belonged to. When the context cannot be closed the
        session keeps its ownership and its egress claim, so the registry's cleanup retry can
        run this again. *evicted* marks automatic retirement so only that close is counted by
        the native context manager.

        *cleanup* carries the registry's own metadata when the app has persistence wired. An
        eviction or shutdown keeps the newest storage state the live context holds; a destroy
        (or an absent *cleanup*) instead forgets the binding's saved state. A snapshot, save or
        delete failure keeps the session, its context and its claim for a deliberate retry.
        """
        async with self._isolated_lock:
            session = self._isolated.get(session_id)
        if session is None:
            await self._forget_missing_session(session_id, cleanup)
            return
        await self._drain_session_task(session)
        if session.owner is None:
            return
        await self._close_session_tabs(session_id)
        preserve = cleanup is not None and cleanup.close_reason != "destroy"
        if preserve:
            # Snapshot before native retirement, while the context can still be read. A failed
            # save propagates and keeps the session, context and claim owned for a retry.
            await self._save_context_state(session)
        if session.context is not None:
            await self._http.close_context(session.context.context)
        await self._close_owner_context(session.owner, session_id, evicted=evicted)
        if not preserve:
            # Forget the binding only once the native close succeeded, so a crash between the two
            # cannot erase state the browser still holds. A failed delete keeps the retry ownership.
            await self._forget_context_state(session.egress, session_id)
        if session.claim_held:
            await self._release(session.egress)
            session.claim_held = False
        async with self._isolated_lock:
            if self._isolated.get(session_id) is session:
                del self._isolated[session_id]

    @staticmethod
    async def _drain_session_task(session: _IsolatedSession) -> None:
        task = session.task
        if task is None or task.done():
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            logger.debug(f"Isolated session {session.session_id} drained a cancelled refresh.")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Isolated cleanup drained {type(exc).__name__} before closing resources.")

    @staticmethod
    async def _close_owner_context(owner: type[Browser], session_id: str, *, evicted: bool) -> None:
        """Close the native context, marked as an eviction only for automatic retirement."""
        if evicted:
            await owner.close_context(session_id, evicted=True)
        else:
            await owner.close_context(session_id)

    async def _get_isolated_context(
        self,
        owner: type[Browser],
        session: _IsolatedSession,
    ) -> BrowserContextHandle:
        """Return the session's native context, restoring a persisted snapshot when configured.

        Without a store this is the existing native call. With one, the binding's saved state
        seeds a newly created context through Playwright's native restore; a live context keeps
        its own state because the browser only consults supplied state when creating. Only the
        native restore error becomes a caller-safe failure; context-cap, ownership and browser
        errors pass through unchanged.
        """
        store = self._state_store
        if store is None:
            return await owner.get_context(session.session_id)
        state = await asyncio.to_thread(store.load, session.egress, session.session_id)
        if state is None:
            return await owner.get_context(session.session_id)
        try:
            return await owner.get_context(session.session_id, storage_state=state)
        except PlaywrightError:
            msg = "persisted context state could not be restored"
            raise ContextStateError(msg) from None

    async def _save_context_state(self, session: _IsolatedSession) -> None:
        """Snapshot a live session context to the store before its native context retires.

        A context that is already closed or whose browser is gone is skipped, so the last saved
        file is retained rather than being overwritten with state read from a dead generation.
        """
        store = self._state_store
        handle = session.context
        if store is None or handle is None or not _context_handle_live(handle):
            return
        state = await handle.context.storage_state(indexed_db=True)
        await _run_blocking_io(store.save, session.egress, session.session_id, state)

    async def _forget_context_state(self, egress: str, session_id: str) -> None:
        """Delete a binding's saved state, if a store is configured."""
        store = self._state_store
        if store is None:
            return
        await _run_blocking_io(store.delete, egress, session_id)

    async def _forget_missing_session(self, session_id: str, cleanup: SessionInfo | None) -> None:
        """Delete an isolated binding's state when a destroy never initialized it this generation.

        Only an explicit destroy of an isolated session with a bound egress erases the saved
        file; an eviction or shutdown, a shared session, and absent metadata all retain it.
        """
        if cleanup is None or cleanup.mode != ISOLATED_MODE or cleanup.close_reason != "destroy":
            return
        if cleanup.egress is None:
            return
        await self._forget_context_state(cleanup.egress, session_id)

    async def _isolated_session(self, session_id: str, egress: str) -> _IsolatedSession:
        """Return the isolated session for *session_id*, initializing or refreshing it on use.

        One task is tracked per generation: concurrent first callers await the same
        initialization, so a failure is seen by all of them and no detached entry is
        re-initialized. Only a first initialization that fails releases the claim and forgets
        the entry; the next caller then starts a fresh generation.

        An existing session whose native context is gone is refreshed under the same rule: one
        task on the existing session, awaited by every concurrent caller, reusing the session's
        owner and held egress claim so no new context is created for a live session and no claim
        is taken or returned. A failed or cancelled refresh keeps the session, its owner and its
        claim in place, so the next caller retries it. An already-ready session with a live
        native context awaits a completed task, which spawns nothing.

        :raises SessionError: when the existing session is bound to a different egress.
        """
        async with self._isolated_lock:
            session = self._isolated.get(session_id)
            if session is None:
                session = _IsolatedSession(session_id=session_id, egress=egress)
                session.task = asyncio.create_task(self._initialize_isolated(session))
                self._isolated[session_id] = session
            elif session.egress != egress:
                msg = f"isolated session {session_id} is bound to a different egress"
                raise SessionError(msg)
            elif session.task is None or session.task.done():
                if not (session.ready and _context_handle_live(session.context)):
                    session.ready = False
                    session.task = asyncio.create_task(self._refresh_isolated(session))
            task = session.task
        if task is not None:
            await task
        return session

    async def _initialize_isolated(self, session: _IsolatedSession) -> None:
        """Create *session*'s context and hold its egress claim for the session's lifetime.

        The claim is recorded as soon as it is taken, so a failure or cancellation between the
        claim and the context can release it and leave no ownership behind.
        """
        owner: type[Browser] | None = None
        claimed = False
        try:
            owner = await self._owner_for(session.egress)
            claimed = session.egress != DEFAULT_EGRESS_NAME
            session.owner = owner
            session.claim_held = claimed
            context = await self._get_isolated_context(owner, session)
        except BaseException:
            if claimed and owner is not None:
                await self._release(session.egress)
                session.claim_held = False
            async with self._isolated_lock:
                if self._isolated.get(session.session_id) is session:
                    del self._isolated[session.session_id]
            raise
        session.owner = owner
        session.context = context
        session.claim_held = claimed
        session.ready = True

    async def _refresh_isolated(self, session: _IsolatedSession) -> None:
        """Refresh a stale context without reacquiring its claim or abandoning retry ownership."""
        owner = session.owner
        if owner is None:
            msg = f"isolated session {session.session_id} has no owner to refresh with"
            raise BrowserContextError(msg)
        stale = session.context
        if stale is not None:
            await self._http.close_context(stale.context)
        await owner.start()
        context = await self._get_isolated_context(owner, session)
        session.context = context
        session.ready = True

    async def _close_session_tabs(self, session_id: str) -> list[str]:
        """Close every interactive tab owned by *session_id*."""
        async with self._tabs_lock:
            keys = [key for key, entry in self._tabs.items() if entry.session_id == session_id]
        closed: list[str] = []
        for key in keys:
            closed.extend(await self.close_interactive(key))
        return closed

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        """Execute a request and close its request-scoped resources on every path.

        The whole fetch, including any interactive-tab takeover, is counted and timed as one
        request, so the active gauge is released on success, failure and cancellation alike.
        """
        with self.metrics.request():
            if request.body_bytes is not None and (request.method != "POST" or request.mode != HTTP_MODE):
                msg = "raw request bodies require explicit HTTP POST"
                raise CallerSafeError(msg)
            if request.mode == HTTP_MODE and (
                request.wait_in_seconds > 0 or request.return_screenshot or request.disable_media
            ):
                msg = "browser response options require browser or auto mode"
                raise CallerSafeError(msg)
            if request.tabs_till_verify is not None and (request.method != "GET" or request.mode == HTTP_MODE):
                msg = "explicit verification requires browser or auto GET"
                raise CallerSafeError(msg)
            if request.solve_captcha and (request.method != "GET" or request.mode == HTTP_MODE):
                msg = "captcha solving requires browser or auto GET"
                raise CallerSafeError(msg)
            egress = request.egress
            if request.mode == BROWSER_MODE:
                await self._steal_least_recent(egress)
            self._in_flight += 1
            self._in_flight_by_egress[egress] = self._in_flight_by_egress.get(egress, 0) + 1
            try:
                result = await self._fetch(request)
            finally:
                self._in_flight -= 1
                remaining = self._in_flight_by_egress[egress] - 1
                if remaining:
                    self._in_flight_by_egress[egress] = remaining
                else:
                    del self._in_flight_by_egress[egress]
            self._count_browser_challenge(result)
            return result

    def _count_browser_challenge(self, result: FetchResult) -> None:
        """Count a challenge the fetch's final browser response still presents.

        Only a browser result adds this count. An HTTP result already counted its own challenge
        when it was classified, so counting it here as well would report one response twice.
        """
        if result.mode == HTTP_MODE:
            return
        classification = result.classification
        if classification is None:
            classification = classify(result.status_code or 0, result.headers, result.response, rendered=True)
        if classification.category in _CHALLENGE_CATEGORIES:
            self.metrics.challenge_detected_total += 1

    async def _fetch(self, request: FetchRequest) -> FetchResult:
        if request.mode != BROWSER_MODE:
            return await self._fetch_routed(request)
        session_id = request.session_id
        if request.session_mode == ISOLATED_MODE and session_id is not None:
            return await self._fetch_isolated(session_id, request)
        owner = await self._owner_for(request.egress)
        group = None
        body_error: BaseException | None = None
        handle: BrowserContextHandle | None = None
        try:
            await owner.start()
            if request.cookie_header:
                # A header is resolved against the shared context before the group exists, so the
                # response's own cookie changes are never overwritten by the caller's values.
                handle = await owner.get_context(None)
                await _apply_request_cookie_header(handle, request)
            group = await self._create_group(owner, handle)
            return await self._fetch_in_group(group, request, handle)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            # A close failure is reported, but the egress claim comes back either way.
            try:
                await self._quit_group(group, body_error)
            finally:
                await self._release(request.egress)

    async def _fetch_isolated(self, session_id: str, request: FetchRequest) -> FetchResult:
        """Fetch inside *session_id*'s isolated context, closing only the request group.

        The session holds its egress claim for its whole lifetime, so nothing is released here;
        the request group lives in a context that outlives it.
        """
        session = await self._isolated_session(session_id, request.egress)
        owner, context = session.live()
        group = None
        body_error: BaseException | None = None
        try:
            await _apply_request_cookie_header(context, request)
            group = await self._create_group(owner, context)
            return await self._fetch_in_group(group, request, context)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            await self._quit_group(group, body_error)

    async def _fetch_routed(self, request: FetchRequest) -> FetchResult:
        """Execute an ``http`` or ``auto`` fetch under one monotonic budget.

        The budget covers owner and context acquisition, the identity read, cookie seeding, the
        HTTP transfer and any browser escalation. Each stage receives the time that is left, so a
        slow acquisition cannot buy the transfer a fresh timeout.
        """
        timeout = max(0.0, float(request.timeout_seconds))
        deadline = time.monotonic() + timeout
        async with asyncio.timeout(timeout):
            return await self._run_route(request, deadline)

    async def _run_route(self, request: FetchRequest, deadline: float) -> FetchResult:
        """Resolve the route's owner and context once, then run its HTTP and browser stages.

        An isolated session keeps the context and lifetime claim it already holds. A shared
        request acquires its egress claim once and releases it once the whole route, HTTP attempt
        and browser escalation included, is finished, so an auto fetch never hands the browser
        back to the idle pool between the two stages.
        """
        session_id = request.session_id
        if request.session_mode == ISOLATED_MODE and session_id is not None:
            session = await self._isolated_session(session_id, request.egress)
            owner, handle = session.live()
            owns_claim = False
        else:
            owner = await self._owner_for(request.egress)
            handle = None
            owns_claim = request.egress != DEFAULT_EGRESS_NAME
        escalate = request.mode == AUTO_MODE
        try:
            await owner.start()
            if handle is None:
                handle = await owner.get_context(None)
            await _apply_request_cookie_header(handle, request)
            if not escalate:
                await self._seed_context(handle, request.cookies)
                return cast("FetchResult", await self._http_stage(owner, handle, request, deadline))
            if (
                request.wait_in_seconds > 0
                or request.return_screenshot
                or request.disable_media
                or request.tabs_till_verify is not None
                or request.solve_captcha
            ):
                # A requested capture, media suppression, explicit verification or captcha solving is
                # browser work by definition: the route goes straight to its own context, so no HTTP
                # attempt is made and no escalation is counted.
                result = await self._browser_stage(owner, handle, request, deadline, apply_cookies=True)
                return _classified(result)
            if request.method == "POST":
                result = await self._browser_stage(owner, handle, request, deadline, apply_cookies=True)
                return _classified(result)
            try:
                await self._seed_context(handle, request.cookies)
            except UnsupportedCookieError:
                result = await self._browser_stage(owner, handle, request, deadline, apply_cookies=True)
                return _classified(result)
            outcome = await self._http_stage(owner, handle, request, deadline, escalate=True)
            if outcome is not None:
                return outcome
            result = await self._browser_stage(owner, handle, request, deadline, apply_cookies=False)
            return _classified(result)
        finally:
            if owns_claim:
                await self._release(request.egress)

    async def _http_stage(
        self,
        owner: type[Browser],
        handle: BrowserContextHandle,
        request: FetchRequest,
        deadline: float,
        *,
        escalate: bool = False,
    ) -> FetchResult | None:
        """Run the HTTP attempt, returning its result or ``None`` when an auto fetch escalates.

        Only an unsupported persona or cookie state and a browser-required classification send an
        auto fetch on to the browser. An ordinary status, a rate, auth, geo or IP refusal, and a
        transport failure are all reported as they are, so no browser is started for them.

        ``http_fastpath_total`` counts HTTP attempts, incremented once the client is in hand and
        the transfer is about to run, so a refusal before the client exists is a bypass rather than
        an attempt. A browser-required decision - or an unsupported state raised by the transfer
        itself - counts one escalation; a transport failure counts none.
        """
        proxy = self._http_proxy(request.egress, owner)
        try:
            client = await self._http.client(handle.context, proxy)
        except (UnsupportedIdentityError, UnsupportedCookieError):
            if escalate:
                return None
            raise
        remaining = max(0.0, deadline - time.monotonic())
        self.metrics.http_fastpath_total += 1
        body = request.body_bytes if request.body_bytes is not None else request.post_data
        try:
            http_result = await client.fetch(
                request.url,
                method=cast("_FetchMethod", request.method),
                content=body if request.method == "POST" else None,
                headers=request.headers,
                header_scope=request.header_scope or HEADER_SCOPE_DOCUMENT,
                deadline_seconds=remaining,
            )
        except (UnsupportedIdentityError, UnsupportedCookieError):
            if escalate:
                self.metrics.browser_escalations_total += 1
                return None
            raise
        classification = http_result.classify()
        if classification.category == SUCCESS:
            self.metrics.http_fastpath_success_total += 1
        elif classification.category in _CHALLENGE_CATEGORIES:
            self.metrics.challenge_detected_total += 1
        if escalate and classification.browser_required:
            self.metrics.browser_escalations_total += 1
            return None
        return FetchResult(
            url=http_result.url,
            status_code=http_result.status_code,
            headers=dict(http_result.headers),
            response=http_result.body,
            cookies=_normalize_cookies(http_result.cookies),
            user_agent=client.identity.user_agent,
            mode=HTTP_MODE,
            classification=classification,
            body_bytes=http_result.body_bytes,
            header_items=http_result.header_items,
        )

    async def _browser_stage(
        self,
        owner: type[Browser],
        handle: BrowserContextHandle,
        request: FetchRequest,
        deadline: float,
        *,
        apply_cookies: bool,
    ) -> FetchResult:
        """Run the browser stage in the route's own context and close only the request group.

        A route that already seeded the native context drops the original cookies here, so an HTTP
        attempt's updated jar is never overwritten by a second application of the supplied ones.
        """
        group = None
        body_error: BaseException | None = None
        group_request = _with_remaining_timeout(request, deadline, drop_cookies=not apply_cookies)
        try:
            await self._steal_least_recent(request.egress)
            group = await self._create_group(owner, handle)
            return await self._fetch_in_group(group, group_request, handle)
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            await self._quit_group(group, body_error)

    async def _seed_context(self, handle: BrowserContextHandle, cookies: list[dict[str, Any]]) -> None:
        """Install *cookies* into the route's native context before the HTTP attempt.

        Every cookie is validated before any is written, so an unsupported one leaves the context
        untouched. The context stays the authority on the cookies; the HTTP path only mirrors back
        the changes the exchange itself made.

        :raises UnsupportedCookieError: when a cookie carries state the fast path cannot seed.
        """
        if not cookies:
            return
        entries = _context_cookies(cookies)
        await handle.context.add_cookies(cast("list[SetCookieParam]", entries))

    def _http_proxy(self, egress: str, owner: type[Browser]) -> str | None:
        """Return the proxy the HTTP client must use so it matches the browser's egress exactly.

        A named egress uses its own configured proxy. The default egress uses the proxy its
        browser was launched with, never an inferred one, so the fast path cannot leave through a
        different address than the browser for the same identity.
        """
        if egress == DEFAULT_EGRESS_NAME:
            return owner.proxy_url()
        return self._pool.definition(egress).proxy_url

    async def _quit_group(self, group: Any, body_error: BaseException | None) -> None:
        """Close a request group, reporting a close failure unless the request already failed.

        A close failure on an otherwise successful request is the caller's problem and is
        raised; when the request already failed, that original failure stays authoritative and
        the close failure is only logged.
        """
        if group is None:
            return
        try:
            await group.quit()
        except Exception as exc:  # the close failure is raised or logged, never swallowed
            if body_error is None:
                raise
            logger.warning(f"Failed to close a request tab group: {type(exc).__name__}")

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

    async def _create_group(
        self,
        owner: type[Browser],
        context: BrowserContextHandle | None = None,
    ) -> TabGroup:
        """Time TabGroup acquisition, preserving shared and isolated context selection."""
        start = time.perf_counter()
        try:
            if context is None:
                return await owner.create()
            return await owner.create(context=context)
        finally:
            self.metrics.browser_acquire_seconds.observe(time.perf_counter() - start)

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

        # An isolated tab belongs to its session's context; a shared tab to the persistent one.
        session_id = request.session_id if request.session_mode == ISOLATED_MODE else None
        group = None
        entry: _LiveTab | None = None
        try:
            if session_id is not None:
                session = await self._isolated_session(session_id, request.egress)
                owner, context = session.live()
                group = await self._create_group(owner, context)
            else:
                owner = await self._owner_for(request.egress)
                await owner.start()
                group = await self._create_group(owner)
            if request.cookies:
                tab = await group.ptab
                await tab.set_cookies(cast("list[CookieParam]", request.cookies))
            site = resolve_page_handler(group, request.url)
            source = await site.get(request.url, request.timeout_seconds)

            tab = InteractiveTab(
                tab_id="",
                url=source.url or request.url,
                title=_extract_title(source.text),
                status_code=source.status_code if isinstance(source.status_code, int) else None,
                requested_url=request.url,
                egress=request.egress,
            )
            # Registration is inside the ownership block: a cancellation while waiting for the
            # tab registry must still close the group rather than leave it unowned.
            async with self._tabs_lock:
                self._tab_counter += 1
                tab.tab_id = f"tab-{self._tab_counter}"
                entry = _LiveTab(tab=tab, group=group, session_id=session_id, owns_claim=session_id is None)
                self._tabs[tab.tab_id] = entry
        except BaseException as exc:
            if entry is not None:
                async with self._tabs_lock:
                    self._tabs.pop(entry.tab.tab_id, None)
            # The egress claim has to come back even when the browser never produced a group. An
            # isolated session keeps its lifetime claim, which its own cleanup releases.
            try:
                await self._quit_group(group, exc)
            finally:
                if session_id is None:
                    await self._release(request.egress)
            raise

        self._touch(entry)
        logger.debug(f"Interactive {entry.tab.tab_id} opened, {len(self._tabs)} open.")
        return InteractiveOpenResult(tab=entry.tab)

    async def _retire_dead_tabs(self) -> list[str]:
        """Close interactive tabs whose native page is gone, releasing the claims they own.

        The dead set is snapshotted under the tab lock and every entry is retired through the
        existing close path outside it, so an isolated tab never releases its session's lifetime
        claim and the first close failure is raised only after the whole dead set is attempted.
        """
        async with self._tabs_lock:
            dead_ids = [key for key, entry in self._tabs.items() if not _interactive_group_live(entry.group)]
        closed: list[str] = []
        failure: BaseException | None = None
        for tab_id in dead_ids:
            try:
                closed.extend(await self.close_interactive(tab_id))
            except Exception as exc:  # noqa: BLE001 - every dead tab is attempted, then the failure is raised
                failure = failure or exc
                logger.warning(f"Failed to retire dead interactive {tab_id}: {type(exc).__name__}")
        if failure is not None:
            raise failure
        return closed

    async def _find_reusable_tab(self, request: InteractiveRequest) -> _LiveTab | None:
        """Return an open tab on the request's egress that already shows its url, if any.

        The tab was requested with this url, or it has navigated to it since, and both count:
        a caller that passes the url a redirect landed on is asking for the page it is already
        showing. Reuse is scoped to the context the tab lives in: a shared named request reuses
        a shared tab, and an isolated request only reuses its own session's tab.
        """
        await self._retire_dead_tabs()
        effective_session = request.session_id if request.session_mode == ISOLATED_MODE else None
        async with self._tabs_lock:
            for entry in self._tabs.values():
                if entry.tab.egress != request.egress:
                    continue
                if entry.session_id != effective_session:
                    continue
                if request.url in (entry.tab.requested_url, entry.tab.url):
                    return entry
        return None

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        """Close one interactive tab, or every one when *tab_id* is None.

        A tab that is not open is ignored, so closing is idempotent. Every group is attempted
        and every owned claim is returned; the first close failure is then raised rather than
        reported as a successful close.
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
        failure: BaseException | None = None
        for key, entry in entries:
            try:
                await entry.group.quit()
            except Exception as exc:  # noqa: BLE001 - every group is attempted, then the failure is raised
                failure = failure or exc
                logger.warning(f"Failed to close interactive {key}: {type(exc).__name__}")
            finally:
                if entry.owns_claim:
                    await self._release(entry.tab.egress)
            if failure is None:
                closed.append(key)
        if failure is not None:
            raise failure
        return closed

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        """List tabs; refresh only the named tab's idle countdown when it exists."""
        await self._retire_dead_tabs()
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
        session_id = query.session_id if query.session_mode == ISOLATED_MODE else None
        if session_id is not None:
            session = await self._isolated_session(session_id, query.egress)
            cookies = _normalize_cookies(await _collect_context_cookies(session.require_context()))
        else:
            owner = await self._owner_for(query.egress)
            try:
                await owner.start()
                cookies = _normalize_cookies(await _collect_cookies(owner))
            finally:
                await self._release(query.egress)
        if query.url is None:
            return cookies
        return _cookies_for_url(cookies, query.url)

    async def _steal_least_recent(self, egress: str = DEFAULT_EGRESS_NAME) -> str | None:
        """Under this egress's concurrency pressure, close its oldest tab but never its only tab."""
        if not self._steal_least_recent_enabled:
            return None
        async with self._tabs_lock:
            candidates = [entry for entry in self._tabs.values() if entry.tab.egress == egress]
            if (
                self._max_concurrency <= 0
                or len(candidates) + self._in_flight_by_egress.get(egress, 0) < self._max_concurrency
            ):
                return None
            if len(candidates) < _MIN_TABS_TO_STEAL:
                return None
            tab_id = min(candidates, key=lambda entry: entry.last_used).tab.tab_id
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
    async def _fetch_in_group(
        group: Any,
        request: FetchRequest,
        context: BrowserContextHandle | None,
    ) -> FetchResult:
        tab = await group.ptab
        if request.cookies:
            await tab.set_cookies(cast("list[CookieParam]", request.cookies))

        site = resolve_page_handler(group, request.url)
        if request.disable_media:
            async with site.media_filter():
                return await _fetch_site_result(site, group, request, context)
        return await _fetch_site_result(site, group, request, context)

    async def aclose(self) -> None:
        """Close every interactive tab and isolated context, then shut every browser down.

        Every session is closed through :meth:`close_session`, so ownership and the egress claim
        survive a failed context close instead of being cleared up front. Every resource is
        attempted before the first failure is raised, so shutdown is not abandoned halfway.
        """
        failure: BaseException | None = None
        try:
            await self.close_interactive(None)
        except Exception as exc:  # noqa: BLE001 - the remaining resources are still closed
            failure = failure or exc
        async with self._isolated_lock:
            sessions = [(session_id, entry.egress) for session_id, entry in self._isolated.items()]
        for session_id, egress in sessions:
            cleanup = (
                SessionInfo(id=session_id, mode=ISOLATED_MODE, egress=egress, ttl_minutes=None, close_reason="shutdown")
                if self._state_store is not None
                else None
            )
            try:
                if cleanup is None:
                    await self.close_session(session_id)
                else:
                    await self.close_session(session_id, cleanup=cleanup)
            except Exception as exc:  # noqa: BLE001 - every session is attempted
                failure = failure or exc
        if failure is not None and self._state_store is not None:
            # A state-store save, delete or context close failed. Surface it before shutting the
            # native owners down, so the retained session, its context and its claim can be retried.
            raise failure
        try:
            await self._http.aclose()
        except Exception as exc:  # noqa: BLE001 - every resource is attempted before the first failure is raised
            failure = failure or exc
        try:
            await Browser.shutdown()
        except Exception as exc:  # noqa: BLE001
            failure = failure or exc
            logger.warning(f"Failed to shut the browser down: {type(exc).__name__}")
        try:
            await self._pool.aclose()
        except Exception as exc:  # noqa: BLE001
            failure = failure or exc
        if failure is not None:
            raise failure


def _extract_title(html: str) -> str:
    """Return the document title carried in *html*, or an empty string."""
    match = _TITLE_RE.search(html or "")
    return " ".join(match.group(1).split()) if match else ""


def _context_handle_live(handle: BrowserContextHandle | None) -> bool:
    """Read native liveness without navigation; legacy doubles may omit the browser projection."""
    if handle is None:
        return False
    context = handle.context
    is_closed = getattr(context, "is_closed", None)
    if callable(is_closed) and is_closed():
        return False
    browser = getattr(context, "browser", None)
    is_connected = getattr(browser, "is_connected", None)
    if not callable(is_connected):
        return True
    return bool(is_connected())


def _interactive_group_live(group: Any) -> bool:
    """Whether an interactive group's native page still exists.

    A group whose native context is closed or whose browser is disconnected is dead, and so is one
    whose own page is closed. A group that exposes no native projection (a legacy double) is
    treated as live, so nothing is retired that cannot be observed.
    """
    context_handle = getattr(group, "context", None)
    if context_handle is not None and not _context_handle_live(context_handle):
        return False
    try:
        page = group.ppage
    except AttributeError:
        return True
    except BrowserTabError:
        return False
    is_closed = getattr(page, "is_closed", None)
    if not callable(is_closed):
        return True
    return not is_closed()


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


async def _collect_context_cookies(handle: BrowserContextHandle) -> list[dict[str, Any]]:
    """Return an isolated context's own cookies in the shape a profile read reports.

    A failure propagates: an isolated cookie read that cannot be answered must not be reported
    as an empty jar, which would look like a context that holds no cookies.
    """
    return [dict(cookie) for cookie in await handle.context.cookies()]


async def _fetch_cookies(group: Any, context: BrowserContextHandle | None) -> list[dict[str, Any]]:
    """Return the cookies a fetch response reports for the context the request ran in.

    An isolated request reads its own context, so a response can never carry the shared
    persistent profile's cookies. A shared request keeps reading the profile it always read.
    """
    if context is not None:
        return _normalize_cookies(await _collect_context_cookies(context))
    return _normalize_cookies(await _collect_cookies(group))


async def _fetch_site_result(
    site: PageHandler,
    group: TabGroup,
    request: FetchRequest,
    context: BrowserContextHandle | None,
) -> FetchResult:
    """Collect the fetched document, optional capture and final context cookies."""
    captcha: dict[str, Any] = {"solve_captcha": True} if request.solve_captcha else {}
    if request.method == "POST":
        source = await site.post(
            request.url,
            request.timeout_seconds,
            post_data=request.post_data,
            headers=request.headers,
        )
    elif request.tabs_till_verify is not None:
        source = await site.get(
            request.url,
            request.timeout_seconds,
            headers=request.headers,
            header_scope=request.header_scope,
            tabs_till_verify=request.tabs_till_verify,
            **captcha,
        )
    elif request.headers:
        source = await site.get(
            request.url,
            request.timeout_seconds,
            headers=request.headers,
            header_scope=request.header_scope,
            **captcha,
        )
    else:
        source = await site.get(request.url, request.timeout_seconds, **captcha)

    if request.wait_in_seconds > 0 or request.return_screenshot:
        source = await site.snapshot(
            source,
            wait_in_seconds=request.wait_in_seconds,
            return_screenshot=request.return_screenshot,
            post_response=request.method == "POST",
        )

    cookies = await _fetch_cookies(group, context)
    return FetchResult(
        url=source.url or request.url,
        status_code=source.status_code if isinstance(source.status_code, int) else None,
        headers=dict(source.headers),
        response=source.text,
        cookies=cookies,
        user_agent=source.user_agent,
        screenshot=source.screenshot if request.return_screenshot else None,
        turnstile_token=source.turnstile_token if request.tabs_till_verify is not None else None,
        captcha_provider=source.captcha_provider if request.solve_captcha else None,
        captcha_token=source.captcha_token if request.solve_captcha else None,
    )


def _classified(result: FetchResult) -> FetchResult:
    """Report a browser fallback's actual mode and an honest classification of its response.

    A browser result that still carries a challenge is classified as that challenge, so a fallback
    never claims a CAPTCHA or a Cloudflare interstitial was solved. The body is a rendered DOM, so
    the HTTP-only JavaScript-shell heuristic does not apply and is skipped.
    """
    return replace(
        result,
        mode=BROWSER_MODE,
        classification=classify(result.status_code or 0, result.headers, result.response, rendered=True),
    )


def _with_remaining_timeout(request: FetchRequest, deadline: float, *, drop_cookies: bool) -> FetchRequest:
    """Return *request* bounded by what is left of the route's budget instead of a fresh one."""
    remaining = max(0.0, deadline - time.monotonic())
    return replace(
        request,
        timeout_seconds=max(1, math.ceil(remaining)),
        cookies=[] if drop_cookies else request.cookies,
    )


def _context_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *cookies* in the shape a native context accepts, refusing unsupported state.

    :raises UnsupportedCookieError: when a cookie carries a field the fast path cannot represent,
        such as a partitioned cookie's ``partitionKey``, or names no domain to scope it to.
    """
    seeded: list[dict[str, Any]] = []
    for cookie in cookies:
        unsupported = sorted(name for name in cookie if name not in _SEED_COOKIE_FIELDS)
        if unsupported:
            msg = f"cookie fields are not supported by the HTTP fast path: {', '.join(unsupported)}"
            raise UnsupportedCookieError(msg)
        domain = cookie.get("domain")
        if not isinstance(domain, str) or not domain:
            msg = "an HTTP fetch needs each supplied cookie to name a domain"
            raise UnsupportedCookieError(msg)
        entry: dict[str, Any] = {
            "name": str(cookie.get("name") or ""),
            "value": str(cookie.get("value") or ""),
            "domain": domain,
            "path": str(cookie.get("path") or "/"),
        }
        expires = cookie.get("expires")
        if isinstance(expires, int | float) and not isinstance(expires, bool) and expires > 0:
            entry["expires"] = int(expires)
        if cookie.get("secure") is True:
            entry["secure"] = True
        if cookie.get("httpOnly") is True:
            entry["httpOnly"] = True
        if cookie.get("sameSite") in _SEED_COOKIE_SAME_SITE:
            entry["sameSite"] = cookie["sameSite"]
        seeded.append(entry)
    return seeded


async def _apply_request_cookie_header(handle: BrowserContextHandle, request: FetchRequest) -> None:
    """Apply caller cookie values once, preserving the selected context's native scopes."""
    header = request.cookie_header
    if not header:
        return
    current = await handle.context.cookies([request.url])
    updates = request_cookie_updates(header, current, request.url)
    if not updates:
        return
    try:
        await handle.context.add_cookies(updates)
    except PlaywrightError:
        # A native rejection is the caller's cookie problem; the native detail is never echoed.
        raise CookieHeaderError from None


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


async def _run_blocking_io[**P, T](func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Join state mutations before propagating cancellation, including repeated cancellation."""
    worker = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - cancellation remains the primary failure
                break
        if not worker.cancelled():
            worker.exception()
        raise


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
