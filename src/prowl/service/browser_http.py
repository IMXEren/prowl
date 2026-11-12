"""One :class:`BrowserHttpClients` helper: identity-matched HTTP clients for live contexts.

A native browser context is the unit of HTTP client ownership. :class:`BrowserHttpClients` hands
out one :class:`~prowl.service.http_transport.HttpClient` per actual context, discovers that
context's own browser identity once per browser process generation, and closes the clients it owns.
The transport itself stays in :mod:`prowl.service.http_transport`; this module only decides which
client belongs to which context and how it is retired.

Identity is read from the running browser, never guessed. A short-lived probe page is opened, every
one of its requests is intercepted, and a fixed reserved HTTPS origin is answered locally, so the
page becomes a secure context without DNS, TLS, external traffic, cookies or storage writes. The
identity the browser reports through ``navigator.userAgentData`` is matched to an impersonation
profile with :func:`~prowl.service.http_transport.resolve_http_identity`, and only the identity
headers the browser actually sent on that navigation are kept, so no language or high-entropy hint
is invented and no request to the synthetic origin can leave the process.

Known boundaries:

- This helper never closes a native context or browser; it only closes the HTTP sessions it owns.
- Identity is cached per browser process generation, and an
  :class:`~prowl.service.http_transport.UnsupportedIdentityError` is cached with it, so a persona
  with no profile is refused at most once per generation.
- A transient or cancelled probe is not cached; its owned page is closed and the next caller
  retries with a fresh one.
- The probe navigation is the only source of identity headers. ``Accept-Language`` is taken from the
  request's own complete header set when the browser sends one, but the synthetic origin does not
  negotiate a language, so a browser that omits it there yields no language header and none is
  fabricated to fill the gap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Final

from loguru import logger

from prowl.service.http_transport import (
    HttpClient,
    HttpIdentity,
    HttpTransportError,
    ObservedBrowserIdentity,
    UnsupportedIdentityError,
    resolve_http_identity,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from playwright.async_api import BrowserContext

#: A reserved, non-resolvable origin used to give the probe page a secure context.
SYNTHETIC_ORIGIN: Final[str] = "https://identity.invalid"

_PROBE_URL: Final[str] = f"{SYNTHETIC_ORIGIN}/"
_PROBE_BODY: Final[str] = "<!doctype html><title>identity</title>"

#: The response status a locally fulfilled probe request carries.
_PROBE_STATUS: Final[int] = 200

#: A brand pair the browser reports is exactly two strings.
_BRAND_PAIR_LEN: Final[int] = 2

#: How many times shutdown drains its owned callback tasks before giving up.
_DRAIN_PASSES: Final[int] = 8

_SHUTDOWN: Final[str] = "the browser HTTP client manager has started shutting down"
_RETIRING: Final[str] = "the browser context's HTTP client is being retired"
_CLOSED_CONTEXT: Final[str] = "the browser context is closed"
_DISCONNECTED_BROWSER: Final[str] = "the browser is not connected"

#: The headers that carry a browser's own identity rather than the shape of one request. Only the
#: ones the browser really sent are reused, so a hint it did not volunteer is never fabricated.
_IDENTITY_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "accept-language",
        "sec-ch-ua",
        "sec-ch-ua-arch",
        "sec-ch-ua-bitness",
        "sec-ch-ua-full-version",
        "sec-ch-ua-full-version-list",
        "sec-ch-ua-mobile",
        "sec-ch-ua-model",
        "sec-ch-ua-platform",
        "sec-ch-ua-platform-version",
        "sec-ch-ua-wow64",
        "user-agent",
    },
)

#: Reads the browser's own low- and high-entropy client hints from the probe page. It runs in a
#: secure context, so ``navigator.userAgentData`` is available; ``None`` means it was not.
_OBSERVE_SCRIPT: Final[str] = """async () => {
  const data = navigator.userAgentData;
  if (!data) {
    return null;
  }
  const hints = await data.getHighEntropyValues([
    "architecture",
    "bitness",
    "platformVersion",
    "fullVersionList",
  ]);
  return {
    userAgent: navigator.userAgent,
    languages: Array.from(navigator.languages),
    platform: data.platform || "",
    mobile: data.mobile === true,
    brands: (data.brands || []).map((brand) => [brand.brand, brand.version]),
    architecture: hints.architecture || "",
    bitness: hints.bitness || "",
    platformVersion: hints.platformVersion || "",
    fullVersionList: (hints.fullVersionList || []).map((brand) => [brand.brand, brand.version]),
  };
}"""


async def observe_identity(context: BrowserContext) -> HttpIdentity:
    """Read *context*'s own browser identity and match it to an impersonation profile.

    The probe page's navigation is answered locally at :data:`SYNTHETIC_ORIGIN` and every other
    request it makes is aborted, so no DNS lookup, TLS handshake or external request happens. The
    page is always closed, whether observation succeeds, fails or is cancelled.

    :raises UnsupportedIdentityError: when the reported persona has no matching profile, when the
        probe page reported no client hints, or when its user agent disagrees with the one it sent.
    """
    page = await context.new_page()
    captured: dict[str, str] = {}
    try:

        async def _intercept(route: Any, request: Any) -> None:
            if request.url == _PROBE_URL:
                headers = await request.all_headers()
                captured.update({name.lower(): value for name, value in headers.items()})
                await route.fulfill(status=_PROBE_STATUS, content_type="text/html", body=_PROBE_BODY)
            else:
                await route.abort()

        await page.route("**/*", _intercept)
        await page.goto(_PROBE_URL, wait_until="domcontentloaded")
        reported = await page.evaluate(_OBSERVE_SCRIPT)
    finally:
        await page.close()
    observed = _observed_identity(reported)
    identity = resolve_http_identity(observed)
    wire_user_agent = captured.get("user-agent", "")
    if wire_user_agent != observed.user_agent:
        msg = "the probe page's user agent disagrees with the user agent it sent on the wire"
        raise UnsupportedIdentityError(msg)
    if not captured.get("accept-language"):
        if reported.get("languages") != ["en-US", "en"]:
            msg = "native Accept-Language is unavailable for this persona; use browser mode"
            raise UnsupportedIdentityError(msg)
        # Verified against a real loopback request; other language variants remain unsupported.
        captured["accept-language"] = "en-US,en;q=0.9"
    return replace(identity, headers=_identity_headers(captured))


def _observed_identity(reported: Mapping[str, Any] | None) -> ObservedBrowserIdentity:
    """Turn the probe page's reported client hints into the browser's observed identity.

    :raises UnsupportedIdentityError: when the probe page reported no client-hint identity.
    """
    if not reported:
        msg = "the probe page reported no client-hint identity"
        raise UnsupportedIdentityError(msg)
    return ObservedBrowserIdentity(
        user_agent=str(reported.get("userAgent") or ""),
        platform=str(reported.get("platform") or ""),
        brands=_brand_pairs(reported.get("brands")),
        full_version_list=_brand_pairs(reported.get("fullVersionList")),
        architecture=str(reported.get("architecture") or ""),
        bitness=str(reported.get("bitness") or ""),
        platform_version=str(reported.get("platformVersion") or ""),
        mobile=reported.get("mobile") is True,
    )


def _brand_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    """Return a reported brand list as ordered string pairs, ignoring anything malformed."""
    if not isinstance(value, list):
        return ()
    return tuple(
        (str(pair[0]), str(pair[1])) for pair in value if isinstance(pair, list) and len(pair) == _BRAND_PAIR_LEN
    )


def _identity_headers(captured: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Keep only the identity headers the browser actually sent, in a stable order."""
    return tuple((name, captured[name]) for name in sorted(_IDENTITY_HEADERS) if name in captured)


@dataclass(slots=True)
class _Generation:
    """One browser process generation and the identity read from it."""

    browser: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    identity: HttpIdentity | None = None
    unsupported: UnsupportedIdentityError | None = None
    states: set[_ContextState] = field(default_factory=set)


@dataclass(slots=True, eq=False)
class _ContextState:
    """One native context's HTTP client and the generation its identity came from."""

    context: BrowserContext
    browser: Any
    generation: _Generation
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: HttpClient | None = None
    proxy: str | None = None
    bound: bool = False
    retiring: bool = False


class BrowserHttpClients:
    """Owns identity-matched HTTP clients, one per live native browser context.

    Identity is shared by every context of one browser process generation; cookie state is not,
    because each client keeps its own context as the single owner of that context's jar. The caller
    is the only one that closes native contexts and browsers.
    """

    def __init__(self) -> None:
        """Create an empty manager; no client, task or listener exists yet."""
        self._lock = asyncio.Lock()
        self._states: dict[BrowserContext, _ContextState] = {}
        self._generations: dict[Any, _Generation] = {}
        self._listeners: dict[Any, list[tuple[str, Callable[..., None]]]] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    async def client(self, context: BrowserContext, proxy: str | None) -> HttpClient:
        """Return the client bound to *context*, creating it on first use.

        The ready path is a lookup plus a liveness check: an existing client is returned without a
        task, navigation, identity read or other repeated work. Only the first caller for a context
        waits for identity discovery and the creation of its session, and it does so under this
        context's own lock, so a context that closes or a manager that shuts down while it waits is
        seen before a client is ever constructed.

        :raises HttpTransportError: when the manager has started shutting down, when *context* is
            closed or its browser disconnected, when *proxy* disagrees with the one already bound,
            or when the context's client is being retired.
        :raises UnsupportedIdentityError: when the browser persona has no matching profile.
        """
        if self._closing:
            msg = _SHUTDOWN
            raise HttpTransportError(msg)
        state = self._states.get(context)
        if state is not None and state.client is not None:
            self._ready(context, state, proxy)
            return state.client
        async with self._lock:
            if self._closing:
                msg = _SHUTDOWN
                raise HttpTransportError(msg)
            state = self._states.get(context)
            if state is None:
                state = self._register(context)
        async with state.lock:
            return await self._locked_client(context, state, proxy)

    def _ready(self, context: BrowserContext, state: _ContextState, proxy: str | None) -> None:
        """Refuse a cached client whose context is retiring, closed, disconnected or mis-proxied."""
        if state.retiring:
            msg = _RETIRING
            raise HttpTransportError(msg)
        self._require_live(context, state)
        self._check_proxy(state, proxy)

    async def _locked_client(
        self,
        context: BrowserContext,
        state: _ContextState,
        proxy: str | None,
    ) -> HttpClient:
        """Create the context's client under its own lock, rechecking shutdown and liveness."""
        if self._closing:
            msg = _SHUTDOWN
            raise HttpTransportError(msg)
        if state.retiring:
            msg = _RETIRING
            raise HttpTransportError(msg)
        if state.client is not None:
            self._require_live(context, state)
            self._check_proxy(state, proxy)
            return state.client
        identity = await self._identity(state)
        if self._closing:
            msg = _SHUTDOWN
            raise HttpTransportError(msg)
        if state.retiring:
            msg = _RETIRING
            raise HttpTransportError(msg)
        self._require_live(context, state)
        state.client = HttpClient(context=context, identity=identity, proxy=proxy)
        state.proxy = proxy
        state.bound = True
        return state.client

    async def close_context(self, context: BrowserContext) -> None:
        """Close the HTTP session owned by *context*, leaving the native context untouched.

        Retirement is fenced before the close is awaited, so no new fetch client is acquired for a
        context that is going away. A close that fails keeps the client owned, and keeps the context
        retired, so a later call retries it instead of dropping a live session. A context that never
        grew a client still has its state and listeners released. Closing a context the manager does
        not own is a no-op.
        """
        state = self._states.get(context)
        if state is None:
            return
        state.retiring = True
        async with state.lock:
            client = state.client
            if client is not None:
                await client.aclose()
            state.client = None
            state.proxy = None
            state.bound = False
            self._remove_state(context, state)

    async def aclose(self) -> None:
        """Freeze callbacks, drain owned work, then close every owned HTTP session.

        Every owned client is attempted before the first failure is raised, and a failed close keeps
        its client owned for a retry.
        """
        self._closing = True
        for target in list(self._listeners):
            self._forget_target(target)
        await self._drain_tasks()
        failures: list[BaseException] = []
        for context in list(self._states):
            try:
                await self.close_context(context)
            except Exception as error:  # noqa: BLE001 - every failure is collected, not raised yet
                failures.append(error)
        await self._drain_tasks()
        if failures:
            raise failures[0]

    def _check_proxy(self, state: _ContextState, proxy: str | None) -> None:
        """Refuse a proxy that disagrees with the one this context's client already uses."""
        if state.bound and state.proxy != proxy:
            msg = "a browser context's HTTP client keeps the proxy it was created with"
            raise HttpTransportError(msg)

    def _require_live(self, context: BrowserContext, state: _ContextState) -> None:
        """Refuse to hand out a client for a closed context or a disconnected browser."""
        if context.is_closed():
            msg = _CLOSED_CONTEXT
            raise HttpTransportError(msg)
        if not state.browser.is_connected():
            msg = _DISCONNECTED_BROWSER
            raise HttpTransportError(msg)

    def _register(self, context: BrowserContext) -> _ContextState:
        """Start owning *context*, attaching the native callbacks that retire it.

        :raises HttpTransportError: when the context is closed, has no browser, or its browser is
            already disconnected.
        """
        browser = getattr(context, "browser", None)
        if browser is None:
            msg = "the browser context is not attached to a browser"
            raise HttpTransportError(msg)
        if context.is_closed():
            msg = _CLOSED_CONTEXT
            raise HttpTransportError(msg)
        if not browser.is_connected():
            msg = _DISCONNECTED_BROWSER
            raise HttpTransportError(msg)
        generation = self._generations.get(browser)
        if generation is None:
            generation = _Generation(browser=browser)
            self._generations[browser] = generation
            self._listen(browser, "disconnected", lambda: self._retire_generation(generation))
        state = _ContextState(context=context, browser=browser, generation=generation)
        generation.states.add(state)
        self._states[context] = state
        self._listen(context, "close", lambda: self.close_context(context))
        return state

    async def _identity(self, state: _ContextState) -> HttpIdentity:
        """Return this generation's identity, probing at most once per generation.

        A transient or cancelled observation is left uncached so the next caller retries, while an
        unsupported persona is cached for the whole generation.
        """
        generation = state.generation
        async with generation.lock:
            if generation.identity is not None:
                return generation.identity
            if generation.unsupported is not None:
                raise generation.unsupported
            try:
                identity = await observe_identity(state.context)
            except UnsupportedIdentityError as error:
                generation.unsupported = error
                raise
            generation.identity = identity
            return identity

    async def _retire_generation(self, generation: _Generation) -> None:
        """Forget a disconnected browser, release its listeners, and close the sessions that used it.

        Every client of the generation is attempted before the first failure is raised.
        """
        generation.identity = None
        generation.unsupported = None
        self._generations.pop(generation.browser, None)
        self._forget_target(generation.browser)
        failures: list[BaseException] = []
        for state in list(generation.states):
            try:
                await self.close_context(state.context)
            except Exception as error:  # noqa: BLE001 - every failure is collected, not raised yet
                failures.append(error)
        if failures:
            raise failures[0]

    def _remove_state(self, context: BrowserContext, state: _ContextState) -> None:
        """Drop one context's ownership, its generation membership and its native listener."""
        self._states.pop(context, None)
        state.generation.states.discard(state)
        self._forget_target(context)

    def _forget_target(self, target: Any) -> None:
        """Detach every native listener the manager still holds for *target*."""
        for event, handler in self._listeners.pop(target, []):
            target.remove_listener(event, handler)

    async def _drain_tasks(self) -> None:
        """Let every owned callback task finish, bounded so a self-rescheduling task cannot spin."""
        for _ in range(_DRAIN_PASSES):
            pending = list(self._tasks)
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def _listen(self, target: Any, event: str, work: Callable[[], Awaitable[None]]) -> None:
        """Attach *work* to a native event, run as an owned, error-reporting task."""

        def handler(*_args: object) -> None:
            self._spawn(work())

        target.on(event, handler)
        self._listeners.setdefault(target, []).append((event, handler))

    def _spawn(self, work: Awaitable[None]) -> None:
        """Run a callback's work as an owned task so a failing callback is never silent."""
        try:
            task = asyncio.ensure_future(work)
        except RuntimeError:
            logger.warning("A browser HTTP callback fired with no running event loop.")
            return
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Report a finished callback task's failure, if any."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.warning(f"A browser HTTP callback failed: {type(error).__name__}: {error}")


__all__ = [
    "SYNTHETIC_ORIGIN",
    "BrowserHttpClients",
    "observe_identity",
]
