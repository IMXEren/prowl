"""Long-lived asyncio HTTP service speaking the FlareSolverr v1 contract.

Requests are dispatched to :class:`~prowl.service.backend.BrowserBackend`,
which drives the browser of the request's egress. Concurrency is bounded by a
semaphore; each logical session serializes its own work and anonymous requests
serialize on the lock of their egress, because an egress owns one persistent
profile.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from aiohttp import web
from loguru import logger

from prowl.browser import Browser, BrowserConfig
from prowl.browser.config import (
    DEFAULT_MAX_CONTEXTS,
    MAX_CONTEXTS_ENV,
    PROXY_URL_ENV,
    default_extensions_dir,
    default_max_contexts,
    default_policy_dir,
    default_profile_archive,
    default_profile_dir,
)
from prowl.browser.proxy.certificates import ProxyCertificates
from prowl.browser.proxy.egress import (
    DEFAULT_EGRESS_IDLE_SECONDS,
    DEFAULT_EGRESS_NAME,
    EGRESS_IDLE_SECONDS_ENV,
    EGRESSES_ENV,
    EgressError,
    parse_egress_spec,
)
from prowl.browser.proxy.server import ProxyServer
from prowl.service.backend import (
    DEFAULT_INTERACTIVE_IDLE_SECONDS,
    DEFAULT_STEAL_LEAST_RECENT,
    INTERACTIVE_IDLE_SECONDS_ENV,
    STEAL_LEAST_RECENT_ENV,
    Backend,
    BrowserBackend,
    CookieQuery,
    FetchRequest,
    FetchResult,
    InteractiveRequest,
)
from prowl.service.context_state import ContextStateStore
from prowl.service.errors import CallerSafeError, ProxyError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from prowl.service.protocol import (
    CMD_BROWSER_CLOSE,
    CMD_BROWSER_LIST,
    CMD_BROWSER_OPEN,
    CMD_COOKIES_LIST,
    CMD_REQUEST_GET,
    CMD_REQUEST_POST,
    CMD_SESSIONS_CREATE,
    CMD_SESSIONS_DESTROY,
    CMD_SESSIONS_LIST,
    DEFAULT_TIMEOUT_MS,
    BrowserCloseCommand,
    BrowserListCommand,
    BrowserOpenCommand,
    Command,
    CookiesListCommand,
    FetchCommand,
    ProtocolError,
    ProxySelection,
    SessionsCreateCommand,
    SessionsDestroyCommand,
    SessionsListCommand,
    cookies_payload,
    error_response,
    execution_payload,
    now_ms,
    ok_response,
    parse_request,
    solution_payload,
    tab_payload,
    validate_configured_proxy_url,
)
from prowl.service.sessions import SHARED_MODE, SessionInfo, SessionRegistry

#: Extra wall-clock slack granted to the browser beyond the caller timeout so
#: the core can run its own cleanup before the service gives up.
_TIMEOUT_SLACK_SECONDS = 10.0

#: A cookie read is a local browser call with no caller-supplied deadline, so it is bounded by
#: the default command timeout plus the usual slack.
_COOKIE_READ_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_MS / 1000 + _TIMEOUT_SLACK_SECONDS

#: Caller-safe messages used when a failure has no declared safe string.
_TIMEOUT_MESSAGE = "request timed out"
_INTERNAL_MESSAGE = "internal error while executing the command"
_PROXY_MESSAGE = "request proxy is not permitted; it must match a configured egress"
_UNKNOWN_EGRESS_MESSAGE = "unknown egress name; it must match a configured egress"

_DEFAULT_MAX_SESSIONS = 32

#: The idle TTL a new isolated session gets when its caller omits one.
_DEFAULT_SESSION_TTL_MINUTES = 60

#: Environment variable enabling the integrated forward proxy listener. Unset means disabled.
FORWARD_PROXY_PORT_ENV: Final[str] = "PROWL_FORWARD_PROXY_PORT"

#: Environment variables naming the operator CA pair used to terminate CONNECT tunnels.
PROXY_CA_CERT_ENV: Final[str] = "PROWL_PROXY_CA_CERT"
PROXY_CA_KEY_ENV: Final[str] = "PROWL_PROXY_CA_KEY"

#: Largest TCP port number the forward proxy listener accepts.
_MAX_PORT: Final[int] = 65535

#: How often the service expires idle sessions. The service owns the timer, not the registry.
SESSION_SWEEP_SECONDS: Final[float] = 30.0

#: Response header carrying the correlation id minted for one HTTP command.
_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"


def _egress_idle_seconds() -> float:
    """Return how long a named egress browser may stay idle.

    :raises EgressError: when the value is not a number.
    """
    raw = os.environ.get(EGRESS_IDLE_SECONDS_ENV, "").strip()
    if not raw:
        return DEFAULT_EGRESS_IDLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        msg = f"{EGRESS_IDLE_SECONDS_ENV} must be a number of seconds, got {raw!r}"
        raise EgressError(msg) from exc


def _steal_least_recent() -> bool:
    """Whether a caller may take over the least recently used interactive tab."""
    raw = os.environ.get(STEAL_LEAST_RECENT_ENV, "").strip().lower()
    if not raw:
        return DEFAULT_STEAL_LEAST_RECENT
    return raw in {"1", "true", "yes", "on"}


def _interactive_idle_seconds() -> float:
    """Return how long an interactive tab may sit untouched before it is closed.

    Unset or zero keeps interactive tabs open until they are closed explicitly, which is the
    default because a person is driving the tab.

    :raises ValueError: when the value is not a number.
    """
    raw = os.environ.get(INTERACTIVE_IDLE_SECONDS_ENV, "").strip()
    if not raw:
        return DEFAULT_INTERACTIVE_IDLE_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        msg = f"{INTERACTIVE_IDLE_SECONDS_ENV} must be a number of seconds, got {raw!r}"
        raise ValueError(msg) from exc


def _forward_proxy_port() -> int | None:
    """Return the configured listener port, or ``None`` when the forward proxy is disabled.

    :raises ValueError: when the value is not an integer inside the port range.
    """
    raw = os.environ.get(FORWARD_PROXY_PORT_ENV, "").strip()
    if not raw:
        return None
    message = f"{FORWARD_PROXY_PORT_ENV} must be an integer between 0 and {_MAX_PORT}"
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not 0 <= port <= _MAX_PORT:
        raise ValueError(message)
    return port


def _session_ttl_minutes() -> int:
    """Return the idle TTL given to a new isolated session whose caller omits one.

    :raises ValueError: when the value is not a positive number of minutes.
    """
    raw = os.environ.get("PROWL_SESSION_TTL_MINUTES", "").strip()
    if not raw:
        return _DEFAULT_SESSION_TTL_MINUTES
    message = f"PROWL_SESSION_TTL_MINUTES must be a positive number of minutes, got {raw!r}"
    try:
        minutes = int(raw)
    except ValueError as exc:
        raise ValueError(message) from exc
    if minutes < 1:
        raise ValueError(message)
    return minutes


@dataclass(slots=True)
class ServiceConfig:
    """Runtime configuration for the HTTP service."""

    host: str = "0.0.0.0"  # noqa: S104 - intentional container bind
    port: int = 8191
    max_concurrency: int = 1
    max_sessions: int = _DEFAULT_MAX_SESSIONS
    proxy_url: str | None = field(default=None, repr=False)
    egresses: dict[str, str] = field(default_factory=dict, repr=False)
    egress_idle_seconds: float = DEFAULT_EGRESS_IDLE_SECONDS
    interactive_idle_seconds: float = DEFAULT_INTERACTIVE_IDLE_SECONDS
    steal_least_recent: bool = DEFAULT_STEAL_LEAST_RECENT
    profile_dir: str = field(default_factory=default_profile_dir)
    profile_archive: str = field(default_factory=default_profile_archive)
    extensions_dir: str | None = field(default_factory=default_extensions_dir)
    policy_dir: str | None = field(default_factory=default_policy_dir)
    #: The per-identity context cap handed to the browser and the session registry. The
    #: shared persistent context consumes one slot, so the registry admits one less
    #: isolated session per egress than this value.
    max_contexts: int = DEFAULT_MAX_CONTEXTS
    #: The idle TTL a new isolated session gets when its caller omits one. Shared sessions
    #: stay unlimited; an explicit TTL or ``None`` on a create still wins.
    session_ttl_minutes: int = _DEFAULT_SESSION_TTL_MINUTES
    #: Optional private root for isolated session storage state. Unset by default, so a process
    #: keeps no session state on disk until an operator names a directory.
    session_state_dir: str | None = None
    #: Port for the integrated forward proxy listener. ``None`` disables it entirely; ``0`` asks
    #: the operating system for an ephemeral port.
    forward_proxy_port: int | None = None
    #: Operator certificate authority used to terminate CONNECT tunnels. Both halves must be set
    #: together and only alongside an enabled listener.
    proxy_ca_cert: str | None = None
    proxy_ca_key: str | None = None

    def __post_init__(self) -> None:
        """Reject an unusable context cap or forward proxy configuration before native work.

        :raises ValueError: when the context cap is below one, the proxy port is outside the port
            range, only one half of the CA pair is given, or a CA is given without a listener.
        """
        if self.max_contexts < 1:
            msg = f"{MAX_CONTEXTS_ENV} must be an integer of at least 1, got {self.max_contexts!r}"
            raise ValueError(msg)
        if self.forward_proxy_port is not None and not 0 <= self.forward_proxy_port <= _MAX_PORT:
            msg = f"{FORWARD_PROXY_PORT_ENV} must be an integer between 0 and {_MAX_PORT}"
            raise ValueError(msg)
        if (self.proxy_ca_cert is None) != (self.proxy_ca_key is None):
            msg = f"{PROXY_CA_CERT_ENV} and {PROXY_CA_KEY_ENV} must be supplied together"
            raise ValueError(msg)
        if self.proxy_ca_cert is not None and self.forward_proxy_port is None:
            msg = f"{PROXY_CA_CERT_ENV} requires {FORWARD_PROXY_PORT_ENV}"
            raise ValueError(msg)

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Build configuration from environment variables.

        :raises EgressError: when the egress list or one of its URLs is unusable.
        :raises ValueError: when the process-wide proxy URL is unusable.
        """
        proxy_url = os.environ.get(PROXY_URL_ENV, "").strip() or None
        if proxy_url is not None:
            validate_configured_proxy_url(proxy_url)
        egresses = parse_egress_spec(os.environ.get(EGRESSES_ENV, ""))
        for name in sorted(egresses):
            try:
                validate_configured_proxy_url(egresses[name])
            except ValueError as exc:
                msg = f"{EGRESSES_ENV} egress {name!r} is unusable: {exc}"
                raise EgressError(msg) from exc
        return cls(
            host=os.environ.get("PROWL_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.environ.get("PROWL_PORT", "8191")),
            max_concurrency=max(1, int(os.environ.get("PROWL_MAX_CONCURRENCY", "1"))),
            max_sessions=max(1, int(os.environ.get("PROWL_MAX_SESSIONS", str(_DEFAULT_MAX_SESSIONS)))),
            proxy_url=proxy_url,
            egresses=egresses,
            egress_idle_seconds=_egress_idle_seconds(),
            interactive_idle_seconds=_interactive_idle_seconds(),
            steal_least_recent=_steal_least_recent(),
            profile_dir=default_profile_dir(),
            profile_archive=default_profile_archive(),
            extensions_dir=default_extensions_dir(),
            policy_dir=default_policy_dir(),
            max_contexts=default_max_contexts(),
            session_ttl_minutes=_session_ttl_minutes(),
            session_state_dir=os.environ.get("PROWL_SESSION_STATE_DIR", "").strip() or None,
            forward_proxy_port=_forward_proxy_port(),
            proxy_ca_cert=os.environ.get(PROXY_CA_CERT_ENV, "").strip() or None,
            proxy_ca_key=os.environ.get(PROXY_CA_KEY_ENV, "").strip() or None,
        )


def _report_shutdown_failure(task: asyncio.Task[None]) -> None:
    """Log the shutdown task's failure so a cancelled waiter cannot hide it."""
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(f"Service shutdown reported a cleanup failure: {type(error).__name__}")


class Service:
    """Command dispatcher shared by the HTTP handlers and the tests."""

    def __init__(
        self,
        config: ServiceConfig,
        backend: Backend,
        *,
        sweep_interval_seconds: float = SESSION_SWEEP_SECONDS,
    ) -> None:
        self.config = config
        self.backend = backend
        if config.proxy_url is not None:
            validate_configured_proxy_url(config.proxy_url)
        self.egresses = self._configured_egress_urls(config)
        # Destroy and expiry release a session's backend resources through this callback once
        # its leases have drained, so the backend close happens behind the registry's fence.
        self.sessions = SessionRegistry(
            max_sessions=config.max_sessions,
            cleanup=self._cleanup_session,
            isolated_per_egress_limit=config.max_contexts - 1,
            default_isolated_ttl_minutes=config.session_ttl_minutes,
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        # Anonymous (session-less) requests share the persistent trust profile of
        # their egress, so they must never overlap within one egress: one lock per
        # egress, not one lock for the process.
        self._anonymous_locks = {name: asyncio.Lock() for name in self.egresses}
        self._sweep_interval = max(0.01, float(sweep_interval_seconds))
        self._sweep_task: asyncio.Task[None] | None = None
        #: The single owned shutdown, shielded so a cancelled waiter cannot make backend
        #: shutdown race session cleanup.
        self._shutdown_task: asyncio.Task[None] | None = None

    async def _cleanup_session(self, info: SessionInfo) -> None:
        """Release a destroyed or expired session's backend resources.

        A concrete backend with a configured state store receives the registry's own metadata, so
        it can preserve or forget the binding according to the cleanup reason. Every other backend
        keeps the original positional contract, with an automatic eviction flagged for the concrete
        backend's native context counting.
        """
        if isinstance(self.backend, BrowserBackend) and self.backend.state_persistence_enabled:
            await self.backend.close_session(info.id, evicted=info.evicted, cleanup=info)
        elif info.evicted and isinstance(self.backend, BrowserBackend):
            await self.backend.close_session(info.id, evicted=True)
        else:
            await self.backend.close_session(info.id)

    def start_periodic_cleanup(self) -> None:
        """Start the owned periodic sweep that expires idle sessions.

        The registry holds no timer, so the service owns this task and stops it on shutdown.
        """
        if self._sweep_task is None:
            self._sweep_task = asyncio.ensure_future(self._sweep_loop())

    async def stop_periodic_cleanup(self) -> None:
        """Cancel and await the periodic sweep, if one is running."""
        task = self._sweep_task
        self._sweep_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _sweep_loop(self) -> None:
        """Expire idle sessions at the fixed interval, logging rather than losing a failure."""
        while True:
            await asyncio.sleep(self._sweep_interval)
            try:
                await self.sessions.purge_expired()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("session sweep failed")

    async def aclose(self) -> None:
        """Stop periodic work, drain every session's cleanup, then close the backend.

        One shutdown task is owned and shielded, so a cancelled waiter cannot make backend
        shutdown race session cleanup and a second caller joins the same shutdown. A cleanup
        failure is reported after the backend is closed, so shutdown always releases browser
        resources.
        """
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown())
            task.add_done_callback(_report_shutdown_failure)
            self._shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown(self) -> None:
        """Run the one shutdown: registry cleanup first, backend last, every failure reported."""
        failure: BaseException | None = None
        try:
            await self.stop_periodic_cleanup()
            await self.sessions.aclose()
        except Exception as exc:  # noqa: BLE001 - the backend is closed either way
            failure = failure or exc
        try:
            await self.backend.aclose()
        except Exception as exc:  # noqa: BLE001
            failure = failure or exc
        if failure is not None:
            raise failure

    @staticmethod
    def _configured_egress_urls(config: ServiceConfig) -> dict[str, str | None]:
        """Return every configured egress URL by name, validating each one.

        :raises ValueError: when a name or URL is unusable.
        """
        if DEFAULT_EGRESS_NAME in config.egresses:
            msg = f"egress name {DEFAULT_EGRESS_NAME!r} is reserved for {PROXY_URL_ENV}"
            raise ValueError(msg)
        urls: dict[str, str | None] = {DEFAULT_EGRESS_NAME: config.proxy_url}
        for name in sorted(config.egresses):
            try:
                validate_configured_proxy_url(config.egresses[name])
            except ValueError as exc:
                msg = f"egress {name!r} is unusable: {exc}"
                raise ValueError(msg) from exc
            urls[name] = config.egresses[name]
        return urls

    async def handle(self, payload: Any, *, request_id: str | None = None) -> tuple[int, dict[str, Any]]:
        """Execute a command with task-local correlation, preserving response and cancellation semantics."""
        request_id = request_id or uuid4().hex
        start = now_ms()
        with logger.contextualize(request_id=request_id):
            try:
                return await self._execute_command(start, payload)
            finally:
                logger.debug("Request {} finished in {}ms", request_id, now_ms() - start)

    async def _execute_command(self, start: int, payload: Any) -> tuple[int, dict[str, Any]]:
        """Validate and run *payload*, mapping each failure onto its error body."""
        try:
            command = parse_request(payload)
        except ProtocolError as exc:
            return exc.http_status, error_response(start, exc.message)

        try:
            body = await self._dispatch(start, command)
        except CallerSafeError as exc:
            body = error_response(start, str(exc))
        except TimeoutError:
            body = error_response(start, _TIMEOUT_MESSAGE)
        except Exception:  # noqa: BLE001
            logger.exception("command failed")
            body = error_response(start, _INTERNAL_MESSAGE)
        return 200, body

    async def _dispatch(self, start: int, command: Command) -> dict[str, Any]:
        """Run *command* and return its response body."""
        if isinstance(command, FetchCommand):
            return await self._handle_fetch(start, command)
        if isinstance(command, BrowserOpenCommand):
            return await self._handle_browser_open(start, command)
        if isinstance(command, BrowserCloseCommand):
            return await self._handle_browser_close(start, command)
        if isinstance(command, SessionsCreateCommand):
            return await self._handle_sessions_create(start, command)
        if isinstance(command, SessionsDestroyCommand):
            return await self._handle_sessions_destroy(start, command)
        if isinstance(command, SessionsListCommand):
            return await self._handle_sessions_list(start)
        if isinstance(command, CookiesListCommand):
            return await self._handle_cookies_list(start, command)
        return await self._handle_browser_list(start, command)

    async def fetch(
        self,
        command: FetchCommand,
        *,
        body_bytes: bytes | None = None,
        cookie_header: str | None = None,
    ) -> FetchResult:
        """Run *command*'s fetch under its session's lease and the global concurrency bound.

        A raw ``Cookie`` header, when the caller supplies one, travels beside the wire headers so
        the backend can resolve it against the native context instead of forwarding it as a header.
        """
        session = command.session
        requested_egress = self._resolve_egress(command.proxy)
        # An omitted proxy is left as None so the lease can reuse a bound session's egress.
        lease_egress = requested_egress if command.proxy is not None else None
        if session is not None:
            await self._bind_session(session, command.session_ttl_minutes, command.session_mode, lease_egress)

        async with self._serialize(session, requested_egress, command.session_mode, lease_egress) as info:
            egress = (info.egress or requested_egress) if info is not None else requested_egress
            request = FetchRequest(
                url=command.url,
                method=command.method,
                timeout_seconds=command.timeout_seconds,
                headers=command.headers,
                header_scope=command.header_scope,
                cookies=command.cookies,
                post_data=command.post_data or "",
                egress=egress,
                session_id=session,
                session_mode=info.mode if info is not None else SHARED_MODE,
                mode=command.mode,
                body_bytes=body_bytes,
                cookie_header=cookie_header,
                wait_in_seconds=0.0 if command.return_only_cookies else command.wait_in_seconds,
                return_screenshot=command.return_screenshot,
                disable_media=command.disable_media,
                tabs_till_verify=command.tabs_till_verify,
                solve_captcha=command.solve_captcha,
            )
            return await asyncio.wait_for(
                self.backend.fetch(session, request),
                timeout=command.timeout_seconds + _TIMEOUT_SLACK_SECONDS,
            )

    async def _handle_fetch(self, start: int, command: FetchCommand) -> dict[str, Any]:
        result = await self.fetch(command)
        solution = solution_payload(
            url=result.url,
            status_code=result.status_code or 0,
            headers=result.headers,
            response="" if command.return_only_cookies else result.response,
            cookies=result.cookies,
            user_agent=result.user_agent or "",
            execution=execution_payload(result.mode, result.classification),
            screenshot=result.screenshot if command.return_screenshot else None,
            turnstile_token=result.turnstile_token if command.tabs_till_verify is not None else None,
            captcha_provider=result.captcha_provider if command.solve_captcha else None,
            captcha_token=result.captcha_token if command.solve_captcha else None,
        )
        return ok_response(start, solution=solution)

    @asynccontextmanager
    async def _serialize(
        self,
        session: str | None,
        egress: str,
        mode: str | None,
        lease_egress: str | None,
    ) -> AsyncIterator[SessionInfo | None]:
        """Hold the request's serialization guard and the global concurrency bound.

        A named session serializes on its own lease, which validates the command's explicit mode
        and egress selector against the generation it actually admits and yields that
        generation's metadata. Anonymous requests serialize on their egress's lock so they never
        overlap within one browser process, even when ``max_concurrency`` is > 1, and yield
        ``None``. Two anonymous requests on different egresses stay independent, because they use
        different profiles.
        """
        if session is None:
            async with self._anonymous_locks[egress], self._semaphore:
                yield None
        else:
            async with self.sessions.lease(session, mode=mode, egress=lease_egress) as info, self._semaphore:
                yield info

    def _resolve_egress(self, selection: ProxySelection | None) -> str:
        """Return the egress that *selection* asks for, defaulting to the process one.

        A url is accepted only when it matches a configured egress, which keeps a
        caller from turning the service into a relay for an arbitrary proxy. A
        name must be one of the configured egresses.

        :raises ProxyError: when the request asks for something not configured.
        """
        if selection is None:
            return DEFAULT_EGRESS_NAME
        if selection.name is not None:
            if selection.name not in self.egresses:
                raise ProxyError(_UNKNOWN_EGRESS_MESSAGE)
            return selection.name
        for name, url in self.egresses.items():
            if url is not None and url == selection.url:
                return name
        raise ProxyError(_PROXY_MESSAGE)

    async def _bind_session(
        self,
        session_id: str,
        ttl_minutes: int | None,
        mode: str | None,
        egress: str | None,
    ) -> None:
        """Create or refresh *session_id* with the command's explicit mode and egress selector.

        An omitted TTL is left unspecified so the configured TTL is preserved; only an explicit
        wire value changes it. Egress binding and the default fallback belong to the lease the
        request actually takes, so this only records what the command named.
        """
        if ttl_minutes is None:
            await self.sessions.ensure(session_id, mode=mode, egress=egress)
        else:
            await self.sessions.ensure(session_id, ttl_minutes, mode=mode, egress=egress)

    async def _handle_browser_open(self, start: int, command: BrowserOpenCommand) -> dict[str, Any]:
        """Open a tab and leave it open on the egress's browser.

        The tab is displayed on the shared X display, so a person watches and drives it there.
        It shares the egress's profile with ordinary fetches, which is what lets a clearance
        earned by clicking through a challenge apply to the app's own requests.

        A url that is already open on that egress is reused, and the reply says so, so a caller
        whose own request timed out cannot leave a tab nobody can close by asking again.
        """
        requested_egress = self._resolve_egress(command.proxy)
        lease_egress = requested_egress if command.proxy is not None else None
        if command.session is not None:
            await self._bind_session(command.session, None, command.session_mode, lease_egress)

        async with self._serialize(command.session, requested_egress, command.session_mode, lease_egress) as info:
            egress = (info.egress or requested_egress) if info is not None else requested_egress
            request = InteractiveRequest(
                url=command.url,
                timeout_seconds=command.timeout_seconds,
                cookies=command.cookies,
                new_tab=command.new_tab,
                egress=egress,
                session_id=command.session,
                session_mode=info.mode if info is not None else SHARED_MODE,
            )
            result = await asyncio.wait_for(
                self.backend.open_interactive(request),
                timeout=command.timeout_seconds + _TIMEOUT_SLACK_SECONDS,
            )

        body = ok_response(start, message="Tab reused" if result.reused else "Tab opened")
        body["tab"] = tab_payload(result.tab)
        body["reused"] = result.reused
        return body

    async def _handle_browser_close(self, start: int, command: BrowserCloseCommand) -> dict[str, Any]:
        """Close one interactive tab, or every one when no tab is named."""
        closed = await self.backend.close_interactive(command.tab)
        body = ok_response(start, message="Tab closed" if len(closed) == 1 else "Tabs closed")
        body["closed"] = closed
        return body

    async def _handle_browser_list(self, start: int, command: BrowserListCommand) -> dict[str, Any]:
        """Report the open interactive tabs.

        Naming the tab a caller is displaying refreshes only that tab's idle countdown, so
        polling one view cannot keep every other forgotten tab open.
        """
        tabs = await self.backend.list_interactive(command.tab)
        body = ok_response(start, message="ok")
        body["tabs"] = [tab_payload(tab) for tab in tabs]
        return body

    async def _handle_cookies_list(self, start: int, command: CookiesListCommand) -> dict[str, Any]:
        """Report the cookies held by an egress's browser profile.

        Reading them is what lets a caller carry a session a person established by browsing in
        the displayed browser into its own requests. The browser is only read, so an open tab is
        left exactly as it is.
        """
        requested_egress = self._resolve_egress(command.proxy)
        lease_egress = requested_egress if command.proxy is not None else None
        async with self._serialize(command.session, requested_egress, None, lease_egress) as info:
            egress = (info.egress or requested_egress) if info is not None else requested_egress
            query = CookieQuery(
                url=command.url,
                egress=egress,
                session_id=info.id if info is not None else None,
                session_mode=info.mode if info is not None else SHARED_MODE,
            )
            cookies = await asyncio.wait_for(
                self.backend.list_cookies(query),
                timeout=_COOKIE_READ_TIMEOUT_SECONDS,
            )
        body = ok_response(start, message="ok")
        body.update(cookies_payload(cookies))
        return body

    async def _handle_sessions_create(self, start: int, command: SessionsCreateCommand) -> dict[str, Any]:
        egress = self._resolve_egress(command.proxy) if command.proxy is not None else None
        if command.session_ttl_minutes is None:
            # An omitted wire TTL is left unspecified so a new session keeps the configured
            # default; only an explicit wire value overrides it.
            session_id = await self.sessions.create(command.session, mode=command.session_mode, egress=egress)
        else:
            session_id = await self.sessions.create(
                command.session,
                command.session_ttl_minutes,
                mode=command.session_mode,
                egress=egress,
            )
        body = ok_response(start, message="Session created")
        body["session"] = session_id
        return body

    async def _handle_sessions_destroy(self, start: int, command: SessionsDestroyCommand) -> dict[str, Any]:
        await self.sessions.destroy(command.session)
        return ok_response(start, message="Session destroyed")

    async def _handle_sessions_list(self, start: int) -> dict[str, Any]:
        body = ok_response(start, message="ok")
        body["sessions"] = await self.sessions.list_sessions()
        return body


# ---------------------------------------------------------------------------
# aiohttp application
# ---------------------------------------------------------------------------

_CONFIG_KEY: web.AppKey[ServiceConfig] = web.AppKey("config", ServiceConfig)
_BACKEND_KEY: web.AppKey[Backend] = web.AppKey("backend", Backend)
_SERVICE_KEY: web.AppKey[Service] = web.AppKey("service", Service)
#: The optional integrated forward proxy listener, registered only while one is configured.
_PROXY_KEY: web.AppKey[ProxyServer] = web.AppKey("proxy_server", ProxyServer)


class _AppState:
    """Mutable readiness and cleanup holder stored on the app before startup.

    aiohttp freezes the application once it starts, so readiness must live on
    a mutable object stored before startup rather than being written back into
    the application mapping.
    """

    __slots__ = ("cleanup_task", "ready")

    def __init__(self) -> None:
        self.ready = False
        #: The single owned cleanup task, shielded so a cancelled waiter cannot make the service
        #: close race an ongoing proxy drain.
        self.cleanup_task: asyncio.Task[None] | None = None


_STATE_KEY: web.AppKey[_AppState] = web.AppKey("state", _AppState)


async def _handle_command(request: web.Request) -> web.Response:
    service = request.app[_SERVICE_KEY]
    request_id = uuid4().hex
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response(
            error_response(now_ms(), "request body must be valid JSON"),
            status=400,
            headers={_REQUEST_ID_HEADER: request_id},
        )
    status, body = await service.handle(payload, request_id=request_id)
    return web.json_response(body, status=status, headers={_REQUEST_ID_HEADER: request_id})


async def _handle_healthz(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _handle_metrics(request: web.Request) -> web.Response:
    """Serve aggregate metrics without launching a browser or exposing identity state."""
    backend = request.app[_BACKEND_KEY]
    if isinstance(backend, BrowserBackend):
        return web.Response(
            text=backend.render_metrics(),
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
        )
    return web.Response(text="metrics are not available for this backend\n", status=501)


async def _handle_readyz(request: web.Request) -> web.Response:
    ready = request.app[_STATE_KEY].ready
    return web.json_response({"status": "ok" if ready else "starting"}, status=200 if ready else 503)


async def _on_startup(app: web.Application) -> None:
    """Validate the CA, start owned resources, then publish readiness."""
    config = app[_CONFIG_KEY]
    state = app[_STATE_KEY]
    if config.forward_proxy_port is None:
        await app[_BACKEND_KEY].start()
        app[_SERVICE_KEY].start_periodic_cleanup()
        state.ready = True
        return
    certificates = (
        await asyncio.to_thread(_load_proxy_certificates, config) if config.proxy_ca_cert is not None else None
    )
    proxy = ProxyServer(
        app[_SERVICE_KEY],
        host=config.host,
        port=config.forward_proxy_port,
        certificates=certificates,
    )
    app[_PROXY_KEY] = proxy
    try:
        await app[_BACKEND_KEY].start()
        await proxy.start()
    except BaseException:
        with contextlib.suppress(Exception):
            await _run_cleanup(app)
        raise
    app[_SERVICE_KEY].start_periodic_cleanup()
    state.ready = True


async def _on_cleanup(app: web.Application) -> None:
    """Mark not ready first, then drain the one owned cleanup sequence."""
    app[_STATE_KEY].ready = False
    await _run_cleanup(app)


def _load_proxy_certificates(config: ServiceConfig) -> ProxyCertificates | None:
    """Load the configured CA pair without changing machine trust."""
    if config.proxy_ca_cert is None or config.proxy_ca_key is None:
        return None
    return ProxyCertificates(Path(config.proxy_ca_cert), Path(config.proxy_ca_key))


async def _run_cleanup(app: web.Application) -> None:
    """Join the owned, shielded proxy-first shutdown sequence."""
    state = app[_STATE_KEY]
    task = state.cleanup_task
    if task is None:
        task = asyncio.create_task(_cleanup(app))
        task.add_done_callback(_report_shutdown_failure)
        state.cleanup_task = task
    await asyncio.shield(task)


async def _cleanup(app: web.Application) -> None:
    """Close the optional listener first, then the service, surfacing the first failure."""
    failure: BaseException | None = None
    proxy = app.get(_PROXY_KEY)
    if proxy is not None:
        try:
            await proxy.aclose()
        except Exception as exc:  # noqa: BLE001 - the service is closed either way
            failure = failure or exc
    try:
        await app[_SERVICE_KEY].aclose()
    except Exception as exc:  # noqa: BLE001
        failure = failure or exc
    if failure is not None:
        raise failure


def create_app(config: ServiceConfig, backend: Backend) -> web.Application:
    """Build the aiohttp application for *config* and *backend*."""
    app = web.Application()
    app[_CONFIG_KEY] = config
    app[_BACKEND_KEY] = backend
    app[_SERVICE_KEY] = Service(config, backend)
    app[_STATE_KEY] = _AppState()
    app.router.add_post("/v1", _handle_command)
    app.router.add_post("/", _handle_command)
    app.router.add_get("/healthz", _handle_healthz)
    app.router.add_get("/readyz", _handle_readyz)
    app.router.add_get("/metrics", _handle_metrics)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def run_server(config: ServiceConfig | None = None, backend: Backend | None = None) -> None:
    """Run the HTTP service until terminated."""
    resolved = config or ServiceConfig.from_env()
    browser_backend = backend or BrowserBackend(
        BrowserConfig(
            proxy_url=resolved.proxy_url,
            profile_dir=resolved.profile_dir,
            profile_archive=resolved.profile_archive,
            extensions_dir=resolved.extensions_dir,
            policy_dir=resolved.policy_dir,
            max_contexts=resolved.max_contexts,
        ),
        egresses=resolved.egresses,
        egress_idle_seconds=resolved.egress_idle_seconds,
        interactive_idle_seconds=resolved.interactive_idle_seconds,
        steal_least_recent=resolved.steal_least_recent,
        max_concurrency=resolved.max_concurrency,
        state_store=(
            ContextStateStore(Path(resolved.session_state_dir)) if resolved.session_state_dir is not None else None
        ),
    )
    app = create_app(resolved, browser_backend)
    logger.info(f"prowl listening on {resolved.host}:{resolved.port}")
    web.run_app(app, host=resolved.host, port=resolved.port, access_log=None, print=None)


__all__ = [
    "CMD_BROWSER_CLOSE",
    "CMD_BROWSER_LIST",
    "CMD_BROWSER_OPEN",
    "CMD_COOKIES_LIST",
    "CMD_REQUEST_GET",
    "CMD_REQUEST_POST",
    "CMD_SESSIONS_CREATE",
    "CMD_SESSIONS_DESTROY",
    "CMD_SESSIONS_LIST",
    "Browser",
    "FetchResult",
    "InteractiveRequest",
    "Service",
    "ServiceConfig",
    "create_app",
    "run_server",
]
