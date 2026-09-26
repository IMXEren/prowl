"""Long-lived asyncio HTTP service speaking the FlareSolverr v1 contract.

Requests are dispatched to :class:`~prowl.service.backend.BrowserBackend`,
which drives the browser of the request's egress. Concurrency is bounded by a
semaphore; each logical session serializes its own work and anonymous requests
serialize on the lock of their egress, because an egress owns one persistent
profile.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aiohttp import web
from loguru import logger

from prowl.browser import Browser, BrowserConfig
from prowl.browser.config import (
    PROXY_URL_ENV,
    default_extensions_dir,
    default_policy_dir,
    default_profile_archive,
    default_profile_dir,
)
from prowl.browser.egress import (
    DEFAULT_EGRESS_IDLE_SECONDS,
    DEFAULT_EGRESS_NAME,
    EGRESS_IDLE_SECONDS_ENV,
    EGRESSES_ENV,
    EgressError,
    parse_egress_spec,
)
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
    now_ms,
    ok_response,
    parse_request,
    solution_payload,
    tab_payload,
    validate_proxy_url,
)
from prowl.service.sessions import SessionRegistry

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


@dataclass(slots=True)
class ServiceConfig:
    """Runtime configuration for the HTTP service."""

    host: str = "0.0.0.0"  # noqa: S104 - intentional container bind
    port: int = 8191
    max_concurrency: int = 1
    max_sessions: int = _DEFAULT_MAX_SESSIONS
    proxy_url: str | None = None
    egresses: dict[str, str] = field(default_factory=dict)
    egress_idle_seconds: float = DEFAULT_EGRESS_IDLE_SECONDS
    interactive_idle_seconds: float = DEFAULT_INTERACTIVE_IDLE_SECONDS
    steal_least_recent: bool = DEFAULT_STEAL_LEAST_RECENT
    profile_dir: str = field(default_factory=default_profile_dir)
    profile_archive: str = field(default_factory=default_profile_archive)
    extensions_dir: str | None = field(default_factory=default_extensions_dir)
    policy_dir: str | None = field(default_factory=default_policy_dir)

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Build configuration from environment variables.

        :raises EgressError: when the egress list or one of its URLs is unusable.
        :raises ValueError: when the process-wide proxy URL is unusable.
        """
        proxy_url = os.environ.get(PROXY_URL_ENV, "").strip() or None
        if proxy_url is not None:
            validate_proxy_url(proxy_url)
        egresses = parse_egress_spec(os.environ.get(EGRESSES_ENV, ""))
        for name in sorted(egresses):
            try:
                validate_proxy_url(egresses[name])
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
        )


class Service:
    """Command dispatcher shared by the HTTP handlers and the tests."""

    def __init__(self, config: ServiceConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        if config.proxy_url is not None:
            validate_proxy_url(config.proxy_url)
        self.egresses = self._configured_egress_urls(config)
        self.sessions = SessionRegistry(max_sessions=config.max_sessions)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        # Anonymous (session-less) requests share the persistent trust profile of
        # their egress, so they must never overlap within one egress: one lock per
        # egress, not one lock for the process.
        self._anonymous_locks = {name: asyncio.Lock() for name in self.egresses}

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
                validate_proxy_url(config.egresses[name])
            except ValueError as exc:
                msg = f"egress {name!r} is unusable: {exc}"
                raise ValueError(msg) from exc
            urls[name] = config.egresses[name]
        return urls

    async def handle(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """Validate and execute *payload*, returning ``(http_status, body)``."""
        start = now_ms()
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

    async def _handle_fetch(self, start: int, command: FetchCommand) -> dict[str, Any]:
        egress = self._resolve_egress(command.proxy)
        session = command.session
        if session is not None:
            await self.sessions.ensure(session, command.session_ttl_minutes, egress=egress)

        request = FetchRequest(
            url=command.url,
            method=command.method,
            timeout_seconds=command.timeout_seconds,
            headers=command.headers,
            header_scope=command.header_scope,
            cookies=command.cookies,
            post_data=command.post_data or "",
            egress=egress,
        )

        async with self._serialize(session, egress):
            result = await asyncio.wait_for(
                self.backend.fetch(session, request),
                timeout=command.timeout_seconds + _TIMEOUT_SLACK_SECONDS,
            )

        solution = solution_payload(
            url=result.url,
            status_code=result.status_code or 0,
            headers=result.headers,
            response="" if command.return_only_cookies else result.response,
            cookies=result.cookies,
            user_agent=result.user_agent or "",
        )
        return ok_response(start, solution=solution)

    @asynccontextmanager
    async def _serialize(self, session: str | None, egress: str) -> AsyncIterator[None]:
        """Hold the fetch's serialization guard and the global concurrency bound.

        Named sessions serialize on their own lease; anonymous requests serialize
        on their egress's lock so they never overlap within one browser process,
        even when ``max_concurrency`` is > 1. Two anonymous requests on different
        egresses stay independent, because they use different profiles.
        """
        if session is None:
            async with self._anonymous_locks[egress], self._semaphore:
                yield
        else:
            async with self.sessions.lease(session), self._semaphore:
                yield

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

    async def _handle_browser_open(self, start: int, command: BrowserOpenCommand) -> dict[str, Any]:
        """Open a tab and leave it open on the egress's browser.

        The tab is displayed on the shared X display, so a person watches and drives it there.
        It shares the egress's profile with ordinary fetches, which is what lets a clearance
        earned by clicking through a challenge apply to the app's own requests.

        A url that is already open on that egress is reused, and the reply says so, so a caller
        whose own request timed out cannot leave a tab nobody can close by asking again.
        """
        egress = self._resolve_egress(command.proxy)
        if command.session is not None:
            await self.sessions.ensure(command.session, None, egress=egress)

        request = InteractiveRequest(
            url=command.url,
            timeout_seconds=command.timeout_seconds,
            cookies=command.cookies,
            new_tab=command.new_tab,
            egress=egress,
        )
        async with self._serialize(command.session, egress):
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
        egress = self._resolve_egress(command.proxy)
        query = CookieQuery(url=command.url, egress=egress)
        async with self._serialize(None, egress):
            cookies = await asyncio.wait_for(
                self.backend.list_cookies(query),
                timeout=_COOKIE_READ_TIMEOUT_SECONDS,
            )
        body = ok_response(start, message="ok")
        body.update(cookies_payload(cookies))
        return body

    async def _handle_sessions_create(self, start: int, command: SessionsCreateCommand) -> dict[str, Any]:
        session_id = await self.sessions.create(command.session, command.session_ttl_minutes)
        body = ok_response(start, message="Session created")
        body["session"] = session_id
        return body

    async def _handle_sessions_destroy(self, start: int, command: SessionsDestroyCommand) -> dict[str, Any]:
        await self.sessions.destroy(command.session)
        await self.backend.close_session(command.session)
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


class _AppState:
    """Mutable readiness holder stored on the app before startup.

    aiohttp freezes the application once it starts, so readiness must live on
    a mutable object stored before startup rather than being written back into
    the application mapping.
    """

    __slots__ = ("ready",)

    def __init__(self) -> None:
        self.ready = False


_STATE_KEY: web.AppKey[_AppState] = web.AppKey("state", _AppState)


async def _handle_command(request: web.Request) -> web.Response:
    service = request.app[_SERVICE_KEY]
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return web.json_response(
            error_response(now_ms(), "request body must be valid JSON"),
            status=400,
        )
    status, body = await service.handle(payload)
    return web.json_response(body, status=status)


async def _handle_healthz(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _handle_readyz(request: web.Request) -> web.Response:
    ready = request.app[_STATE_KEY].ready
    return web.json_response({"status": "ok" if ready else "starting"}, status=200 if ready else 503)


async def _on_startup(app: web.Application) -> None:
    await app[_BACKEND_KEY].start()
    app[_STATE_KEY].ready = True


async def _on_cleanup(app: web.Application) -> None:
    app[_STATE_KEY].ready = False
    await app[_BACKEND_KEY].aclose()


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
        ),
        egresses=resolved.egresses,
        egress_idle_seconds=resolved.egress_idle_seconds,
        interactive_idle_seconds=resolved.interactive_idle_seconds,
        steal_least_recent=resolved.steal_least_recent,
        max_concurrency=resolved.max_concurrency,
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
