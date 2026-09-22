"""Long-lived asyncio HTTP service speaking the FlareSolverr v1 contract.

Requests are dispatched to :class:`~prowl.service.backend.BrowserBackend`,
which drives the shared browser process. Concurrency is bounded by a
semaphore; each logical session serializes its own work and anonymous
requests serialize on one shared lock.
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
from prowl.browser.config import default_profile_archive, default_profile_dir
from prowl.service.backend import Backend, BrowserBackend, FetchRequest, FetchResult
from prowl.service.errors import CallerSafeError, ProxyError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
from prowl.service.protocol import (
    CMD_REQUEST_GET,
    CMD_REQUEST_POST,
    CMD_SESSIONS_CREATE,
    CMD_SESSIONS_DESTROY,
    CMD_SESSIONS_LIST,
    FetchCommand,
    ProtocolError,
    SessionsCreateCommand,
    SessionsDestroyCommand,
    error_response,
    now_ms,
    ok_response,
    parse_request,
    solution_payload,
    validate_proxy_url,
)
from prowl.service.sessions import SessionRegistry

#: Extra wall-clock slack granted to the browser beyond the caller timeout so
#: the core can run its own cleanup before the service gives up.
_TIMEOUT_SLACK_SECONDS = 10.0

#: Caller-safe messages used when a failure has no declared safe string.
_TIMEOUT_MESSAGE = "request timed out"
_INTERNAL_MESSAGE = "internal error while executing the command"
_PROXY_MESSAGE = "request proxy is not permitted; egress is fixed by the service configuration"

_DEFAULT_MAX_SESSIONS = 32


@dataclass(slots=True)
class ServiceConfig:
    """Runtime configuration for the HTTP service."""

    host: str = "0.0.0.0"  # noqa: S104 - intentional container bind
    port: int = 8191
    max_concurrency: int = 1
    max_sessions: int = _DEFAULT_MAX_SESSIONS
    proxy_url: str | None = None
    profile_dir: str = field(default_factory=default_profile_dir)
    profile_archive: str = field(default_factory=default_profile_archive)

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Build configuration from environment variables."""
        proxy_url = os.environ.get("PROWL_PROXY_URL", "").strip() or None
        if proxy_url is not None:
            validate_proxy_url(proxy_url)
        return cls(
            host=os.environ.get("PROWL_HOST", "0.0.0.0"),  # noqa: S104
            port=int(os.environ.get("PROWL_PORT", "8191")),
            max_concurrency=max(1, int(os.environ.get("PROWL_MAX_CONCURRENCY", "1"))),
            max_sessions=max(1, int(os.environ.get("PROWL_MAX_SESSIONS", str(_DEFAULT_MAX_SESSIONS)))),
            proxy_url=proxy_url,
            profile_dir=default_profile_dir(),
            profile_archive=default_profile_archive(),
        )


class Service:
    """Command dispatcher shared by the HTTP handlers and the tests."""

    def __init__(self, config: ServiceConfig, backend: Backend) -> None:
        self.config = config
        self.backend = backend
        if config.proxy_url is not None:
            validate_proxy_url(config.proxy_url)
        self.sessions = SessionRegistry(max_sessions=config.max_sessions)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        # Anonymous (session-less) requests share the one persistent trust profile,
        # so they must never overlap: this lock serializes them regardless of how
        # high max_concurrency is.
        self._anonymous_lock = asyncio.Lock()

    async def handle(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """Validate and execute *payload*, returning ``(http_status, body)``."""
        start = now_ms()
        try:
            command = parse_request(payload)
        except ProtocolError as exc:
            return exc.http_status, error_response(start, exc.message)

        try:
            if isinstance(command, FetchCommand):
                body = await self._handle_fetch(start, command)
            elif isinstance(command, SessionsCreateCommand):
                body = await self._handle_sessions_create(start, command)
            elif isinstance(command, SessionsDestroyCommand):
                body = await self._handle_sessions_destroy(start, command)
            else:
                body = await self._handle_sessions_list(start)
        except CallerSafeError as exc:
            body = error_response(start, str(exc))
        except TimeoutError:
            body = error_response(start, _TIMEOUT_MESSAGE)
        except Exception:  # noqa: BLE001
            logger.exception("command failed")
            body = error_response(start, _INTERNAL_MESSAGE)
        return 200, body

    async def _handle_fetch(self, start: int, command: FetchCommand) -> dict[str, Any]:
        self._resolve_proxy(command.proxy_url)
        session = command.session
        if session is not None:
            await self.sessions.ensure(session, command.session_ttl_minutes)

        request = FetchRequest(
            url=command.url,
            method=command.method,
            timeout_seconds=command.timeout_seconds,
            headers=command.headers,
            cookies=command.cookies,
            post_data=command.post_data or "",
        )

        async with self._serialize(session):
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
    async def _serialize(self, session: str | None) -> AsyncIterator[None]:
        """Hold the fetch's serialization guard and the global concurrency bound.

        Named sessions serialize on their own lease; anonymous requests share one
        dedicated lock so they never overlap even when ``max_concurrency`` is > 1.
        """
        if session is None:
            async with self._anonymous_lock, self._semaphore:
                yield
        else:
            async with self.sessions.lease(session), self._semaphore:
                yield

    def _resolve_proxy(self, requested: str | None) -> None:
        """Accept a request proxy only when it matches the process-wide egress."""
        if requested is None:
            return
        configured = self.config.proxy_url
        if configured is None or requested != configured:
            raise ProxyError(_PROXY_MESSAGE)

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
        ),
    )
    app = create_app(resolved, browser_backend)
    logger.info(f"prowl listening on {resolved.host}:{resolved.port}")
    web.run_app(app, host=resolved.host, port=resolved.port, access_log=None, print=None)


__all__ = [
    "CMD_REQUEST_GET",
    "CMD_REQUEST_POST",
    "CMD_SESSIONS_CREATE",
    "CMD_SESSIONS_DESTROY",
    "CMD_SESSIONS_LIST",
    "Browser",
    "FetchResult",
    "Service",
    "ServiceConfig",
    "create_app",
    "run_server",
]
