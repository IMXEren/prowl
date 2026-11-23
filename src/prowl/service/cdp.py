"""Authenticated CDP tunnel for a reserved browser tab.

The browser debugging port stays on loopback. Only callers holding the configured
service token can attach, and a reservation is released when its tunnel closes.
"""

from __future__ import annotations

import asyncio
import hmac
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, WSServerHandshakeError, web
from loguru import logger

from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME

if TYPE_CHECKING:
    from prowl.service.backend import BrowserBackend

_MAX_CDP_MESSAGE = 16 * 1024 * 1024
_MIN_TOKEN_LENGTH = 32


class CdpGateway:
    """Own the authenticated reservation and tunnel routes."""

    def __init__(self, backend: BrowserBackend, token: str) -> None:
        if len(token) < _MIN_TOKEN_LENGTH:
            msg = "the CDP token must be at least 32 characters"
            raise ValueError(msg)
        self.backend = backend
        self.token = token
        self._active: dict[str, int] = {}
        self._browser_tasks: dict[str, asyncio.Task] = {}
        self._page_tasks: dict[str, set[asyncio.Task]] = {}
        self._active_lock = asyncio.Lock()

    def register(self, app: web.Application) -> None:
        """Mount the routes after a valid token is supplied."""
        app.router.add_post("/cdp/reservations", self.create)
        app.router.add_get("/cdp/reservations/{tab}/socket", self.cdp)
        app.router.add_get("/cdp/reservations/{tab}/json/version", self.version)
        app.router.add_get("/cdp/reservations/{tab}/json/list", self.targets)
        app.router.add_get("/devtools/page/{target}", self.page)
        app.router.add_delete("/cdp/reservations/{tab}", self.release)

    def authorize(self, request: web.Request) -> None:
        """Reject requests before allocating a tab or upgrading the socket."""
        if not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {self.token}"):
            raise web.HTTPUnauthorized

    async def create(self, request: web.Request) -> web.Response:
        """Reserve a blank page under the selected egress."""
        self.authorize(request)
        payload = await request.json()
        egress = payload.get("egress", DEFAULT_EGRESS_NAME) if isinstance(payload, dict) else None
        if not isinstance(egress, str):
            raise web.HTTPBadRequest
        tab, target = await self.backend.reserve_tab(egress)
        return web.json_response({"tab": tab, "target": target})

    async def release(self, request: web.Request) -> web.Response:
        """Release a reservation even when no client attached."""
        self.authorize(request)
        tab = request.match_info["tab"]
        async with self._active_lock:
            closed = await self.backend.release_reserved_tab(tab)
            if closed and (task := self._browser_tasks.get(tab)) is not None:
                task.cancel()
        return web.json_response({"closed": closed})

    async def cdp(self, request: web.Request) -> web.WebSocketResponse:
        """Tunnel an attached reservation until either peer disconnects."""
        self.authorize(request)
        tab = request.match_info["tab"]
        async with self._active_lock:
            port = await self.backend.attach_reserved_tab(tab)
            if port is None:
                raise web.HTTPNotFound
            self._active[tab] = 1
            self._browser_tasks[tab] = asyncio.current_task()
        reason = "connection not established"
        downstream: web.WebSocketResponse | None = None
        try:
            async with ClientSession() as session:
                upstream = await self._connect_upstream(session, port)
                try:
                    downstream = web.WebSocketResponse(max_msg_size=_MAX_CDP_MESSAGE, heartbeat=30)
                    await downstream.prepare(request)
                    try:
                        reason = await self._relay(downstream, upstream)
                    finally:
                        await downstream.close()
                    return downstream
                finally:
                    await upstream.close()
        except asyncio.CancelledError:
            if reason == "connection not established":
                reason = "cancelled"
                if downstream is not None and downstream.closed:
                    reason = f"caller disconnected (code={downstream.close_code})"
            raise
        except Exception as exc:
            reason = f"error ({type(exc).__name__})"
            raise
        finally:
            logger.info("Reserved CDP tunnel closed: {}", reason)
            async with self._active_lock:
                self._browser_tasks.pop(tab, None)
                pages = tuple(self._page_tasks.get(tab, ()))
            for page in pages:
                page.cancel()
            await asyncio.gather(*pages, return_exceptions=True)
            await self._release_socket(tab)

    async def _release_socket(self, tab: str) -> None:
        async with self._active_lock:
            remaining = self._active[tab] - 1
            if remaining:
                self._active[tab] = remaining
                return
            del self._active[tab]
        await self.backend.release_reserved_tab(tab)

    async def _reservation(self, tab: str) -> tuple[int, str]:
        info = await self.backend.reserved_cdp_target(tab)
        if info is None:
            raise web.HTTPNotFound
        return info

    @staticmethod
    def _socket_url(request: web.Request, path: str, tab: str) -> str:
        scheme = "wss" if request.secure else "ws"
        return str(request.url.with_scheme(scheme).with_path(path).with_query(reservation=tab))

    async def version(self, request: web.Request) -> web.Response:
        """Publish the reserved browser's authenticated discovery URL."""
        self.authorize(request)
        tab = request.match_info["tab"]
        port, _ = await self._reservation(tab)
        async with ClientSession() as session, session.get(f"http://127.0.0.1:{port}/json/version") as response:
            response.raise_for_status()
            version = await response.json()
        version["webSocketDebuggerUrl"] = self._socket_url(request, f"/cdp/reservations/{tab}/socket", tab)
        return web.json_response(version)

    async def targets(self, request: web.Request) -> web.Response:
        """Publish page URLs for the reservation's browser."""
        self.authorize(request)
        tab = request.match_info["tab"]
        port, _ = await self._reservation(tab)
        async with ClientSession() as session, session.get(f"http://127.0.0.1:{port}/json/list") as response:
            response.raise_for_status()
            targets = await response.json()
        for target in targets:
            if "webSocketDebuggerUrl" in target:
                target["webSocketDebuggerUrl"] = self._socket_url(request, f"/devtools/page/{target['id']}", tab)
        return web.json_response(targets)

    async def page(self, request: web.Request) -> web.WebSocketResponse:
        """Tunnel a page socket through the reservation's owning browser."""
        self.authorize(request)
        tab = request.query.get("reservation", "")
        port, _ = await self._reservation(tab)
        task = asyncio.current_task()
        async with self._active_lock:
            if tab not in self._active:
                raise web.HTTPNotFound
            self._active[tab] += 1
            self._page_tasks.setdefault(tab, set()).add(task)
        try:
            async with ClientSession() as session:
                try:
                    upstream = await session.ws_connect(
                        f"ws://127.0.0.1:{port}/devtools/page/{request.match_info['target']}",
                        max_msg_size=_MAX_CDP_MESSAGE,
                        heartbeat=30,
                    )
                except WSServerHandshakeError as exc:
                    if exc.status == web.HTTPNotFound.status_code:
                        raise web.HTTPNotFound from exc
                    raise
                try:
                    downstream = web.WebSocketResponse(max_msg_size=_MAX_CDP_MESSAGE, heartbeat=30)
                    await downstream.prepare(request)
                    try:
                        await self._relay(downstream, upstream)
                    finally:
                        await downstream.close()
                    return downstream
                finally:
                    await upstream.close()
        finally:
            async with self._active_lock:
                self._page_tasks[tab].discard(task)
                if not self._page_tasks[tab]:
                    del self._page_tasks[tab]
            await self._release_socket(tab)

    @staticmethod
    async def _connect_upstream(session: ClientSession, port: int) -> ClientWebSocketResponse:
        async with session.get(f"http://127.0.0.1:{port}/json/version") as response:
            response.raise_for_status()
            version = await response.json()
        path = urlsplit(version["webSocketDebuggerUrl"]).path
        if not path.startswith("/devtools/browser/"):
            msg = "CDP endpoint is not a browser target"
            raise ValueError(msg)
        return await session.ws_connect(
            f"ws://127.0.0.1:{port}{path}",
            max_msg_size=_MAX_CDP_MESSAGE,
            heartbeat=30,
        )

    @staticmethod
    async def _pipe(
        source: web.WebSocketResponse | ClientWebSocketResponse,
        destination: web.WebSocketResponse | ClientWebSocketResponse,
    ) -> str:
        async for message in source:
            if message.type == WSMsgType.TEXT:
                await destination.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await destination.send_bytes(message.data)
            else:
                break
        reason = f"closed (code={source.close_code})"
        if error := source.exception():
            reason += f" ({type(error).__name__})"
        return reason

    async def _relay(self, downstream: web.WebSocketResponse, upstream: ClientWebSocketResponse) -> str:
        tasks = {
            asyncio.create_task(self._pipe(downstream, upstream)): "caller",
            asyncio.create_task(self._pipe(upstream, downstream)): "browser",
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            task = next(iter(done))
            return f"{tasks[task]} {task.result()}"
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def register_cdp(app: web.Application, backend: BrowserBackend, token: str) -> None:
    """Enable the privileged CDP gateway when a token is configured."""
    CdpGateway(backend, token).register(app)
