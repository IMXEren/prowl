"""Authenticated CDP tunnel for a reserved browser tab.

The browser debugging port stays on loopback. Only callers holding the configured
service token can attach, and a reservation is released when its tunnel closes.
"""

from __future__ import annotations

import asyncio
import hmac
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web

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

    def register(self, app: web.Application) -> None:
        """Mount the routes after a valid token is supplied."""
        app.router.add_post("/cdp/reservations", self.create)
        app.router.add_get("/cdp/reservations/{tab}/socket", self.cdp)
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
        closed = await self.backend.release_reserved_tab(request.match_info["tab"])
        return web.json_response({"closed": closed})

    async def cdp(self, request: web.Request) -> web.WebSocketResponse:
        """Tunnel an attached reservation until either peer disconnects."""
        self.authorize(request)
        tab = request.match_info["tab"]
        port = await self.backend.attach_reserved_tab(tab)
        if port is None:
            raise web.HTTPNotFound
        try:
            async with ClientSession() as session:
                upstream = await self._connect_upstream(session, port)
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
            await self.backend.release_reserved_tab(tab)

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
    ) -> None:
        async for message in source:
            if message.type == WSMsgType.TEXT:
                await destination.send_str(message.data)
            elif message.type == WSMsgType.BINARY:
                await destination.send_bytes(message.data)
            else:
                break

    async def _relay(self, downstream: web.WebSocketResponse, upstream: ClientWebSocketResponse) -> None:
        tasks = [
            asyncio.create_task(self._pipe(downstream, upstream)),
            asyncio.create_task(self._pipe(upstream, downstream)),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def register_cdp(app: web.Application, backend: BrowserBackend, token: str) -> None:
    """Enable the privileged CDP gateway when a token is configured."""
    CdpGateway(backend, token).register(app)
