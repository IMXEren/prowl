"""The reserved-tab CDP tunnel only admits its trusted caller and releases its lease."""

from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase, TestCase

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from prowl.service.cdp import register_cdp

_TOKEN = "a-token-long-enough-to-protect-the-debugging-port"  # noqa: S105


class FakeReservations:
    """Record one-shot reservation attachment and close."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.attached = False
        self.closed = asyncio.Event()
        self.egress: str | None = None

    async def reserve_tab(self, egress: str) -> tuple[str, str]:
        self.egress = egress
        return "tab-1", "target-1"

    async def attach_reserved_tab(self, tab: str) -> int | None:
        if tab != "tab-1" or self.attached:
            return None
        self.attached = True
        return self.port

    async def release_reserved_tab(self, tab: str) -> list[str]:
        if tab == "tab-1":
            self.closed.set()
            return [tab]
        return []


class CdpGatewayTests(IsolatedAsyncioTestCase):
    """Exercise HTTP auth, two-way WebSocket relaying and disconnect cleanup."""

    async def asyncSetUp(self) -> None:
        async def version(_request: web.Request) -> web.Response:
            return web.json_response(
                {"webSocketDebuggerUrl": f"ws://127.0.0.1:{self.browser.port}/devtools/browser/mock"},
            )

        async def browser_socket(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await ws.send_str(message.data)
            return ws

        app = web.Application()
        app.router.add_get("/json/version", version)
        app.router.add_get("/devtools/browser/mock", browser_socket)
        self.browser = TestServer(app)
        await self.browser.start_server()
        self.backend = FakeReservations(self.browser.port)
        gateway = web.Application()
        register_cdp(gateway, self.backend, _TOKEN)  # type: ignore[arg-type]
        self.client = TestClient(TestServer(gateway))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.browser.close()

    async def test_unauthorized_request_cannot_reserve_or_attach(self) -> None:
        response = await self.client.post("/cdp/reservations", json={})
        self.assertEqual(response.status, 401)
        response = await self.client.get("/cdp/reservations/tab-1/socket")
        self.assertEqual(response.status, 401)
        self.assertIsNone(self.backend.egress)
        self.assertFalse(self.backend.attached)

    async def test_reserved_tab_can_be_released_before_attach(self) -> None:
        headers = {"Authorization": f"Bearer {_TOKEN}"}
        await self.client.post("/cdp/reservations", json={}, headers=headers)
        unauthorized = await self.client.delete("/cdp/reservations/tab-1")
        self.assertEqual(unauthorized.status, 401)
        self.assertFalse(self.backend.closed.is_set())
        response = await self.client.delete("/cdp/reservations/tab-1", headers=headers)
        self.assertEqual(await response.json(), {"closed": ["tab-1"]})

    async def test_reserved_tab_is_relayed_once_and_released_on_disconnect(self) -> None:
        headers = {"Authorization": f"Bearer {_TOKEN}"}
        response = await self.client.post("/cdp/reservations", json={"egress": "isp"}, headers=headers)
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.json(), {"tab": "tab-1", "target": "target-1"})
        self.assertEqual(self.backend.egress, "isp")

        ws = await self.client.ws_connect("/cdp/reservations/tab-1/socket", headers=headers)
        response = await self.client.get("/cdp/reservations/tab-1/socket", headers=headers)
        self.assertEqual(response.status, 404)
        await ws.send_str('{"id":1,"method":"Runtime.enable"}')
        reply = await ws.receive(timeout=2)
        self.assertEqual(reply.data, '{"id":1,"method":"Runtime.enable"}')
        await ws.close()
        await asyncio.wait_for(self.backend.closed.wait(), timeout=2)


class CdpTokenTests(TestCase):
    """A short token must not expose a full browser debugging channel."""

    def test_short_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32"):
            register_cdp(web.Application(), object(), "short")  # type: ignore[arg-type]
