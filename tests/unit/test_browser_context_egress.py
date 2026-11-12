"""Context-scoped egress ownership and authenticated SOCKS5 bridge tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from prowl.browser.browser import Browser
from prowl.browser.driver.contexts import BrowserContextHandle
from prowl.browser.exceptions import BrowserContextError
from prowl.browser.proxy.bridge import BrowserProxyBridge, BrowserProxyBridgeError, configured_proxy_bridge
from prowl.browser.proxy.context import create_context_egress_browser
from prowl.browser.proxy.egress import EgressBoundary
from prowl.service.backend import BrowserBackend
from prowl.service.context_state import ContextStateStore

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext as PWBrowserCtx


class SocksBridgeTests(IsolatedAsyncioTestCase):
    async def test_credentials_are_kept_out_of_the_browser_proxy(self) -> None:
        bridge = configured_proxy_bridge("socks5://user:secret@127.0.0.1:1080")
        assert bridge is not None
        self.assertIsNone(bridge.listen_url)
        self.assertNotIn("secret", repr(bridge))
        await bridge.start()
        try:
            self.assertRegex(bridge.listen_url or "", r"^http://127\.0\.0\.1:\d+$")
        finally:
            await bridge.aclose()


class SocksValidationTests(TestCase):
    def test_invalid_auth_is_not_echoed(self) -> None:
        with self.assertRaises(BrowserProxyBridgeError) as error:
            BrowserProxyBridge("socks5://user:secret@127.0.0.1")
        self.assertNotIn("secret", str(error.exception))

    def test_unauthenticated_socks_also_uses_http_bridge(self) -> None:
        self.assertIsInstance(configured_proxy_bridge("socks5://127.0.0.1:1080"), BrowserProxyBridge)


class ContextOwnerTests(IsolatedAsyncioTestCase):
    async def test_shared_and_isolated_contexts_pin_proxy_and_close_in_order(self) -> None:
        contexts: dict[str, Any] = {}
        events: list[str] = []

        async def get_context(
            session_id: str, *, proxy_server: str | None = None, storage_state: object | None = None
        ) -> Any:
            self.assertIsNone(storage_state)
            if session_id in contexts:
                return contexts[session_id]
            self.assertEqual(proxy_server, "http://127.0.0.1:12345")
            handle = SimpleNamespace(session_id=session_id, context=SimpleNamespace(is_closed=lambda: False))
            contexts[session_id] = handle
            return handle

        async def close_context(session_id: str) -> None:
            events.append("browser")
            contexts.pop(session_id, None)

        async def close_http(_context: object) -> None:
            events.append("http")

        bridge = MagicMock(listen_url="http://127.0.0.1:12345")
        bridge.start = AsyncMock()
        bridge.aclose = AsyncMock(side_effect=lambda: events.append("bridge"))
        with (
            patch("prowl.browser.proxy.context.configured_proxy_bridge", return_value=bridge),
            patch.object(Browser, "_runtime", SimpleNamespace(contexts=SimpleNamespace(isolated=contexts.get))),
            patch.object(Browser, "start", new_callable=AsyncMock),
            patch.object(Browser, "is_running", return_value=True),
            patch.object(Browser, "get_context", new_callable=AsyncMock, side_effect=get_context),
            patch.object(Browser, "close_context", new_callable=AsyncMock, side_effect=close_context),
        ):
            owner = create_context_egress_browser(
                name="one",
                proxy_url="socks5://user:secret@127.0.0.1:1080",
                base_browser=Browser,
                close_http_context=close_http,
            )
            shared = await owner.get_context()
            isolated = await owner.get_context("session")
            self.assertNotEqual(shared.session_id, isolated.session_id)
            assert shared.session_id is not None
            self.assertTrue(shared.session_id.startswith("\x00egress:one:"))
            with self.assertRaises(BrowserContextError):
                await owner.get_context(proxy_server="socks5://wrong:1234")
            with self.assertRaises(BrowserContextError):
                await owner.create(
                    BrowserContextHandle("\x00egress:other:shared", cast("PWBrowserCtx", object()), object())
                )
            await owner.shutdown()
        self.assertEqual(events, ["http", "browser", "bridge"])

    async def test_http_close_failure_retains_native_context_and_bridge(self) -> None:
        context = object()
        shared = SimpleNamespace(session_id="\x00egress:one:shared", context=context)
        bridge = MagicMock(listen_url="http://127.0.0.1:12345")
        bridge.start = AsyncMock()
        bridge.aclose = AsyncMock()
        close_http = AsyncMock(side_effect=OSError("close failed"))
        close_context = AsyncMock()
        with (
            patch("prowl.browser.proxy.context.configured_proxy_bridge", return_value=bridge),
            patch.object(Browser, "_runtime", SimpleNamespace(contexts=SimpleNamespace(isolated=lambda _: shared))),
            patch.object(Browser, "start", new_callable=AsyncMock),
            patch.object(Browser, "is_running", return_value=True),
            patch.object(Browser, "get_context", new_callable=AsyncMock, return_value=shared),
            patch.object(Browser, "close_context", close_context),
        ):
            owner = create_context_egress_browser(
                name="one",
                proxy_url="socks5://user:secret@127.0.0.1:1080",
                base_browser=Browser,
                close_http_context=close_http,
            )
            await owner.get_context()
            with self.assertRaises(OSError):
                await owner.shutdown()
        close_context.assert_not_awaited()
        bridge.aclose.assert_not_awaited()

    async def test_shared_context_storage_restores_after_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContextStateStore(Path(directory))
            state = {
                "cookies": [{"name": "pref", "value": "saved", "domain": "example.test", "path": "/"}],
                "origins": [{"origin": "https://example.test", "localStorage": [{"name": "theme", "value": "dark"}]}],
            }
            contexts: dict[str, Any] = {}
            restored: list[object] = []
            native = MagicMock()
            native.is_closed.return_value = False
            native.storage_state = AsyncMock(return_value=state)
            bridge = MagicMock(listen_url="http://127.0.0.1:12345")
            bridge.start = AsyncMock()
            bridge.aclose = AsyncMock()

            async def get_context(
                session_id: str, *, proxy_server: str | None = None, storage_state: object | None = None
            ) -> Any:
                if session_id not in contexts:
                    self.assertEqual(proxy_server, bridge.listen_url)
                    restored.append(storage_state)
                    contexts[session_id] = SimpleNamespace(session_id=session_id, context=native)
                return contexts[session_id]

            async def close_context(session_id: str) -> None:
                contexts.pop(session_id)

            async def close_http(_context: object) -> None:
                return None

            with (
                patch("prowl.browser.proxy.context.configured_proxy_bridge", return_value=bridge),
                patch.object(Browser, "_runtime", SimpleNamespace(contexts=SimpleNamespace(isolated=contexts.get))),
                patch.object(Browser, "start", new_callable=AsyncMock),
                patch.object(Browser, "is_running", return_value=True),
                patch.object(Browser, "get_context", new_callable=AsyncMock, side_effect=get_context),
                patch.object(Browser, "close_context", new_callable=AsyncMock, side_effect=close_context),
                patch.object(store, "load", wraps=store.load) as load,
            ):
                options = {
                    "name": "one",
                    "proxy_url": "socks5://user:secret@127.0.0.1:1080",
                    "base_browser": Browser,
                    "close_http_context": close_http,
                    "state_store": store,
                }
                first = create_context_egress_browser(**options)
                await first.get_context()
                await first.get_context()
                self.assertEqual(load.call_count, 1)
                await first.shutdown()
                second = create_context_egress_browser(**options)
                await second.get_context()
                self.assertEqual(restored, [None, state])
                self.assertEqual(load.call_count, 2)
                await second.shutdown()

    async def test_shared_context_save_failure_retains_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContextStateStore(Path(directory))
            native = MagicMock()
            native.is_closed.return_value = False
            native.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
            shared = SimpleNamespace(session_id="\x00egress:one:shared", context=native)
            bridge = MagicMock(listen_url="http://127.0.0.1:12345")
            bridge.start = AsyncMock()
            bridge.aclose = AsyncMock()
            close_http = AsyncMock()
            close_context = AsyncMock()
            with (
                patch("prowl.browser.proxy.context.configured_proxy_bridge", return_value=bridge),
                patch.object(Browser, "_runtime", SimpleNamespace(contexts=SimpleNamespace(isolated=lambda _: shared))),
                patch.object(Browser, "start", new_callable=AsyncMock),
                patch.object(Browser, "is_running", return_value=True),
                patch.object(Browser, "get_context", new_callable=AsyncMock, return_value=shared),
                patch.object(Browser, "close_context", close_context),
                patch.object(store, "save", side_effect=OSError("disk failure")),
            ):
                owner = create_context_egress_browser(
                    name="one",
                    proxy_url="socks5://user:secret@127.0.0.1:1080",
                    base_browser=Browser,
                    close_http_context=close_http,
                    state_store=store,
                )
                with self.assertRaises(OSError):
                    await owner.shutdown()
            close_http.assert_not_awaited()
            close_context.assert_not_awaited()
            bridge.aclose.assert_not_awaited()

    async def test_http_proxy_uses_remote_dns_on_same_socks_upstream(self) -> None:
        backend = BrowserBackend(
            egresses={"one": "socks5://user:secret@127.0.0.1:1080"},
            egress_boundary=EgressBoundary.CONTEXT,
        )
        self.assertEqual(backend._http_proxy("one", Browser), "socks5h://user:secret@127.0.0.1:1080")
        self.assertIs(backend._pool._boundary, EgressBoundary.CONTEXT)
