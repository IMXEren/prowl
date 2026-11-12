"""Named egress owners that share a browser process but keep separate proxy contexts."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from playwright.async_api import Error as PlaywrightError

from prowl.browser.browser import Browser  # noqa: TC001 - runtime subclass base
from prowl.browser.exceptions import BrowserContextError
from prowl.browser.proxy.bridge import configured_proxy_bridge
from prowl.service.context_state import ContextStateError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from playwright.async_api import BrowserContext as PWBrowserCtx
    from playwright.async_api import Page as PWPage
    from playwright.async_api import StorageState

    from prowl.browser.browser import TabGroup
    from prowl.browser.driver.contexts import BrowserContextHandle
    from prowl.service.context_state import ContextStateStore

_SHARED_STATE_ID = "\x00named-egress-shared"


# Prefixes are private context-manager keys, not caller session identifiers.
def create_context_egress_browser(  # noqa: C901 - the returned class owns each egress operation
    *,
    name: str,
    proxy_url: str,
    base_browser: type[Browser],
    close_http_context: Callable[[PWBrowserCtx], Awaitable[None]],
    state_store: ContextStateStore | None = None,
) -> type[Browser]:
    """Bind a named egress to isolated contexts within the configured base browser."""
    prefix = f"\x00egress:{name}:"
    shared_id = f"{prefix}shared"
    bridge = configured_proxy_bridge(proxy_url)

    class ContextEgressBrowser(base_browser):
        _EGRESS_NAME = name
        _runtime = base_browser._runtime  # noqa: SLF001
        _lifecycle = base_browser._lifecycle  # noqa: SLF001
        _bridge_lock = asyncio.Lock()

        @classmethod
        async def start(cls) -> None:
            await base_browser.start()
            async with cls._bridge_lock:
                if bridge is not None and bridge.listen_url is None:
                    await bridge.start()
                proxy = bridge.listen_url if bridge is not None else proxy_url
                handle = base_browser._runtime.contexts.isolated(shared_id)  # noqa: SLF001
                if state_store is None or (handle is not None and not handle.context.is_closed()):
                    await base_browser.get_context(shared_id, proxy_server=proxy)
                    return
                state = await asyncio.to_thread(state_store.load, name, _SHARED_STATE_ID)
                try:
                    await base_browser.get_context(shared_id, proxy_server=proxy, storage_state=state)
                except PlaywrightError:
                    msg = "persisted egress context state could not be restored"
                    raise ContextStateError(msg) from None

        @classmethod
        def proxy_url(cls) -> str:
            return proxy_url

        @classmethod
        async def get_context(
            cls,
            session_id: str | None = None,
            *,
            storage_state: StorageState | None = None,
            proxy_server: str | None = None,
        ) -> BrowserContextHandle:
            if proxy_server is not None:
                msg = "egress proxy cannot be overridden"
                raise BrowserContextError(msg)
            await cls.start()
            if session_id is None:
                if storage_state is not None:
                    msg = "shared egress context does not accept storage state"
                    raise BrowserContextError(msg)
                return await base_browser.get_context(shared_id)
            return await base_browser.get_context(
                f"{prefix}session:{session_id}", storage_state=storage_state, proxy_server=cls._effective_proxy()
            )

        @classmethod
        def _effective_proxy(cls) -> str:
            proxy = bridge.listen_url if bridge is not None else proxy_url
            if proxy is None:
                msg = "egress proxy bridge is not listening"
                raise BrowserContextError(msg)
            return proxy

        @classmethod
        async def close_context(cls, session_id: str, *, evicted: bool = False) -> None:
            await base_browser.close_context(f"{prefix}session:{session_id}", evicted=evicted)

        @classmethod
        async def create(cls, context: BrowserContextHandle | None = None) -> TabGroup:
            handle = context if context is not None else await cls.get_context(None)
            if handle.session_id is None or not handle.session_id.startswith(prefix):
                msg = "browser context belongs to a different egress"
                raise BrowserContextError(msg)
            return await super().create(handle)

        @classmethod
        async def _new_page(cls, context: BrowserContextHandle | None = None) -> tuple[str, PWPage]:
            if context is None or context.session_id is None or not context.session_id.startswith(prefix):
                msg = "browser context belongs to a different egress"
                raise BrowserContextError(msg)
            return await super()._new_page(context)

        @classmethod
        def resource_metrics(cls) -> dict[str, int]:
            # The shared base runtime already contributes these counters once.
            return dict.fromkeys(base_browser.resource_metrics(), 0)

        @classmethod
        async def shutdown(cls) -> None:
            handle = base_browser._runtime.contexts.isolated(shared_id)  # noqa: SLF001
            if handle is not None:
                if state_store is not None and base_browser.is_running() and not handle.context.is_closed():
                    state = await handle.context.storage_state(indexed_db=True)
                    await asyncio.to_thread(state_store.save, name, _SHARED_STATE_ID, state)
                await close_http_context(handle.context)
                await base_browser.close_context(shared_id)
            if bridge is not None:
                await bridge.aclose()

        @classmethod
        async def retry_shutdown(cls) -> None:
            await cls.shutdown()

    ContextEgressBrowser.__name__ = f"ContextEgressBrowser_{name}"
    return ContextEgressBrowser
