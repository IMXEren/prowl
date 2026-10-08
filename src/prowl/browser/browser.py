"""Browser process manager, tab group abstraction, and utilities."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Self

from loguru import logger

from prowl.browser.driver.runtime import BrowserRuntimeState, resolve_cdp_ws_url
from prowl.browser.exceptions import BrowserContextError, BrowserError, BrowserStartError, BrowserTabError
from prowl.browser.lifecycle.startup import BrowserLifecycle, BrowserShutdownState

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Browser as PWBrowser
    from playwright.async_api import BrowserContext as PWBrowserCtx
    from playwright.async_api import Page as PWPage
    from playwright.async_api import StorageState
    from pydoll.browser import Chrome
    from pydoll.browser.tab import Tab as PDTab

    from prowl.browser.config import BrowserConfig
    from prowl.browser.driver.contexts import BrowserContextHandle
    from prowl.browser.fingerprint import FingerprintManager


class Browser:
    """Singleton process manager for the shared browser instance.

    Manages the browser lifecycle (start, shutdown), CDP WebSocket connection,
    fingerprint configuration, shared Playwright persistent context, and pydoll
    Chrome instance. All methods are classmethods - there is only one browser
    process.

    Tab groups (concurrent browsing sessions) are created via :meth:`create`,
    which returns a :class:`TabGroup` instance.
    """

    _cdp_port: ClassVar[int] = 9222

    _MAX_GROUPS: ClassVar[int] = 3
    _runtime: ClassVar[BrowserRuntimeState] = BrowserRuntimeState(max_groups=_MAX_GROUPS)
    _lifecycle: ClassVar[BrowserLifecycle] = BrowserLifecycle(_runtime)

    @classmethod
    def _ensure_lifecycle(cls: type[Self]) -> None:
        """Ensure concrete runtime and lifecycle composition roots exist."""
        if cls._lifecycle is None:
            cls._lifecycle = BrowserLifecycle(cls._runtime)

    @classmethod
    def configure(cls: type[Self], config: BrowserConfig) -> None:
        """Adopt explicit launch configuration before the browser starts.

        The proxy URL is never logged. Refuses to reconfigure a running or
        shutting-down browser, and rejects a context cap below one before any
        configuration is mutated.

        :raises BrowserStartError: when the browser is running or shutting down, or when
            the context cap is below one.
        """
        if config.max_contexts < 1:
            msg = f"BrowserConfig.max_contexts must be at least 1, got {config.max_contexts!r}."
            raise BrowserStartError(msg)
        cls._lifecycle.apply_config(config, is_running=cls.is_running)
        cls._runtime.contexts.configure_limit(config.max_contexts)

    @classmethod
    def proxy_url(cls: type[Self]) -> str | None:
        """Return the proxy configured for this identity."""
        return cls._lifecycle.proxy_url

    @classmethod
    def _webdata_path(cls: type[Self]) -> Path:
        return cls._lifecycle.webdata_path()

    @classmethod
    def unpack_profile(cls: type[Self], archive: str | Path | None = None) -> bool:
        """Restore ``_profile_dir`` from *archive* (zip)."""
        return cls._lifecycle.unpack_profile(archive)

    @classmethod
    def pack_profile(cls: type[Self], archive: str | Path | None = None) -> Path | None:
        """Zip ``_profile_dir`` into *archive*, skipping caches and journals."""
        return cls._lifecycle.pack_profile(archive)

    def __repr__(self: Self) -> str:
        """Return a human-readable representation of the browser state."""
        cls = type(self)
        state = "running" if cls.is_running() else "stopped"
        return f"<Browser({state}) port={cls._cdp_port} groups={len(cls._runtime.active_groups)}>"

    @classmethod
    def pw(cls: type[Self]) -> PWBrowser:
        """Shared Playwright Browser instance.

        Only use when the browser is already running.
        """
        return cls._runtime.get_pw_browser()

    @classmethod
    def pw_main_ctx(cls: type[Self]) -> PWBrowserCtx:
        """Shared Playwright Browser persistent ctx instance.

        Only use when the browser is already running.
        """
        return cls._runtime.get_pw_main_ctx()

    @classmethod
    def pd(cls: type[Self]) -> Chrome:
        """Shared pydoll Chrome instance.

        Only use when the browser is already running.
        """
        return cls._runtime.get_pd()

    @classmethod
    async def _get_cdp_ws_url(cls: type[Self]) -> str:
        """Queries the local debugging endpoint to fetch the active DevTools WebSocket URL."""
        return await resolve_cdp_ws_url(cls._lifecycle.cdp_port)

    @classmethod
    def is_running(cls: type[Self]) -> bool:
        """Is browser running."""
        runtime_is_running = getattr(cls._runtime, "is_running", None)
        if callable(runtime_is_running):
            return bool(runtime_is_running())
        return (
            cls._runtime.main_ctx is not None
            and not cls._runtime.main_ctx.is_closed()
            and cls._runtime.shared_pd is not None
        )

    @classmethod
    async def start(
        cls: type[Self],
    ) -> None:
        """Start the browser with a persistent ctx.

        Restores a cached profile package if available, primes a fresh profile
        if needed, injects Google as the default search engine, then launches.
        """
        await cls._lifecycle.start(
            is_running=cls.is_running,
            popup_handler=cls._handle_popup_page,
        )
        cls._cdp_port = cls._lifecycle.cdp_port

    @classmethod
    async def connect(cls: type[Self], ws_url: str) -> None:
        """Attach this process's drivers to a caller-owned remote CDP websocket."""
        if not isinstance(ws_url, str) or not ws_url.strip():
            msg = "Browser.connect requires a non-empty ws_url."
            raise BrowserStartError(msg)
        await cls._lifecycle.connect(
            is_running=cls.is_running,
            ws_url=ws_url,
            popup_handler=cls._handle_popup_page,
        )

    @classmethod
    async def get_context(
        cls: type[Self],
        session_id: str | None = None,
        *,
        storage_state: StorageState | None = None,
    ) -> BrowserContextHandle:
        """Return this browser's context handle for *session_id*.

        ``None`` returns the shared persistent context, which carries the browsing state
        every request already inherits. A session id returns a long-lived isolated context
        in the same browser process, created on first use and reused afterwards, so a
        session keeps its own cookies and storage on the same device identity.
        *storage_state* seeds a newly created isolated context through Playwright's native
        restore; the shared persistent context never accepts it, and a live session handle
        keeps its own state because reuse never calls the factory again.

        :raises BrowserStartError: when the browser is shutting down or cannot start.
        :raises BrowserContextError: when the session id cannot name a context, or when
            storage state is supplied for the shared persistent context.
        """
        if session_id is None:
            if storage_state is not None:
                msg = "The shared persistent context does not accept supplied storage state."
                raise BrowserContextError(msg)
            return cls._runtime.shared_context()
        if cls._lifecycle.shutdown_state is BrowserShutdownState.IN_PROGRESS:
            msg = "Browser is shutting down - cannot create new contexts."
            raise BrowserStartError(msg)
        if not cls.is_running():
            await cls.start()
        if storage_state is None:
            return await cls._runtime.isolated_context(session_id)
        return await cls._runtime.isolated_context(session_id, storage_state=storage_state)

    @classmethod
    async def close_context(cls: type[Self], session_id: str, *, evicted: bool = False) -> None:
        """Close this browser's isolated context for *session_id*.

        Idempotent: closing an unknown session is a no-op. The shared persistent context is
        never closed here. *evicted* marks automatic cleanup retiring the context rather than
        an explicit caller close; only evictions are counted in :meth:`resource_metrics`.

        :raises BrowserContextError: when the session id cannot name a context.
        """
        if evicted:
            await cls._runtime.close_isolated_context(session_id, evicted=True)
        else:
            await cls._runtime.close_isolated_context(session_id)

    @classmethod
    def resource_metrics(cls: type[Self]) -> dict[str, int]:
        """Return this identity's native resource counters as a plain integer snapshot.

        The cumulative counters survive browser generations of this identity class and the
        gauges describe what it currently owns. The snapshot reads driver and lifecycle state
        only, so it never probes the live browser, takes a lock or labels an identity.
        """
        contexts = cls._runtime.contexts
        return {
            "context_count": contexts.count,
            "context_created_total": contexts.context_created_total,
            "context_evicted_total": contexts.context_evicted_total,
            "browser_restart_total": cls._lifecycle.browser_restart_total,
            "tabgroups_active": len(cls._runtime.active_groups),
        }

    @classmethod
    async def create(cls: type[Self], context: BrowserContextHandle | None = None) -> TabGroup:
        """Create a new :class:`TabGroup` in the shared browser process.

        Acquires a semaphore slot (max *MAX_GROUPS* concurrent groups).
        Starts the browser automatically if not already running. *context* selects the
        browser context the group's pages are created in; the default is the shared
        persistent context.

        :raises BrowserStartError: if the browser is shutting down
            or group creation fails.
        :raises BrowserContextError: when *context* is not this browser's own context.
        """
        if cls._lifecycle.shutdown_state is BrowserShutdownState.IN_PROGRESS:
            msg = "Browser is shutting down - cannot create new tab groups."
            raise BrowserStartError(msg)

        try:
            # The group has to belong to the class that created it. A tab group resolves its
            # tabs through its owner, so a group made through an egress browser subclass must
            # bind to that subclass, or its tabs are looked up in the default browser's runtime
            # where they do not exist.
            group_factory = partial(TabGroup, owner=cls)
            instance = await cls._runtime.create_tab_group(group_factory, cls.start, cls.is_running, context)
        except BrowserContextError:
            raise
        except Exception as e:
            msg = f"Failed to start the browser due to {e}"
            raise BrowserStartError(msg) from e
        else:
            return instance

    @classmethod
    async def _create_from_running(cls: type[Self]) -> TabGroup:
        """Create a tab group in the running browser, bound to *cls* as its owner."""
        return await cls._runtime.create_group(partial(TabGroup, owner=cls))

    @classmethod
    async def _new_page(cls: type[Self], context: BrowserContextHandle | None = None) -> tuple[str, PWPage]:
        """Create a new tab using PW in *context* and add it to page map."""
        if cls._lifecycle.shutdown_state is BrowserShutdownState.IN_PROGRESS:
            msg = "Browser is shutting down - cannot create new pages."
            raise BrowserTabError(msg)

        return await cls._runtime.create_page(context)

    @classmethod
    def _add_tab_to_pd(cls: type[Self], target_id: str) -> PDTab:
        return cls._runtime._add_tab_to_pd(target_id)  # noqa: SLF001

    @classmethod
    def _remove_tab_from_pd(cls: type[Self], target_id: str) -> None:
        cls._runtime._remove_tab_from_pd(target_id)  # noqa: SLF001

    @classmethod
    async def get_pd_tab(cls: type[Self], target_id: str) -> PDTab | None:
        """Resolve the live Pydoll Tab for *target_id*, or ``None``."""
        return await cls._runtime.get_pd_tab(target_id)

    @classmethod
    def get_pw_page(cls: type[Self], target_id: str) -> PWPage | None:
        """Resolve the live Playwright Page for *target_id*, or ``None``."""
        return cls._runtime.get_pw_page(target_id)

    @classmethod
    async def minimize_main_window(cls: type[Self]) -> None:
        """Uses the active PyDoll CDP connection to minimize the browser window."""
        await cls._runtime.minimize_main_window()

    @classmethod
    async def _handle_popup_page(cls: type[Self], page: PWPage) -> None:
        """Handle involuntary popup pages (window.open, target=_blank).

        Attaches the new page to the :class:`TabGroup` that owns the opener,
        so it is tracked and cleaned up on quit/shutdown.
        """
        await cls._runtime.attach_popup_page(page)

    @classmethod
    async def _cleanup_resources(cls: type[Self]) -> None:
        """Compatibility facade for lifecycle-owned cleanup."""
        await cls._lifecycle._cleanup_resources()  # noqa: SLF001

    @classmethod
    async def _do_shutdown(cls: type[Self]) -> None:
        """Compatibility facade for lifecycle-owned shutdown finalization."""
        await cls._lifecycle._do_shutdown()  # noqa: SLF001

    @classmethod
    async def shutdown(cls: type[Self]) -> None:
        """Gracefully tear down all browser resources via lifecycle."""
        await cls._lifecycle.shutdown()

    @classmethod
    async def retry_shutdown(cls: type[Self]) -> None:
        """Retry a failed browser cleanup via lifecycle."""
        await cls._lifecycle.retry_shutdown()

    @classmethod
    async def finally_cleanup(cls: type[Self], tg: TabGroup | None = None) -> None:
        """As the name suggests, use in finally blocks to quit as well as shutdown the Browser."""
        if tg is not None:
            try:
                await tg.quit()
            except Exception as e:  # noqa: BLE001
                logger.error(f"TabGroup quit failed during source cleanup: {e}")
        await cls.shutdown()

    @classmethod
    def _do_sync_chores_before_exit(cls: type[Self]) -> None:
        """The left out synchronous chores that need to be done before exiting."""
        cls._lifecycle.do_sync_chores_before_exit()

    @classmethod
    def _sync_atexit_fallback(cls) -> None:
        """Compatibility facade for lifecycle-owned synchronous fallback."""
        cls._lifecycle.sync_atexit_fallback()


class TabGroup:
    """Instance-level tab group.

    Represents a group of tabs (1 parent + n children) within one browser process.
    Created via ``owner.create()``. Holds per-group state and delegates to its
    owner, which is :class:`Browser` for the process-wide egress and an egress
    subclass of it for a named egress, so a group always belongs to exactly one
    browser process.
    """

    def __init__(
        self: Self,
        target_id: str,
        gid: int,
        owner: type[Browser] | None = None,
        context: BrowserContextHandle | None = None,
    ) -> None:
        """Initialize tab group with a parent tab.

        Private constructor. Always use ``owner.create()`` to instantiate.
        *context* is the browser context the group's pages are created in; the owning
        runtime binds it at creation, and the shared persistent context is used otherwise.
        """
        self.gid: int = gid
        self.target_id: str = target_id
        self._owner: type[Browser] = owner if owner is not None else Browser
        self._context: BrowserContextHandle | None = context
        self.child_target_ids: list[str] = []
        self._lock: asyncio.Lock = asyncio.Lock()
        self._quitting: bool = False

    def bind_context(self: Self, context: BrowserContextHandle) -> None:
        """Bind this group to the browser context its pages are created in."""
        self._context = context

    @property
    def context(self: Self) -> BrowserContextHandle:
        """The browser context this group's pages live in."""
        if self._context is None:
            self._context = self._owner._runtime.shared_context()  # noqa: SLF001
        return self._context

    def __repr__(self: Self) -> str:
        """Return a human-readable representation of the tab group."""
        return f"<TabGroup #{self.gid} ({len(self.child_target_ids)} children)>"

    def pd(self: Self) -> Chrome:
        """Delegates to the owning browser's pd()."""
        return self._owner.pd()

    @property
    def fp(self: Self) -> FingerprintManager:
        """Delegates to Browser lifecycle fingerprint state."""
        fp = self._owner._lifecycle.fingerprint  # noqa: SLF001
        if fp is None:
            msg = "Browser fingerprint is not configured - call start() first."
            raise BrowserError(msg)
        return fp

    async def new_tab(self: Self) -> PDTab:
        """Spawns a dependent sub-tab in this group's context and links it to this tab group."""
        async with self._lock:
            if self._quitting:
                msg = "Tab group is closing - cannot create new pages."
                raise BrowserTabError(msg)
            target_id, page = await self._owner._new_page(self.context)  # noqa: SLF001
            if self._quitting:
                await page.close()
                self._owner._runtime._forget_target(target_id, page)  # noqa: SLF001
                msg = "Tab group closed while creating a page."
                raise BrowserTabError(msg)
            self._owner._runtime.attach_page_to_group(page, self)  # noqa: SLF001
            self.child_target_ids.append(target_id)
        tab = await self._owner.get_pd_tab(target_id)
        if tab is None:
            msg = f"Failed to resolve newly created tab {target_id} in group #{self.gid}."
            raise BrowserTabError(msg)
        return tab

    @property
    def ppage(self: Self) -> PWPage:
        """Access to the parent Playwright Page for this group."""
        page = self._owner._runtime.target_to_page_map.get(self.target_id)  # noqa: SLF001
        if page is None:
            msg = f"Parent page {self.target_id} in group #{self.gid} is no longer available."
            raise BrowserTabError(msg)
        return page

    @property
    async def ptab(self: Self) -> PDTab:
        """Access to the parent Pydoll Tab for this group."""
        tab = await self._owner.get_pd_tab(self.target_id)
        if tab is None:
            msg = f"Parent tab {self.target_id} in group #{self.gid} is no longer available."
            raise BrowserTabError(msg)
        return tab

    async def close(self: Self, target_id: str) -> None:
        """Tears down a page associated with `target_id`.

        If the parent tab is targeted, the entire group is torn down
        via :meth:`quit`. Child tabs are closed individually and removed
        from the child list.
        """
        await self._owner._runtime.close_group_target(self, target_id)  # noqa: SLF001

    async def quit(self: Self) -> None:
        """Tears down all pages linked to this tab group and returns the pool slot."""
        await self._owner._runtime.close_group(self)  # noqa: SLF001
