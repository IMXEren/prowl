"""Concrete runtime state container for browser driver ownership."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import curl_cffi
from cloakbrowser import launch_persistent_context_async
from loguru import logger
from playwright.async_api import async_playwright
from pydoll.browser import Chrome
from pydoll.browser.tab import Tab as PDTab

from prowl.browser.driver.contexts import (
    BrowserContextHandle,
    BrowserContextManager,
    apply_human_patch,
    persona_context_options,
    require_session_id,
)
from prowl.browser.exceptions import BrowserContextError, BrowserError, BrowserStartError, BrowserTabError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from playwright.async_api import Browser as PWBrowser
    from playwright.async_api import BrowserContext as PWBrowserCtx
    from playwright.async_api import Page as PWPage
    from playwright.async_api import Playwright, StorageState
    from pydoll.browser.options import ChromiumOptions

    from prowl.browser.browser import TabGroup

    #: Handler a context calls for an involuntary popup page.
    PopupHandler = Callable[[PWPage], Coroutine[None, None, None] | None]


@dataclass(frozen=True, slots=True)
class DriverStartupConfig:
    """Explicit startup inputs passed from lifecycle to the concrete driver."""

    profile_dir: str
    user_data_dir: str
    cdp_port: int
    fingerprint_options: ChromiumOptions
    launch_arguments: list[str]
    viewport: dict[str, int]
    locale: str
    popup_handler: Callable[[PWPage], Coroutine[None, None, None] | None]
    headless: bool = False
    geoip: bool = True
    humanize: bool = True
    color_scheme: Literal["light", "dark", "no-preference"] = "dark"


@dataclass(frozen=True, slots=True)
class DriverRemoteAttachConfig:
    """Explicit inputs for attaching driver clients to a remote CDP websocket."""

    ws_url: str
    popup_handler: Callable[[PWPage], Coroutine[None, None, None] | None]
    main_ctx: PWBrowserCtx | None = None


async def resolve_cdp_ws_url(cdp_port: int) -> str:
    """Resolve the active DevTools websocket URL for *cdp_port*."""
    try:
        endpoint = f"http://127.0.0.1:{cdp_port}/json/version"
        async with curl_cffi.AsyncSession() as session:
            response = await session.get(endpoint, timeout=15)
            response.raise_for_status()
            data = response.json()
            return cast("str", data["webSocketDebuggerUrl"])
    except Exception as exc:
        msg = f"Failed to resolve WebSocket debugger URL from port {cdp_port}. Is CloakBrowser running?"
        raise RuntimeError(msg) from exc


@dataclass(slots=True)
class BrowserRuntimeState:
    """Concrete owner for live browser driver state."""

    max_groups: int
    main_ctx: PWBrowserCtx | None = None
    shared_pd: Chrome | None = None
    target_to_page_map: dict[str, PWPage] = field(default_factory=dict)
    target_page_owned: dict[str, bool] = field(default_factory=dict)
    page_to_group: dict[PWPage, TabGroup | None] = field(default_factory=dict)
    active_groups: set[TabGroup] = field(default_factory=set)
    spawn_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    group_semaphore: asyncio.Semaphore = field(init=False)
    next_group_id: int = 1
    cdp_playwright: Playwright | None = None
    cdp_browser: PWBrowser | None = None
    main_ctx_owned: bool = False
    contexts: BrowserContextManager = field(default_factory=BrowserContextManager)
    popup_handler: PopupHandler | None = None
    #: Whether the running browser was launched with CloakBrowser humanization, so
    #: contexts created beside its persistent one get the same patch.
    humanize_contexts: bool = False

    def __post_init__(self) -> None:
        """Initialize semaphore after max_groups is available."""
        self.group_semaphore = asyncio.Semaphore(self.max_groups)

    async def start_live(self, config: DriverStartupConfig) -> None:
        """Start concrete browser clients from explicit lifecycle-owned config."""
        # Reset the group id
        self.next_group_id = 1
        self.shared_pd = Chrome(options=config.fingerprint_options)
        self.shared_pd._set_browser_preferences_in_user_data_dir(config.user_data_dir)  # noqa: SLF001

        self.main_ctx = await launch_persistent_context_async(
            headless=config.headless,
            args=config.launch_arguments,
            viewport=config.viewport,
            locale=config.locale,
            geoip=config.geoip,
            humanize=config.humanize,
            color_scheme=config.color_scheme,
            user_data_dir=config.user_data_dir,
        )
        self.main_ctx = cast("PWBrowserCtx", self.main_ctx)
        self.main_ctx_owned = True
        self.main_ctx.on("page", config.popup_handler)
        self.popup_handler = config.popup_handler
        self.humanize_contexts = config.humanize
        self.contexts.bind_shared(self.main_ctx)
        await asyncio.sleep(0.5)

        ws_url = await resolve_cdp_ws_url(config.cdp_port)
        main_tab = await self.shared_pd.connect(ws_url)
        main_page = self.main_ctx.pages[0]
        if main_tab._target_id is None:  # noqa: SLF001
            msg = "Failed to resolve target ID for the main browser tab."
            raise BrowserTabError(msg)
        self.target_to_page_map[main_tab._target_id] = main_page  # noqa: SLF001
        self.target_page_owned[main_tab._target_id] = True  # noqa: SLF001
        self.page_to_group[main_page] = None
        await self.minimize_main_window()

    async def attach_remote(self, config: DriverRemoteAttachConfig) -> None:
        """Attach concrete browser clients to a caller-owned remote CDP websocket."""
        self.next_group_id = 1
        self.shared_pd = Chrome()
        try:
            if config.main_ctx is None:
                cdp_playwright = await async_playwright().start()
                cdp_browser = await cdp_playwright.chromium.connect_over_cdp(config.ws_url)
                self.cdp_playwright = cdp_playwright
                self.cdp_browser = cdp_browser
                contexts = cdp_browser.contexts
                if contexts:
                    self.main_ctx = contexts[0]
                    self.main_ctx_owned = False
                else:
                    self.main_ctx = await cdp_browser.new_context()
                    self.main_ctx_owned = True
            else:
                self.main_ctx = cast("PWBrowserCtx", config.main_ctx)
                self.main_ctx_owned = False
            self.main_ctx.on("page", config.popup_handler)
            self.popup_handler = config.popup_handler
            # A remote browser is not ours to humanize: its launch settings are unknown,
            # so contexts created here keep whatever behavior it already runs.
            self.contexts.bind_shared(self.main_ctx)

            main_tab = await self.shared_pd.connect(config.ws_url)
            page_owned = self.main_ctx_owned
            if self.main_ctx.pages:
                main_page = self.main_ctx.pages[0]
            else:
                main_page = await self.main_ctx.new_page()
                page_owned = True
            cdp = await self.main_ctx.new_cdp_session(main_page)
            try:
                result = await cdp.send("Target.getTargetInfo")
            finally:
                await cdp.detach()
            target_id = result["targetInfo"]["targetId"]
            if main_tab._target_id is None or target_id is None:  # noqa: SLF001
                msg = "Failed to resolve target ID for the remote browser tab."
                raise BrowserTabError(msg)
            self.target_to_page_map[target_id] = main_page
            self.target_page_owned[target_id] = page_owned
            self.page_to_group[main_page] = None
            await self.minimize_main_window()
        except BaseException as e:
            await self.rollback_start()
            msg = "failed to attach remote cdp"
            raise BrowserStartError(msg) from e

    async def rollback_start(self) -> None:
        """Retire the local generation so a fresh one can be launched cleanly.

        Registered groups and pages give their admission slots back exactly once, before the
        browser they run in is closed, then the isolated contexts and the native projections
        are retired. Failed native closes retain unfinished ownership for a subsequent retry.
        """
        await self.close_all_groups_and_pages()
        await self.close_isolated_contexts()
        if self.shared_pd is not None:
            await self.shared_pd.close()
            self.shared_pd = None
        if self.main_ctx is not None and self.main_ctx_owned:
            await self.main_ctx.close()
        self.main_ctx = None
        if self.cdp_browser is not None:
            await self.cdp_browser.close()
            self.cdp_browser = None
        if self.cdp_playwright is not None:
            await self.cdp_playwright.stop()
            self.cdp_playwright = None
        self.main_ctx_owned = False
        self.target_to_page_map.clear()
        self.target_page_owned.clear()
        self.page_to_group.clear()
        self.contexts.reset()

    def allocate_group_id(self) -> int:
        """Return the next concrete group id and advance the sequence."""
        group_id = self.next_group_id
        self.next_group_id += 1
        return group_id

    def _require_main_ctx(self) -> PWBrowserCtx:
        """Return the live Playwright context or raise a concrete runtime error."""
        if self.main_ctx is None:
            msg = "Browser is not running - call Browser.start() first."
            raise BrowserError(msg)
        return self.main_ctx

    def is_running(self) -> bool:
        """Return whether concrete runtime clients are live.

        A Playwright context can keep reporting itself open after the browser process behind
        it has gone away, so the native connection is consulted too when the projection
        exposes one.
        """
        if self.main_ctx is None or self.shared_pd is None:
            return False
        if self.main_ctx.is_closed():
            return False
        browser = self.main_ctx.browser
        is_connected = getattr(browser, "is_connected", None)
        return not (callable(is_connected) and not is_connected())

    def get_pw_browser(self) -> PWBrowser:
        """Return the shared Playwright browser projection."""
        ctx = self._require_main_ctx()
        browser = ctx.browser
        if browser is None:
            msg = "Playwright browser disconnected."
            raise BrowserError(msg)
        return browser

    def get_pw_main_ctx(self) -> PWBrowserCtx:
        """Return the shared Playwright persistent context projection."""
        return self._require_main_ctx()

    def get_pd(self) -> Chrome:
        """Return the shared PyDoll Chrome projection."""
        if self.shared_pd is None:
            msg = "Browser is not running - call Browser.start() first."
            raise BrowserError(msg)
        return self.shared_pd

    # -- Owned browser contexts ------------------------------------------------------

    def shared_context(self) -> BrowserContextHandle:
        """Return the handle wrapping the shared persistent context.

        The handle is bound on first use, so a runtime whose main context was installed
        directly still resolves one identity, and a restarted browser replaces a handle
        that pointed at the previous context.
        """
        handle = self.contexts.shared()
        if handle is not None and handle.context is self.main_ctx:
            return handle
        return self.contexts.bind_shared(self._require_main_ctx())

    def resolve_context(self, context: BrowserContextHandle | None) -> BrowserContextHandle:
        """Return the context handle a request group should use.

        ``None`` selects the shared persistent context.

        :raises BrowserContextError: for a handle owned by another browser identity or
            whose context has already been closed.
        """
        if context is None:
            return self.shared_context()
        self.contexts.require_own(context)
        if context.context.is_closed():
            msg = f"The browser context for session {context.session_id!r} is closed."
            raise BrowserContextError(msg)
        return context

    async def isolated_context(
        self,
        session_id: str,
        *,
        storage_state: StorageState | None = None,
    ) -> BrowserContextHandle:
        """Return the isolated context for *session_id*, creating it on first use.

        The context lives in the running browser process, so it keeps the identity's
        proxy, fingerprint, locale and timezone while holding its own cookies and storage.
        *storage_state* seeds a newly created context through the native API; an existing
        live handle wins and ignores it, because the factory only runs on first creation.

        :raises BrowserContextError: for a session id that cannot name a context.
        """
        require_session_id(session_id)
        if storage_state is None:
            return await self.contexts.get_or_create(session_id, lambda: self._open_isolated_context(session_id))
        return await self.contexts.get_or_create(
            session_id,
            lambda: self._open_isolated_context(session_id, storage_state=storage_state),
        )

    async def _open_isolated_context(
        self,
        session_id: str,
        *,
        storage_state: StorageState | None = None,
    ) -> PWBrowserCtx:
        """Create one isolated context beside the persistent one in this browser.

        *storage_state* is handed to the native ``new_context`` unchanged, so cookies,
        localStorage and IndexedDB are restored by Playwright itself.
        """
        main_ctx = self._require_main_ctx()
        browser = main_ctx.browser
        if browser is None:
            msg = "Playwright browser disconnected."
            raise BrowserError(msg)
        options = persona_context_options(main_ctx)
        if storage_state is None:
            context = await browser.new_context(**options)
        else:
            context = await browser.new_context(**options, storage_state=storage_state)
        try:
            if self.popup_handler is not None:
                context.on("page", self.popup_handler)
            if self.humanize_contexts:
                apply_human_patch(context)
        except BaseException:
            await context.close()
            raise
        logger.debug(f"Isolated browser context created for session {session_id!r}.")
        return context

    async def close_isolated_context(self, session_id: str, *, evicted: bool = False) -> None:
        """Close the isolated context for *session_id* and the groups that live in it.

        Idempotent: an unknown session is a no-op. The shared persistent context is never
        closed here. *evicted* says automatic cleanup, not an explicit caller, is retiring the
        context, and is forwarded to the manager's eviction counter.
        """
        require_session_id(session_id)
        handle = self.contexts.isolated(session_id)
        if handle is None:
            return
        await self._close_groups_in_context(handle)
        if evicted:
            await self.contexts.close_isolated(session_id, evicted=True)
        else:
            await self.contexts.close_isolated(session_id)
        for target_id, page in list(self.target_to_page_map.items()):
            if page.context is handle.context:
                self._forget_target(target_id, page)

    async def close_isolated_contexts(self) -> None:
        """Close every isolated context this runtime owns and their request groups."""
        for handle in self.contexts.isolated_handles():
            if handle.session_id is not None:
                await self.close_isolated_context(handle.session_id)

    async def _close_groups_in_context(self, handle: BrowserContextHandle) -> None:
        """Close every request group whose pages live in *handle*."""
        async with self.spawn_lock:
            groups = [group for group in self.active_groups if group.context is handle]
        if groups:
            await asyncio.gather(*(self.close_group(group) for group in groups), return_exceptions=True)

    def attach_page_to_group(self, page: PWPage, group: TabGroup) -> None:
        """Associate a concrete Playwright page projection with a tab group."""
        self.page_to_group[page] = group

    async def create_tab_group(
        self,
        group_factory: Callable[[str, int], TabGroup],
        ensure_running: Callable[[], Awaitable[None]],
        is_running: Callable[[], bool],
        context: BrowserContextHandle | None = None,
    ) -> TabGroup:
        """Own group semaphore admission and create a group after startup if needed.

        The admission slot is returned exactly once on every path, including cancellation
        while the browser starts or the parent page is created, and a group that was
        registered but never handed to the caller has its slot returned here.
        """
        if context is not None:
            self.contexts.require_own(context)
        await self.group_semaphore.acquire()
        registered: list[TabGroup] = []
        try:
            if not is_running():
                await ensure_running()
            if not is_running():
                msg = "Failed to start from already running: Master browser process is not active."
                raise BrowserError(msg)
            return await self.create_group(group_factory, context, registered)
        except BaseException:
            for group in registered:
                try:
                    await self.close_group(group)
                finally:
                    self._release_group_slot(group)
            if not registered:
                self.group_semaphore.release()
            raise

    def _release_group_slot(self, group: TabGroup) -> None:
        """Return a registered group's admission slot exactly once, without awaiting.

        Releasing without awaiting is what keeps a cancelled close from leaking the slot.
        """
        if group in self.active_groups:
            self.active_groups.discard(group)
            self.group_semaphore.release()

    def _forget_target(self, target_id: str, page: PWPage | None = None) -> None:
        """Drop the runtime bookkeeping for a target without awaiting."""
        self._remove_tab_from_pd(target_id)
        self.target_to_page_map.pop(target_id, None)
        self.target_page_owned.pop(target_id, None)
        if page is not None:
            self.page_to_group.pop(page, None)

    def _add_tab_to_pd(self, target_id: str, browser_context_id: str | None = None) -> PDTab:
        """Add a PyDoll tab entry for *target_id* in its browser context and return it."""
        if self.shared_pd is None:
            msg = "Browser is not running - call Browser.start() first."
            raise BrowserError(msg)
        tab = PDTab(
            self.shared_pd,
            **self.shared_pd._get_tab_kwargs(target_id, browser_context_id=browser_context_id),  # noqa: SLF001
        )
        self.shared_pd._tabs_opened[target_id] = tab  # noqa: SLF001
        return tab

    def _remove_tab_from_pd(self, target_id: str) -> None:
        """Remove a PyDoll tab entry if the concrete client is present."""
        if self.shared_pd is None:
            return
        self.shared_pd._tabs_opened.pop(target_id, None)  # noqa: SLF001

    async def create_page(self, context: BrowserContextHandle | None = None) -> tuple[str, PWPage]:
        """Create a concrete Playwright page in *context* and track its target id.

        The CDP target's browser context id is kept on the PyDoll tab, so storage and
        cookie operations made through that tab reach the context the page really lives in.
        """
        handle = self.resolve_context(context)
        async with self.spawn_lock:
            self.contexts.require_own(handle)
            page = await handle.context.new_page()
            try:
                cdp = await handle.context.new_cdp_session(page)
                try:
                    result = await cdp.send("Target.getTargetInfo")
                finally:
                    await cdp.detach()
                info = result["targetInfo"]
                target_id = info["targetId"]
                self._add_tab_to_pd(target_id, info.get("browserContextId"))
                self.target_to_page_map[target_id] = page
                self.target_page_owned[target_id] = True
            except BaseException:
                await page.close()
                raise
        return (target_id, page)

    async def create_group(
        self,
        group_factory: Callable[[str, int], TabGroup],
        context: BrowserContextHandle | None = None,
        registered: list[TabGroup] | None = None,
    ) -> TabGroup:
        """Create and register a concrete tab group from a new parent page.

        *registered* receives the group as soon as it owns an admission slot, so a caller
        cancelled between creation and hand-off can return that slot exactly once.
        """
        handle = self.resolve_context(context)
        target_id, page = await self.create_page(handle)
        try:
            group = group_factory(target_id, self.allocate_group_id())
            group.bind_context(handle)
        except BaseException:
            self._forget_target(target_id, page)
            with contextlib.suppress(Exception):
                await page.close()
            raise
        self.page_to_group[page] = group
        self.active_groups.add(group)
        if registered is not None:
            registered.append(group)
        return group

    async def get_pd_tab(self, target_id: str) -> PDTab | None:
        """Resolve the live PyDoll tab for *target_id*, or ``None``.

        The tab this runtime registered for the target is authoritative: resolution from
        the browser's target list rebuilds a tab without its browser context id, which
        would point storage and cookie operations at the default context of a page that
        lives in an isolated one.
        """
        if self.shared_pd is None:
            msg = "Browser is not running - call Browser.start() first."
            raise BrowserError(msg)
        cached = self.shared_pd._tabs_opened.get(target_id)  # noqa: SLF001
        if cached is not None:
            return cast("PDTab", cached)
        for target in await self.shared_pd.get_targets():
            if target["targetId"] == target_id and target["type"] == "page":
                return self._add_tab_to_pd(target_id, target.get("browserContextId"))
        return None

    def get_pw_page(self, target_id: str) -> PWPage | None:
        """Resolve the live Playwright page for *target_id*, or ``None``."""
        return self.target_to_page_map.get(target_id)

    async def minimize_main_window(self) -> None:
        """Minimize the browser window when a PyDoll client is connected."""
        if self.shared_pd is None:
            return
        await self.shared_pd.set_window_minimized()

    async def attach_popup_page(self, page: PWPage) -> None:
        """Attach an involuntary popup page to its opener's tab group.

        The page's own context is used to resolve its target, so a popup opened from a page
        in an isolated context is registered against that context.
        """
        opener = await page.opener()
        if opener is None:
            return
        group = self.page_to_group.get(opener)
        if group is None:
            return

        try:
            cdp = await page.context.new_cdp_session(page)
            try:
                result = await cdp.send("Target.getTargetInfo")
            finally:
                await cdp.detach()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to resolve target ID for popup page; discarding.")
            await page.close()
            return

        info = result["targetInfo"]
        target_id = info["targetId"]
        async with self.spawn_lock:
            self._add_tab_to_pd(target_id, info.get("browserContextId"))
            self.target_to_page_map[target_id] = page
            self.target_page_owned[target_id] = True
        async with group._lock:  # noqa: SLF001
            if group._quitting:  # noqa: SLF001
                await page.close()
                self._forget_target(target_id, page)
                return
            self.page_to_group[page] = group
            group.child_target_ids.append(target_id)
        logger.debug(f"Popup tab {target_id} attached to group {group.gid}")

    async def close_group_target(self, group: TabGroup, target_id: str) -> None:
        """Close one target owned by a group, or the whole group for parent target."""
        if target_id == group.target_id:
            await self.close_group(group)
            return

        async with group._lock:  # noqa: SLF001
            if target_id not in group.child_target_ids:
                msg = f"Target {target_id} does not belong to this tab group."
                raise BrowserTabError(msg)

        target_page = self.get_pw_page(target_id)
        if target_page is not None and self.target_page_owned.get(target_id, True):
            await target_page.close()

        self._forget_target(target_id, target_page)
        async with group._lock:  # noqa: SLF001
            if target_id in group.child_target_ids:
                group.child_target_ids.remove(target_id)

    async def close_group(self, group: TabGroup) -> None:
        """Close all targets for a group and release its concurrency slot once."""
        if group._quitting:  # noqa: SLF001
            return
        group._quitting = True  # noqa: SLF001
        errors: list[BaseException] = []
        try:
            async with group._lock:  # noqa: SLF001
                target_ids = [*group.child_target_ids, group.target_id]
            for target_id in target_ids:
                page = self.get_pw_page(target_id)
                try:
                    if page is not None and self.target_page_owned.get(target_id, True):
                        await page.close()
                except asyncio.CancelledError as exc:
                    errors.append(exc)
                except Exception as exc:  # noqa: BLE001 - finish other pages before re-raising
                    logger.exception("Request page cleanup failed")
                    errors.append(exc)
                else:
                    self._forget_target(target_id, page)
                    if target_id in group.child_target_ids:
                        group.child_target_ids.remove(target_id)
        finally:
            self._release_group_slot(group)
        if errors:
            raise errors[0]

    async def close_all_groups_and_pages(self) -> None:
        """Close all registered groups and orphan pages, then clear runtime maps."""
        async with self.spawn_lock:
            running_groups = list(self.active_groups)

        if running_groups:
            logger.debug(f"Force-closing {len(running_groups)} dangling tab groups...")
            await asyncio.gather(*(self.close_group(group) for group in running_groups), return_exceptions=True)

        async with self.spawn_lock:
            self.active_groups.clear()
            active_pages = [
                page
                for target_id, page in self.target_to_page_map.items()
                if self.target_page_owned.get(target_id, True)
            ]

        if active_pages:
            logger.debug("Closing orphan tabs...")
            await asyncio.gather(*(page.close() for page in active_pages), return_exceptions=True)

        async with self.spawn_lock:
            self.target_to_page_map.clear()
            self.target_page_owned.clear()
            self.page_to_group.clear()
