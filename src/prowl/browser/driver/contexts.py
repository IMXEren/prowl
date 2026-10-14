"""Owned Playwright contexts: the shared persistent one and isolated sessions.

The persistent context carries the browsing state every caller inherits. An isolated
session context is a second context in the same browser process: it keeps its own cookies,
storage and permissions while staying on the same device identity, because fingerprint,
proxy, locale and timezone are process-wide launch properties. This module owns the
handles that say which context belongs to which browser identity, so a handle from one
identity's runtime is never accepted by another.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, cast

from cloakbrowser.human import patch_context_async
from cloakbrowser.human import resolve_config as resolve_human_config

from prowl.browser.config import DEFAULT_MAX_CONTEXTS
from prowl.browser.exceptions import BrowserContextError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from playwright.async_api import BrowserContext as PWBrowserCtx
    from playwright.async_api import ViewportSize


class PersonaContextOptions(TypedDict, total=False):
    """The browser persona options accepted by Playwright's public API."""

    viewport: ViewportSize | None
    no_viewport: bool
    screen: ViewportSize
    color_scheme: Literal["dark", "light", "no-preference", "null"]


_PERSONA_CONTEXT_OPTIONS = {
    "viewport": "viewport",
    "noDefaultViewport": "no_viewport",
    "screen": "screen",
    "colorScheme": "color_scheme",
}
_HUMAN_PRESET = "default"


def persona_context_options(context: PWBrowserCtx) -> PersonaContextOptions:
    """Translate actual launch options to public API names, without identity emulation."""
    # Playwright stores wire-format launch options on the implementation, not the wrapper.
    options = getattr(getattr(context, "_impl_obj", None), "_options", {})
    return cast(
        "PersonaContextOptions",
        {public: options[wire] for wire, public in _PERSONA_CONTEXT_OPTIONS.items() if wire in options},
    )


def require_session_id(session_id: object) -> str:
    """Return *session_id* when it can name an isolated session context.

    :raises BrowserContextError: for a value that is not a non-empty string.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        msg = "A session context needs a non-empty session id."
        raise BrowserContextError(msg)
    return session_id


def apply_human_patch(context: PWBrowserCtx) -> None:
    """Apply CloakBrowser's public humanize patch to a context created outside its launch.

    CloakBrowser patches the persistent context on its own launch path; a context Prowl
    creates beside it gets the same patch through the same helper, so interaction is
    humanized in both.
    """
    patch_context_async(context, resolve_human_config(_HUMAN_PRESET, None))


@dataclass(frozen=True, slots=True)
class BrowserContextHandle:
    """One Playwright context owned by one browser identity.

    ``session_id`` is ``None`` for the shared persistent context and the session id for an
    isolated session context. ``owner`` is the identity token of the manager that created
    the handle: a handle from another browser identity is rejected instead of silently
    resolving pages against the wrong browser process.
    """

    session_id: str | None
    context: PWBrowserCtx
    owner: object

    @property
    def is_shared(self) -> bool:
        """Whether this handle wraps the shared persistent context."""
        return self.session_id is None


class BrowserContextManager:
    """Own the shared persistent context and the isolated contexts built beside it.

    The manager also owns the identity's context cap: the shared persistent context and every
    live isolated context count against it, so one identity cannot grow contexts without bound.
    """

    def __init__(self, max_contexts: int = DEFAULT_MAX_CONTEXTS) -> None:
        """Create an empty manager with its own identity token and context cap."""
        self._owner = object()
        self._shared: BrowserContextHandle | None = None
        self._isolated: dict[str, BrowserContextHandle] = {}
        self._closing: set[str] = set()
        #: Cumulative managed-context registrations, kept across generation resets.
        self.context_created_total = 0
        #: Cumulative automatic evictions; an explicit close never counts here.
        self.context_evicted_total = 0
        self.configure_limit(max_contexts)
        self._lock = asyncio.Lock()

    # -- Context cap -----------------------------------------------------------------

    @property
    def max_contexts(self) -> int:
        """Return the most contexts this identity may own, the shared one included."""
        return self._max_contexts

    def configure_limit(self, max_contexts: int) -> None:
        """Set the context cap for this identity.

        :raises BrowserContextError: when *max_contexts* is below one.
        """
        if max_contexts < 1:
            msg = f"The browser context cap must be at least 1, got {max_contexts!r}."
            raise BrowserContextError(msg)
        self._max_contexts = max_contexts

    def _owned_count(self) -> int:
        """Return how many contexts this identity owns now, the shared one included."""
        return (1 if self._shared is not None else 0) + len(self._isolated)

    @property
    def count(self) -> int:
        """Return the managed contexts this identity currently owns."""
        return self._owned_count()

    # -- Shared persistent context --------------------------------------------------

    def bind_shared(self, context: PWBrowserCtx) -> BrowserContextHandle:
        """Wrap *context* as this identity's shared persistent context.

        The shared context takes one slot of the cap, so it is refused when the live isolated
        contexts already fill every slot.

        :raises BrowserContextError: when the cap leaves no room for the shared context.
        """
        if self._shared is None and len(self._isolated) >= self._max_contexts:
            msg = f"The browser identity already owns {self._max_contexts} contexts, the limit."
            raise BrowserContextError(msg)
        if self._shared is None or self._shared.context is not context:
            self.context_created_total += 1
        self._shared = BrowserContextHandle(session_id=None, context=context, owner=self._owner)
        return self._shared

    def shared(self) -> BrowserContextHandle | None:
        """Return the shared handle, or ``None`` while no context is bound."""
        return self._shared

    # -- Isolated session contexts --------------------------------------------------

    def isolated(self, session_id: str) -> BrowserContextHandle | None:
        """Return the isolated handle for *session_id*, or ``None`` when there is none."""
        return self._isolated.get(session_id)

    def isolated_handles(self) -> list[BrowserContextHandle]:
        """Return the live isolated handles."""
        return list(self._isolated.values())

    def isolated_ids(self) -> tuple[str, ...]:
        """Return the ids of the live isolated contexts."""
        return tuple(self._isolated)

    async def get_or_create(
        self,
        session_id: str,
        create: Callable[[], Awaitable[PWBrowserCtx]],
    ) -> BrowserContextHandle:
        """Return the isolated context for *session_id*, creating it on first use.

        Creation runs under the manager lock so concurrent callers for one session id get
        one context instead of two, and so the identity's context cap is checked before the
        factory runs. Reuse at the cap still works, a refused creation never consumes a slot,
        and a failed or cancelled creation leaves the owned count unchanged.

        :raises BrowserContextError: when the session context is still closing or creating it
            would exceed the identity's context cap.
        """
        async with self._lock:
            if session_id in self._closing:
                msg = f"The browser context for session {session_id!r} is still closing."
                raise BrowserContextError(msg)
            existing = self._isolated.get(session_id)
            if existing is not None and not existing.context.is_closed():
                return existing
            if existing is not None:
                del self._isolated[session_id]
            if self._owned_count() >= self._max_contexts:
                msg = (
                    f"The browser identity already owns {self._max_contexts} contexts, the limit;"
                    f" cannot create session {session_id!r}."
                )
                raise BrowserContextError(msg)
            creation = asyncio.ensure_future(create())
            try:
                context = await asyncio.shield(creation)
            except asyncio.CancelledError:
                context = await creation
                await context.close()
                raise
            handle = BrowserContextHandle(session_id=session_id, context=context, owner=self._owner)
            self._isolated[session_id] = handle
            self.context_created_total += 1
            return handle

    async def close_isolated(self, session_id: str, *, evicted: bool = False) -> bool:
        """Close and forget the isolated context for *session_id*.

        Idempotent: an unknown or already closed session is a no-op, and the shared
        persistent context is never touched here. *evicted* marks this close as automatic
        cleanup retiring a context, counted once per removed handle after its native close; a
        failed close keeps the handle and counts nothing, and an explicit close never counts.

        :return: whether a live context was closed.
        """
        async with self._lock:
            handle = self._isolated.get(session_id)
            if handle is None:
                return False
            self._closing.add(session_id)
            await handle.context.close()
            del self._isolated[session_id]
            self._closing.discard(session_id)
            if evicted:
                self.context_evicted_total += 1
            return True

    async def close_all_isolated(self) -> None:
        """Close every isolated context this manager still owns."""
        for session_id in self.isolated_ids():
            await self.close_isolated(session_id)

    def require_own(self, handle: BrowserContextHandle) -> BrowserContextHandle:
        """Return *handle* when this manager owns it.

        :raises BrowserContextError: for a handle from another browser identity.
        """
        if not isinstance(handle, BrowserContextHandle) or handle.owner is not self._owner:
            msg = "Context handle belongs to a different browser identity."
            raise BrowserContextError(msg)
        current = self._shared if handle.session_id is None else self._isolated.get(handle.session_id)
        if current is not handle or handle.context.is_closed() or handle.session_id in self._closing:
            msg = f"The browser context for session {handle.session_id!r} is closed or stale."
            raise BrowserContextError(msg)
        return handle

    def reset(self) -> None:
        """Forget every handle without touching the contexts.

        Used when the browser itself is going away: its contexts die with it, and the next
        generation binds its own. The cumulative counters survive, because they describe the
        identity rather than the generation.
        """
        self._shared = None
        self._isolated.clear()
        self._closing.clear()


__all__ = [
    "BrowserContextHandle",
    "BrowserContextManager",
    "apply_human_patch",
    "persona_context_options",
    "require_session_id",
]
