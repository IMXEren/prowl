"""Logical session registry: names, modes, TTLs, capacity, and per-session leases.

Logical sessions are not isolated browser profiles. They are names layered over
browser contexts: a ``shared`` session uses the one persistent profile while an
``isolated`` session names its own context inside the same browser process. The
registry serializes work per session id with an async lease held across the whole
backend fetch, so a context is never navigated concurrently.

Destroy and expiry fence the id before any cleanup runs. The id stays unavailable
until active leases drain and the cleanup callback finishes, so a re-created
session cannot overlap the older backend cleanup. New leases during destroy fail,
while leases admitted before the fence drain normally. Cleanup runs in its own
shielded task, so cancelling a destroy waiter never abandons it. A failed cleanup
leaves an unavailable tombstone that only :meth:`SessionRegistry.retry_cleanup`
can clear.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Final, Literal, cast

from loguru import logger

from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME
from prowl.service.errors import SessionError, SessionLimitError, SessionNotFoundError

_SECONDS_PER_MINUTE = 60.0

#: How a logical session maps onto browser contexts.
type SessionMode = Literal["shared", "isolated"]

#: Why a session's cleanup ran: an explicit destroy, automatic retirement, or shutdown.
type SessionCloseReason = Literal["destroy", "evicted", "shutdown"]

#: The default mode: one shared persistent profile.
SHARED_MODE: Final[SessionMode] = "shared"

#: The opt-in mode: a dedicated context beside the persistent one.
ISOLATED_MODE: Final[SessionMode] = "isolated"

#: Called once per destroyed or expired id to release that session's backend resources.
type SessionCleanup = Callable[[SessionInfo], Awaitable[None]]

_MODES: Final[frozenset[str]] = frozenset({"shared", "isolated"})


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Immutable metadata for one logical session generation.

    ``egress`` is ``None`` while a shared session created without an explicit
    egress stays unbound; the first use that names one binds it for good.
    """

    id: str
    mode: SessionMode
    egress: str | None
    ttl_minutes: int | None
    #: Whether this cleanup is automatic retirement (TTL expiry or admission eviction) rather
    #: than an explicit destroy or shutdown.
    evicted: bool = False
    #: The origin of this cleanup attempt, immutable across retries.
    close_reason: SessionCloseReason = "destroy"


class _Unset(enum.Enum):
    VALUE = "unset"


_UNSET: Final = _Unset.VALUE


class _Status(enum.Enum):
    """Lifecycle state of one session entry."""

    LIVE = "live"
    CLOSING = "closing"
    FAILED = "failed"


@dataclass(slots=True)
class _Entry:
    """One live logical session with its serialization lock and lease count."""

    session_id: str
    mode: SessionMode
    created_at: float
    expires_at: float | None
    ttl_minutes: int | None
    egress: str | None
    last_used: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: int = 0
    status: _Status = _Status.LIVE
    destroying: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None
    close_task: asyncio.Task[None] | None = None
    #: Whether the in-flight cleanup is automatic retirement rather than an explicit close.
    evicted: bool = False
    #: The origin of the in-flight cleanup attempt, immutable across retries.
    close_reason: SessionCloseReason = "destroy"

    def info(self) -> SessionInfo:
        """Return an immutable snapshot of this generation's metadata."""
        return SessionInfo(
            id=self.session_id,
            mode=self.mode,
            egress=self.egress,
            ttl_minutes=self.ttl_minutes,
            evicted=self.evicted,
            close_reason=self.close_reason,
        )


class SessionRegistry:
    """Track logical session names, TTL expiry, capacity, and per-session leases."""

    def __init__(
        self,
        *,
        max_sessions: int,
        cleanup: SessionCleanup | None = None,
        isolated_per_egress_limit: int | None = None,
        default_isolated_ttl_minutes: int | None = None,
    ) -> None:
        """Create a registry bounded to *max_sessions* live generations.

        *cleanup* is awaited once per destroyed or expired id, with that
        generation's metadata, to release its backend resources.

        *isolated_per_egress_limit* bounds isolated sessions sharing one egress.
        ``None`` keeps the legacy behaviour where a full registry always rejects a
        new session; a non-negative value instead evicts the oldest idle session to
        make room, and ``0`` disables isolated admissions entirely.

        *default_isolated_ttl_minutes* is the idle TTL a new isolated session gets when
        its caller omits one. ``None`` leaves it unlimited, which is how shared sessions
        always start; an explicit TTL still wins and an explicit ``None`` still clears it.
        """
        if isolated_per_egress_limit is not None and isolated_per_egress_limit < 0:
            msg = "isolated_per_egress_limit must be non-negative"
            raise ValueError(msg)
        if default_isolated_ttl_minutes is not None and default_isolated_ttl_minutes < 1:
            msg = "default_isolated_ttl_minutes must be a positive number of minutes"
            raise ValueError(msg)
        self.max_sessions = max(1, int(max_sessions))
        self.isolated_per_egress_limit = isolated_per_egress_limit
        self.default_isolated_ttl_minutes = default_isolated_ttl_minutes
        self._cleanup = cleanup
        self._entries: dict[str, _Entry] = {}
        self._closing: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        #: Set once shutdown starts; no new generation may be admitted afterwards.
        self._shutdown = False

    async def ensure(
        self,
        session_id: str,
        ttl_minutes: int | _Unset | None = _UNSET,
        *,
        mode: str | None = None,
        egress: str | _Unset | None = _UNSET,
    ) -> SessionInfo:
        """Create *session_id* if absent, refresh it otherwise, and return its metadata.

        Mode and egress bind once: an existing session keeps them and a mismatch is
        an error. An omitted TTL preserves the configured TTL and refreshes the idle
        deadline from it; an explicit ``None`` clears the TTL. A shared session
        created without an explicit egress stays unbound until a use names one.
        Waits out a concurrent destroy so a re-created session cannot overlap the
        older lease.

        :raises SessionLimitError: when a new session would exceed the cap.
        :raises SessionError: when mode or egress disagrees with the existing session.
        """
        resolved_mode = _resolve_mode(mode)
        while True:
            waiter: asyncio.Event | None = None
            victim: _Entry | None = None
            async with self._lock:
                self._refuse_after_shutdown()
                now = time.monotonic()
                self._expire_due(now)
                entry = self._entries.get(session_id)
                if entry is not None and entry.status is _Status.LIVE:
                    self._refresh(entry, now, resolved_mode, egress, ttl_minutes)
                    return entry.info()
                waiter = self._fenced_waiter(session_id, entry)
                if waiter is None:
                    victim = self._plan_admission(resolved_mode, egress)
                    if victim is None:
                        return self._create_entry(session_id, now, resolved_mode, egress, ttl_minutes).info()
                    self._start_close(victim.session_id, victim, by_destroy=False, evicted=True)
            if waiter is not None:
                await waiter.wait()
            elif victim is not None:
                await self._await_close(victim)

    async def create(
        self,
        session_id: str | None,
        ttl_minutes: int | _Unset | None = _UNSET,
        *,
        mode: str | None = None,
        egress: str | None = None,
    ) -> str:
        """Create a named or generated session and return its id.

        An omitted TTL leaves a new isolated session on the configured default and a
        shared session unlimited; an explicit ``None`` creates it with no expiry.

        :raises SessionError: when the named session already exists.
        :raises SessionLimitError: when the cap is already reached.
        """
        resolved_mode = _resolve_mode(mode)
        while True:
            waiter: asyncio.Event | None = None
            victim: _Entry | None = None
            async with self._lock:
                self._refuse_after_shutdown()
                now = time.monotonic()
                self._expire_due(now)
                resolved = session_id or uuid.uuid4().hex
                entry = self._entries.get(resolved)
                if entry is not None and entry.status is _Status.LIVE:
                    msg = f"session already exists: {resolved}"
                    raise SessionError(msg)
                waiter = self._fenced_waiter(resolved, entry)
                if waiter is None:
                    victim = self._plan_admission(resolved_mode, egress)
                    if victim is None:
                        self._create_entry(resolved, now, resolved_mode, egress, ttl_minutes)
                        return resolved
                    self._start_close(victim.session_id, victim, by_destroy=False, evicted=True)
            if waiter is not None:
                await waiter.wait()
            elif victim is not None:
                await self._await_close(victim)

    async def destroy(self, session_id: str) -> None:
        """Fence *session_id*, drain its leases, release its resources, then forget it.

        Destroying an id whose earlier cleanup failed retries that cleanup rather than
        telling the caller about a retry it has no way to reach.

        :raises SessionNotFoundError: when the session is unknown or already expired.
        :raises SessionError: when the cleanup failed again.
        """
        entry = await self._fence_for_destroy(session_id)
        await self._await_close(entry)

    @asynccontextmanager
    async def lease(
        self,
        session_id: str | None,
        *,
        mode: str | None = None,
        egress: str | _Unset | None = _UNSET,
    ) -> AsyncIterator[SessionInfo | None]:
        """Serialize work for *session_id* and yield the leased generation's metadata.

        A ``None`` session is not serialized here; the service owns the dedicated
        anonymous-request lock. The metadata is captured when the lease is admitted, so
        callers build backend requests inside the context rather than from an earlier
        ``ensure``.

        *mode* is checked against the generation actually admitted, so a destroy and
        recreate between an ``ensure`` and this lease cannot smuggle a stale mode through.
        *egress* binds once or verifies an existing binding: a name binds an unbound
        session and rejects a different binding, ``None`` selects the existing binding or
        the default egress, and omitting it leaves the binding untouched.

        :raises SessionNotFoundError: when the session is unknown, expired, or closing.
        :raises SessionError: when mode or egress disagrees with the admitted generation.
        """
        if session_id is None:
            yield None
            return
        resolved_mode = _resolve_mode(mode)
        async with self._lock:
            self._refuse_after_shutdown()
            now = time.monotonic()
            self._expire_due(now)
            entry = self._entries.get(session_id)
            if entry is None or entry.status is not _Status.LIVE:
                if entry is not None and entry.status is _Status.FAILED:
                    msg = f"session {session_id} cleanup failed and it cannot be reused"
                    raise SessionError(msg)
                closing = self._closing.get(session_id)
                if closing is not None and closing.status is _Status.FAILED:
                    msg = f"session {session_id} cleanup failed and it cannot be reused"
                    raise SessionError(msg)
                msg = f"unknown session: {session_id}"
                raise SessionNotFoundError(msg)
            if resolved_mode is not None and resolved_mode != entry.mode:
                msg = f"session {session_id} already exists as {entry.mode!r}"
                raise SessionError(msg)
            if egress is not _Unset.VALUE:
                _bind_egress(entry, egress)
            entry.active += 1
            entry.last_used = now
            entry.drained.clear()
            lease_entry = entry
            info = entry.info()
            lock = entry.lock
        try:
            async with lock:
                yield info
        finally:
            async with self._lock:
                lease_entry.active -= 1
                if lease_entry.active <= 0:
                    lease_entry.last_used = time.monotonic()
                    # Use is a refresh: the idle countdown restarts once the last active
                    # operation finishes, so a long operation does not expire on return.
                    if lease_entry.ttl_minutes is not None:
                        lease_entry.expires_at = _deadline(time.monotonic(), lease_entry.ttl_minutes)
                    lease_entry.drained.set()

    async def list_sessions(self) -> list[str]:
        """Return the ids of live sessions, omitting expired and cleaned-up ones."""
        async with self._lock:
            self._expire_due(time.monotonic())
            return sorted(self._entries)

    async def purge_expired(self) -> list[str]:
        """Expire idle sessions now, run their cleanup, and return the expired ids.

        The service owns the periodic call; the registry holds no timer. Every expired id's
        cleanup is awaited before a failure is re-raised, so the caller logs a real failure
        instead of an empty success.

        :raises BaseException: the first cleanup failure, after every expired id is awaited.
        """
        async with self._lock:
            expired = self._expire_due(time.monotonic())
            entries = [self._closing[session_id] for session_id in expired if session_id in self._closing]
        failure: BaseException | None = None
        for entry in entries:
            try:
                await self._await_close(entry)
            except Exception as exc:  # noqa: BLE001 - every expired id is still awaited
                failure = failure or exc
        if failure is not None:
            raise failure
        return expired

    async def drain(self) -> None:
        """Wait until no session lease is active, so resources can be closed in order."""
        while True:
            async with self._lock:
                pending = [entry.drained for entry in self._entries.values() if entry.active > 0]
                if not pending:
                    return
            for event in pending:
                await event.wait()

    async def aclose(self) -> None:
        """Fence every live session, drain its leases, and run each cleanup once.

        Shutdown ownership: the service calls this rather than the registry holding a
        timer. Shutdown also fences the registry permanently, so no new generation can be
        admitted while it runs. Every cleanup is awaited before a failure is re-raised.

        :raises BaseException: the first cleanup failure, after every cleanup is awaited.
        """
        async with self._lock:
            self._shutdown = True
            self._expire_due(time.monotonic())
            for session_id in list(self._entries):
                entry = self._entries[session_id]
                if entry.status is _Status.LIVE:
                    self._start_close(session_id, entry, by_destroy=False, shutdown=True)
            entries = [*self._entries.values(), *self._closing.values()]
        seen: set[int] = set()
        failure: BaseException | None = None
        for entry in entries:
            if id(entry) in seen:
                continue
            seen.add(id(entry))
            try:
                await self._await_close(entry)
            except Exception as exc:  # noqa: BLE001 - every cleanup is awaited before raising
                failure = failure or exc
        if failure is not None:
            raise failure

    async def retry_cleanup(self, session_id: str) -> None:
        """Re-run a failed cleanup for *session_id*, keeping the id fenced meanwhile.

        :raises SessionNotFoundError: when no entry or tombstone exists for *session_id*.
        :raises SessionError: when the cleanup is not in a failed state.
        """
        async with self._lock:
            entry = self._closing.get(session_id) or self._entries.get(session_id)
            if entry is None:
                msg = f"unknown session: {session_id}"
                raise SessionNotFoundError(msg)
            if entry.status is not _Status.FAILED:
                msg = f"session {session_id} cleanup is already in progress"
                raise SessionError(msg)
            self._restart_close(session_id, entry, by_destroy=entry.destroying)
        await self._await_close(entry)

    # -- internals ------------------------------------------------------------------

    async def _fence_for_destroy(self, session_id: str) -> _Entry:
        async with self._lock:
            self._expire_due(time.monotonic())
            entry = self._entries.get(session_id)
            if entry is not None:
                if entry.status is _Status.LIVE:
                    self._start_close(session_id, entry, by_destroy=True)
                elif entry.status is _Status.FAILED:
                    self._restart_close(session_id, entry, by_destroy=True)
                return entry
            closing = self._closing.get(session_id)
            if closing is None:
                msg = f"unknown session: {session_id}"
                raise SessionNotFoundError(msg)
            if closing.status is _Status.FAILED:
                self._restart_close(session_id, closing, by_destroy=True)
            return closing

    async def _await_close(self, entry: _Entry) -> None:
        """Await *entry*'s cleanup and re-raise the failure it recorded, if any.

        A cleanup task cancelled on its own is a recorded tombstone, so its error is raised. A
        caller cancelled while waiting leaves the task running with no error recorded, so the
        cancellation propagates instead.
        """
        task = entry.close_task
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if entry.error is None:
                    raise
        if entry.error is not None:
            raise entry.error

    def _refuse_after_shutdown(self) -> None:
        """Reject admission once shutdown started. Caller holds the lock."""
        if self._shutdown:
            msg = "session registry is shutting down"
            raise SessionError(msg)

    def _fenced_waiter(self, session_id: str, entry: _Entry | None) -> asyncio.Event | None:
        """Return the event to wait on when *session_id* is fenced, else ``None``.

        An ``_entries`` entry that is not live is a destroy in progress; an
        ``_closing`` entry is an expiry or a completed destroy. A failed tombstone is
        unavailable and raises instead of waiting on its already-set event, which would
        spin the caller forever.
        """
        if entry is not None:
            if entry.status is _Status.FAILED:
                msg = f"session {session_id} cleanup failed and it cannot be reused"
                raise SessionError(msg)
            return entry.closed
        closing = self._closing.get(session_id)
        if closing is None:
            return None
        if closing.status is _Status.FAILED:
            msg = f"session {session_id} cleanup failed and it cannot be reused"
            raise SessionError(msg)
        return closing.closed

    def _create_entry(
        self,
        session_id: str,
        now: float,
        mode: SessionMode | None,
        egress: str | _Unset | None,
        ttl_minutes: int | _Unset | None,
    ) -> _Entry:
        resolved_mode = mode or SHARED_MODE
        if ttl_minutes is _Unset.VALUE:
            resolved_ttl = self.default_isolated_ttl_minutes if resolved_mode == ISOLATED_MODE else None
        else:
            resolved_ttl = ttl_minutes
        resolved_egress = None if egress is _Unset.VALUE else egress
        entry = _Entry(
            session_id=session_id,
            mode=resolved_mode,
            created_at=now,
            expires_at=_deadline(now, resolved_ttl),
            ttl_minutes=resolved_ttl,
            egress=_new_egress(resolved_mode, resolved_egress),
            last_used=now,
        )
        self._entries[session_id] = entry
        return entry

    def _refresh(
        self,
        entry: _Entry,
        now: float,
        mode: SessionMode | None,
        egress: str | _Unset | None,
        ttl_minutes: int | _Unset | None,
    ) -> None:
        """Refresh a live entry in place, binding mode and egress only on first use."""
        if mode is not None and mode != entry.mode:
            msg = f"session {entry.session_id} already exists as {entry.mode!r}"
            raise SessionError(msg)
        if egress is not _Unset.VALUE and egress is not None:
            if entry.egress is None:
                entry.egress = egress
            elif entry.egress != egress:
                msg = f"session {entry.session_id} is bound to a different egress"
                raise SessionError(msg)
        if ttl_minutes is _Unset.VALUE:
            entry.expires_at = _deadline(now, entry.ttl_minutes)
        else:
            entry.ttl_minutes = ttl_minutes
            entry.expires_at = _deadline(now, ttl_minutes)
        entry.last_used = now

    def _start_close(
        self,
        session_id: str,
        entry: _Entry,
        *,
        by_destroy: bool,
        evicted: bool = False,
        shutdown: bool = False,
    ) -> None:
        """Fence a live entry and start its cleanup driver. Caller holds the lock.

        *evicted* marks automatic retirement (TTL expiry or admission eviction) rather than an
        explicit destroy or shutdown; *shutdown* marks registry close rather than an explicit
        destroy. Either way the origin is recorded in the entry's ``close_reason``.
        """
        entry.status = _Status.CLOSING
        entry.destroying = by_destroy
        entry.evicted = evicted
        entry.close_reason = "evicted" if evicted else "shutdown" if shutdown else "destroy"
        if entry.active <= 0:
            entry.drained.set()
        entry.close_task = asyncio.create_task(self._drive_close(session_id, entry))

    def _restart_close(self, session_id: str, entry: _Entry, *, by_destroy: bool) -> None:
        """Re-run a failed cleanup for *entry*. Caller holds the lock.

        The entry's automatic-retirement marker and close reason are preserved, so retrying a
        failed eviction cleanup (even through a destroy) does not turn it into an explicit close.
        """
        entry.error = None
        entry.status = _Status.CLOSING
        entry.destroying = by_destroy
        entry.closed = asyncio.Event()
        if entry.active <= 0:
            entry.drained.set()
        else:
            entry.drained.clear()
        entry.close_task = asyncio.create_task(self._drive_close(session_id, entry))

    async def _drive_close(self, session_id: str, entry: _Entry) -> None:
        """Drain leases, run cleanup once, and settle the entry out of the live set.

        A failure leaves an unavailable tombstone and records the error on the entry; the
        driver returns normally so every caller observes the failure through
        :meth:`_await_close` and an opportunistic expiry started by a read is still logged.
        """
        try:
            await entry.drained.wait()
            async with self._lock:
                if self._entries.get(session_id) is entry:
                    del self._entries[session_id]
                self._closing[session_id] = entry
            if self._cleanup is not None:
                await self._cleanup(entry.info())
        except asyncio.CancelledError:
            async with self._lock:
                entry.status = _Status.FAILED
                entry.error = SessionError(f"session {session_id} cleanup was cancelled")
            entry.closed.set()
            logger.error(f"Session {session_id} cleanup was cancelled; the id stays fenced.")
        except Exception as exc:  # noqa: BLE001 - any cleanup failure fences the id
            async with self._lock:
                entry.status = _Status.FAILED
                entry.error = exc
            entry.closed.set()
            logger.error(f"Session {session_id} cleanup failed: {type(exc).__name__}")
        else:
            async with self._lock:
                if self._closing.get(session_id) is entry:
                    del self._closing[session_id]
            entry.closed.set()

    def _expire_due(self, now: float) -> list[str]:
        """Fence idle expired entries and start their cleanup. Caller holds the lock."""
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if entry.status is _Status.LIVE and entry.active <= 0 and _is_expired(entry, now)
        ]
        for session_id in expired:
            entry = self._entries.pop(session_id)
            self._closing[session_id] = entry
            self._start_close(session_id, entry, by_destroy=False, evicted=True)
        return expired

    def _admit_new(self) -> None:
        if len(self._entries) + len(self._closing) >= self.max_sessions:
            msg = "session limit reached; destroy an existing session before creating another"
            raise SessionLimitError(msg)

    def _plan_admission(self, mode: SessionMode | None, egress: str | _Unset | None) -> _Entry | None:
        """Reserve capacity for a new entry. Caller holds the lock.

        Returns ``None`` when there is room, or the idle entry to evict first. Without
        an isolated budget a full registry raises instead, keeping the legacy
        no-eviction behaviour. Global pressure admits any mode; isolated pressure
        admits only an isolated session bound to the same effective egress.

        :raises SessionLimitError: when full and no idle session can be evicted.
        """
        if self.isolated_per_egress_limit is None:
            self._admit_new()
            return None
        resolved_mode = mode or SHARED_MODE
        resolved_egress = None if egress is _Unset.VALUE else egress
        effective_egress = _new_egress(resolved_mode, resolved_egress)
        global_full = len(self._entries) + len(self._closing) >= self.max_sessions
        isolated_full = (
            resolved_mode == ISOLATED_MODE and self._isolated_count(effective_egress) >= self.isolated_per_egress_limit
        )
        if not global_full and not isolated_full:
            return None
        victim = self._select_victim(global_full=global_full and not isolated_full, egress=effective_egress)
        if victim is None:
            msg = "session limit reached; destroy an existing session before creating another"
            raise SessionLimitError(msg)
        return victim

    def _select_victim(self, *, global_full: bool, egress: str | None) -> _Entry | None:
        """Return the oldest idle session eligible for eviction. Caller holds the lock.

        Global pressure admits any mode; isolated pressure admits only an isolated
        session bound to *egress*.
        """
        oldest: _Entry | None = None
        for entry in self._entries.values():
            if entry.status is not _Status.LIVE or entry.active > 0:
                continue
            if not global_full and not (entry.mode == ISOLATED_MODE and entry.egress == egress):
                continue
            if oldest is None or entry.last_used < oldest.last_used:
                oldest = entry
        return oldest

    def _isolated_count(self, egress: str | None) -> int:
        """Count isolated generations bound to *egress*, including pending cleanup."""
        count = 0
        for entry in (*self._entries.values(), *self._closing.values()):
            if entry.mode == ISOLATED_MODE and entry.egress == egress:
                count += 1
        return count


def _resolve_mode(mode: str | None) -> SessionMode | None:
    if mode is None:
        return None
    if mode not in _MODES:
        msg = f"unknown session mode: {mode!r}"
        raise SessionError(msg)
    return cast("SessionMode", mode)


def _bind_egress(entry: _Entry, egress: str | None) -> None:
    """Bind *entry* to *egress*, or verify its existing binding. Caller holds the lock.

    ``None`` selects the existing binding, or the default egress when the entry is unbound.
    """
    if egress is None:
        if entry.egress is None:
            entry.egress = DEFAULT_EGRESS_NAME
        return
    if entry.egress is None:
        entry.egress = egress
    elif entry.egress != egress:
        msg = f"session {entry.session_id} is bound to a different egress"
        raise SessionError(msg)


def _new_egress(mode: SessionMode, egress: str | None) -> str | None:
    """Return the egress a new session binds at creation.

    A shared session stays unbound when no egress is named; an isolated session
    always names a browser process, so it falls back to the default egress.
    """
    if egress is not None:
        return egress
    if mode == ISOLATED_MODE:
        return DEFAULT_EGRESS_NAME
    return None


def _deadline(now: float, ttl_minutes: int | None) -> float | None:
    if ttl_minutes is None:
        return None
    return now + ttl_minutes * _SECONDS_PER_MINUTE


def _is_expired(entry: _Entry, now: float) -> bool:
    return entry.expires_at is not None and entry.expires_at <= now


__all__ = [
    "ISOLATED_MODE",
    "SHARED_MODE",
    "SessionCleanup",
    "SessionCloseReason",
    "SessionInfo",
    "SessionMode",
    "SessionRegistry",
]
