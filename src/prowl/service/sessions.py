"""Logical session registry: names, TTLs, capacity, and per-session leases.

Logical sessions are not isolated browser profiles. They are names layered
over the one shared persistent profile; the registry serializes work per
session id with an async lease held across the whole backend fetch, so the
shared profile is never navigated concurrently. Destroy fences active work
before removal, expiry never removes a session that has active work, and a
re-created session cannot overlap an older lease for the same id. The final
lease out of a destroying entry removes that entry under the registry lock
before releasing waiters, so a waiter never re-observes a dead entry.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prowl.browser.egress import DEFAULT_EGRESS_NAME
from prowl.service.errors import SessionError, SessionLimitError, SessionNotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_SECONDS_PER_MINUTE = 60.0


@dataclass(slots=True)
class _Entry:
    """One live logical session with its serialization lock and lease count."""

    created_at: float
    expires_at: float | None
    egress: str = DEFAULT_EGRESS_NAME
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: int = 0
    destroying: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)


class SessionRegistry:
    """Track logical session names, TTL expiry, and per-session leases."""

    def __init__(self, *, max_sessions: int) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def ensure(
        self,
        session_id: str,
        ttl_minutes: int | None,
        *,
        egress: str = DEFAULT_EGRESS_NAME,
    ) -> None:
        """Create *session_id* if absent, refresh its TTL otherwise.

        Waits out a concurrent destroy so a re-created session cannot overlap
        the older lease for the same id. A session is bound to the egress of its
        first use, because serializing it only means something within one browser
        process.

        :raises SessionLimitError: when a new session would exceed the cap.
        :raises SessionError: when the session is already bound to another egress.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_expired(now)
                entry = self._entries.get(session_id)
                if entry is not None and not entry.destroying:
                    if entry.egress != egress:
                        msg = f"session {session_id} is bound to a different egress"
                        raise SessionError(msg)
                    entry.expires_at = _deadline(now, ttl_minutes)
                    return
                if entry is None:
                    self._admit_new()
                    self._entries[session_id] = _Entry(
                        created_at=now,
                        expires_at=_deadline(now, ttl_minutes),
                        egress=egress,
                    )
                    return
                drained = entry.drained
            await drained.wait()

    async def create(
        self,
        session_id: str | None,
        ttl_minutes: int | None,
        *,
        egress: str = DEFAULT_EGRESS_NAME,
    ) -> str:
        """Create a named or generated session and return its id.

        :raises SessionError: when the named session already exists.
        :raises SessionLimitError: when the cap is already reached.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_expired(now)
                resolved = session_id or uuid.uuid4().hex
                entry = self._entries.get(resolved)
                if entry is None:
                    self._admit_new()
                    self._entries[resolved] = _Entry(
                        created_at=now,
                        expires_at=_deadline(now, ttl_minutes),
                        egress=egress,
                    )
                    return resolved
                if not entry.destroying:
                    msg = f"session already exists: {resolved}"
                    raise SessionError(msg)
                drained = entry.drained
            await drained.wait()

    async def destroy(self, session_id: str) -> None:
        """Fence active work for *session_id*, then remove it.

        :raises SessionNotFoundError: when the session is unknown or expired.
        """
        async with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            entry = self._entries.get(session_id)
            if entry is None:
                msg = f"unknown session: {session_id}"
                raise SessionNotFoundError(msg)
            entry.destroying = True
            if entry.active <= 0:
                del self._entries[session_id]
                return
            drained = entry.drained
        await drained.wait()
        async with self._lock:
            if self._entries.get(session_id) is entry:
                del self._entries[session_id]

    @asynccontextmanager
    async def lease(self, session_id: str | None) -> AsyncIterator[None]:
        """Serialize work for *session_id* for the duration of the context.

        A ``None`` session is not serialized here; the service owns the
        dedicated anonymous-request lock.

        :raises SessionNotFoundError: when the session is unknown or expired.
        """
        if session_id is None:
            yield
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._purge_expired(now)
                entry = self._entries.get(session_id)
                if entry is None:
                    msg = f"unknown session: {session_id}"
                    raise SessionNotFoundError(msg)
                if not entry.destroying:
                    entry.active += 1
                    lease_entry = entry
                    lock = entry.lock
                    break
                drained = entry.drained
            await drained.wait()
        try:
            async with lock:
                yield
        finally:
            async with self._lock:
                lease_entry.active -= 1
                if lease_entry.active <= 0 and lease_entry.destroying:
                    if self._entries.get(session_id) is lease_entry:
                        del self._entries[session_id]
                    lease_entry.drained.set()

    async def list_sessions(self) -> list[str]:
        """Return the ids of live sessions, omitting expired ones."""
        async with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            return sorted(self._entries)

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if not entry.destroying and entry.active <= 0 and _is_expired(entry, now)
        ]
        for session_id in expired:
            del self._entries[session_id]

    def _admit_new(self) -> None:
        if len(self._entries) >= self.max_sessions:
            msg = "session limit reached; destroy an existing session before creating another"
            raise SessionLimitError(msg)


def _deadline(now: float, ttl_minutes: int | None) -> float | None:
    if ttl_minutes is None:
        return None
    return now + ttl_minutes * _SECONDS_PER_MINUTE


def _is_expired(entry: _Entry, now: float) -> bool:
    return entry.expires_at is not None and entry.expires_at <= now


__all__ = ["SessionRegistry"]
