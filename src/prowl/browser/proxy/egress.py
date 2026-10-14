"""Named egresses: one browser process per configured egress.

A configured egress is a named proxy that a browser is launched with. The egress
named :data:`DEFAULT_EGRESS_NAME` is the process-wide one configured by
``PROWL_PROXY_URL``, and it keeps using the
:class:`~prowl.browser.browser.Browser` singleton, so a deployment that configures
a single proxy behaves exactly as it did before named egresses existed.

Every other egress is a :class:`~prowl.browser.browser.Browser` subclass whose
class-level browser state is its own. Subclassing rather than copying keeps one
implementation of startup, tab groups, and shutdown; each egress merely carries
its own runtime, lifecycle, profile directory, and profile archive, so each keeps
its own warm trust, cookies, and clearance. An egress browser starts on first use
and shuts down again once it has been idle, so an unused egress costs no memory.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger

from prowl.browser.browser import Browser
from prowl.browser.config import DEFAULT_MAX_CONTEXTS, PROXY_URL_ENV, BrowserConfig
from prowl.browser.driver import BrowserRuntimeState
from prowl.browser.lifecycle import BrowserLifecycle

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Name of the egress configured process-wide by ``PROWL_PROXY_URL``.
DEFAULT_EGRESS_NAME: Final[str] = "default"

#: Environment variable holding the ``name=url`` list of named egresses.
EGRESSES_ENV: Final[str] = "PROWL_EGRESSES"

#: Environment variable holding how long a named egress may stay idle.
EGRESS_IDLE_SECONDS_ENV: Final[str] = "PROWL_EGRESS_IDLE_SECONDS"

#: A named egress browser is shut down after this long without work.
DEFAULT_EGRESS_IDLE_SECONDS: Final[float] = 300.0

#: Egress names become path components, so they use a deliberately narrow alphabet.
_EGRESS_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Preferred CDP port for the first named egress; the default browser keeps 9222.
_CDP_PORT_BASE: Final[int] = 9300

#: Spacing between preferred CDP ports so two egresses cannot pick the same one.
_CDP_PORT_STRIDE: Final[int] = 110

#: Native counters summed over a pool's whole lifetime, retired egresses included.
_COUNTER_METRICS: Final = (
    "context_created_total",
    "context_evicted_total",
    "browser_restart_total",
)

#: Native gauges summed over only the egresses a pool currently has live.
_GAUGE_METRICS: Final = ("context_count", "tabgroups_active")


def _owner_resource_metrics(owner: type[Browser]) -> dict[str, int]:
    """Return *owner*'s native resource snapshot, or zeros for a legacy owner.

    Only the fixed metric names are read. A fake or legacy owner that predates the
    native counters contributes nothing rather than failing the aggregation.
    """
    getter = getattr(owner, "resource_metrics", None)
    if getter is None:
        return dict.fromkeys(_COUNTER_METRICS + _GAUGE_METRICS, 0)
    return getter()


class EgressError(ValueError):
    """A named egress is malformed or missing."""


def parse_egress_spec(raw: str) -> dict[str, str]:
    """Parse a comma separated ``name=url`` list into a name to url mapping.

    Only the shape is checked here; the proxy URL policy is validated by the
    caller that owns proxy configuration, so a single place decides which
    schemes and credentials are acceptable.

    :raises EgressError: when an entry is malformed, duplicated, or uses a
        reserved or path-unsafe name.
    """
    egresses: dict[str, str] = {}
    for raw_entry in raw.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        name, separator, url = entry.partition("=")
        name = name.strip()
        url = url.strip()
        if not separator or not name or not url:
            msg = f"{EGRESSES_ENV} entries must be name=url pairs"
            raise EgressError(msg)
        _check_egress_name(name)
        if name in egresses:
            msg = f"duplicate egress name: {name}"
            raise EgressError(msg)
        egresses[name] = url
    return egresses


def derive_egress_paths(
    profile_dir: str,
    profile_archive: str,
    name: str,
) -> tuple[str, str]:
    """Return the profile directory and archive that belong to *name*.

    The default egress keeps the configured paths untouched. Every other egress
    gets its own directory beside the configured one and its own archive beside
    the configured archive, so profiles are never shared between egresses and no
    egress profile is nested inside another's tree.
    """
    if name == DEFAULT_EGRESS_NAME:
        return profile_dir, profile_archive
    directory = Path(profile_dir)
    archive = Path(profile_archive)
    return (
        str(directory.with_name(f"{directory.name}-{name}")),
        str(archive.with_name(f"{archive.stem}-{name}{archive.suffix}")),
    )


def _check_egress_name(name: str) -> None:
    """Validate the same portable path component for environment and direct configuration."""
    if not _EGRESS_NAME_PATTERN.fullmatch(name) or name.endswith("."):
        msg = "invalid egress name: use letters, digits, dot, dash, or underscore"
        raise EgressError(msg)
    if name == DEFAULT_EGRESS_NAME:
        msg = f"egress name {DEFAULT_EGRESS_NAME!r} is reserved for {PROXY_URL_ENV}"
        raise EgressError(msg)


def _owner_path(raw: str) -> Path:
    """Return *raw* as a normalized absolute path, reading only filesystem metadata."""
    return Path(raw).resolve(strict=False)


def _check_owners_disjoint(owners: list[tuple[Path, Path]]) -> None:
    """Reject profile roots and archives that let one identity reach another's files.

    Every entry is a resolved ``(profile_dir, profile_archive)`` pair. Two profile
    directories may not coincide or nest, two archives may not coincide, and an
    archive may not sit inside a different identity's profile directory. The checks
    compare native paths, so on a case-insensitive filesystem two paths that differ
    only in case are one path.

    :raises EgressError: when *owners* collide.
    """
    for index, (profile_dir, profile_archive) in enumerate(owners):
        for other_dir, other_archive in owners[index + 1 :]:
            if profile_dir.is_relative_to(other_dir) or other_dir.is_relative_to(profile_dir):
                msg = "egress profile directories must not coincide or contain each other"
                raise EgressError(msg)
            if profile_archive == other_archive:
                msg = "egress profile archives must not coincide"
                raise EgressError(msg)
            if profile_archive.is_relative_to(other_dir) or other_archive.is_relative_to(profile_dir):
                msg = "an egress profile archive must not live inside another egress profile"
                raise EgressError(msg)


# The arguments are the browser's whole identity: which egress it serves, which profile it
# owns, which port it debugs on, and which extensions the deployment loads.
def create_egress_browser(  # noqa: PLR0913
    *,
    name: str,
    proxy_url: str,
    profile_dir: str,
    profile_archive: str,
    preferred_cdp_port: int,
    extensions_dir: str | None = None,
    policy_dir: str | None = None,
    max_contexts: int = DEFAULT_MAX_CONTEXTS,
) -> type[Browser]:
    """Return a :class:`Browser` subclass whose state belongs to one egress.

    The subclass carries its own runtime, lifecycle, profile, and proxy, while
    every operation still comes from :class:`Browser`. The extensions and policy
    directories are deployment-wide, so every egress browser gets the same set.
    """
    max_groups = Browser._MAX_GROUPS  # noqa: SLF001
    egress_browser = type(
        f"EgressBrowser_{name}",
        (Browser,),
        {"_EGRESS_NAME": name, "_MAX_GROUPS": max_groups},
    )
    runtime = BrowserRuntimeState(max_groups=max_groups)
    lifecycle = BrowserLifecycle(
        runtime,
        profile_dir=profile_dir,
        profile_archive=Path(profile_archive),
        proxy_url=proxy_url,
        extensions_dir=extensions_dir,
        policy_dir=policy_dir,
    )
    lifecycle.cdp_port = preferred_cdp_port
    # The egress obeys the same configured context cap as the default identity; it owns
    # its own manager, so the limit is set before any launch rather than inherited.
    runtime.contexts.configure_limit(max_contexts)
    egress_browser._runtime = runtime  # noqa: SLF001
    egress_browser._lifecycle = lifecycle  # noqa: SLF001
    return egress_browser


@dataclass(slots=True)
class _EgressDefinition:
    """Everything needed to build one egress browser, resolved up front."""

    name: str
    proxy_url: str = field(repr=False)
    profile_dir: str
    profile_archive: str
    cdp_port: int
    extensions_dir: str | None = None
    policy_dir: str | None = None
    max_contexts: int = DEFAULT_MAX_CONTEXTS


@dataclass(slots=True)
class _LiveEgress:
    """A created egress browser with its work count and its tracked close.

    ``closing`` is set once the entry has committed to shutting the browser down. The entry
    stays listed until that shutdown returns successfully, so a request arriving meanwhile
    waits for it instead of starting a second browser against the same profile directory and
    archive. ``idle_task`` is only the idle delay: when it elapses it hands the entry to
    ``close_task`` and finishes, so cancelling the timer can never interrupt a close.

    ``close_task`` is the pool's single native close for this entry. It is owned by the pool
    and shielded from a caller's cancellation, and its result is that shutdown's failure or
    ``None`` when it succeeded. ``shutdown_done`` is set when the close finishes, whatever
    its outcome, so a waiting acquire wakes rather than parking forever.
    """

    browser: type[Browser]
    inflight: int = 0
    idle_task: asyncio.Task[None] | None = None
    closing: bool = False
    close_task: asyncio.Task[BaseException | None] | None = None
    shutdown_done: asyncio.Event = field(default_factory=asyncio.Event)


class EgressPool:
    """Own one browser per named egress, starting it on use and idling it out.

    Definitions are resolved at construction, so a bad configuration fails at
    startup rather than on the first request. Browsers themselves are created on
    first use and shut down after ``idle_seconds`` without work.
    """

    def __init__(
        self,
        base_config: BrowserConfig,
        egresses: Mapping[str, str],
        *,
        idle_seconds: float = DEFAULT_EGRESS_IDLE_SECONDS,
    ) -> None:
        self._idle_seconds = max(0.0, float(idle_seconds))
        self._definitions: dict[str, _EgressDefinition] = {}
        owners: list[tuple[Path, Path]] = [
            (_owner_path(base_config.profile_dir), _owner_path(base_config.profile_archive)),
        ]
        for index, name in enumerate(sorted(egresses)):
            _check_egress_name(name)
            profile_dir, profile_archive = derive_egress_paths(
                base_config.profile_dir,
                base_config.profile_archive,
                name,
            )
            owners.append((_owner_path(profile_dir), _owner_path(profile_archive)))
            self._definitions[name] = _EgressDefinition(
                name=name,
                proxy_url=egresses[name],
                profile_dir=profile_dir,
                profile_archive=profile_archive,
                cdp_port=_CDP_PORT_BASE + index * _CDP_PORT_STRIDE,
                extensions_dir=base_config.extensions_dir,
                policy_dir=base_config.policy_dir,
                max_contexts=base_config.max_contexts,
            )
        _check_owners_disjoint(owners)
        self._live: dict[str, _LiveEgress] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        # Cumulative counters of every owner this pool has retired, so they outlive
        # individual generations and the pool's own close.
        self._retired: dict[str, int] = dict.fromkeys(_COUNTER_METRICS, 0)

    def names(self) -> tuple[str, ...]:
        """Return the configured egress names."""
        return tuple(sorted(self._definitions))

    def live_names(self) -> tuple[str, ...]:
        """Return the names of egresses with a created browser.

        An egress whose browser is still shutting down is listed until that shutdown
        completes, because its profile directory and archive are still in use.
        """
        return tuple(sorted(self._live))

    def definition(self, name: str) -> _EgressDefinition:
        """Return the resolved definition for *name*.

        :raises EgressError: when *name* is not configured.
        """
        definition = self._definitions.get(name)
        if definition is None:
            msg = f"unknown egress: {name}"
            raise EgressError(msg)
        return definition

    def resource_metrics(self) -> dict[str, int]:
        """Return this pool's cumulative counters and its live gauges.

        The three cumulative counters cover every owner the pool has ever created,
        retired ones included, so idle retirement and ``aclose`` never reset them. The
        two gauges cover only the owners currently live, so a shut-down egress stops
        contributing to them. The snapshot reads native state only; it takes no lock and
        never probes the live browser.
        """
        snapshot: dict[str, int] = dict.fromkeys(_COUNTER_METRICS + _GAUGE_METRICS, 0)
        for name in _COUNTER_METRICS:
            snapshot[name] = self._retired[name]
        for entry in self._live.values():
            owner_metrics = _owner_resource_metrics(entry.browser)
            for name in (*_COUNTER_METRICS, *_GAUGE_METRICS):
                snapshot[name] += owner_metrics[name]
        return snapshot

    async def acquire(self, name: str) -> type[Browser]:
        """Return the browser for *name*, creating it on first use.

        Cancels a pending idle teardown and marks the egress busy. When the browser for
        this egress is still shutting down, the call waits for that shutdown to finish
        before creating a replacement, because the browser being closed still owns the
        profile directory and the profile archive. A close that finished without success
        keeps ownership of the egress, so the call raises instead of starting a replacement
        against a profile the failed close may still hold.

        :raises EgressError: when *name* is not configured, the pool has been closed, or the
            egress's last close failed.
        """
        while True:
            entry, usable = await self._claim(name)
            if usable:
                return entry.browser
            failure = self._close_failure(entry)
            if failure is not None:
                msg = f"egress {name!r} browser close failed; ownership is retained until the pool is closed"
                raise EgressError(msg) from failure
            # Waiting happens outside the pool lock, which the teardown needs in order to
            # retire the entry once its shutdown is done.
            task = entry.close_task
            if task is None:
                await entry.shutdown_done.wait()
            else:
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    if caller is not None and caller.cancelling():
                        raise
                    msg = f"egress {name!r} browser close was cancelled; ownership is retained"
                    raise EgressError(msg) from None

    @staticmethod
    def _close_failure(entry: _LiveEgress) -> BaseException | None:
        """Return the failure of *entry*'s finished close, or ``None`` while it is open."""
        task = entry.close_task
        if task is None or not task.done():
            return None
        if task.cancelled():
            return asyncio.CancelledError()
        return task.result()

    async def _claim(self, name: str) -> tuple[_LiveEgress, bool]:
        """Claim *name* for one request, or report that its browser is still closing.

        Returns the entry and whether it may be used now. When it may not, the browser is
        shutting down and the caller has to wait on its ``shutdown_done`` before trying
        again, so no request ever holds a browser that is being closed.

        :raises EgressError: when *name* is not configured or the pool has been closed.
        """
        async with self._lock:
            if self._closed:
                msg = f"egress pool is closed; cannot acquire {name!r}"
                raise EgressError(msg)
            definition = self.definition(name)
            entry = self._live.get(name)
            if entry is None:
                entry = _LiveEgress(
                    browser=create_egress_browser(
                        name=definition.name,
                        proxy_url=definition.proxy_url,
                        profile_dir=definition.profile_dir,
                        profile_archive=definition.profile_archive,
                        preferred_cdp_port=definition.cdp_port,
                        extensions_dir=definition.extensions_dir,
                        policy_dir=definition.policy_dir,
                        max_contexts=definition.max_contexts,
                    ),
                )
                self._live[name] = entry
                logger.debug(f"Egress {name!r} browser created.")
            if entry.closing:
                return entry, False
            if entry.idle_task is not None:
                entry.idle_task.cancel()
                entry.idle_task = None
            entry.inflight += 1
            return entry, True

    async def release(self, name: str) -> None:
        """Mark *name* idle, scheduling its teardown once nothing is running."""
        async with self._lock:
            entry = self._live.get(name)
            # Nothing to release when the entry is gone, and a closing entry has no work
            # left to count: its shutdown is already in flight.
            if entry is None or entry.closing:
                return
            entry.inflight = max(0, entry.inflight - 1)
            if entry.inflight == 0 and entry.idle_task is None:
                entry.idle_task = asyncio.ensure_future(self._teardown_when_idle(definition_name=name, entry=entry))

    async def _teardown_when_idle(self, *, definition_name: str, entry: _LiveEgress) -> None:
        """Wait out the idle delay, then hand *entry* to its tracked close.

        The commit happens under the pool lock and marks the entry closing before the
        native shutdown is started, so an acquire arriving meanwhile waits for that
        shutdown rather than starting a second browser against the same profile directory
        and archive. The timer ends as soon as it has handed the entry over; the close
        itself runs in ``close_task``, which no cancellation of this timer can reach.
        """
        await asyncio.sleep(self._idle_seconds)
        async with self._lock:
            entry.idle_task = None
            if entry.inflight > 0 or self._live.get(definition_name) is not entry:
                return
            entry.closing = True
            self._start_close_locked(definition_name, entry)
        logger.debug(f"Idling out the egress {definition_name!r} browser.")

    def _start_close_locked(
        self,
        definition_name: str,
        entry: _LiveEgress,
    ) -> asyncio.Task[BaseException | None]:
        """Start, or return, the single tracked native close for *entry*.

        The caller holds the pool lock. A close that is still running is reused, so a
        retry never runs a second shutdown against a browser that is already closing. A
        fresh close clears ``shutdown_done`` before it starts so waiters parked on a
        previous attempt wait for this one instead of returning against a stale event.
        An entry whose close already finished is retried through the owner's retry entry
        point, so the native side retires what the failed close left live instead of
        replaying the cached failure.
        """
        task = entry.close_task
        if task is not None and not task.done():
            return task
        retry = task is not None
        entry.shutdown_done.clear()
        task = asyncio.ensure_future(self._close_entry(definition_name, entry, retry=retry))
        entry.close_task = task
        return task

    async def _close_entry(
        self,
        definition_name: str,
        entry: _LiveEgress,
        *,
        retry: bool = False,
    ) -> BaseException | None:
        """Run one native shutdown and retire *entry* only when it returns successfully.

        The failure is returned as the task's result rather than raised, so no background
        close is ever an unobserved task exception. Until a close succeeds the entry keeps
        its browser and stays listed, so the profile it may still hold is never handed to a
        second browser. A retried attempt uses the owner's retry entry point when it has one,
        so a native browser retries its failed cleanup rather than replaying the failure.
        """
        retry_shutdown = getattr(entry.browser, "retry_shutdown", None) if retry else None
        try:
            if retry_shutdown is not None:
                await retry_shutdown()
            else:
                await entry.browser.shutdown()
            async with self._lock:
                if self._live.get(definition_name) is entry:
                    self._bank_retired_locked(entry)
                    del self._live[definition_name]
            logger.debug(f"Egress {definition_name!r} browser shut down.")
        except BaseException as error:  # noqa: BLE001 - recorded as the close's result for retry
            logger.warning(f"Egress {definition_name!r} browser shutdown failed: {error!r}")
            return error
        else:
            return None
        finally:
            entry.shutdown_done.set()

    def _bank_retired_locked(self, entry: _LiveEgress) -> None:
        """Add *entry*'s cumulative owner counters to the pool's retired totals.

        Called under the pool lock only as *entry* is removed on a successful close, so
        each owner's totals are banked exactly once and a failed close banks nothing.
        """
        owner_metrics = _owner_resource_metrics(entry.browser)
        for name in _COUNTER_METRICS:
            self._retired[name] += owner_metrics[name]

    async def aclose(self) -> None:
        """Drain every live egress: fence the pool, then shut every browser down.

        The pool is fenced first, so no new acquire can win the egresses being drained.
        An idle timer that has not yet committed is cancelled, a close that is already
        running is joined rather than skipped, and every other owned entry is handed to its
        own tracked close. All of them are awaited, and only then is the first failure
        surfaced, so one browser that will not close never hides the others.

        Cancelling a caller of this method does not abandon the closes: they are shielded
        and remain tracked, so a later ``aclose`` retries whatever is still owned.

        :raises EgressError: when one or more egress browsers could not be shut down.
        """
        async with self._lock:
            self._closed = True
            items = list(self._live.items())
            tasks = []
            for name, entry in items:
                entry.closing = True
                if entry.idle_task is not None:
                    entry.idle_task.cancel()
                    entry.idle_task = None
                tasks.append(self._start_close_locked(name, entry))
        results = await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
        failures = [failure for failure in results if isinstance(failure, BaseException)]
        if failures:
            msg = f"{len(failures)} egress browser(s) could not be shut down"
            raise EgressError(msg) from failures[0]


__all__ = [
    "DEFAULT_EGRESS_IDLE_SECONDS",
    "DEFAULT_EGRESS_NAME",
    "EGRESSES_ENV",
    "EGRESS_IDLE_SECONDS_ENV",
    "EgressError",
    "EgressPool",
    "create_egress_browser",
    "derive_egress_paths",
    "parse_egress_spec",
]
