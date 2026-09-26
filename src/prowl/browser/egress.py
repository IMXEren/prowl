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
import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger

from prowl.browser.browser import Browser
from prowl.browser.config import PROXY_URL_ENV, BrowserConfig
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
            msg = f"{EGRESSES_ENV} entries must be name=url pairs, got {entry!r}"
            raise EgressError(msg)
        if not _EGRESS_NAME_PATTERN.match(name):
            msg = f"invalid egress name {name!r}: use letters, digits, dot, dash, or underscore"
            raise EgressError(msg)
        if name == DEFAULT_EGRESS_NAME:
            msg = f"egress name {DEFAULT_EGRESS_NAME!r} is reserved for {PROXY_URL_ENV}"
            raise EgressError(msg)
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
    gets its own directory inside the configured one and its own archive beside
    the configured archive, so profiles are never shared between egresses.
    """
    if name == DEFAULT_EGRESS_NAME:
        return profile_dir, profile_archive
    archive = Path(profile_archive)
    return (
        str(Path(profile_dir) / name),
        str(archive.with_name(f"{archive.stem}-{name}{archive.suffix}")),
    )


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
    egress_browser._runtime = runtime  # noqa: SLF001
    egress_browser._lifecycle = lifecycle  # noqa: SLF001
    return egress_browser


@dataclass(slots=True)
class _EgressDefinition:
    """Everything needed to build one egress browser, resolved up front."""

    name: str
    proxy_url: str
    profile_dir: str
    profile_archive: str
    cdp_port: int
    extensions_dir: str | None = None
    policy_dir: str | None = None


@dataclass(slots=True)
class _LiveEgress:
    """A created egress browser with its work count and pending teardown.

    ``closing`` is set once the teardown has committed to shutting the browser down. The
    entry stays listed until that shutdown finishes, so a request arriving meanwhile waits
    for it instead of starting a second browser against the same profile directory and
    archive. ``shutdown_done`` is set in the teardown's ``finally`` block, so even a
    cancelled shutdown releases whoever is waiting.
    """

    browser: type[Browser]
    inflight: int = 0
    idle_task: asyncio.Task[None] | None = None
    closing: bool = False
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
        for index, name in enumerate(sorted(egresses)):
            profile_dir, profile_archive = derive_egress_paths(
                base_config.profile_dir,
                base_config.profile_archive,
                name,
            )
            self._definitions[name] = _EgressDefinition(
                name=name,
                proxy_url=egresses[name],
                profile_dir=profile_dir,
                profile_archive=profile_archive,
                cdp_port=_CDP_PORT_BASE + index * _CDP_PORT_STRIDE,
                extensions_dir=base_config.extensions_dir,
                policy_dir=base_config.policy_dir,
            )
        self._live: dict[str, _LiveEgress] = {}
        self._lock = asyncio.Lock()

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

    async def acquire(self, name: str) -> type[Browser]:
        """Return the browser for *name*, creating it on first use.

        Cancels a pending idle teardown and marks the egress busy. When the browser for
        this egress is still shutting down, the call waits for that shutdown to finish
        before creating a replacement, because the browser being closed still owns the
        profile directory and the profile archive.

        :raises EgressError: when *name* is not configured.
        """
        while True:
            entry, usable = await self._claim(name)
            if usable:
                return entry.browser
            # Waiting happens outside the pool lock, which the teardown needs in order to
            # retire the entry once its shutdown is done.
            await entry.shutdown_done.wait()

    async def _claim(self, name: str) -> tuple[_LiveEgress, bool]:
        """Claim *name* for one request, or report that its browser is still closing.

        Returns the entry and whether it may be used now. When it may not, the browser is
        shutting down and the caller has to wait on its ``shutdown_done`` before trying
        again, so no request ever holds a browser that is being closed.

        :raises EgressError: when *name* is not configured.
        """
        async with self._lock:
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
        """Shut *entry* down once the idle delay elapses without new work.

        The entry is marked closing and stays listed for the whole shutdown, so an acquire
        that arrives meanwhile waits on ``shutdown_done`` rather than starting a second
        browser against the same profile directory and archive. It is retired in the
        ``finally`` block, so a cancelled shutdown cannot leave the egress stuck in the
        closing state with its waiters parked forever.
        """
        closing = False
        try:
            await asyncio.sleep(self._idle_seconds)
            async with self._lock:
                if entry.inflight > 0 or self._live.get(definition_name) is not entry:
                    return
                closing = True
                entry.closing = True
            logger.debug(f"Idling out the egress {definition_name!r} browser.")
            with contextlib.suppress(Exception):
                await entry.browser.shutdown()
        finally:
            if closing:
                async with self._lock:
                    if self._live.get(definition_name) is entry:
                        del self._live[definition_name]
                entry.shutdown_done.set()

    async def aclose(self) -> None:
        """Cancel pending teardowns and shut every live egress browser down.

        An egress whose shutdown is already in flight is left alone: its own teardown is
        closing that browser, so shutting it down again here would close one browser twice
        and cancelling its task would interrupt a close. Waiting for it would also stall
        service shutdown behind a browser close.
        """
        async with self._lock:
            entries = [entry for entry in self._live.values() if not entry.closing]
            self._live.clear()
            for entry in entries:
                if entry.idle_task is not None:
                    entry.idle_task.cancel()
                    entry.idle_task = None
        for entry in entries:
            with contextlib.suppress(Exception):
                await entry.browser.shutdown()


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
