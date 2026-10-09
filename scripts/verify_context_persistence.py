r"""Standalone offline gate for isolated-session context-state persistence.

Run explicitly with the project virtualenv on a machine that already has the
CloakBrowser binary and a display:

    .\.venv\Scripts\python.exe -m scripts.verify_context_persistence

It uses a fresh ``TemporaryDirectory`` profile, one loopback HTTPS server on 127.0.0.1 only, and
downloads and installs nothing. The controlled requests are local; it never adds a synthetic
locale, timezone, user agent or proxy. The offline claim covers the traffic this probe controls:
its own loopback server, GeoIP resolution, and the binary download/update path (pinned to the
already-installed binary). It is not a host firewall, so unrelated background traffic the
operating system or the browser itself initiates is outside its scope.

The probe reuses the isolated-session fixture (its loopback server, offline launch wrapper,
state helpers and result reporting) and drives the real service registry and backend across
three browser generations over one temporary profile:

* the enabled generation over a ``ContextStateStore`` seeds shared, ``one`` and ``two`` markers,
  expires ``one`` through the real TTL path, recreates it, proves a fresh id starts empty, then
  restarts the browser over the same profile and store and reads every marker back;
* an explicit destroy forgets ``one``'s file while its sibling keeps state;
* a final generation with no store must neither restore nor write any saved state.

Every state operation navigates ``<url>/state`` in the same origin, which is retained across a
browser restart. No HTTP transport or curl is involved.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

from prowl.browser import Browser, BrowserConfig
from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import BrowserBackend
from prowl.service.context_state import ContextStateStore
from scripts.verify_isolated_sessions import (
    _EMPTY_STATE,
    ProbePrerequisiteError,
    _cleanup_servers,
    _emit,
    _offline_launch_patches,
    _Probe,
    _read_state,
    _resolve_binary_path,
    _Result,
    _serve,
    _set_state,
    _tls_context,
    _with_group,
)

_SITE_HOST = "127.0.0.1"
_STATE_PATH = "/state"

#: A navigable page that writes no cookie, storage or IndexedDB state, so a read never resets it.
_STATE_PAGE = "<!doctype html><html><body>state probe</body></html>"

#: Logical isolated session ids and the marker written into each context's three surfaces.
_ISOLATED_ONE = "one"
_ISOLATED_TWO = "two"
_ISOLATED_FRESH = "three"
_MARKER_SHARED = "shared"


class _OwnershipError(RuntimeError):
    """The launched browser is not the probe's temporary profile."""


@dataclass(slots=True)
class _Rig:
    """One probe run's temporary paths and its loopback base URL."""

    directory: Path
    url: str
    state_root: Path
    store: ContextStateStore


# -- State helpers -----------------------------------------------------------------


def _state_ok(read: dict[str, Any], marker: str) -> bool:
    """Whether *read* carries *marker* on the cookie, localStorage and IndexedDB surfaces alike."""
    return read == {"local": marker, "cookie": marker, "db": marker}


async def _seed(page: Any, marker: str, *, persistent_cookie: bool = False) -> None:
    """Write *marker* to localStorage, a cookie and IndexedDB, awaiting the IndexedDB commit.

    ``persistent_cookie`` adds ``Max-Age`` because the fixture cookie is a session cookie, which a
    real browser profile drops on restart; the isolated contexts keep theirs through the state store.
    """
    await _set_state(page, marker)
    if persistent_cookie:
        await page.evaluate(
            "(marker) => { document.cookie = 'probe-cookie=' + marker + '; Max-Age=3600; path=/'; }",
            marker,
        )


def _root_json_bytes(root: Path) -> dict[str, bytes]:
    """Return every stored state file's bytes keyed by name, for a byte-for-byte comparison."""
    return {path.name: path.read_bytes() for path in sorted(root.glob("*.json"))}


# -- Startup, ownership and cleanup ------------------------------------------------


async def _start_owned(probe: _Probe, backend: BrowserBackend, directory: Path) -> bool:
    """Start *backend* and refuse to continue unless it owns the probe's temporary profile.

    The profile fence is checked only after ``backend.start()`` applies the launch configuration,
    because before that the browser is still pointed at whatever profile it had. A mismatch raises
    so no cookie, storage or IndexedDB write can touch the wrong profile; the caller must then leave
    the unowned browser alone.

    :raises _OwnershipError: when the launched browser is not the probe's temporary profile.
    """
    await backend.start()
    expected = (directory / "profile").resolve()
    actual = Browser._webdata_path().parent.parent.resolve()  # noqa: SLF001 - temporary-profile fence
    ok = actual == expected
    probe.record(_Result("owned browser profile fence", ok, f"expected={expected} actual={actual}"))
    if not ok:
        msg = "refusing to run outside the probe's temporary profile"
        raise _OwnershipError(msg)
    return ok


async def _shutdown_owned(probe: _Probe, backend: BrowserBackend, *, owned: bool) -> None:
    """Close the owned browser, recording a cleanup failure instead of masking a primary error.

    An unowned browser is left untouched, so a failed profile fence never shuts down a default profile.
    """
    if not owned:
        return
    try:
        await backend.aclose()
    except Exception as exc:  # noqa: BLE001 - a cleanup failure is recorded, not raised over the primary error
        probe.error("owned browser cleanup", exc)


def _status(probe: _Probe) -> int:
    """Return the process exit code implied by *probe*: nonzero when any check failed."""
    return 1 if probe.failures else 0


# -- Backend construction ----------------------------------------------------------


def _browser_config(directory: Path) -> BrowserConfig:
    """Return the explicit temporary-profile launch configuration shared by every generation."""
    return BrowserConfig(
        profile_dir=str(directory / "profile"),
        profile_archive=str(directory / "profile.zip"),
        extensions_dir=None,
        policy_dir=None,
        proxy_url=None,
    )


def _new_backend(rig: _Rig, *, enabled: bool) -> tuple[BrowserBackend, Service]:
    """Build a backend generation over the same profile, with or without the state store."""
    backend = BrowserBackend(_browser_config(rig.directory), state_store=rig.store if enabled else None)
    return backend, Service(ServiceConfig(), backend)


# -- Checks ------------------------------------------------------------------------


class _Checks:
    """Ordered context-state persistence checks across successive backend generations."""

    def __init__(self, probe: _Probe, rig: _Rig) -> None:
        """Record the probe reporter and the run's paths and store."""
        self.probe = probe
        self.rig = rig
        self.live: list[BrowserBackend] = []
        self.owned = False
        self.shared: Any = None
        self.one: Any = None
        self.two: Any = None

    def _state_url(self) -> str:
        """Return the shared-origin state page for this run's loopback server."""
        return f"{self.rig.url}{_STATE_PATH}"

    async def run(self, backend: BrowserBackend, service: Service) -> None:
        """Start the owned browser and run every check in order."""
        self.live.append(backend)
        await _start_owned(self.probe, backend, self.rig.directory)
        self.owned = True
        self.shared = await Browser.get_context(None)
        self.one = await backend._isolated_session(_ISOLATED_ONE, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        self.two = await backend._isolated_session(_ISOLATED_TWO, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        await service.sessions.ensure(_ISOLATED_ONE, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        await service.sessions.ensure(_ISOLATED_TWO, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        await self._seed()
        await self._evict_one(service)
        await self._recreate_one(backend, service)
        await self._fresh_id(backend, service)
        backend2, service2 = await self._restart(service)
        await self._destroy_one(backend2, service2)
        await self._disabled(service2)
        self._final()

    async def _seed(self) -> None:
        """Write a distinct marker into every context and read all three surfaces back."""
        url = self._state_url()
        await _with_group(self.shared, url, partial(_seed, marker=_MARKER_SHARED, persistent_cookie=True))
        await _with_group(self.one.context, url, partial(_seed, marker=_ISOLATED_ONE))
        await _with_group(self.two.context, url, partial(_seed, marker=_ISOLATED_TWO))
        reads = {
            "shared": await _with_group(self.shared, url, _read_state),
            "one": await _with_group(self.one.context, url, _read_state),
            "two": await _with_group(self.two.context, url, _read_state),
        }
        ok = (
            _state_ok(reads["shared"], _MARKER_SHARED)
            and _state_ok(reads["one"], _ISOLATED_ONE)
            and _state_ok(reads["two"], _ISOLATED_TWO)
        )
        self.probe.record(
            _Result("state writes are visible on cookie, localStorage and IndexedDB", ok, f"reads={reads!r}"),
        )

    async def _evict_one(self, service: Service) -> None:
        """Expire ``one`` through the real registry and verify its native close and saved state."""
        before = Browser.resource_metrics()["context_evicted_total"]
        handle = self.one.context
        service.sessions._entries[_ISOLATED_ONE].expires_at = 0.0  # noqa: SLF001
        expired = await service.sessions.purge_expired()
        after = Browser.resource_metrics()["context_evicted_total"]
        native_closed = handle.context.is_closed()
        saved = self.rig.store.load(DEFAULT_EGRESS_NAME, _ISOLATED_ONE) is not None
        ok = _ISOLATED_ONE in expired and native_closed and saved and after > before
        self.probe.record(
            _Result(
                "a real TTL eviction closes the native context and preserves its saved state",
                ok,
                f"expired={expired!r} native_closed={native_closed} saved={saved} evicted_total={before}->{after}",
            ),
        )

    async def _recreate_one(self, backend: BrowserBackend, service: Service) -> None:
        """Recreate the evicted id and verify it and its siblings keep their own markers."""
        await service.sessions.ensure(_ISOLATED_ONE, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        one = await backend._isolated_session(_ISOLATED_ONE, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        url = self._state_url()
        reads = {
            "one": await _with_group(one.context, url, _read_state),
            "two": await _with_group(self.two.context, url, _read_state),
            "shared": await _with_group(self.shared, url, _read_state),
        }
        ok = (
            _state_ok(reads["one"], _ISOLATED_ONE)
            and _state_ok(reads["two"], _ISOLATED_TWO)
            and _state_ok(reads["shared"], _MARKER_SHARED)
        )
        self.probe.record(
            _Result(
                "a recreated isolated id restores its saved state without disturbing its siblings",
                ok,
                f"reads={reads!r}",
            ),
        )

    async def _fresh_id(self, backend: BrowserBackend, service: Service) -> None:
        """A never-seeded id must start empty, and destroying it leaves the seeded files intact."""
        before = _root_json_bytes(self.rig.state_root)
        await service.sessions.ensure(_ISOLATED_FRESH, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        fresh = await backend._isolated_session(_ISOLATED_FRESH, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        read = await _with_group(fresh.context, self._state_url(), _read_state)
        self.probe.record(
            _Result(
                "a fresh isolated id starts empty instead of restoring another id's saved state",
                read == dict(_EMPTY_STATE),
                f"read={read!r}",
            ),
        )
        await service.sessions.destroy(_ISOLATED_FRESH)
        after = _root_json_bytes(self.rig.state_root)
        sibling = await _with_group(self.two.context, self._state_url(), _read_state)
        kept = bool(before) and before == after and _state_ok(sibling, _ISOLATED_TWO)
        self.probe.record(
            _Result("destroying a fresh id preserves saved files and live sibling state", kept, "files unchanged")
        )

    async def _restart(self, service: Service) -> tuple[BrowserBackend, Service]:
        """Restart over the same profile and store, then read every marker back from the new generation."""
        await service.aclose()
        preserved = self.rig.store.load(DEFAULT_EGRESS_NAME, _ISOLATED_ONE) is not None
        preserved = preserved and self.rig.store.load(DEFAULT_EGRESS_NAME, _ISOLATED_TWO) is not None
        self.probe.record(
            _Result("an orderly shutdown preserves the isolated saved state", preserved, "saved files present")
        )
        backend, service2 = _new_backend(self.rig, enabled=True)
        self.live.append(backend)
        await _start_owned(self.probe, backend, self.rig.directory)
        shared = await Browser.get_context(None)
        one = await backend._isolated_session(_ISOLATED_ONE, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        self.two = await backend._isolated_session(_ISOLATED_TWO, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        await service2.sessions.ensure(_ISOLATED_ONE, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        await service2.sessions.ensure(_ISOLATED_TWO, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        url = self._state_url()
        reads = {
            "shared": await _with_group(shared, url, _read_state),
            "one": await _with_group(one.context, url, _read_state),
            "two": await _with_group(self.two.context, url, _read_state),
        }
        restored = _state_ok(reads["one"], _ISOLATED_ONE) and _state_ok(reads["two"], _ISOLATED_TWO)
        self.probe.record(
            _Result(
                "isolated ids restore their markers from the saved state after a restart",
                restored,
                f"one={reads['one']!r} two={reads['two']!r}",
            ),
        )
        self.probe.record(
            _Result(
                "shared persistent cookie, localStorage and IndexedDB survive the restart",
                _state_ok(reads["shared"], _MARKER_SHARED),
                f"shared={reads['shared']!r}",
            ),
        )
        return backend, service2

    async def _destroy_one(self, backend: BrowserBackend, service: Service) -> None:
        """An explicit destroy must close the context, forget its file, and leave the sibling's file."""
        created = await backend._isolated_session(_ISOLATED_ONE, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        handle = created.context
        if handle is None:
            msg = "isolated context initialization returned no handle"
            raise RuntimeError(msg)
        await service.sessions.destroy(_ISOLATED_ONE)
        forgotten = self.rig.store.load(DEFAULT_EGRESS_NAME, _ISOLATED_ONE) is None
        self.probe.record(
            _Result(
                "an explicit destroy closes the native context and forgets its saved state",
                handle.context.is_closed() and forgotten,
                f"native_closed={handle.context.is_closed()} forgotten={forgotten}",
            ),
        )
        await service.sessions.ensure(_ISOLATED_ONE, mode="isolated", egress=DEFAULT_EGRESS_NAME)
        again = await backend._isolated_session(_ISOLATED_ONE, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        url = self._state_url()
        reads = {
            "one": await _with_group(again.context, url, _read_state),
            "two": await _with_group(self.two.context, url, _read_state),
        }
        ok = reads["one"] == dict(_EMPTY_STATE) and _state_ok(reads["two"], _ISOLATED_TWO)
        self.probe.record(
            _Result(
                "a recreated id after destroy starts empty while its sibling keeps its state",
                ok,
                f"one={reads['one']!r} two={reads['two']!r}",
            ),
        )

    async def _disabled(self, service: Service) -> None:
        """A backend without a store must not read or write the saved files across its whole lifetime."""
        await service.aclose()
        before = _root_json_bytes(self.rig.state_root)
        backend, _ = _new_backend(self.rig, enabled=False)
        self.live.append(backend)
        await _start_owned(self.probe, backend, self.rig.directory)
        two = await backend._isolated_session(_ISOLATED_TWO, DEFAULT_EGRESS_NAME)  # noqa: SLF001
        read = await _with_group(two.context, self._state_url(), _read_state)
        self.probe.record(
            _Result(
                "a disabled state store does not restore saved isolated state",
                read == dict(_EMPTY_STATE),
                f"two={read!r}",
            ),
        )
        await backend.aclose()
        after = _root_json_bytes(self.rig.state_root)
        self.probe.record(
            _Result(
                "a disabled state store leaves the saved files byte-for-byte unchanged",
                bool(before) and before == after,
                f"files={sorted(before)}",
            ),
        )

    def _final(self) -> None:
        """The final shutdown must retire every native context, tab group and runtime handle."""
        metrics = Browser.resource_metrics()
        handles_gone = Browser._runtime.main_ctx is None and Browser._runtime.shared_pd is None  # noqa: SLF001
        ok = not metrics["context_count"] and not metrics["tabgroups_active"] and handles_gone
        self.probe.record(
            _Result(
                "the final shutdown retires every native context, tab group and runtime handle",
                ok,
                f"metrics={metrics!r} handles_gone={handles_gone}",
            ),
        )


# -- Orchestration -----------------------------------------------------------------


async def _run_owned(directory: Path, url: str, probe: _Probe) -> None:
    """Drive every generation, closing the live browser unless a profile fence refused to own it."""
    state_root = directory / "state"
    rig = _Rig(directory=directory, url=url, state_root=state_root, store=ContextStateStore(state_root))
    backend = BrowserBackend(_browser_config(directory), state_store=rig.store)
    service = Service(ServiceConfig(), backend)
    checks = _Checks(probe, rig)
    try:
        await checks.run(backend, service)
    except _OwnershipError:
        checks.owned = False
    except Exception as exc:  # noqa: BLE001 - one failed run must still run the owned cleanup below
        probe.error("context persistence checks", exc)
    finally:
        current = checks.live[-1] if checks.live else backend
        await _shutdown_owned(probe, current, owned=checks.owned)


async def _run(directory: Path, binary_path: str, probe: _Probe) -> None:
    """Bring up the loopback server and the browser generations, cleaning both up on every path."""
    servers: list[Any] = []
    try:
        tls = _tls_context(directory)
        server, state, url = _serve(_SITE_HOST, tls)
        servers.append(server)
        state.pages[_STATE_PATH] = _STATE_PAGE
        with (
            mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary_path}),
            _offline_launch_patches(),
        ):
            await _run_owned(directory, url, probe)
    finally:
        if not _cleanup_servers(*servers):
            probe.record(_Result("server cleanup", ok=False, detail="see cleanup errors above"))


def main() -> int:
    """Run the probe, print each check as it completes, and return a nonzero status on any failure.

    :return: ``0`` when every check passed, ``1`` when a check or cleanup failed or the run
        aborted, and ``2`` when a local prerequisite (the CloakBrowser binary) is missing.
    """
    try:
        binary_path = _resolve_binary_path()
    except ProbePrerequisiteError as exc:
        _emit(f"[PREREQ-FAIL] {exc}", stream=sys.stderr)
        return 2

    probe = _Probe()
    try:
        with tempfile.TemporaryDirectory(
            prefix="prowl-context-persistence-",
            ignore_cleanup_errors=True,
        ) as raw_directory:
            asyncio.run(_run(Path(raw_directory), binary_path, probe))
    except ProbePrerequisiteError as exc:
        _emit(f"[PREREQ-FAIL] {exc}", stream=sys.stderr)
        return 2
    except BaseException as exc:  # noqa: BLE001 - report the abort rather than a traceback only
        _emit(f"[ERROR] the probe aborted: {type(exc).__name__}: {exc}", stream=sys.stderr)
        return 1

    _emit(
        f"{probe.total - probe.failures}/{probe.total} checks passed. "
        "This probe covers loopback traffic only and is not a host firewall.",
        stream=sys.stderr,
    )
    return _status(probe)


if __name__ == "__main__":
    sys.exit(main())
