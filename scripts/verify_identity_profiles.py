"""Verify native cookie isolation across temporary profile relocation and archive restore."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from prowl.browser import Browser, BrowserConfig
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle.startup import BrowserLifecycle
from prowl.browser.proxy.egress import EgressPool
from prowl.service.backend import BrowserBackend
from scripts.migrate_identity_profiles import MigrationResult, apply_migration
from scripts.verify_isolated_sessions import (
    ProbePrerequisiteError,
    _emit,
    _offline_launch_patches,
    _Probe,
    _resolve_binary_path,
    _Result,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from playwright._impl._api_structures import SetCookieParam

    from prowl.browser.driver.contexts import BrowserContextHandle

#: The named egress under test. It is the name the legacy nested layout used as a subdirectory.
_NAMED_EGRESS = "one"

#: The reserved synthetic origin the identity cookies are scoped to, and the cookie name.
_PROFILE_URL = "https://profile.test/"
_COOKIE_NAME = "identity"

#: The two identity marker values written into the default and the named profile.
_MAIN = "main"
_NAMED = "named"

#: Persistent-cookie lifetime (seconds) and the streaming chunk used to hash an archive.
_COOKIE_LIFETIME_SECONDS = 3600
_HASH_CHUNK = 1 << 20


class _OwnershipError(RuntimeError):
    """The launched browser is not the probe's temporary profile."""


@dataclass(frozen=True, slots=True)
class _Paths:
    """Every path this run owns, all of them inside one temporary directory."""

    root: Path
    default_root: Path
    default_archive: Path
    legacy_named_root: Path
    named_root: Path
    named_archive: Path


def _paths(directory: Path) -> _Paths:
    """Return the temporary layout: the default roots plus the legacy nested and sibling roots."""
    default_root = directory / "profile"
    return _Paths(
        root=directory,
        default_root=default_root,
        default_archive=directory / "profile.zip",
        legacy_named_root=default_root / _NAMED_EGRESS,
        named_root=directory / f"profile-{_NAMED_EGRESS}",
        named_archive=directory / f"profile-{_NAMED_EGRESS}.zip",
    )


def _default_config(paths: _Paths) -> BrowserConfig:
    """Return the explicit temporary launch configuration for the default identity."""
    return BrowserConfig(
        proxy_url=None,
        profile_dir=str(paths.default_root),
        profile_archive=str(paths.default_archive),
        extensions_dir=None,
        policy_dir=None,
    )


def _legacy_config(paths: _Paths) -> BrowserConfig:
    """Return the explicit configuration of the legacy nested named root and its standalone archive."""
    return BrowserConfig(
        proxy_url=None,
        profile_dir=str(paths.legacy_named_root),
        profile_archive=str(paths.named_archive),
        extensions_dir=None,
        policy_dir=None,
    )


def _lifecycle(profile_dir: Path, archive: Path) -> BrowserLifecycle:
    """Return a real lifecycle over one identity's own profile directory and archive."""
    return BrowserLifecycle(BrowserRuntimeState(max_groups=1), profile_dir=str(profile_dir), profile_archive=archive)


def _persistent_identity(value: str) -> SetCookieParam:
    """Return the probe's persistent identity cookie for *value*, scoped to the synthetic origin."""
    return {
        "name": _COOKIE_NAME,
        "value": value,
        "url": _PROFILE_URL,
        "secure": True,
        "httpOnly": True,
        "sameSite": "Lax",
        "expires": time.time() + _COOKIE_LIFETIME_SECONDS,
    }


async def _write_identity(handle: BrowserContextHandle, value: str) -> None:
    """Install the persistent identity cookie for *value* into *handle*'s native context."""
    await handle.context.add_cookies([_persistent_identity(value)])


async def _read_identity(handle: BrowserContextHandle) -> str | None:
    """Return the identity cookie value the native context would send to the synthetic origin."""
    for cookie in await handle.context.cookies(_PROFILE_URL):
        if cookie.get("name") == _COOKIE_NAME:
            return cookie.get("value")
    return None


def _require_owned(probe: _Probe, owner: type[Browser], expected_root: Path) -> None:
    """Fence the configured profile before lazy native acquisition."""
    actual = owner._webdata_path().parent.parent.resolve()  # noqa: SLF001 - temporary-profile fence
    expected = Path(expected_root).resolve()
    ok = actual == expected
    probe.record(_Result("owned browser profile fence", ok, f"expected={expected} actual={actual}"))
    if not ok:
        msg = "refusing to run outside the probe's temporary profile"
        raise _OwnershipError(msg)


async def _open_owned(
    probe: _Probe,
    starter: Callable[[], Awaitable[None]],
    owner: type[Browser],
    expected_root: Path,
    *,
    config: BrowserConfig | None = None,
) -> BrowserContextHandle:
    """Fence configured ownership before startup and shared-context acquisition."""
    if config is not None:
        owner.configure(config)
    _require_owned(probe, owner, expected_root)
    await starter()
    return await owner.get_context(None)


def _archive_digest(path: Path) -> str:
    """Return the SHA-256 of *path*, read in bounded chunks so no whole archive is buffered."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_members(path: Path) -> frozenset[str]:
    """Return the member names of *path* without reading any member's payload."""
    with zipfile.ZipFile(path) as package:
        return frozenset(package.namelist())


def _profile_state(root: Path) -> dict[str, str]:
    """Hash only persistent native cookie/key files while their owner is closed."""
    candidates = (Path("Local State"), Path("Default/Cookies"), Path("Default/Network/Cookies"))
    state = {str(path): _archive_digest(root / path) for path in candidates if (root / path).is_file()}
    if "Local State" not in state or len(state) == 1:
        message = "native persistent profile files are missing"
        raise RuntimeError(message)
    return state


def _transition_named(default_root: Path, default_archive: Path, names: Sequence[str]) -> tuple[MigrationResult, str]:
    """Bundle *default_root* the legacy way, then apply the real migration for *names*.

    The bundle is written with the nested named root included, exactly as the legacy default
    archive carried it. The returned digest is of that archive immediately before the migration,
    so the caller can prove the retained backup is the same archive.

    :raises scripts.migrate_identity_profiles.MigrationError: when the migration cannot be applied.
    """
    _lifecycle(default_root, default_archive).pack_profile()
    members = _archive_members(default_archive)
    if not all(any(member.startswith(f"{name}/") for member in members) for name in names):
        message = "legacy archive does not contain the expected nested identity"
        raise RuntimeError(message)
    original = _archive_digest(default_archive)
    result = apply_migration(str(default_root), str(default_archive), list(names))
    return result, original


def _gauges_zero(*owners: type[Browser]) -> bool:
    """Whether every *owner* currently owns no native context and no active tab group."""
    return all(
        owner.resource_metrics()["context_count"] == 0 and owner.resource_metrics()["tabgroups_active"] == 0
        for owner in owners
    )


def _record_gauges(probe: _Probe, label: str, *owners: type[Browser]) -> None:
    """Record whether every *owner*'s resource gauges returned to zero."""
    closed = _gauges_zero(*owners)
    probe.record(_Result(f"{label}: owned resource gauges are back to zero", closed, "all owners checked"))
    if not closed:
        message = "native resources remain; refusing further archive operations"
        raise _OwnershipError(message)


async def _close_owners(pool: EgressPool, egress: str, backend: BrowserBackend, *, acquired: bool) -> None:
    """Attempt every owned cleanup in order without swallowing failure or cancellation."""
    try:
        if acquired:
            await pool.release(egress)
    finally:
        try:
            await pool.aclose()
        finally:
            await backend.aclose()


async def _initial_default(probe: _Probe, paths: _Paths) -> None:
    """Write the default identity into the default root, then shut the backend down."""
    config = _default_config(paths)
    backend = BrowserBackend(config)
    try:
        handle = await _open_owned(probe, backend.start, Browser, paths.default_root, config=config)
        await _write_identity(handle, _MAIN)
        value = await _read_identity(handle)
        probe.record(
            _Result(
                "default identity is written into its own profile root",
                value == _MAIN,
                "persistent native cookie read back",
            ),
        )
    finally:
        try:
            await backend.aclose()
        finally:
            _record_gauges(probe, "default backend after its first shutdown", Browser)


async def _legacy_named(probe: _Probe, paths: _Paths) -> None:
    """Write the named identity into the legacy nested root, then shut the backend down."""
    config = _legacy_config(paths)
    backend = BrowserBackend(config)
    try:
        handle = await _open_owned(probe, backend.start, Browser, paths.legacy_named_root, config=config)
        await _write_identity(handle, _NAMED)
        value = await _read_identity(handle)
        probe.record(
            _Result(
                "legacy nested named identity is written into its own nested profile root",
                value == _NAMED,
                "persistent native cookie read back",
            ),
        )
    finally:
        try:
            await backend.aclose()
        finally:
            _record_gauges(probe, "legacy nested backend", Browser)


def _migrate_bundle(probe: _Probe, paths: _Paths) -> None:
    """Bundle the default root, then move the nested named root to its sibling via the real migration."""
    named_digest = _archive_digest(paths.named_archive)
    result, original = _transition_named(paths.default_root, paths.default_archive, [_NAMED_EGRESS])
    backup_ok = result.backup is not None and result.backup.is_file() and _archive_digest(result.backup) == original
    members = _archive_members(paths.default_archive)
    bundle_clean = not any(member.startswith(f"{_NAMED_EGRESS}/") for member in members)
    moved = paths.named_root.is_dir() and not paths.legacy_named_root.exists()
    probe.record(
        _Result(
            "the real migration moves the nested named root and cleans the default bundle",
            result.archive_rewritten
            and backup_ok
            and bundle_clean
            and moved
            and _archive_digest(paths.named_archive) == named_digest,
            (f"rewritten={result.archive_rewritten} backup={backup_ok} bundle_clean={bundle_clean} moved={moved}"),
        ),
    )


async def _restored_owners(probe: _Probe, paths: _Paths, label: str) -> None:
    """Restart the default and the named identity as two distinct real browser generations."""
    config = _default_config(paths)
    backend = BrowserBackend(config)
    pool = EgressPool(_default_config(paths), {_NAMED_EGRESS: ""})
    acquired = False
    named_owner: type[Browser] | None = None
    try:
        default_handle = await _open_owned(probe, backend.start, Browser, paths.default_root, config=config)
        main_value = await _read_identity(default_handle)
        named_owner = await pool.acquire(_NAMED_EGRESS)
        acquired = True
        named_handle = await _open_owned(probe, named_owner.start, named_owner, paths.named_root)
        named_value = await _read_identity(named_handle)
        main_again = await _read_identity(default_handle)
        main_browser = default_handle.context.browser
        named_browser = named_handle.context.browser
        distinct = (
            default_handle.context is not named_handle.context
            and main_browser is not None
            and named_browser is not None
            and main_browser is not named_browser
        )
        ok = main_value == _MAIN and named_value == _NAMED and main_again == _MAIN and distinct
        probe.record(
            _Result(
                f"{label}: distinct default and named identities each read only their own cookie",
                ok,
                f"default={main_value == _MAIN} named={named_value == _NAMED} "
                f"stable={main_again == _MAIN} distinct={distinct}",
            ),
        )
    finally:
        try:
            await _close_owners(pool, _NAMED_EGRESS, backend, acquired=acquired)
        finally:
            owners = [Browser] if named_owner is None else [Browser, named_owner]
            _record_gauges(probe, label, *owners)


def _restore_archives(probe: _Probe, paths: _Paths) -> None:
    """Restore the cleaned default archive and the named standalone archive on their own roots."""
    sibling_before = _profile_state(paths.named_root)
    restored = _lifecycle(paths.default_root, paths.default_archive).unpack_profile()
    resurrected = paths.legacy_named_root.exists()
    sibling_after = _profile_state(paths.named_root)
    probe.record(
        _Result(
            "restoring the cleaned default archive cannot resurrect the nested named root",
            restored and not resurrected and sibling_before == sibling_after,
            f"restored={restored} resurrected={resurrected} sibling_unchanged={sibling_before == sibling_after}",
        ),
    )
    if not _lifecycle(paths.named_root, paths.named_archive).unpack_profile():
        message = "named archive restore failed"
        raise RuntimeError(message)


async def _run(directory: Path, binary_path: str, probe: _Probe) -> None:
    """Run every stage under the offline launch patch, all of it inside *directory*."""
    paths = _paths(directory)
    with (
        mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary_path}),
        _offline_launch_patches(),
    ):
        await _initial_default(probe, paths)
        await _legacy_named(probe, paths)
        _migrate_bundle(probe, paths)
        await _restored_owners(probe, paths, "after migration")
        _restore_archives(probe, paths)
        await _restored_owners(probe, paths, "after archive restore")


async def _bounded_run(directory: Path, binary_path: str, probe: _Probe) -> None:
    async with asyncio.timeout(480):
        await _run(directory, binary_path, probe)


def main() -> int:
    """Run the probe, print each check as it completes, and return a nonzero status on failure.

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
            prefix="prowl-identity-profiles-",
            ignore_cleanup_errors=True,
        ) as raw_directory:
            asyncio.run(_bounded_run(Path(raw_directory), binary_path, probe))
    except ProbePrerequisiteError as exc:
        _emit(f"[PREREQ-FAIL] {exc}", stream=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level probe failure boundary
        _emit(f"[ERROR] the probe aborted: {type(exc).__name__}: {exc}", stream=sys.stderr)
        return 1

    _emit(
        f"{probe.total - probe.failures}/{probe.total} checks passed. "
        "This probe covers temporary-profile relocation and isolation only.",
        stream=sys.stderr,
    )
    return 1 if probe.failures else 0


if __name__ == "__main__":
    sys.exit(main())
