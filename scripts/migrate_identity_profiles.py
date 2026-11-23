"""Plan or apply the owner-run migration of nested named egress profiles.

Planning is always read-only. Applying requires Prowl to be fully stopped with
exclusive access to the profile roots; nothing here performs a process check.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import lzma
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from prowl.browser.config import BrowserConfig, default_profile_archive, default_profile_dir
from prowl.browser.proxy.egress import EGRESSES_ENV, EgressError, EgressPool, parse_egress_spec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO

#: Planned statuses a selected entry can carry. Fixed strings, part of the JSON plan.
STATUS_MOVE: Final[str] = "move"
STATUS_MIGRATED: Final[str] = "migrated"
STATUS_ARCHIVE_ONLY: Final[str] = "archive-only"
STATUS_NO_STATE: Final[str] = "no-state"

#: Fixed, non-echoing messages. No path, URL, name, or cause is interpolated.
DUPLICATE_MESSAGE: Final[str] = "duplicate egress name in selection"
NAME_MESSAGE: Final[str] = "egress name or target ownership is invalid"
LINK_MESSAGE: Final[str] = "legacy profile is a link, not an owned directory"
ESCAPE_MESSAGE: Final[str] = "legacy profile path escapes its identity root"
AMBIGUOUS_MESSAGE: Final[str] = "ambiguous default profile ownership"
INVALID_SOURCE_MESSAGE: Final[str] = "legacy profile is not a Chromium user-data root"
INVALID_TARGET_MESSAGE: Final[str] = "target profile is not a Chromium user-data root"
CONFLICT_MESSAGE: Final[str] = "migration conflict: both the legacy and the target profile exist"
BUNDLED_MESSAGE: Final[str] = "bundled legacy profile requires manual recovery"
ARCHIVE_MALFORMED_MESSAGE: Final[str] = "default profile archive is malformed"
ARCHIVE_UNREADABLE_MESSAGE: Final[str] = "default profile archive could not be read"
MOVE_MESSAGE: Final[str] = "profile migration could not be completed; the original layout was restored"
ROLLBACK_MESSAGE: Final[str] = "profile migration failed and the original layout could not be restored; review required"
STAGING_MESSAGE: Final[str] = "temporary filtered archive could not be prepared"
BACKUP_LOCATION_MESSAGE: Final[str] = "default profile archive is not in a safe backup location"
BACKUP_WRITE_MESSAGE: Final[str] = "default profile archive backup could not be created"

#: The two markers a genuine Chromium user-data root carries.
_LOCAL_STATE_NAME: Final[str] = "Local State"
_DEFAULT_DIR_NAME: Final[str] = "Default"

#: Suffixes and the stream chunk used when a filtered archive or a backup is written.
_STAGING_SUFFIX: Final[str] = ".prowl-staging"
_BACKUP_SUFFIX: Final[str] = ".before-layout"
_COPY_CHUNK: Final[int] = 1 << 20


class MigrationError(RuntimeError):
    """A migration plan cannot be produced safely; the message is always fixed text."""


@dataclass(frozen=True, slots=True)
class MigrationEntry:
    """One selected egress profile and what the plan intends for it."""

    name: str
    source: Path
    destination: Path
    archive: Path
    status: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """The resolved default identity roots plus one read-only entry per selected name."""

    profile: Path
    archive: Path
    entries: tuple[MigrationEntry, ...]


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """A completed apply: the fresh plan, the roots moved, the backup and the rewrite flag."""

    plan: MigrationPlan
    moved: tuple[MigrationEntry, ...]
    backup: Path | None
    archive_rewritten: bool


def _resolved(raw: str | Path) -> Path:
    """Return *raw* as an absolute native path, reading only filesystem metadata."""
    return Path(raw).resolve(strict=False)


def _same_native_name(left: str, right: str) -> bool:
    """Compare two single path components with the host's own case semantics."""
    return os.path.normcase(left) == os.path.normcase(right)


def _deduplicate(names: Sequence[str]) -> list[str]:
    """Return *names* in order, rejecting a repeat before it is flattened into a mapping.

    :raises MigrationError: when a name is selected twice.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            raise MigrationError(DUPLICATE_MESSAGE)
        seen.add(name)
        ordered.append(name)
    return ordered


def _archive_top_levels(archive: Path) -> set[str]:
    """Return the first path component of every member of *archive*.

    Only the member names are read; nothing is extracted or decoded. A missing archive
    is empty state, not an error.

    :raises MigrationError: when *archive* is unreadable or not a ZIP.
    """
    if not archive.exists():
        return set()
    try:
        with zipfile.ZipFile(archive) as package:
            members = package.namelist()
    except (zipfile.BadZipFile, UnicodeError):
        raise MigrationError(ARCHIVE_MALFORMED_MESSAGE) from None
    except OSError:
        raise MigrationError(ARCHIVE_UNREADABLE_MESSAGE) from None
    tops: set[str] = set()
    for member in members:
        head = _member_root(member)
        if head:
            tops.add(head)
    return tops


def _is_junction(path: Path) -> bool:
    """Return whether *path* is a directory junction, on hosts that support them."""
    return os.path.isjunction(path)


def _looks_like_user_data_root(source: Path) -> bool:
    """Return whether *source* carries the markers of a Chromium user-data root."""
    return (source / _LOCAL_STATE_NAME).is_file() and (source / _DEFAULT_DIR_NAME).is_dir()


def _check_legacy_source(name: str, source: Path, destination: Path, profile: Path) -> None:
    if not _resolved(source).is_relative_to(profile):
        raise MigrationError(ESCAPE_MESSAGE)
    if _same_native_name(name, _DEFAULT_DIR_NAME):
        raise MigrationError(AMBIGUOUS_MESSAGE)
    if not source.is_dir() or not _looks_like_user_data_root(source):
        raise MigrationError(INVALID_SOURCE_MESSAGE)
    if destination.exists():
        raise MigrationError(CONFLICT_MESSAGE)


def _plan_entry(
    name: str,
    destination: Path,
    archive: Path,
    roots: MigrationPlan,
    bundled: set[str],
) -> MigrationEntry:
    """Preflight one legacy directory and its exact target without changing either."""
    source = roots.profile / name
    for path in (source, destination, archive):
        if path.is_symlink() or _is_junction(path):
            raise MigrationError(LINK_MESSAGE)
    destination = _resolved(destination)
    archive = _resolved(archive)
    if os.path.lexists(source):
        _check_legacy_source(name, source, destination, roots.profile)
        status = STATUS_MOVE
    elif destination.exists():
        if not destination.is_dir() or not _looks_like_user_data_root(destination):
            raise MigrationError(INVALID_TARGET_MESSAGE)
        status = STATUS_MIGRATED
    elif archive.exists():
        if not archive.is_file():
            raise MigrationError(ARCHIVE_UNREADABLE_MESSAGE)
        status = STATUS_ARCHIVE_ONLY
    else:
        if any(_same_native_name(top, name) for top in bundled):
            raise MigrationError(BUNDLED_MESSAGE)
        status = STATUS_NO_STATE
    return MigrationEntry(name, _resolved(source), destination, archive, status)


def plan_migration(
    profile_dir: str,
    profile_archive: str,
    names: Sequence[str],
) -> MigrationPlan:
    """Return the read-only migration plan for the selected *names*.

    The default identity's own profile and archive are the roots a legacy nested layout
    lived under, and each selected destination is the reviewed sibling path. Target
    ownership and name policy are validated once through a real :class:`EgressPool`, which
    is pure construction and never acquires a browser.

    :raises MigrationError: when a name, an ownership relationship, or a selected entry
        cannot be planned safely.
    """
    selected = _deduplicate(names)
    config = BrowserConfig(profile_dir=profile_dir, profile_archive=profile_archive)
    try:
        pool = EgressPool(config, dict.fromkeys(selected, ""))
    except EgressError:
        raise MigrationError(NAME_MESSAGE) from None
    raw_archive = Path(profile_archive).absolute()
    if raw_archive.is_symlink() or _is_junction(raw_archive):
        raise MigrationError(LINK_MESSAGE)
    roots = MigrationPlan(profile=_resolved(profile_dir), archive=_resolved(raw_archive), entries=())
    bundled = _archive_top_levels(roots.archive) if selected else set()
    entries: list[MigrationEntry] = []
    for name in selected:
        definition = pool.definition(name)
        entries.append(
            _plan_entry(
                name,
                destination=Path(definition.profile_dir).absolute(),
                archive=Path(definition.profile_archive).absolute(),
                roots=roots,
                bundled=bundled,
            ),
        )
    return MigrationPlan(profile=roots.profile, archive=roots.archive, entries=tuple(entries))


def _member_root(member: str) -> str | None:
    head, separator, _tail = member.replace("\\", "/").partition("/")
    return head if separator else None


def _drops(head: str | None, names: Sequence[str]) -> bool:
    """Return whether an archive member under *head* belongs to a selected legacy name.

    The default identity's own ``Default`` subtree is never dropped, even when a selected
    name compares equal to it under the host's name semantics.
    """
    if not head or _same_native_name(head, _DEFAULT_DIR_NAME):
        return False
    return any(_same_native_name(head, name) for name in names)


def _needs_rewrite(archive: Path, names: Sequence[str]) -> bool:
    """Return whether *archive* still bundles a selected legacy name's subtree."""
    if not names:
        return False
    return any(_drops(top, names) for top in _archive_top_levels(archive))


def _backup_target(plan: MigrationPlan) -> Path:
    """Return an unused backup path beside the archive, outside every profile root.

    :raises MigrationError: when the archive sits inside an identity profile root.
    """
    parent = plan.archive.parent
    roots = (
        plan.profile,
        *(entry.source for entry in plan.entries),
        *(entry.destination for entry in plan.entries),
    )
    if any(parent.is_relative_to(root) for root in roots):
        raise MigrationError(BACKUP_LOCATION_MESSAGE)
    while True:
        candidate = plan.archive.with_name(f"{plan.archive.name}{_BACKUP_SUFFIX}-{uuid.uuid4().hex}")
        if not os.path.lexists(candidate):
            return candidate


def _discard(path: Path | None) -> None:
    """Remove a staging file if it exists, ignoring a missing or locked file."""
    if path is None:
        return
    with contextlib.suppress(OSError):
        path.unlink()


def _stage_filtered_archive(archive: Path, names: Sequence[str]) -> Path:
    """Stream *archive* into a closed temp file beside it, dropping selected subtrees.

    Each retained member is copied from the source ZIP to the staged ZIP without
    buffering the whole member, so payloads, per-member metadata and the archive comment
    survive unchanged.

    :raises MigrationError: when the staged archive cannot be prepared.
    """
    descriptor, filename = tempfile.mkstemp(prefix=f"{archive.name}{_STAGING_SUFFIX}-", dir=archive.parent)
    staged = Path(filename)
    try:
        with (
            os.fdopen(descriptor, "w+b") as sink,
            zipfile.ZipFile(archive) as source,
            zipfile.ZipFile(sink, "w", allowZip64=True) as target,
        ):
            target.comment = source.comment
            for info in source.infolist():
                head = _member_root(info.filename)
                if _drops(head, names):
                    continue
                if info.is_dir():
                    target.writestr(info, b"")
                    continue
                with source.open(info) as reader, target.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer, length=_COPY_CHUNK)
    except (zipfile.BadZipFile, UnicodeError, RuntimeError, EOFError, zlib.error, lzma.LZMAError):
        _discard(staged)
        raise MigrationError(ARCHIVE_MALFORMED_MESSAGE) from None
    except OSError:
        _discard(staged)
        raise MigrationError(STAGING_MESSAGE) from None
    except BaseException:
        _discard(staged)
        raise
    return staged


def _write_backup(archive: Path, backup: Path) -> None:
    """Copy *archive* byte for byte to *backup*, appearing complete only once it is done.

    The bytes are written to a distinct partial name that is renamed to *backup* after a
    flush, so an interruption cannot leave a partial file under the completed name.

    :raises MigrationError: when the backup cannot be written.
    """
    descriptor, filename = tempfile.mkstemp(prefix=f"{backup.name}.partial-", dir=backup.parent)
    pending = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as sink, archive.open("rb") as source:
            shutil.copyfileobj(source, sink, length=_COPY_CHUNK)
            sink.flush()
            os.fsync(sink.fileno())
        pending.replace(backup)
    except OSError:
        _discard(pending)
        raise MigrationError(BACKUP_WRITE_MESSAGE) from None
    except BaseException:
        _discard(pending)
        raise


def _rollback(done: Sequence[MigrationEntry]) -> None:
    """Reverse owned moves, including an interrupted rename that completed."""
    failed = False
    for entry in reversed(done):
        if os.path.lexists(entry.source):
            failed |= os.path.lexists(entry.destination)
            continue
        try:
            entry.destination.rename(entry.source)
        except OSError:
            failed = True
    if failed:
        raise MigrationError(ROLLBACK_MESSAGE) from None


def _apply_transaction(plan: MigrationPlan, moves: Sequence[MigrationEntry], staged: Path | None) -> None:
    """Track attempted renames before their syscalls; never undo a completed archive commit."""
    attempted: list[MigrationEntry] = []
    try:
        for entry in moves:
            if os.path.lexists(entry.destination):
                raise MigrationError(CONFLICT_MESSAGE)
            attempted.append(entry)
            entry.source.rename(entry.destination)
        if staged is not None:
            staged.replace(plan.archive)
    except BaseException as error:
        if staged is not None and not os.path.lexists(staged):
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            message = "archive commit completed; review migration result"
            raise MigrationError(message) from None
        _rollback(attempted)
        if isinstance(error, OSError):
            raise MigrationError(MOVE_MESSAGE) from None
        raise


def apply_migration(profile_dir: str, profile_archive: str, names: Sequence[str]) -> MigrationResult:
    """Apply the reviewed migration for *names*, moving owned roots and cleaning the bundle.

    Requires Prowl to be fully stopped with exclusive access to both profile roots; this
    function does not verify that and performs no process check. The plan is produced
    freshly here, so a previously produced or serialized plan is never applied. All
    preflight runs before the first mutation.

    :raises MigrationError: when the plan is unsafe, a move fails, or the archive cannot
        be rewritten. A failed run restores the original layout, or reports that manual
        review is required when that is not possible.
    """
    plan = plan_migration(profile_dir, profile_archive, names)
    selected = tuple(entry.name for entry in plan.entries)
    moves = tuple(entry for entry in plan.entries if entry.status == STATUS_MOVE)
    backup = _backup_target(plan) if _needs_rewrite(plan.archive, selected) else None
    staged: Path | None = None
    try:
        if backup is not None:
            staged = _stage_filtered_archive(plan.archive, selected)
            _write_backup(plan.archive, backup)
        _apply_transaction(plan, moves, staged)
    finally:
        _discard(staged)
    return MigrationResult(plan=plan, moved=moves, backup=backup, archive_rewritten=backup is not None)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the planner or apply arguments, defaulting to the configured identity."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.migrate_identity_profiles",
        description="Plan or apply the move of nested named egress profiles to the sibling layout.",
    )
    parser.add_argument(
        "--profile-dir",
        default=default_profile_dir(),
        help="default identity profile directory (default: PROWL_PROFILE_DIR)",
    )
    parser.add_argument(
        "--profile-archive",
        default=default_profile_archive(),
        help="default identity profile archive (default: PROWL_PROFILE_ARCHIVE)",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="egress names to plan or apply (default: the names in PROWL_EGRESSES)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the migration; requires Prowl stopped with exclusive profile access",
    )
    return parser.parse_args(argv)


def _selected_names(requested: Sequence[str] | None) -> list[str]:
    """Return the requested names, or the names configured in ``PROWL_EGRESSES``."""
    if requested is not None:
        return list(requested)
    return list(parse_egress_spec(os.environ.get(EGRESSES_ENV, "")))


def _render(plan: MigrationPlan) -> str:
    """Return the plan as concise JSON carrying paths, name and status only."""
    return json.dumps(
        {
            "profile": str(plan.profile),
            "archive": str(plan.archive),
            "entries": [
                {
                    "name": entry.name,
                    "source": str(entry.source),
                    "destination": str(entry.destination),
                    "archive": str(entry.archive),
                    "status": entry.status,
                }
                for entry in plan.entries
            ],
        },
        indent=2,
    )


def _render_result(result: MigrationResult) -> str:
    """Return an applied result as concise JSON carrying paths and flags only."""
    return json.dumps(
        {
            "applied": True,
            "archive": str(result.plan.archive),
            "backup": None if result.backup is None else str(result.backup),
            "archive_rewritten": result.archive_rewritten,
            "moved": [
                {
                    "name": entry.name,
                    "source": str(entry.source),
                    "destination": str(entry.destination),
                }
                for entry in result.moved
            ],
        },
        indent=2,
    )


def _emit(text: str, *, stream: TextIO | None = None) -> None:
    output = sys.stdout if stream is None else stream
    output.write(f"{text}\n")
    output.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Print the read-only plan, or apply it with ``--apply``; nonzero on a fixed error."""
    args = _parse_args(argv)
    try:
        names = _selected_names(args.names)
        if args.apply:
            _emit(_render_result(apply_migration(args.profile_dir, args.profile_archive, names)))
        else:
            _emit(_render(plan_migration(args.profile_dir, args.profile_archive, names)))
    except (MigrationError, EgressError) as exc:
        _emit(f"[ERROR] {exc}", stream=sys.stderr)
        return 1
    except (OSError, zipfile.BadZipFile):
        _emit(f"[ERROR] {ARCHIVE_UNREADABLE_MESSAGE}", stream=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
