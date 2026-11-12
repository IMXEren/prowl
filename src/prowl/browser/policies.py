"""Managed Chromium policies applied to the browser at launch.

Chromium reads administrative policy from a system directory, and which directory that is
depends on how the build is branded: a Chromium build reads ``/etc/chromium/policies`` and a
Chrome-branded build reads ``/etc/opt/chrome/policies``. A deployment cannot mount its own
policy files without knowing that, and Prowl cannot ask the build, so the configured directory
is copied into every candidate that is writable instead. A policy file under the path a build
does not read is inert, so covering both is safe and removes the need to know.

Policies only set what policies can set. For an unpacked extension that means who may run it,
which hosts it may touch, and whether it is pinned to the toolbar; it does not cover a
permission a user normally grants by clicking, which is a value in the profile rather than a
policy. See the README for that distinction.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Environment variable naming the directory of policy JSON files to apply.
POLICY_DIR_ENV: Final[str] = "PROWL_POLICY_DIR"

#: Directories a Chromium build may read managed policy from, Chromium first.
MANAGED_POLICY_DIRS: Final[tuple[str, ...]] = (
    "/etc/chromium/policies/managed",
    "/etc/opt/chrome/policies/managed",
)

_SUFFIX: Final[str] = ".json"


@dataclass(frozen=True, slots=True)
class PolicyFile:
    """One policy file, where it came from, and where it was written."""

    name: str
    origin: Path
    target: Path


def policy_files(policy_dir: str | Path) -> list[Path]:
    """Return the policy JSON files in *policy_dir*, ordered by name.

    Only files ending in ``.json`` are policy; anything else in the directory is left alone so
    a readme or a mount marker cannot make Chromium reject the whole policy set.
    """
    root = Path(policy_dir)
    if not root.is_dir():
        return []
    return sorted(
        (entry for entry in root.iterdir() if entry.is_file() and entry.suffix == _SUFFIX), key=lambda p: p.name
    )


def apply_managed_policies(
    policy_dir: str | None,
    *,
    targets: Sequence[str] = MANAGED_POLICY_DIRS,
) -> list[PolicyFile]:
    """Write every policy file in *policy_dir* into the managed policy directories.

    A target that cannot be created or written is reported and skipped rather than failing the
    launch, because a container that runs as an unprivileged user has to mount the policy
    directory at the managed path instead, and that is a deployment choice rather than an error
    in the browser.

    :param policy_dir: the directory holding policy JSON, or ``None`` when none is configured.
    :param targets: the managed policy directories to write to, in preference order.
    :return: one entry per written file, or an empty list when nothing was applied.
    """
    if not policy_dir:
        return []
    sources = policy_files(policy_dir)
    if not sources:
        logger.debug(f"No policy file in {policy_dir}.")
        return []

    written: list[PolicyFile] = []
    unwritable: list[str] = []
    for target in targets:
        directory = Path(target)
        written.extend(_apply_to(directory, sources, unwritable))
    if written:
        logger.info(
            f"Applied {len(sources)} managed policy file(s) to "
            f"{', '.join(sorted({str(entry.target.parent) for entry in written}))}.",
        )
    if unwritable:
        logger.warning(
            f"Could not write managed policy to {', '.join(unwritable)}. Mount the policy "
            f"directory there if this build reads it.",
        )
    return written


def _apply_to(directory: Path, sources: Iterable[Path], unwritable: list[str]) -> list[PolicyFile]:
    """Copy *sources* into *directory*, recording the directories that refused the write.

    A file that is not a JSON policy object is skipped with a warning rather than copied,
    because Chromium discards the whole managed policy set when it finds one, and the rest of
    the mounted set is then still applied.
    """
    written: list[PolicyFile] = []
    for source in sources:
        target = directory / source.name
        if source.resolve() == target.resolve():
            # The deployment mounted the policy directory at the managed path already.
            continue
        if not _is_policy_object(source):
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        except OSError:
            if str(directory) not in unwritable:
                unwritable.append(str(directory))
            return written
        written.append(PolicyFile(name=source.name, origin=source, target=target))
    return written


def _is_policy_object(source: Path) -> bool:
    """Return whether *source* holds a JSON object, warning when it does not."""
    try:
        parsed = json.loads(source.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Skipping policy file {source.name}: unreadable ({type(exc).__name__}).")
        return False
    if not isinstance(parsed, dict):
        logger.warning(f"Skipping policy file {source.name}: a policy file holds a JSON object.")
        return False
    return True


__all__ = [
    "MANAGED_POLICY_DIRS",
    "POLICY_DIR_ENV",
    "PolicyFile",
    "apply_managed_policies",
    "policy_files",
]
