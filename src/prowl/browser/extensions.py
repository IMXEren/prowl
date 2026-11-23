"""Unpacked Chromium extensions loaded into the browser at launch.

A deployment can mount a directory of unpacked extensions, such as an ad blocker, and
every browser the process starts then loads them. Extensions are deployment
configuration on the same footing as the egress proxy: the browser and its persistent
profile are shared, so a caller cannot choose them per request, and a change takes
effect on the next launch.

Loading an extension is not free. An extension runs inside the pages it applies to and
is one more thing that distinguishes this browser from a stock one, which matters for a
browser whose value is passing bot checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger

#: Environment variable naming the directory of unpacked extensions.
EXTENSIONS_DIR_ENV: Final[str] = "PROWL_EXTENSIONS_DIR"

#: File that makes a directory an unpacked extension.
_MANIFEST_NAME: Final[str] = "manifest.json"


@dataclass(frozen=True, slots=True)
class Extension:
    """One unpacked extension: the name to report and the path to hand to Chromium."""

    name: str
    path: str


def discover_extensions(extensions_dir: str | None) -> list[Extension]:
    """Return the unpacked extensions under *extensions_dir*, ordered by directory name.

    A subdirectory counts as an extension when it holds a ``manifest.json`` that parses
    as a JSON object. Anything else is skipped with a warning rather than failing the
    launch, because one unusable extension in a mounted directory should not stop the
    browser from starting. The reported name comes from the manifest when it has one, so
    a directory renamed for convenience still names the extension it holds.

    :param extensions_dir: the directory to scan, or ``None`` when none is configured.
    :return: the extensions found, or an empty list when there are none.
    """
    if not extensions_dir:
        return []
    root = Path(extensions_dir)
    if not root.is_dir():
        logger.warning(f"{EXTENSIONS_DIR_ENV} is not a readable directory: {root}")
        return []

    extensions: list[Extension] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue
        manifest = entry / _MANIFEST_NAME
        if not manifest.is_file():
            continue
        name = _manifest_name(manifest, fallback=entry.name)
        if name is None:
            continue
        extensions.append(Extension(name=name, path=str(entry)))
    return extensions


def extension_launch_arguments(extensions_dir: str | None) -> list[str]:
    """Return the Chromium arguments that load every extension in *extensions_dir*.

    Both flags are part of the answer. ``--load-extension`` adds the extensions, and
    ``--disable-extensions-except`` limits the browser to exactly those, so an extension
    a profile happened to carry cannot end up loaded alongside them.

    An absent, empty, or unusable directory contributes nothing, so a deployment that
    does not use extensions launches exactly as it did before this existed.

    :param extensions_dir: the directory to load from, or ``None`` when none is configured.
    :return: the launch arguments, or an empty list when no extension is loadable.
    """
    extensions = discover_extensions(extensions_dir)
    if not extensions:
        if extensions_dir:
            logger.debug(f"No usable extension in {extensions_dir}.")
        return []

    paths = ",".join(extension.path for extension in extensions)
    logger.info(
        f"Loading {len(extensions)} browser extension(s): {', '.join(extension.name for extension in extensions)}.",
    )
    return [f"--load-extension={paths}", f"--disable-extensions-except={paths}"]


def _manifest_name(manifest: Path, *, fallback: str) -> str | None:
    """Return the name *manifest* records, or *fallback* when it records none.

    :return: the name to report, or ``None`` when the manifest cannot be used at all.
    """
    try:
        parsed = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Skipping extension {manifest.parent}: unusable manifest ({type(exc).__name__}).")
        return None
    if not isinstance(parsed, dict):
        logger.warning(f"Skipping extension {manifest.parent}: manifest is not a JSON object.")
        return None
    name = parsed.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    # A manifest without a name is still a loadable extension, so it is named after its
    # directory rather than skipped.
    return fallback


__all__ = [
    "EXTENSIONS_DIR_ENV",
    "Extension",
    "discover_extensions",
    "extension_launch_arguments",
]
