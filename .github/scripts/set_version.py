#!/usr/bin/env python3
"""Set Prowl's single-source package version from a semantic-release version.

semantic-release decides ``nextRelease.version``; this rewrites the committed
``__version__`` in ``src/prowl/__about__.py`` so the built wheel/sdist and the
committed release metadata carry exactly that version.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ABOUT = Path(__file__).resolve().parents[2] / "src" / "prowl" / "__about__.py"
VERSION_LINE = re.compile(r'^__version__ = "[^"]*"$', re.MULTILINE)
EXPECTED_ARG_COUNT = 2


def set_version(version: str, path: Path = ABOUT) -> None:
    """Rewrite the single ``__version__`` assignment in ``path`` to ``version``."""
    text = path.read_text(encoding="utf-8")
    updated, replacements = VERSION_LINE.subn(f'__version__ = "{version}"', text)
    if replacements != 1:
        msg = f"expected exactly one __version__ assignment in {path}, found {replacements}"
        raise SystemExit(msg)
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Apply the semantic-release version supplied on the command line."""
    if len(sys.argv) != EXPECTED_ARG_COUNT or not sys.argv[1]:
        msg = "usage: set_version.py <version>"
        raise SystemExit(msg)
    version = sys.argv[1]
    set_version(version)
    print(f"set {ABOUT} __version__ = {version}")


if __name__ == "__main__":
    main()
