"""Audit built Prowl wheel/sdist artifacts for forbidden payloads and version drift.

A released Prowl artifact must contain source only: never the CloakBrowser
binary, the persistent browser profile or its caches, credentials, fonts, the
GeoIP database, or any other runtime state. The audit also fails when a built
artifact's version does not match the version semantic-release is about to tag.
"""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path

FORBIDDEN_SUFFIXES = (
    ".mmdb",  # GeoIP database
    ".ttf",
    ".otf",
    ".ttc",
    ".woff",
    ".woff2",  # fonts
    ".pem",
    ".key",
    ".crt",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",  # credentials
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".node",  # native binaries
    ".zip",  # packed browser profile / runtime archive
    ".db",
    ".sqlite",
    ".sqlite3",  # browser profile / caches
)

FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"(^|/)\.env(\.|$)", re.IGNORECASE),
    re.compile(r"(^|/)node_modules/", re.IGNORECASE),
    re.compile(r"cloakbrowser", re.IGNORECASE),
    re.compile(r"browser[-_]?profile", re.IGNORECASE),
    re.compile(r"(^|/)chrome([-_.]|$)", re.IGNORECASE),
    re.compile(r"geoip|geolite", re.IGNORECASE),
    re.compile(r"cookies?\.json$", re.IGNORECASE),
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.py[co]$"),
)


def archive_members(artifact: Path) -> list[str]:
    """Return every member name recorded in a wheel or sdist."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            return archive.namelist()
    with tarfile.open(artifact) as archive:
        return [member.name for member in archive.getmembers()]


def forbidden_members(members: list[str]) -> list[str]:
    """Return members that must never be shipped in a release artifact."""
    hits: list[str] = []
    for name in members:
        lowered = name.lower()
        if lowered.endswith(FORBIDDEN_SUFFIXES) or any(pattern.search(name) for pattern in FORBIDDEN_NAME_PATTERNS):
            hits.append(name)
    return hits


def metadata_version(artifact: Path) -> str:
    """Read the declared version from a wheel's METADATA or an sdist's PKG-INFO."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            text = archive.read(metadata).decode("utf-8")
    else:
        with tarfile.open(artifact) as archive:
            member = next(member for member in archive.getmembers() if member.name.endswith("PKG-INFO"))
            extracted = archive.extractfile(member)
            if extracted is None:
                msg = f"could not read {member.name} from {artifact.name}"
                raise SystemExit(msg)
            text = extracted.read().decode("utf-8")
    for line in text.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    msg = f"{artifact.name} has no Version field in its metadata"
    raise SystemExit(msg)


def audit_dist(dist: Path, expect_version: str | None = None) -> list[str]:
    """Audit the wheel and sdist in ``dist``; return a list of failures (empty when clean)."""
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [
            (
                f"expected exactly one wheel and one sdist in {dist}, "
                f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
            )
        ]

    failures: list[str] = []
    for artifact in (*wheels, *sdists):
        members = archive_members(artifact)
        hits = forbidden_members(members)
        if hits:
            failures.append(f"{artifact.name} contains forbidden entries: {', '.join(hits)}")
        version = metadata_version(artifact)
        if expect_version is not None and version != expect_version:
            failures.append(f"{artifact.name} version {version!r} != expected {expect_version!r}")
        print(f"audited {artifact.name}: version={version}, {len(members)} entries")
    return failures


def main() -> None:
    """Run the release artifact audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="Directory holding the built wheel and sdist")
    parser.add_argument("--expect-version", help="Fail when a built artifact's version differs")
    args = parser.parse_args()

    failures = audit_dist(args.dist, args.expect_version)
    if failures:
        raise SystemExit("\n".join(failures))
    print("artifact audit passed")


if __name__ == "__main__":
    main()
