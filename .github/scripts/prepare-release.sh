#!/usr/bin/env bash
# Build and audit the release wheel/sdist for the version semantic-release chose.
# Called from .releaserc's @semantic-release/exec prepareCmd with
# $nextRelease.version so the attached assets are version-accurate and clean.
set -euo pipefail

version="${1:?semantic-release version is required}"

uv run python .github/scripts/set_version.py "$version"

rm -rf dist
uv run python -m build

uv run python .github/scripts/audit_artifacts.py dist --expect-version "$version"

echo "Prepared audited release artifacts for version ${version}"
