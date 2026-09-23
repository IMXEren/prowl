from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import TestCase

_SCRIPT = Path(__file__).resolve().parent.parent / ".github" / "scripts" / "ghcr_package.py"
_SPEC = importlib.util.spec_from_file_location("ghcr_package", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
ghcr_package = importlib.util.module_from_spec(_SPEC)
sys.modules["ghcr_package"] = ghcr_package
_SPEC.loader.exec_module(ghcr_package)


def _version(version_id: int, created: int, *tags: str) -> object:
    return ghcr_package.PackageVersion(version_id, f"2026-01-{created:02d}T00:00:00Z", tags)


class GhcrRetentionTests(TestCase):
    def test_retains_versions_not_individual_tags(self) -> None:
        versions = [
            _version(1, 1, "1.0.0"),
            _version(2, 2, "1.1.0"),
            _version(3, 3, "1.2.0", "latest"),
            _version(4, 4, "1.3.0-dev.1"),
            _version(5, 5, "1.3.0-dev.2"),
            _version(6, 6, "1.3.0-dev.3"),
            _version(7, 7, "1.3.0-dev.4", "dev"),
            _version(8, 8, "unclassified"),
        ]

        protected, deletions = ghcr_package.retention_plan(
            versions,
            "1.3.0-dev.4",
            keep_prerelease=3,
            keep_stable=2,
        )

        self.assertEqual(protected, {2, 3, 5, 6, 7})
        self.assertEqual([version.version_id for version in deletions], [1, 4])

    def test_refuses_when_current_manifest_is_not_in_newest_window(self) -> None:
        versions = [
            _version(1, 1, "1.0.0", "latest"),
            _version(2, 2, "1.1.0"),
            _version(3, 3, "1.2.0"),
        ]

        with self.assertRaisesRegex(ghcr_package.RetentionSafetyError, "Current manifest"):
            ghcr_package.retention_plan(versions, "1.0.0", keep_prerelease=3, keep_stable=2)

    def test_refuses_mixed_stable_and_prerelease_version_tags(self) -> None:
        version = _version(1, 1, "1.0.0", "1.1.0-dev.1")

        with self.assertRaisesRegex(ghcr_package.RetentionSafetyError, "mixes"):
            ghcr_package.classify_version(version)

    def test_refuses_ambiguous_current_tag(self) -> None:
        versions = [_version(1, 1, "1.0.0"), _version(2, 2, "1.0.0", "latest")]

        with self.assertRaisesRegex(ghcr_package.RetentionSafetyError, "exactly one"):
            ghcr_package.retention_plan(versions, "1.0.0", keep_prerelease=3, keep_stable=2)
