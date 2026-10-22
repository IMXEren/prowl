"""Checks for the named-profile migration planner and its explicit apply step.

Every case builds only synthetic :class:`tempfile.TemporaryDirectory` trees; no real
profile, environment default, or native resource is touched. Planner cases pin that the
inspected tree is byte-for-byte unchanged; apply cases mutate only the synthetic tree.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts import migrate_identity_profiles as planner


def _write_old_root(root: Path, name: str) -> Path:
    """Create a nested legacy ``root/name`` that looks like a Chromium user-data root."""
    source = root / name
    (source / "Default").mkdir(parents=True)
    (source / "Local State").write_bytes(b"local-state-bytes")
    (source / "Default" / "Cookies").write_bytes(b"cookie-bytes")
    (source / "Default" / "Preferences").write_bytes(b"preferences-bytes")
    return source


def _snapshot(root: Path) -> dict[str, bytes]:
    """Return every file under *root* keyed by relative path, with its exact bytes."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


class MigrationPlanTests(TestCase):
    def test_genuine_old_profile_plans_a_sibling_move_and_preserves_bytes(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            source = _write_old_root(root, "decodo")
            before = _snapshot(base)

            plan = planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["decodo"])

            self.assertEqual(plan.profile, root.resolve())
            self.assertEqual(plan.archive, (base / "browser-profile.zip").resolve())
            entry = plan.entries[0]
            self.assertEqual(entry.status, planner.STATUS_MOVE)
            self.assertEqual(entry.source, source.resolve())
            self.assertEqual(entry.destination, (base / "profile-decodo").resolve())
            self.assertEqual(entry.archive, (base / "browser-profile-decodo.zip").resolve())
            self.assertFalse(entry.destination.exists())
            self.assertEqual(_snapshot(base), before)

    def test_existing_destination_conflicts_without_changes(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            (base / "profile-decodo" / "Default").mkdir(parents=True)
            (base / "profile-decodo" / "Local State").write_bytes(b"newer")
            before = _snapshot(base)

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["decodo"])

            self.assertEqual(_snapshot(base), before)

    def test_source_named_default_is_ambiguous(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "Default")

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["Default"])

    def test_markerless_source_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            (root / "decodo").mkdir(parents=True)
            (root / "decodo" / "not-a-profile").write_bytes(b"x")

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["decodo"])

    def test_linked_source_is_rejected_when_the_host_supports_it(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            outside = base / "outside"
            _write_old_root(outside.parent, "outside")
            root.mkdir(parents=True)
            try:
                (root / "decodo").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("host cannot create a directory symlink")
            before = _snapshot(base)

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["decodo"])

            self.assertEqual(_snapshot(base), before)

    def test_absent_source_reports_migrated_archive_only_and_no_state(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            root.mkdir()
            _write_old_root(base, "profile-one")
            (base / "browser-profile-two.zip").write_bytes(b"standalone-archive")

            plan = planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["one", "two", "three"])

            statuses = {entry.name: entry.status for entry in plan.entries}
            self.assertEqual(statuses["one"], planner.STATUS_MIGRATED)
            self.assertEqual(statuses["two"], planner.STATUS_ARCHIVE_ONLY)
            self.assertEqual(statuses["three"], planner.STATUS_NO_STATE)

    def test_bundled_legacy_state_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            root.mkdir()
            archive = base / "browser-profile.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("decodo/Default/Cookies", b"cookie-bytes")
                package.writestr("decodo/Local State", b"local-state-bytes")

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(archive), ["decodo"])

    def test_corrupt_default_archive_is_a_fixed_safe_error(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            root.mkdir()
            archive = base / "browser-profile.zip"
            archive.write_bytes(b"not-a-zip-payload")

            with self.assertRaises(planner.MigrationError) as error:
                planner.plan_migration(str(root), str(archive), ["decodo"])

            self.assertNotIn("not-a-zip-payload", str(error.exception))

    def test_duplicate_selection_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")

            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "browser-profile.zip"), ["decodo", "decodo"])

    def test_cli_main_prints_a_read_only_plan_without_credentials(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            before = _snapshot(base)
            output = io.StringIO()

            with patch("sys.stdout", output):
                code = planner.main(
                    [
                        "--profile-dir",
                        str(root),
                        "--profile-archive",
                        str(base / "browser-profile.zip"),
                        "--names",
                        "decodo",
                    ],
                )

            self.assertEqual(code, 0)
            self.assertEqual(_snapshot(base), before)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["entries"][0]["name"], "decodo")
            self.assertEqual(payload["entries"][0]["status"], "move")
            self.assertNotIn("://", output.getvalue())
            self.assertNotIn("socks", output.getvalue())

    def test_corrupt_archive_is_rejected_even_with_a_live_source(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            archive = base / "profile.zip"
            archive.write_bytes(b"invalid")
            before = _snapshot(base)
            with self.assertRaises(planner.MigrationError) as error:
                planner.plan_migration(str(root), str(archive), ["one"])
            self.assertTrue(error.exception.__suppress_context__)
            self.assertEqual(_snapshot(base), before)

    def test_dangling_destination_link_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            destination = base / "profile-one"
            try:
                destination.symlink_to(base / "missing", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("host cannot create directory symlinks")
            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(root), str(base / "profile.zip"), ["one"])
            self.assertTrue(destination.is_symlink())
            self.assertTrue((root / "one" / "Local State").is_file())

    def test_existing_target_file_is_not_reported_as_migrated(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            (base / "profile-one").write_bytes(b"not-a-profile")
            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(base / "profile"), str(base / "profile.zip"), ["one"])

    def test_empty_target_does_not_hide_sole_bundled_state(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            (base / "profile-one").mkdir()
            archive = base / "profile.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("one/Default/Cookies", b"sole-copy")
            before = _snapshot(base)
            with self.assertRaises(planner.MigrationError):
                planner.plan_migration(str(base / "profile"), str(archive), ["one"])
            self.assertEqual(_snapshot(base), before)


class MigrationApplyTests(TestCase):
    def _bundle(self, archive: Path, members: dict[str, bytes], comment: bytes = b"") -> None:
        with zipfile.ZipFile(archive, "w") as package:
            for member, payload in members.items():
                package.writestr(member, payload)
            package.comment = comment

    def test_apply_moves_old_named_root_and_preserves_standalone_bytes(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            source = _write_old_root(root, "decodo")
            (source / "Default" / "Cookies").write_bytes(b"latest-cookie-bytes")
            standalone = base / "browser-profile-two.zip"
            standalone.write_bytes(b"standalone-named-archive")

            result = planner.apply_migration(str(root), str(base / "browser-profile.zip"), ["decodo", "two"])

            destination = base / "profile-decodo"
            self.assertTrue(destination.is_dir())
            self.assertEqual((destination / "Default" / "Cookies").read_bytes(), b"latest-cookie-bytes")
            self.assertEqual((destination / "Default" / "Preferences").read_bytes(), b"preferences-bytes")
            self.assertFalse(source.exists())
            self.assertEqual(standalone.read_bytes(), b"standalone-named-archive")
            self.assertIsNone(result.backup)
            self.assertFalse(result.archive_rewritten)
            self.assertEqual([entry.name for entry in result.moved], ["decodo"])
            self.assertFalse((base / "browser-profile-decodo.zip").exists())

    def test_apply_rewrites_bundle_drops_selected_subtree_and_keeps_original_backup(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            archive = base / "browser-profile.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("Local State", b"local-state-bytes")
                package.writestr(
                    zipfile.ZipInfo("Default/Cookies", date_time=(2021, 5, 6, 7, 8, 10)), b"native-cookie-bytes"
                )
                package.writestr("decodo/Default/Cookies", b"foreign-cookie-bytes")
                package.writestr("keepme/Default/data", b"unselected-bytes")
                package.comment = b"identity-bundle"
            original = archive.read_bytes()

            result = planner.apply_migration(str(root), str(archive), ["decodo"])

            self.assertTrue(result.archive_rewritten)
            backups = list(base.glob("browser-profile.zip.before-layout-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(result.backup, backups[0])
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertTrue((base / "profile-decodo").is_dir())
            self.assertFalse((root / "decodo").exists())
            with zipfile.ZipFile(archive) as package:
                names = package.namelist()
                self.assertIn("Default/Cookies", names)
                self.assertNotIn("decodo/Default/Cookies", names)
                self.assertIn("keepme/Default/data", names)
                self.assertIn("Local State", names)
                self.assertEqual(package.comment, b"identity-bundle")
                self.assertEqual(package.getinfo("Default/Cookies").date_time, (2021, 5, 6, 7, 8, 10))
                self.assertEqual(package.read("Default/Cookies"), b"native-cookie-bytes")
                self.assertEqual(package.read("keepme/Default/data"), b"unselected-bytes")

    def test_default_cli_remains_read_only_without_the_apply_flag(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            archive = base / "browser-profile.zip"
            self._bundle(archive, {"Default/Cookies": b"native", "decodo/Default/Cookies": b"foreign"})
            before = _snapshot(base)
            output = io.StringIO()

            with patch("sys.stdout", output):
                code = planner.main(
                    [
                        "--profile-dir",
                        str(root),
                        "--profile-archive",
                        str(archive),
                        "--names",
                        "decodo",
                    ],
                )

            self.assertEqual(code, 0)
            self.assertEqual(_snapshot(base), before)
            self.assertEqual(list(base.glob("*.before-layout-*")), [])
            self.assertFalse((base / "profile-decodo").exists())
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["entries"][0]["status"], "move")
            self.assertNotIn("applied", payload)

    def test_apply_leaves_a_bundle_without_selected_subtrees_untouched(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            archive = base / "browser-profile.zip"
            self._bundle(archive, {"Default/Cookies": b"native", "Local State": b"state"})
            original = archive.read_bytes()

            result = planner.apply_migration(str(root), str(archive), ["decodo"])

            self.assertFalse(result.archive_rewritten)
            self.assertIsNone(result.backup)
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(list(base.glob("*.before-layout-*")), [])
            self.assertEqual(list(base.glob("*prowl-staging*")), [])
            self.assertTrue((base / "profile-decodo").is_dir())

    def test_apply_without_a_default_archive_moves_and_creates_nothing(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")

            result = planner.apply_migration(str(root), str(base / "browser-profile.zip"), ["decodo"])

            self.assertFalse(result.archive_rewritten)
            self.assertIsNone(result.backup)
            self.assertFalse((base / "browser-profile.zip").exists())
            self.assertTrue((base / "profile-decodo").is_dir())

    def test_apply_cleans_bundle_for_an_already_migrated_destination(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            root.mkdir()
            destination = base / "profile-decodo"
            (destination / "Default").mkdir(parents=True)
            (destination / "Local State").write_bytes(b"state")
            (destination / "Default" / "Cookies").write_bytes(b"latest")
            archive = base / "browser-profile.zip"
            self._bundle(archive, {"decodo/Default/Cookies": b"stale", "Default/Cookies": b"native"})
            original = archive.read_bytes()

            result = planner.apply_migration(str(root), str(archive), ["decodo"])

            self.assertTrue(result.archive_rewritten)
            self.assertEqual(result.moved, ())
            self.assertEqual((destination / "Default" / "Cookies").read_bytes(), b"latest")
            self.assertFalse((root / "decodo").exists())
            backups = list(base.glob("browser-profile.zip.before-layout-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            with zipfile.ZipFile(archive) as package:
                self.assertNotIn("decodo/Default/Cookies", package.namelist())
                self.assertIn("Default/Cookies", package.namelist())

    def test_case_colliding_selection_never_removes_native_default_data(self) -> None:
        if os.path.normcase("Default") != os.path.normcase("dEfAuLt"):
            self.skipTest("host compares names case-sensitively")
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            root.mkdir()
            destination = base / "profile-dEfAuLt"
            (destination / "Default").mkdir(parents=True)
            (destination / "Local State").write_bytes(b"state")
            archive = base / "browser-profile.zip"
            self._bundle(archive, {"Default/Cookies": b"native", "Local State": b"state"})
            original = archive.read_bytes()

            result = planner.apply_migration(str(root), str(archive), ["dEfAuLt"])

            self.assertFalse(result.archive_rewritten)
            self.assertIsNone(result.backup)
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(list(base.glob("*.before-layout-*")), [])
            with zipfile.ZipFile(archive) as package:
                self.assertIn("Default/Cookies", package.namelist())

    def test_second_move_failure_rolls_back_and_retains_backup(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            _write_old_root(root, "two")
            archive = base / "browser-profile.zip"
            self._bundle(
                archive,
                {
                    "Default/Cookies": b"native",
                    "one/Default/Cookies": b"one",
                    "two/Default/Cookies": b"two",
                },
            )
            original = archive.read_bytes()
            real_rename = os.rename
            calls = {"count": 0}

            def flaky(source: str | bytes, destination: str | bytes) -> None:
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError
                real_rename(source, destination)

            with patch("scripts.migrate_identity_profiles.os.rename", flaky), self.assertRaises(planner.MigrationError):
                planner.apply_migration(str(root), str(archive), ["one", "two"])

            self.assertTrue((root / "one").is_dir())
            self.assertTrue((root / "two").is_dir())
            self.assertFalse((base / "profile-one").exists())
            self.assertFalse((base / "profile-two").exists())
            self.assertEqual(archive.read_bytes(), original)
            backups = list(base.glob("browser-profile.zip.before-layout-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(list(base.glob("*prowl-staging*")), [])
            self.assertEqual(list(base.glob("*.partial-*")), [])

    def test_final_archive_commit_failure_rolls_back_and_retains_backup(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            _write_old_root(root, "two")
            archive = base / "browser-profile.zip"
            self._bundle(
                archive,
                {
                    "Default/Cookies": b"native",
                    "one/Default/Cookies": b"one",
                    "two/Default/Cookies": b"two",
                },
            )
            original = archive.read_bytes()
            real_replace = os.replace

            def flaky_replace(source: str | bytes, destination: str | bytes) -> None:
                if Path(os.fsdecode(destination)) == archive:
                    raise OSError
                real_replace(source, destination)

            with (
                patch("scripts.migrate_identity_profiles.os.replace", flaky_replace),
                self.assertRaises(planner.MigrationError),
            ):
                planner.apply_migration(str(root), str(archive), ["one", "two"])

            self.assertTrue((root / "one").is_dir())
            self.assertTrue((root / "two").is_dir())
            self.assertFalse((base / "profile-one").exists())
            self.assertFalse((base / "profile-two").exists())
            self.assertEqual(archive.read_bytes(), original)
            backups = list(base.glob("browser-profile.zip.before-layout-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(list(base.glob("*prowl-staging*")), [])
            self.assertEqual(list(base.glob("*.partial-*")), [])

    def test_cli_apply_reports_and_performs_the_migration(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "decodo")
            archive = base / "browser-profile.zip"
            self._bundle(archive, {"Default/Cookies": b"native", "decodo/Default/Cookies": b"foreign"})
            output = io.StringIO()

            with patch("sys.stdout", output):
                code = planner.main(
                    [
                        "--profile-dir",
                        str(root),
                        "--profile-archive",
                        str(archive),
                        "--names",
                        "decodo",
                        "--apply",
                    ],
                )

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["applied"])
            self.assertTrue(payload["archive_rewritten"])
            self.assertEqual(payload["moved"][0]["name"], "decodo")
            self.assertEqual(payload["moved"][0]["destination"], str((base / "profile-decodo").resolve()))
            self.assertTrue(Path(payload["backup"]).is_file())
            self.assertTrue((base / "profile-decodo").is_dir())
            self.assertFalse((root / "decodo").exists())
            with zipfile.ZipFile(archive) as package:
                self.assertNotIn("decodo/Default/Cookies", package.namelist())
            self.assertNotIn("://", output.getvalue())

    def test_interrupt_after_rename_restores_the_attempted_move(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            real_rename = os.rename

            def interrupted(source: str, destination: str) -> None:
                real_rename(source, destination)
                if Path(destination) == base / "profile-one":
                    raise KeyboardInterrupt

            with (
                patch("scripts.migrate_identity_profiles.os.rename", interrupted),
                self.assertRaises(KeyboardInterrupt),
            ):
                planner.apply_migration(str(root), str(base / "profile.zip"), ["one"])
            self.assertTrue((root / "one" / "Default" / "Cookies").is_file())
            self.assertFalse((base / "profile-one").exists())

    def test_interrupt_after_archive_commit_keeps_the_committed_layout(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            archive = base / "profile.zip"
            self._bundle(archive, {"Default/Cookies": b"default", "one/Default/Cookies": b"old"})
            original = archive.read_bytes()
            real_replace = os.replace

            def interrupted(source: str, destination: str) -> None:
                real_replace(source, destination)
                if Path(destination) == archive:
                    raise KeyboardInterrupt

            with (
                patch("scripts.migrate_identity_profiles.os.replace", interrupted),
                self.assertRaises(KeyboardInterrupt),
            ):
                planner.apply_migration(str(root), str(archive), ["one"])
            self.assertFalse((root / "one").exists())
            self.assertTrue((base / "profile-one" / "Default" / "Cookies").is_file())
            with zipfile.ZipFile(archive) as package:
                self.assertEqual(package.namelist(), ["Default/Cookies"])
            backups = list(base.glob("profile.zip.before-layout-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)

    def test_matching_root_file_is_not_a_selected_directory_subtree(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            _write_old_root(base, "profile-one")
            archive = base / "profile.zip"
            self._bundle(archive, {"one": b"default-owned-file", "Default/Cookies": b"default"})
            original = archive.read_bytes()
            result = planner.apply_migration(str(base / "profile"), str(archive), ["one"])
            self.assertFalse(result.archive_rewritten)
            self.assertIsNone(result.backup)
            self.assertEqual(archive.read_bytes(), original)

    def test_invalid_environment_selection_is_reported_without_traceback(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            error = io.StringIO()
            with (
                patch.dict(os.environ, {"PROWL_EGRESSES": "socks5://user:secret@127.0.0.1:1080"}),
                patch("sys.stderr", error),
            ):
                code = planner.main(
                    ["--profile-dir", str(base / "profile"), "--profile-archive", str(base / "profile.zip")]
                )
            self.assertEqual(code, 1)
            self.assertNotIn("secret", error.getvalue())
            self.assertNotIn("Traceback", error.getvalue())

    def test_backup_is_private_on_posix(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX file modes do not establish Windows ACL behavior")
        with TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "profile"
            _write_old_root(root, "one")
            archive = base / "profile.zip"
            self._bundle(archive, {"Default/Cookies": b"default", "one/Default/Cookies": b"old"})
            archive.chmod(0o600)
            result = planner.apply_migration(str(root), str(archive), ["one"])
            assert result.backup is not None
            self.assertEqual(result.backup.stat().st_mode & 0o777, 0o600)
            self.assertEqual(archive.stat().st_mode & 0o777, 0o600)
