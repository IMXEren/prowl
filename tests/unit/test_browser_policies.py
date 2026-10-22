"""Managed policy: which files are applied, and how a bad mount is handled.

Chromium reads managed policy from a system directory whose path depends on the build branding,
so the configured directory is copied into every candidate. These pin that behaviour and the way
an unusable file is refused without losing the rest of the set.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Self
from unittest import TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.policies import POLICY_DIR_ENV, apply_managed_policies, policy_files


def _policy(root: Path, name: str, body: object | str) -> Path:
    """Write one policy file into *root* and return it."""
    path = root / name
    path.write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")
    return path


class PolicyApplicationTests(TestCase):
    """Applying a policy directory to the managed locations."""

    def setUp(self: Self) -> None:
        """Create a source directory and two target directories per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.source = base / "policies"
        self.source.mkdir()
        self.first = base / "chromium-managed"
        self.second = base / "chrome-managed"

    def _apply(self) -> list[object]:
        return list(apply_managed_policies(str(self.source), targets=[str(self.first), str(self.second)]))

    def test_no_directory_configured_writes_nothing(self) -> None:
        """Default: an unconfigured deployment touches no policy directory."""
        self.assertEqual(apply_managed_policies(None, targets=[str(self.first)]), [])
        self.assertFalse(self.first.exists())

    def test_empty_directory_writes_nothing(self) -> None:
        """An empty mount is not a policy set."""
        self.assertEqual(self._apply(), [])
        self.assertFalse(self.first.exists())

    def test_one_policy_file_reaches_every_candidate_directory(self) -> None:
        """Every candidate is written, because the branding decides which one is read."""
        _policy(self.source, "extensions.json", {"ExtensionSettings": {"*": {"toolbar_pin": "force_pinned"}}})

        written = self._apply()

        self.assertEqual(len(written), 2)
        for target in (self.first, self.second):
            body = json.loads((target / "extensions.json").read_text(encoding="utf-8"))
            self.assertEqual(body["ExtensionSettings"]["*"]["toolbar_pin"], "force_pinned")

    def test_several_policy_files_are_all_applied(self) -> None:
        """Chromium merges a directory, so every file is copied."""
        _policy(self.source, "a.json", {"HomepageLocation": "https://example.com"})
        _policy(self.source, "b.json", {"PasswordManagerEnabled": False})

        self._apply()

        self.assertEqual(sorted(path.name for path in self.first.iterdir()), ["a.json", "b.json"])

    def test_non_json_files_are_ignored(self) -> None:
        """A readme in the mount is not a policy."""
        _policy(self.source, "note.txt", "not policy")
        _policy(self.source, "real.json", {"HomepageLocation": "https://example.com"})

        self._apply()

        self.assertEqual([path.name for path in self.first.iterdir()], ["real.json"])

    def test_unparseable_policy_is_skipped_and_the_rest_still_applies(self) -> None:
        """One broken file must not cost the browser every other policy."""
        _policy(self.source, "broken.json", "{not json")
        _policy(self.source, "not-an-object.json", "[1, 2]")
        _policy(self.source, "good.json", {"HomepageLocation": "https://example.com"})

        self._apply()

        self.assertEqual([path.name for path in self.first.iterdir()], ["good.json"])

    def test_an_unwritable_target_does_not_fail_the_launch(self) -> None:
        """An unprivileged container reports the directory instead of raising."""
        _policy(self.source, "one.json", {"HomepageLocation": "https://example.com"})

        with patch("prowl.browser.policies.shutil.copyfile", side_effect=PermissionError("denied")):
            written = apply_managed_policies(str(self.source), targets=[str(self.first)])

        self.assertEqual(written, [])

    def test_policy_files_are_listed_in_name_order(self) -> None:
        """The scan is deterministic so the same mount always applies in the same order."""
        _policy(self.source, "b.json", {"PasswordManagerEnabled": False})
        _policy(self.source, "a.json", {"HomepageLocation": "https://example.com"})

        self.assertEqual([path.name for path in policy_files(self.source)], ["a.json", "b.json"])

    def test_a_directory_that_is_not_there_has_no_policy(self) -> None:
        """A missing mount is not an error."""
        self.assertEqual(policy_files(self.source / "absent"), [])
        self.assertEqual(apply_managed_policies(str(self.source / "absent"), targets=[str(self.first)]), [])

    def test_config_reads_the_policy_directory_from_the_environment(self) -> None:
        """The service and the browser core read the same variable."""
        with patch.dict("os.environ", {POLICY_DIR_ENV: str(self.source)}, clear=True):
            self.assertEqual(BrowserConfig.from_env().policy_dir, str(self.source))

        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(BrowserConfig.from_env().policy_dir)
