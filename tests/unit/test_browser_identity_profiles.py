"""Disjoint identity roots retain archive ownership without launching browsers."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle.startup import BrowserLifecycle
from prowl.browser.proxy.egress import (
    DEFAULT_EGRESS_NAME,
    EgressError,
    EgressPool,
    derive_egress_paths,
    parse_egress_spec,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock

_URL = "socks5://127.0.0.1:10001"


class DerivedProfileRootTests(TestCase):
    """A named root is a sibling of the default, never a child of it."""

    def test_default_selector_keeps_the_configured_paths(self) -> None:
        paths = derive_egress_paths("/state/profile", "/state/browser-profile.zip", DEFAULT_EGRESS_NAME)
        self.assertEqual(paths, ("/state/profile", "/state/browser-profile.zip"))

    def test_named_root_is_a_sibling_not_a_child(self) -> None:
        directory, archive = derive_egress_paths("/state/profile", "/state/browser-profile.zip", "decodo")
        default = Path("/state/profile")
        derived = Path(directory)
        self.assertEqual(derived.parent, default.parent)
        self.assertNotEqual(derived, default)
        self.assertFalse(derived.is_relative_to(default))
        self.assertEqual(Path(archive), Path("/state/browser-profile-decodo.zip"))

    def test_named_root_derivation_is_stable(self) -> None:
        first = derive_egress_paths("/state/profile", "/state/browser-profile.zip", "decodo")
        second = derive_egress_paths("/state/profile", "/state/browser-profile.zip", "decodo")
        self.assertEqual(first, second)

    def test_two_named_roots_are_disjoint_siblings(self) -> None:
        default = Path("/state/profile")
        one = Path(derive_egress_paths("/state/profile", "/state/browser-profile.zip", "one")[0])
        two = Path(derive_egress_paths("/state/profile", "/state/browser-profile.zip", "two")[0])
        self.assertNotEqual(one, two)
        self.assertFalse(one.is_relative_to(two))
        self.assertFalse(two.is_relative_to(one))
        self.assertFalse(one.is_relative_to(default))
        self.assertFalse(two.is_relative_to(default))

    def test_native_default_folder_name_is_a_separate_sibling(self) -> None:
        directory, _archive = derive_egress_paths("/state/profile", "/state/profile.zip", "Default")
        self.assertEqual(Path(directory), Path("/state/profile-Default"))
        self.assertFalse(Path(directory).is_relative_to(Path("/state/profile")))

    def test_environment_names_use_the_same_policy_without_credential_echo(self) -> None:
        for raw in ("a.=socks5://user:secret@127.0.0.1:1080", "socks5://user:secret@127.0.0.1:1080"):
            with self.assertRaises(EgressError) as error:
                parse_egress_spec(raw)
            self.assertNotIn("secret", str(error.exception))


class ProfilePackIsolationTests(TestCase):
    """The real profile archive code keeps each identity's state to itself."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _lifecycle(self, profile_dir: Path, archive: Path) -> BrowserLifecycle:
        return BrowserLifecycle(
            BrowserRuntimeState(max_groups=1),
            profile_dir=str(profile_dir),
            profile_archive=archive,
        )

    def _write(self, profile_dir: Path, cookie: bytes) -> None:
        target = profile_dir / "Default" / "Cookies"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cookie)

    def _contents(self, archive: Path) -> dict[str, bytes]:
        with zipfile.ZipFile(archive) as zf:
            return {name: zf.read(name) for name in zf.namelist()}

    def test_default_pack_excludes_a_sibling_named_profile(self) -> None:
        default_dir = self.root / "profile"
        named_dir = self.root / "profile-decodo"
        self._write(default_dir, b"default-cookie")
        self._write(named_dir, b"named-cookie")

        archive = self._lifecycle(default_dir, self.root / "profile.zip").pack_profile()

        assert archive is not None
        entries = self._contents(archive)
        self.assertEqual(entries, {"Default/Cookies": b"default-cookie"})

    def test_named_pack_contains_only_its_own_state(self) -> None:
        default_dir = self.root / "profile"
        named_dir = self.root / "profile-decodo"
        self._write(default_dir, b"default-cookie")
        self._write(named_dir, b"named-cookie")

        archive = self._lifecycle(named_dir, self.root / "profile-decodo.zip").pack_profile()

        assert archive is not None
        entries = self._contents(archive)
        self.assertEqual(entries, {"Default/Cookies": b"named-cookie"})

    def test_default_restore_leaves_a_sibling_named_profile_untouched(self) -> None:
        default_dir = self.root / "profile"
        named_dir = self.root / "profile-decodo"
        self._write(default_dir, b"restore-me")
        lifecycle = self._lifecycle(default_dir, self.root / "profile.zip")
        archive = lifecycle.pack_profile()
        assert archive is not None

        self._write(default_dir, b"changed")
        (default_dir / "Default" / "Extra").write_bytes(b"extra")
        self._write(named_dir, b"sibling")

        self.assertTrue(lifecycle.unpack_profile())

        self.assertEqual((default_dir / "Default" / "Cookies").read_bytes(), b"restore-me")
        self.assertFalse((default_dir / "Default" / "Extra").exists())
        self.assertEqual((named_dir / "Default" / "Cookies").read_bytes(), b"sibling")


class EgressOwnershipValidationTests(IsolatedAsyncioTestCase):
    """Names and resolved paths are checked once, before any native factory runs."""

    def _config(self, root: Path) -> BrowserConfig:
        return BrowserConfig(profile_dir=str(root / "profile"), profile_archive=str(root / "profile.zip"))

    def _root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    async def test_reserved_default_selector_is_rejected(self) -> None:
        with self.assertRaises(EgressError):
            EgressPool(self._config(self._root()), {DEFAULT_EGRESS_NAME: _URL})

    async def test_path_unsafe_programmatic_names_are_rejected(self) -> None:
        for name in ("../evil", "with space", "slash/name", ".hidden", "a\\b", "a\n", "a."):
            with self.assertRaises(EgressError, msg=name):
                EgressPool(self._config(self._root()), {name: _URL})

    async def test_ownership_is_validated_before_the_native_factory(self) -> None:
        factory: MagicMock
        with patch("prowl.browser.proxy.egress.create_egress_browser") as factory:
            with self.assertRaises(EgressError):
                EgressPool(self._config(self._root()), {"../evil": _URL})
            factory.assert_not_called()

    async def test_a_named_archive_under_the_default_profile_is_rejected_without_echoing_paths(self) -> None:
        root = self._root()
        profile = root / "profile"
        config = BrowserConfig(profile_dir=str(profile), profile_archive=str(profile / "nested.zip"))
        with self.assertRaises(EgressError) as ctx:
            EgressPool(config, {"decodo": "socks5://user:secret@127.0.0.1:1080"})
        message = str(ctx.exception)
        self.assertNotIn(str(profile), message)
        self.assertNotIn("nested", message)
        self.assertNotIn("secret", message)

    async def test_case_colliding_identities_follow_native_path_semantics(self) -> None:
        config = self._config(self._root())
        folds_case = os.path.normcase("a") == os.path.normcase("A")
        if folds_case:
            with self.assertRaises(EgressError):
                EgressPool(config, {"a": _URL, "A": _URL})
        else:
            pool = EgressPool(config, {"a": _URL, "A": _URL})
            self.assertEqual(pool.names(), ("A", "a"))

    async def test_coincident_archives_are_rejected(self) -> None:
        root = self._root()
        config = self._config(root)
        with (
            patch(
                "prowl.browser.proxy.egress.derive_egress_paths",
                return_value=(str(root / "named"), config.profile_archive),
            ),
            self.assertRaisesRegex(EgressError, "archives must not coincide"),
        ):
            EgressPool(config, {"one": _URL})

    async def test_nested_profile_aliases_are_rejected(self) -> None:
        root = self._root()
        config = self._config(root)
        paths = (str(Path(config.profile_dir) / "nested"), str(root / "named.zip"))
        with (
            patch("prowl.browser.proxy.egress.derive_egress_paths", return_value=paths),
            self.assertRaisesRegex(EgressError, "directories must not coincide or contain"),
        ):
            EgressPool(config, {"one": _URL})
