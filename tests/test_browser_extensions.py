"""Unpacked extensions: discovery, launch flags, and the configuration seam.

A deployment may mount a directory of unpacked Chromium extensions, such as an ad
blocker. These pin the contract that matters: the default launch is unchanged when no
directory is configured, an unusable extension in a mounted directory never stops the
browser from starting, and the flags are added on the one launch path so the priming
launch and the live launch agree.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Self, TypedDict, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from loguru import logger

from prowl.browser.config import BrowserConfig
from prowl.browser.driver import BrowserRuntimeState, DriverStartupConfig
from prowl.browser.egress import create_egress_browser
from prowl.browser.extensions import Extension, discover_extensions, extension_launch_arguments
from prowl.browser.lifecycle import BrowserLifecycle

# ruff: noqa: S108


def _write_extension(root: Path, directory: str, manifest: object | str) -> Path:
    """Create one extension directory under *root* and return it."""
    extension = root / directory
    extension.mkdir(parents=True, exist_ok=True)
    body = manifest if isinstance(manifest, str) else json.dumps(manifest)
    (extension / "manifest.json").write_text(body, encoding="utf-8")
    return extension


def _manifest(name: str | None = "Ad Blocker") -> dict[str, object]:
    """Return a minimal Manifest V3 manifest, optionally without a name."""
    manifest: dict[str, object] = {"manifest_version": 3, "version": "1.0"}
    if name is not None:
        manifest["name"] = name
    return manifest


class _ListSink:
    """Loguru sink that collects the messages it is given."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = messages

    def write(self, message: str) -> None:
        """Collect one log message."""
        self.messages.append(message)


class ExtensionDiscoveryTests(TestCase):
    """What the directory scan accepts, skips, and names."""

    def setUp(self: Self) -> None:
        """Create a temporary extensions root per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_no_directory_configured_loads_nothing(self) -> None:
        """Default: an unconfigured deployment contributes no argument at all."""
        self.assertEqual(extension_launch_arguments(None), [])
        self.assertEqual(extension_launch_arguments(""), [])

    def test_empty_directory_loads_nothing(self) -> None:
        """An empty mounted directory launches exactly as an absent one does."""
        self.assertEqual(extension_launch_arguments(str(self.root)), [])

    def test_directory_that_does_not_exist_loads_nothing(self) -> None:
        """A path that is not a directory is reported and ignored, not fatal."""
        missing = self.root / "absent"
        self.assertEqual(extension_launch_arguments(str(missing)), [])
        self.assertEqual(extension_launch_arguments(str(self.root / "manifest.json")), [])

    def test_one_extension_gets_both_flags(self) -> None:
        """One extension is added, and the browser is limited to exactly that set."""
        extension = _write_extension(self.root, "ublock", _manifest("uBlock Origin Lite"))

        arguments = extension_launch_arguments(str(self.root))

        self.assertEqual(
            arguments,
            [f"--load-extension={extension}", f"--disable-extensions-except={extension}"],
        )

    def test_several_extensions_share_one_flag_pair(self) -> None:
        """Every extension is listed in the same two arguments, ordered by directory name."""
        first = _write_extension(self.root, "aaa", _manifest("First"))
        second = _write_extension(self.root, "bbb", _manifest("Second"))

        arguments = extension_launch_arguments(str(self.root))

        self.assertEqual(len(arguments), 2)
        self.assertEqual(arguments[0], f"--load-extension={first},{second}")
        self.assertEqual(arguments[1], f"--disable-extensions-except={first},{second}")

    def test_only_the_two_extension_flags_are_added(self) -> None:
        """No extra launch argument sneaks in with the feature."""
        _write_extension(self.root, "one", _manifest())

        flags = [argument.split("=", 1)[0] for argument in extension_launch_arguments(str(self.root))]

        self.assertEqual(flags, ["--load-extension", "--disable-extensions-except"])

    def test_subdirectory_without_manifest_is_skipped(self) -> None:
        """A directory that is not an extension is skipped, and does not stop the scan."""
        kept = _write_extension(self.root, "real", _manifest())
        (self.root / "not-an-extension").mkdir()
        (self.root / "not-an-extension" / "readme.txt").write_text("data", encoding="utf-8")

        self.assertEqual([extension.path for extension in discover_extensions(str(self.root))], [str(kept)])

    def test_nested_extension_is_not_discovered(self) -> None:
        """Only immediate subdirectories are extensions, so a nested copy is not loaded."""
        _write_extension(self.root / "outer", "inner", _manifest())

        self.assertEqual(discover_extensions(str(self.root)), [])

    def test_unparseable_manifest_is_skipped_and_the_rest_still_load(self) -> None:
        """One broken manifest never aborts the launch or hides a usable extension."""
        kept = _write_extension(self.root, "good", _manifest("Good"))
        _write_extension(self.root, "broken-json", "{not json")
        _write_extension(self.root, "not-an-object", "[1, 2, 3]")

        found = discover_extensions(str(self.root))

        self.assertEqual([extension.path for extension in found], [str(kept)])

    def test_manifest_name_is_reported_while_the_flag_uses_the_directory(self) -> None:
        """A renamed directory still reports the extension's own name, and loads from its path."""
        directory = _write_extension(self.root, "mv3-adblock", _manifest("uBlock Origin Lite"))

        found = discover_extensions(str(self.root))

        self.assertEqual(found, [Extension(name="uBlock Origin Lite", path=str(directory))])
        self.assertEqual(extension_launch_arguments(str(self.root))[0], f"--load-extension={directory}")

    def test_manifest_without_a_name_falls_back_to_the_directory_name(self) -> None:
        """A nameless manifest is still loadable, so it is named after its directory."""
        _write_extension(self.root, "unnamed-extension", _manifest(name=None))

        found = discover_extensions(str(self.root))

        self.assertEqual([extension.name for extension in found], ["unnamed-extension"])

    def test_loaded_count_is_logged_by_name(self) -> None:
        """The launch reports how many extensions loaded, naming each from its manifest."""
        _write_extension(self.root, "one", _manifest("uBlock Origin Lite"))
        _write_extension(self.root, "two", _manifest("Other Extension"))
        messages: list[str] = []
        sink = logger.add(_ListSink(messages), level="INFO")
        self.addCleanup(logger.remove, sink)

        extension_launch_arguments(str(self.root))

        logged = "\n".join(messages)
        self.assertIn("Loading 2 browser extension(s)", logged)
        self.assertIn("uBlock Origin Lite", logged)
        self.assertIn("Other Extension", logged)


def _fingerprint() -> MagicMock:
    fp = MagicMock()
    fp.options.arguments = ["--remote-debugging-port=9999", "--window-size=1920,980"]
    return fp


class _DriverDouble:
    """Mutable test double for lifecycle-to-driver calls."""

    def __init__(self) -> None:
        """Create async driver edge mocks."""
        self.start_live = AsyncMock()


async def _noop_popup_handler(_page: object) -> None:
    return None


class _StartResult(TypedDict):
    request: DriverStartupConfig
    prime_arguments: list[str]


class ExtensionLaunchWiringTests(IsolatedAsyncioTestCase):
    """Extensions reach Chromium through the one launch path, and only when configured."""

    def setUp(self: Self) -> None:
        """Create a temporary extensions root and a driver double per test."""
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.driver = _DriverDouble()
        self.lifecycle = BrowserLifecycle(cast("BrowserRuntimeState", self.driver))
        self.primed: list[list[str]] = []

    async def _start(self) -> _StartResult:
        with (
            patch.object(BrowserLifecycle, "unpack_profile"),
            patch.object(BrowserLifecycle, "webdata_path", return_value=Path("/tmp/missing-web-data")),
            patch.object(
                BrowserLifecycle,
                "_prime_profile",
                AsyncMock(side_effect=lambda arguments: self.primed.append(list(arguments))),
            ),
            patch.object(BrowserLifecycle, "_inject_search_engine"),
            patch("prowl.browser.lifecycle.startup.ensure_binary"),
            patch("prowl.browser.lifecycle.startup.get_free_port", return_value=9999),
            patch("prowl.browser.lifecycle.startup.FingerprintManager", return_value=_fingerprint()),
            patch.object(BrowserLifecycle, "_register_atexit"),
        ):
            await self.lifecycle.start(is_running=lambda: False, popup_handler=_noop_popup_handler)
        await_args = self.driver.start_live.await_args
        if await_args is None:
            msg = "driver.start_live was not awaited"
            raise AssertionError(msg)
        return {
            "request": cast("DriverStartupConfig", await_args.args[0]),
            "prime_arguments": self.primed[-1] if self.primed else [],
        }

    async def test_managed_policies_are_applied_before_the_launch(self) -> None:
        """Chromium reads policy as it starts, so it is written before any launch."""
        self.lifecycle.policy_dir = str(self.root)

        with patch("prowl.browser.lifecycle.startup.apply_managed_policies") as apply_policies:
            await self._start()

        apply_policies.assert_called_once_with(str(self.root))

    async def test_extensions_reach_both_the_priming_and_the_live_launch(self) -> None:
        """One launch path: the flags are built once and used by both launches."""
        extension = _write_extension(self.root, "adblock", _manifest("uBlock Origin Lite"))
        self.lifecycle.extensions_dir = str(self.root)

        result = await self._start()

        expected = [f"--load-extension={extension}", f"--disable-extensions-except={extension}"]
        live = result["request"].launch_arguments
        self.assertEqual(live[-2:], expected)
        self.assertEqual(result["prime_arguments"], live)

    async def test_no_extensions_leaves_the_launch_arguments_unchanged(self) -> None:
        """Default: only the fingerprint arguments and the egress proxy are passed."""
        self.lifecycle.extensions_dir = None
        self.lifecycle.proxy_url = "socks5://127.0.0.1:1080"

        result = await self._start()

        self.assertEqual(
            result["request"].launch_arguments,
            [
                "--remote-debugging-port=9999",
                "--window-size=1920,980",
                "--window-position=0,0",
                "--proxy-server=socks5://127.0.0.1:1080",
            ],
        )

    async def test_empty_directory_leaves_the_launch_arguments_unchanged(self) -> None:
        """An empty mounted directory is indistinguishable from none at launch time."""
        self.lifecycle.extensions_dir = str(self.root)

        result = await self._start()

        self.assertEqual(
            result["request"].launch_arguments,
            ["--remote-debugging-port=9999", "--window-size=1920,980", "--window-position=0,0"],
        )

    def test_apply_config_adopts_the_extensions_directory(self) -> None:
        """The extensions directory is a launch input on the configuration seam."""
        lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))

        lifecycle.apply_config(
            BrowserConfig(
                proxy_url="socks5://127.0.0.1:1080",
                profile_dir="/x",
                profile_archive="/y.zip",
                extensions_dir=str(self.root),
            ),
            is_running=lambda: False,
        )

        self.assertEqual(lifecycle.extensions_dir, str(self.root))

    def test_browser_config_reads_the_directory_from_the_environment(self) -> None:
        """The service and the browser core read the same variable."""
        with patch.dict("os.environ", {"PROWL_EXTENSIONS_DIR": str(self.root)}, clear=True):
            self.assertEqual(BrowserConfig.from_env().extensions_dir, str(self.root))

        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(BrowserConfig.from_env().extensions_dir)

    def test_browser_lifecycle_defaults_to_the_environment(self) -> None:
        """A lifecycle built without explicit config still honours the variable."""
        with patch.dict("os.environ", {"PROWL_EXTENSIONS_DIR": str(self.root)}, clear=True):
            lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))

        self.assertEqual(lifecycle.extensions_dir, str(self.root))

    def test_egress_browsers_load_the_same_extensions(self) -> None:
        """A named egress carries the deployment-wide extensions directory too."""
        egress_browser = create_egress_browser(
            name="decodo",
            proxy_url="socks5://127.0.0.1:1080",
            profile_dir=str(self.root / "profile"),
            profile_archive=str(self.root / "profile.zip"),
            preferred_cdp_port=9300,
            extensions_dir=str(self.root),
        )

        self.assertEqual(egress_browser._lifecycle.extensions_dir, str(self.root))
