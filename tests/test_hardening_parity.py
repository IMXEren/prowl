"""Hardening parity tests: browser config seam, launch flags, and packaging.

These pin the cross-cutting hardening contracts that are easy to regress:
environment-driven profile paths, the explicit reconfiguration seam, the
``/dev/shm`` flag removal, and the package/Docker wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.driver import BrowserRuntimeState
from prowl.browser.exceptions import BrowserStartError
from prowl.browser.lifecycle import BrowserLifecycle
from prowl.browser.options import BrowserOptions

_ROOT = Path(__file__).resolve().parent.parent


class BrowserLifecycleConfigTests(TestCase):
    """Launch inputs come from the environment and can be set explicitly."""

    def test_profile_paths_come_from_environment(self) -> None:
        env = {"PROWL_PROFILE_DIR": "/state/profile", "PROWL_PROFILE_ARCHIVE": "/state/browser-profile.zip"}
        with patch.dict("os.environ", env, clear=True):
            lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))
        self.assertEqual(lifecycle.profile_dir, "/state/profile")
        self.assertEqual(lifecycle.profile_archive, Path("/state/browser-profile.zip"))

    def test_apply_config_sets_launch_inputs(self) -> None:
        lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))
        lifecycle.apply_config(
            BrowserConfig(proxy_url="socks5://127.0.0.1:1080", profile_dir="/x", profile_archive="/y.zip"),
            is_running=lambda: False,
        )
        self.assertEqual(lifecycle.proxy_url, "socks5://127.0.0.1:1080")
        self.assertEqual(lifecycle.profile_dir, "/x")
        self.assertEqual(lifecycle.profile_archive, Path("/y.zip"))

    def test_apply_config_refuses_while_running(self) -> None:
        lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))
        with self.assertRaises(BrowserStartError):
            lifecycle.apply_config(BrowserConfig(), is_running=lambda: True)

    def test_apply_config_refuses_while_shutting_down(self) -> None:
        from prowl.browser.lifecycle import BrowserShutdownState  # noqa: PLC0415

        lifecycle = BrowserLifecycle(BrowserRuntimeState(max_groups=1))
        lifecycle._shutdown_state = BrowserShutdownState.IN_PROGRESS
        with self.assertRaises(BrowserStartError):
            lifecycle.apply_config(BrowserConfig(), is_running=lambda: False)


class BrowserOptionParityTests(TestCase):
    """The shm workaround must stay out so a large Compose /dev/shm is effective."""

    def test_disable_dev_shm_usage_is_not_set(self) -> None:
        options = BrowserOptions.new()
        self.assertNotIn("--disable-dev-shm-usage", options.arguments)


class PackagingParityTests(TestCase):
    """Package metadata and Docker wiring stay in lockstep with the project."""

    def test_package_declares_inline_types(self) -> None:
        self.assertTrue((_ROOT / "src" / "prowl" / "py.typed").is_file())

    def test_sdist_includes_license(self) -> None:
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"LICENSE"', pyproject)
        self.assertIn('name = "prowl"', pyproject)

    def test_dockerfile_installs_git_and_copies_license(self) -> None:
        dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("git \\", dockerfile)
        self.assertIn("COPY pyproject.toml README.md LICENSE ./", dockerfile)
        self.assertIn("PROWL_PROFILE_ARCHIVE=/state/browser-profile.zip", dockerfile)

    def test_compose_uses_prowl_state_volume(self) -> None:
        compose = (_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("prowl:local", compose)
        self.assertNotIn("ghcr.io/imxeren/prowl", compose)
        self.assertIn("PROWL_PROFILE_DIR=/state/profile", compose)
        self.assertIn("prowl-state:/state", compose)
        self.assertIn('shm_size: "2gb"', compose)

    def test_env_example_documents_archive_path(self) -> None:
        env_example = (_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("PROWL_PROFILE_ARCHIVE=/state/browser-profile.zip", env_example)
