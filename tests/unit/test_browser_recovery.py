"""Regression tests for native browser recovery and profile preservation.

Coverage:
  - A live, non-empty profile directory is authoritative over a stale archive.
  - A missing or empty profile directory restores from the configured archive.
  - A dead local generation is retired before a relaunch; an externally attached dead
    browser refuses a local launch instead of being replaced.
  - Retiring a dead generation returns its group admission slots exactly once and rejects
    its stale context handles.
  - A failed isolated-context retirement keeps live ownership and can be retried.
  - Native connectivity, not just the Playwright projection, decides liveness.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from prowl.browser.driver import BrowserRuntimeState
from prowl.browser.exceptions import BrowserContextError, BrowserStartError
from prowl.browser.lifecycle import BrowserLifecycle

if TYPE_CHECKING:
    from collections.abc import Iterator


def _write_archive(path: Path, files: dict[str, str]) -> None:
    """Write a profile archive that ``unpack_profile`` can extract."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _profile_lifecycle(profile: Path, archive: Path) -> BrowserLifecycle:
    """Return a lifecycle owning *profile* and restoring from *archive*."""
    return BrowserLifecycle(
        cast("BrowserRuntimeState", MagicMock()),
        profile_dir=str(profile),
        profile_archive=archive,
    )


class ProfileRestoreTests(TestCase):
    """Automatic startup restores the archive only into a missing or empty profile."""

    def test_live_profile_overrides_a_stale_archive(self) -> None:
        """A non-empty profile directory is authoritative; the archive is not unpacked."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            profile.mkdir()
            (profile / "live-sentinel").write_text("live", encoding="utf-8")
            archive = root / "profile.zip"
            _write_archive(archive, {"stale-sentinel": "stale"})
            lifecycle = _profile_lifecycle(profile, archive)

            restored = lifecycle.restore_profile_from_archive()

            self.assertFalse(restored)
            self.assertTrue((profile / "live-sentinel").exists())
            self.assertFalse((profile / "stale-sentinel").exists())

    def test_missing_profile_directory_restores_the_archive(self) -> None:
        """A profile directory that does not exist yet is restored from the archive."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            archive = root / "profile.zip"
            _write_archive(archive, {"stale-sentinel": "stale"})
            lifecycle = _profile_lifecycle(profile, archive)

            restored = lifecycle.restore_profile_from_archive()

            self.assertTrue(restored)
            self.assertTrue((profile / "stale-sentinel").exists())

    def test_empty_profile_directory_restores_the_archive(self) -> None:
        """An empty profile directory carries no state, so the archive is restored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile"
            profile.mkdir()
            archive = root / "profile.zip"
            _write_archive(archive, {"stale-sentinel": "stale"})
            lifecycle = _profile_lifecycle(profile, archive)

            restored = lifecycle.restore_profile_from_archive()

            self.assertTrue(restored)
            self.assertTrue((profile / "stale-sentinel").exists())


def _native_context(*, closed: bool = False, connected: bool | None = None) -> MagicMock:
    """Return a context double whose native browser connection is controlled."""
    context = MagicMock()
    context.is_closed.return_value = closed
    if connected is None:
        context.browser = None
    else:
        context.browser.is_connected.return_value = connected
    return context


class NativeConnectivityTests(TestCase):
    """Liveness follows the native browser connection when it is exposed."""

    def test_missing_clients_are_not_running(self) -> None:
        """Without a context or a PyDoll client the runtime is not live."""
        runtime = BrowserRuntimeState(max_groups=1)

        self.assertFalse(runtime.is_running())

    def test_disconnected_native_browser_is_not_running(self) -> None:
        """An open-looking context over a disconnected native browser is not live."""
        runtime = BrowserRuntimeState(max_groups=1)
        runtime.main_ctx = _native_context(connected=False)
        runtime.shared_pd = MagicMock()

        self.assertFalse(runtime.is_running())

    def test_connected_native_browser_is_running(self) -> None:
        """A connected native browser keeps the runtime live."""
        runtime = BrowserRuntimeState(max_groups=1)
        runtime.main_ctx = _native_context(connected=True)
        runtime.shared_pd = MagicMock()

        self.assertTrue(runtime.is_running())

    def test_context_without_a_native_projection_stays_running(self) -> None:
        """A context that exposes no native browser keeps the previous liveness contract."""
        runtime = BrowserRuntimeState(max_groups=1)
        runtime.main_ctx = _native_context(connected=None)
        runtime.shared_pd = MagicMock()

        self.assertTrue(runtime.is_running())

    def test_closed_context_is_not_running(self) -> None:
        """A closed Playwright context is not live even over a connected browser."""
        runtime = BrowserRuntimeState(max_groups=1)
        runtime.main_ctx = _native_context(closed=True, connected=True)
        runtime.shared_pd = MagicMock()

        self.assertFalse(runtime.is_running())


def _owned_group(target_id: str = "t1") -> MagicMock:
    """Return a group double the runtime can close and release."""
    group = MagicMock()
    group.target_id = target_id
    group.child_target_ids = []
    group._lock = asyncio.Lock()
    group._quitting = False
    return group


class RollbackRetirementTests(IsolatedAsyncioTestCase):
    """Retiring a dead generation releases its groups, slots and stale handles."""

    async def test_rollback_releases_group_slots_once_and_rejects_stale_handles(self) -> None:
        """Registered groups give their slot back exactly once; their handles become stale."""
        runtime = BrowserRuntimeState(max_groups=1)
        context = _native_context(connected=True)
        context.close = AsyncMock()
        page = MagicMock()
        page.close = AsyncMock()
        group = _owned_group()

        runtime.shared_pd = MagicMock()
        runtime.shared_pd.close = AsyncMock()
        runtime.main_ctx = context
        runtime.main_ctx_owned = True
        handle = runtime.contexts.bind_shared(context)
        await runtime.group_semaphore.acquire()
        runtime.active_groups.add(group)
        runtime.page_to_group[page] = group
        runtime.target_to_page_map["t1"] = page
        runtime.target_page_owned["t1"] = True

        await runtime.rollback_start()

        page.close.assert_awaited_once()
        self.assertEqual(runtime.active_groups, set())
        # The single admission slot came back once, not twice.
        self.assertEqual(runtime.group_semaphore._value, 1)
        with self.assertRaises(BrowserContextError):
            runtime.contexts.require_own(handle)
        self.assertIsNone(runtime.contexts.shared())
        self.assertIsNone(runtime.main_ctx)
        self.assertIsNone(runtime.shared_pd)

    async def test_rollback_failure_keeps_ownership_and_can_retry(self) -> None:
        """A failed isolated-context retirement keeps live ownership and succeeds on retry."""
        runtime = BrowserRuntimeState(max_groups=1)
        context = _native_context(connected=True)
        context.close = AsyncMock()
        runtime.shared_pd = MagicMock()
        runtime.shared_pd.close = AsyncMock()
        runtime.main_ctx = context
        runtime.main_ctx_owned = True
        runtime.contexts.bind_shared(context)

        isolated = _native_context(connected=True)
        isolated.close = AsyncMock(side_effect=[RuntimeError("isolated close failed"), None])

        async def create() -> MagicMock:
            return isolated

        await runtime.contexts.get_or_create("session-a", create)

        with self.assertRaisesRegex(RuntimeError, "isolated close failed"):
            await runtime.rollback_start()

        # The live resources whose retirement failed stay owned for a retry.
        self.assertIs(runtime.main_ctx, context)
        self.assertIsNotNone(runtime.shared_pd)
        context.close.assert_not_awaited()

        await runtime.rollback_start()

        self.assertIsNone(runtime.main_ctx)
        self.assertIsNone(runtime.shared_pd)
        self.assertIsNone(runtime.contexts.shared())
        self.assertEqual(runtime.contexts.isolated_ids(), ())


class _RecoveryDriver:
    """Driver double modelling a dead generation retiring into a live one."""

    def __init__(self, *, attached: bool = False) -> None:
        self.main_ctx = _native_context(connected=False)
        self.shared_pd: MagicMock | None = MagicMock()
        self.main_ctx_owned = not attached
        self._live = False
        self.rollback_calls = 0
        self.start_calls = 0

    def is_running(self) -> bool:
        """Return whether a launch has already brought the runtime live."""
        return self._live

    async def rollback_start(self) -> None:
        """Retire the dead generation."""
        self.rollback_calls += 1
        self.main_ctx = None
        self.shared_pd = None
        self._live = False

    async def start_live(self, _config: object) -> None:
        """Bring a fresh generation live."""
        self.start_calls += 1
        self.main_ctx = _native_context(connected=True)
        self.shared_pd = MagicMock()
        self._live = True


async def _noop_popup_handler(_page: object) -> None:
    return None


@contextlib.contextmanager
def _patched_startup() -> Iterator[None]:
    """Neutralize the external startup dependencies for a lifecycle-only test."""
    fingerprint = MagicMock()
    fingerprint.options.arguments = []
    webdata = MagicMock()
    webdata.exists.return_value = True
    with (
        patch("prowl.browser.lifecycle.startup.get_free_port", return_value=9999),
        patch("prowl.browser.lifecycle.startup.FingerprintManager", return_value=fingerprint),
        patch("prowl.browser.lifecycle.startup.apply_managed_policies"),
        patch("prowl.browser.lifecycle.startup.extension_launch_arguments", return_value=[]),
        patch("prowl.browser.lifecycle.startup.ensure_binary"),
        patch("prowl.browser.lifecycle.startup.clear_stale_singleton_files"),
        patch("prowl.browser.lifecycle.startup.SearchEngineInjector"),
        patch("prowl.browser.lifecycle.startup.atexit.register"),
        patch("prowl.browser.lifecycle.startup.get_coordinator", return_value=MagicMock(guarantees_cleanup=False)),
        patch.object(BrowserLifecycle, "restore_profile_from_archive", return_value=False),
        patch.object(BrowserLifecycle, "webdata_path", return_value=webdata),
        patch.object(BrowserLifecycle, "_inject_search_engine"),
    ):
        yield


class LifecycleRecoveryTests(IsolatedAsyncioTestCase):
    """Startup retires a dead generation before relaunching, without replacing an external one."""

    def _lifecycle(self, driver: _RecoveryDriver) -> BrowserLifecycle:
        return BrowserLifecycle(cast("BrowserRuntimeState", driver))

    async def test_start_retires_a_dead_local_generation_before_launching(self) -> None:
        """A dead local generation is retired first, and its profile is not packaged."""
        driver = _RecoveryDriver()
        lifecycle = self._lifecycle(driver)
        packaged_during_recovery: list[bool] = []
        original_start_live = driver.start_live

        async def record_ownership(config: object) -> None:
            packaged_during_recovery.append(lifecycle._owns_local_profile)
            await original_start_live(config)

        driver.start_live = record_ownership  # type: ignore[method-assign]

        with _patched_startup():
            await lifecycle.start(is_running=driver.is_running, popup_handler=_noop_popup_handler)

        self.assertEqual(driver.rollback_calls, 1)
        self.assertEqual(driver.start_calls, 1)
        # The crashed generation was not treated as an owned profile to package.
        self.assertEqual(packaged_during_recovery, [False])

    async def test_start_without_a_stale_generation_does_not_retire(self) -> None:
        """A clean runtime relaunches without running a needless retirement."""
        driver = _RecoveryDriver()
        driver.main_ctx = None
        driver.shared_pd = None
        lifecycle = self._lifecycle(driver)

        with _patched_startup():
            await lifecycle.start(is_running=driver.is_running, popup_handler=_noop_popup_handler)

        self.assertEqual(driver.rollback_calls, 0)
        self.assertEqual(driver.start_calls, 1)

    async def test_simultaneous_restart_retires_and_launches_once(self) -> None:
        """Two concurrent restarts of one dead generation retire and launch exactly once."""
        driver = _RecoveryDriver()
        lifecycle = self._lifecycle(driver)

        with _patched_startup():
            await asyncio.gather(
                lifecycle.start(is_running=driver.is_running, popup_handler=_noop_popup_handler),
                lifecycle.start(is_running=driver.is_running, popup_handler=_noop_popup_handler),
            )

        self.assertEqual(driver.rollback_calls, 1)
        self.assertEqual(driver.start_calls, 1)

    async def test_externally_attached_dead_browser_refuses_local_launch(self) -> None:
        """A dead externally attached browser is never closed or replaced by a local one."""
        driver = _RecoveryDriver(attached=True)
        lifecycle = self._lifecycle(driver)

        with _patched_startup(), self.assertRaisesRegex(BrowserStartError, "Reconnect explicitly"):
            await lifecycle.start(is_running=driver.is_running, popup_handler=_noop_popup_handler)

        self.assertEqual(driver.rollback_calls, 0)
        self.assertEqual(driver.start_calls, 0)
        self.assertIsNotNone(driver.main_ctx)
