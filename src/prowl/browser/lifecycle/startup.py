"""Concrete owner for browser startup, profile, and launch configuration."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import ctypes
import enum
import os
import shutil
import socket
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from cloakbrowser import ensure_binary, launch_persistent_context_async
from loguru import logger

from prowl.browser.config import (
    BrowserConfig,
    default_extensions_dir,
    default_policy_dir,
    default_profile_archive,
    default_profile_dir,
    default_window_size,
)
from prowl.browser.driver import (
    BrowserRuntimeState,
    DriverRemoteAttachConfig,
    DriverStartupConfig,
)
from prowl.browser.exceptions import BrowserShutdownError, BrowserStartError
from prowl.browser.extensions import extension_launch_arguments
from prowl.browser.fingerprint import FingerprintManager
from prowl.browser.policies import apply_managed_policies
from prowl.browser.profile import SearchEngineInjector
from prowl.shutdown import CoordinatorStateError, RegistrationToken, get_coordinator

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from playwright.async_api import Page as PWPage
    from pydoll.browser.options import ChromiumOptions


class BrowserShutdownState(enum.Enum):
    """Shutdown state machine for the concrete browser lifecycle."""

    NOT_STARTED = enum.auto()
    IN_PROGRESS = enum.auto()
    SUCCEEDED = enum.auto()
    FAILED = enum.auto()


def get_free_port(preferred: int, fallback_range: range | None = None) -> int:
    """Return *preferred* if available, else the first free port in *fallback_range*."""
    candidates = (preferred, *(fallback_range or range(0)))
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    msg = "No free TCP port found for browser CDP."
    raise RuntimeError(msg)


#: Files Chromium writes to claim a profile directory, relative to the profile root.
_SINGLETON_FILES: tuple[str, ...] = ("SingletonLock", "SingletonSocket", "SingletonCookie")


def _pid_is_alive(pid: int) -> bool:
    """Return whether *pid* names a live process on this machine."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists, this user may just not signal it.
        return True
    except OSError:
        return False
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Return whether *pid* exists, without terminating it the way os.kill would.

    On Windows ``os.kill`` terminates the target for any signal other than the two console
    events, so a liveness probe has to ask for a handle instead.
    """
    process_query_limited_information = 0x1000
    error_access_denied = 5
    still_active = 259
    inherit_handle = False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(process_query_limited_information, inherit_handle, pid)
    if not handle:
        # A process this user may not open still exists.
        return ctypes.get_last_error() == error_access_denied
    exit_code = ctypes.c_ulong()
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
        kernel32.CloseHandle(handle)
        return False
    kernel32.CloseHandle(handle)
    # A finished process keeps its pid reserved while a handle to it is still held, so the exit
    # code decides liveness here, not whether a handle could be opened at all.
    return exit_code.value == still_active


def _singleton_lock_owner(lock: Path) -> str | None:
    """Return the ``host-pid`` owner recorded in *lock*, or ``None`` when there is none."""
    try:
        if lock.is_symlink():
            return str(lock.readlink())
        if lock.exists():
            return lock.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return None


def _singleton_owner_is_alive(owner: str) -> bool:
    """Return whether the ``host-pid`` claim in *owner* is a process on this machine."""
    host, separator, pid_text = owner.rpartition("-")
    if not separator:
        return False
    if host != socket.gethostname():
        # Another host wrote this claim. A container that was replaced reports the id of the
        # container that is gone, so the process it names cannot be running here.
        return False
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    return _pid_is_alive(pid)


def clear_stale_singleton_files(profile_dir: str | Path) -> list[str]:
    """Remove Chromium's profile claim when the process that wrote it is gone.

    Chromium records the owning host and process in a ``SingletonLock`` and refuses to start
    while it finds a claim it cannot disprove, so a profile left behind by a killed process, or
    by a container that no longer exists, fails with "the profile appears to be in use by another
    process on another computer" until the claim is cleared by hand. A claim whose owner is
    still alive is left alone, because that really is a running browser.

    :return: the names of the files that were removed.
    """
    profile = Path(profile_dir)
    owner = _singleton_lock_owner(profile / "SingletonLock")
    if owner is None or _singleton_owner_is_alive(owner):
        return []

    removed: list[str] = []
    for name in _SINGLETON_FILES:
        with contextlib.suppress(OSError):
            (profile / name).unlink()
            removed.append(name)
    if removed:
        logger.info(f"Cleared a stale browser profile claim: {', '.join(removed)}.")
    return removed


def default_viewport() -> dict[str, int]:
    """Return the page viewport, following a configured window size when there is one.

    The viewport is what the page lays out in, and it is not the same thing as the window: the window
    can be fitted to a display while the viewport stays at the persona's size, and then the layout is
    wider than the window with its right-hand side out of reach. A configured window size therefore
    sets the viewport too, so the page lays out at the size that is actually shown.
    """
    size = default_window_size()
    if size is not None:
        return {"width": size[0], "height": size[1]}
    return {"width": 1920, "height": 980}


@dataclass(slots=True)
class BrowserLifecycle:
    """Concrete lifecycle owner for startup and profile configuration."""

    driver: BrowserRuntimeState
    profile_dir: str = field(default_factory=default_profile_dir)
    profile_archive: Path = field(default_factory=lambda: Path(default_profile_archive()))
    checked_binary: bool = False
    cdp_port: int = 9222
    fingerprint: FingerprintManager | None = None
    fingerprint_options: ChromiumOptions | None = None
    viewport: dict[str, int] = field(default_factory=default_viewport)
    locale: str = "en-US,en"
    proxy_url: str | None = field(
        default_factory=lambda: os.environ.get("PROWL_PROXY_URL", "").strip() or None,
    )
    #: Directory of unpacked extensions every browser loads, or ``None`` for none.
    extensions_dir: str | None = field(default_factory=default_extensions_dir)
    #: Directory of managed policy JSON applied before every launch, or ``None``.
    policy_dir: str | None = field(default_factory=default_policy_dir)
    #: Window size the browser is launched with, or ``None`` for the fingerprint screen size.
    window_size: tuple[int, int] | None = field(default_factory=default_window_size)
    _startup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _shutdown_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _shutdown_state: BrowserShutdownState = BrowserShutdownState.NOT_STARTED
    _shutdown_error: BrowserShutdownError | None = None
    _shutdown_task: asyncio.Task[None] | None = None
    _atexit_registered: bool = False
    _owns_local_profile: bool = False
    _signal_registration: RegistrationToken | None = None

    _SKIP_PROFILE_DIRS: ClassVar[set[str]] = {
        "Cache",
        "Code Cache",
        "GPUCache",
        "DawnCache",
        "DawnGraphiteCache",
        "DawnWebGPUCache",
        "blob_storage",
        "ShaderCache",
        "GrShaderCache",
    }

    # -- Explicit configuration seam ------------------------------------------------

    def apply_config(self, config: BrowserConfig, *, is_running: Callable[[], bool]) -> None:
        """Adopt *config* for the next launch.

        Refuses to reconfigure a running or shutting-down browser so the launch
        inputs cannot diverge from the values already in use.

        :raises BrowserStartError: when the browser is running or shutting down.
        """
        if is_running() or self._shutdown_state is BrowserShutdownState.IN_PROGRESS:
            msg = "Cannot reconfigure the browser while it is running or shutting down."
            raise BrowserStartError(msg)
        self.proxy_url = config.proxy_url
        self.profile_dir = config.profile_dir
        self.profile_archive = Path(config.profile_archive)
        self.extensions_dir = config.extensions_dir
        self.policy_dir = config.policy_dir
        self.window_size = config.window_size

    # -- Profile helpers ------------------------------------------------------------

    def webdata_path(self) -> Path:
        """Return the Chrome Web Data path in the owned profile directory."""
        return Path(self.profile_dir) / "Default" / "Web Data"

    def unpack_profile(self, archive: str | Path | None = None) -> bool:
        """Restore the owned profile directory from *archive* when available."""
        pkg = Path(archive) if archive else self.profile_archive
        if not pkg.exists():
            return False
        target = Path(self.profile_dir)
        if target.exists():
            shutil.rmtree(target)
        logger.info(f"Extracting profile from {pkg} ...")
        shutil.unpack_archive(str(pkg), str(target))
        return True

    def pack_profile(self, archive: str | Path | None = None) -> Path | None:
        """Zip the owned profile directory into *archive*, skipping caches."""
        profile_dir = Path(self.profile_dir)
        if not profile_dir.exists():
            return None

        pkg = (Path(archive) if archive else self.profile_archive).resolve()
        profile_dir = profile_dir.resolve()

        if pkg.is_relative_to(profile_dir):
            pkg = profile_dir.parent / pkg.name

        stale = profile_dir / pkg.name
        stale.unlink(missing_ok=True)
        pkg.unlink(missing_ok=True)

        files: list[Path] = []
        for entry in profile_dir.rglob("*"):
            if not entry.is_file():
                continue
            if any(part in self._SKIP_PROFILE_DIRS for part in entry.parts):
                continue
            if entry.name.endswith("-journal"):
                continue
            if entry.resolve() == pkg.resolve():
                continue
            files.append(entry)

        with zipfile.ZipFile(str(pkg), "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, file_path.relative_to(profile_dir))
        logger.info(f"Profile archived to {pkg} ({len(files)} files).")
        return pkg

    @property
    def shutdown_state(self) -> BrowserShutdownState:
        """Return the current shutdown state for facade admission checks."""
        return self._shutdown_state

    # -- Admission helpers ----------------------------------------------------------

    def _reset_after_shutdown(self) -> None:
        """Reset terminal shutdown state before a new lifecycle generation."""
        self._shutdown_state = BrowserShutdownState.NOT_STARTED
        self._shutdown_error = None
        self._shutdown_task = None

    def _admit_start(self) -> None:
        """Validate and normalize shutdown state before startup side effects."""
        if self._shutdown_state is BrowserShutdownState.IN_PROGRESS:
            msg = "Browser is shutting down - cannot start."
            raise BrowserStartError(msg)
        if self._shutdown_state in {BrowserShutdownState.SUCCEEDED, BrowserShutdownState.FAILED}:
            self._reset_after_shutdown()

    # -- Start / connect ------------------------------------------------------------

    async def start(
        self,
        *,
        is_running: Callable[[], bool],
        popup_handler: Callable[[PWPage], Coroutine[None, None, None]],
    ) -> None:
        """Run concrete startup sequencing and delegate live launch to the driver.

        OS signal installation is owned by prowl.shutdown; this method only
        performs browser-specific launch and atexit registration.
        """
        self._admit_start()
        if is_running():
            self._register_signal_cleanup()
            return

        async with self._startup_lock:
            self._admit_start()
            if is_running():
                self._register_signal_cleanup()
                return

            self.unpack_profile()
            # Clear a claim left by a process that is no longer running, before any launch,
            # including the priming launch below, or that launch fails the same way.
            clear_stale_singleton_files(self.profile_dir)
            if not self.checked_binary:
                ensure_binary()
                self.checked_binary = True

            self.cdp_port = get_free_port(self.cdp_port, range(9223, 9323))
            profile = {
                "screen": self.viewport,
                "user_data_dir": self.profile_dir,
                "port": self.cdp_port,
            }
            self.fingerprint = FingerprintManager(profile)
            self.fingerprint_options = self.fingerprint.options
            # Chromium reads managed policy as it starts, so the files are in place first.
            apply_managed_policies(self.policy_dir)
            launch_arguments = list(self.fingerprint.options.arguments)
            # The window is placed at the origin and, when a size is configured, launched at that
            # size instead of the fingerprint screen size. Either one is what stops Chromium
            # restoring the window bounds it saved the last time this profile ran: a restored
            # placement is honoured even when it is larger than the display, which puts the right
            # edge and the bottom of the window out of reach, and that is what happened once the
            # display was resized. An explicit position skips the restore, leaving the window to
            # the window manager, which fits it to the screen.
            if self.window_size is not None:
                launch_arguments = [
                    argument for argument in launch_arguments if not argument.startswith("--window-size=")
                ]
                launch_arguments.append(f"--window-size={self.window_size[0]},{self.window_size[1]}")
            launch_arguments.append("--window-position=0,0")
            if self.proxy_url:
                # One browser and one profile have exactly one egress. The URL is
                # never logged or echoed so proxy credentials cannot leak.
                launch_arguments.append(f"--proxy-server={self.proxy_url}")
            # Extensions are deployment configuration, added on the one launch path so the
            # priming launch and the live launch always agree on them.
            launch_arguments.extend(extension_launch_arguments(self.extensions_dir))

            if not self.webdata_path().exists():
                await self._prime_profile(launch_arguments)

            self._inject_search_engine()

            try:
                await self.driver.start_live(
                    DriverStartupConfig(
                        profile_dir=self.profile_dir,
                        user_data_dir=self.profile_dir,
                        cdp_port=self.cdp_port,
                        fingerprint_options=self.fingerprint.options,
                        launch_arguments=launch_arguments,
                        viewport=dict(self.viewport),
                        locale=self.locale,
                        popup_handler=popup_handler,
                    ),
                )

                self._owns_local_profile = True

                self._register_signal_cleanup()
                self._register_atexit()
            except BaseException as e:
                self._unregister_atexit()
                self._unregister_signal_cleanup()
                await self.driver.rollback_start()
                msg = "failed to start the browser"
                raise BrowserStartError(msg) from e

    async def connect(
        self,
        *,
        is_running: Callable[[], bool],
        ws_url: str,
        popup_handler: Callable[[PWPage], Coroutine[None, None, None]],
    ) -> None:
        """Attach concrete drivers to a caller-owned remote CDP websocket.

        OS signal installation is owned by prowl.shutdown; this method only
        performs remote driver attach and atexit registration.
        """
        self._admit_start()

        async with self._startup_lock:
            self._admit_start()
            if is_running():
                self._register_signal_cleanup()
                return

            try:
                await self.driver.attach_remote(DriverRemoteAttachConfig(ws_url=ws_url, popup_handler=popup_handler))
                self._register_signal_cleanup()
            except BaseException as e:
                self._unregister_signal_cleanup()
                await self.driver.rollback_start()
                msg = "failed to connect browser"
                raise BrowserStartError(msg) from e
            self._owns_local_profile = False

    # -- Cleanup resources ----------------------------------------------------------

    async def _cleanup_resources(self) -> None:
        """Close browser resources and run final synchronous profile chores."""
        try:
            await self.driver.close_all_groups_and_pages()
        except BaseException:  # noqa: BLE001
            logger.error("Failed to close browser groups and pages")

        if self.driver.shared_pd is not None:
            logger.debug("Closing the pydoll connection...")
            try:
                await self.driver.shared_pd.close()
            except BaseException:  # noqa: BLE001
                logger.error("Failed to close PyDoll during cleanup")
            finally:
                self.driver.shared_pd = None

        if self.driver.main_ctx is not None and self.driver.main_ctx_owned:
            logger.debug("Closing main persistent context...")
            try:
                await self.driver.main_ctx.close()
            except BaseException:  # noqa: BLE001
                logger.error("Failed to close Playwright context during cleanup")
        self.driver.main_ctx = None
        self.driver.main_ctx_owned = False

        if self.driver.cdp_browser is not None:
            logger.debug("Closing Playwright CDP browser...")
            try:
                await self.driver.cdp_browser.close()
            except BaseException:  # noqa: BLE001
                logger.error("Failed to close Playwright CDP browser during cleanup")
            self.driver.cdp_browser = None

        if self.driver.cdp_playwright is not None:
            logger.debug("Stopping Playwright CDP owner...")
            try:
                await self.driver.cdp_playwright.stop()
            except BaseException:  # noqa: BLE001
                logger.error("Failed to stop Playwright during cleanup")
            self.driver.cdp_playwright = None

        self.do_sync_chores_before_exit()

    async def _do_shutdown(self) -> None:
        """Own cleanup execution and terminal state finalization."""
        try:
            await self._cleanup_resources()
        except BaseException as exc:  # noqa: BLE001
            self._shutdown_state = BrowserShutdownState.FAILED
            self._shutdown_error = BrowserShutdownError(exc)
        else:
            self._shutdown_state = BrowserShutdownState.SUCCEEDED
        finally:
            self._unregister_signal_cleanup()
            self._unregister_atexit()
            self._shutdown_task = None

    # -- Signal cleanup ownership ---------------------------------------------------

    def _register_signal_cleanup(self) -> None:
        """Register asynchronous browser cleanup when coordination is available."""
        token = self._signal_registration

        if token is not None and token.active:
            return

        coordinator = get_coordinator()
        if not coordinator.guarantees_cleanup:
            return

        try:
            self._signal_registration = coordinator.register(
                self.shutdown,
                owner_loop=asyncio.get_running_loop(),
                name="browser",
            )
        except CoordinatorStateError:
            # A signal may have started shutdown between the availability check
            # and registration. Browser remains usable without coordination.
            logger.debug(
                "Browser signal cleanup was not registered because "
                "the coordinator is no longer accepting registrations.",
            )

    def _unregister_signal_cleanup(self) -> None:
        """Remove the browser cleanup registration after a normal shutdown."""
        token = self._signal_registration
        self._signal_registration = None

        if token is not None:
            token.unregister()

    def _register_atexit(self) -> None:
        """Register the lifecycle-owned synchronous fallback once."""
        if not self._atexit_registered:
            atexit.register(self.sync_atexit_fallback)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        """Unregister the lifecycle-owned synchronous fallback once."""
        if self._atexit_registered:
            atexit.unregister(self.sync_atexit_fallback)
            self._atexit_registered = False

    # -- Shutdown with cancellation resilience --------------------------------------

    async def shutdown(self) -> None:
        """Gracefully tear down all browser resources once per generation.

        The owner must tolerate repeated CancelledError while awaiting the
        same shielded cleanup task to terminal SUCCEEDED / FAILED state.
        Cancellation is immediately propagated; the shielded cleanup task
        continues independently to its terminal outcome.
        """
        shutdown_state = self._shutdown_state
        if shutdown_state is BrowserShutdownState.SUCCEEDED:
            return
        if shutdown_state is BrowserShutdownState.FAILED:
            if self._shutdown_error is None:
                msg = "Invariant: FAILED without _shutdown_error"
                raise BrowserShutdownError(msg)
            raise self._shutdown_error

        async with self._shutdown_lock:
            # Re-check terminal states after acquiring the lock.
            shutdown_state = self._shutdown_state
            if shutdown_state is BrowserShutdownState.SUCCEEDED:
                return
            if shutdown_state is BrowserShutdownState.FAILED:
                if self._shutdown_error is None:
                    msg = "Invariant: FAILED without _shutdown_error"
                    raise BrowserShutdownError(msg)
                raise self._shutdown_error

            existing_task = self._shutdown_task
            if existing_task is not None and not existing_task.done():
                # Capture reference for awaiting outside the lock.
                cleanup_to_await: asyncio.Task[None] = existing_task
            else:
                self._shutdown_state = BrowserShutdownState.IN_PROGRESS
                cleanup_to_await = asyncio.get_running_loop().create_task(
                    self._do_shutdown(),
                    name="browser-cleanup",
                )
                self._shutdown_task = cleanup_to_await

        # Await the shielded cleanup outside the lock so waiters can enter.
        # Cancellation propagates immediately; the shielded cleanup task
        # survives and will reach terminal independently.
        await asyncio.shield(cleanup_to_await)

        if self._shutdown_state is BrowserShutdownState.FAILED:
            if self._shutdown_error is None:  # pragma: no cover
                msg = "Invariant: FAILED without _shutdown_error"
                raise BrowserShutdownError(msg)
            raise self._shutdown_error

    # -- Synchronous chores ---------------------------------------------------------

    def do_sync_chores_before_exit(self) -> None:
        """Run bounded synchronous cleanup needed before process exit."""
        if self._owns_local_profile:
            self.pack_profile()
            self._owns_local_profile = False

    def sync_atexit_fallback(self) -> None:
        """Last synchronous safety net for profile preservation."""
        if self._shutdown_state in (BrowserShutdownState.SUCCEEDED, BrowserShutdownState.FAILED):
            return

        logger.warning(
            "Process exited before asynchronous browser cleanup completed. Running synchronous preservation fallback.",
        )

        try:
            self.do_sync_chores_before_exit()
        except BaseException:  # noqa: BLE001
            logger.error("Sync last-minute chores failed!")

    # -- Profile priming ------------------------------------------------------------

    async def _prime_profile(self, launch_arguments: list[str]) -> None:
        """Launch and close a headless context before Web Data injection."""
        logger.debug("Priming fresh profile to generate User Data...")
        prime_ctx = await launch_persistent_context_async(
            headless=True,
            args=launch_arguments,
            viewport=self.viewport,
            locale=self.locale,
            geoip=True,
            humanize=True,
            user_data_dir=self.profile_dir,
        )
        try:
            await asyncio.sleep(1)
        finally:
            await prime_ctx.close()

    def _inject_search_engine(self) -> None:
        """Inject Google after Web Data exists in the primed Chrome profile."""
        with SearchEngineInjector(self.profile_dir) as injector:
            injector.inject(
                short_name="Google",
                keyword="google.com",
                url="https://www.google.com/search?q={searchTerms}",
                suggest_url="https://www.google.com/complete/search?client=chrome&q={searchTerms}",
            )
