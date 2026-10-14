"""Fingerprint management."""

from typing import TYPE_CHECKING, Any, Self

from pydoll.constants import PageLoadState

from prowl.browser.options import BrowserOptions

if TYPE_CHECKING:
    from pydoll.browser.options import ChromiumOptions


class FingerprintManager:
    """Comprehensive fingerprint evasion using browser options and JavaScript."""

    def __init__(self: Self, profile: dict[str, Any]) -> None:
        """Initialize with target profile (OS, location, device, etc.)."""
        self.profile = profile
        self.options: ChromiumOptions = BrowserOptions.new()
        self._configure_browser_options()

    def _configure_browser_options(self: Self) -> None:
        """Configure browser launch options based on profile."""
        port = self.profile["port"]
        self.options.add_argument(f"--remote-debugging-port={port}")

        screen = self.profile["screen"]

        # The patched build renders the page at the fingerprint screen width, not at the window
        # width, so a window smaller than the screen clips the layout: the window can be resized
        # to fit a display while the page inside it still lays out for the larger screen. When a
        # window size is configured, the screen follows it, so the persona, the window and the
        # display agree and the page fits. Unset leaves the persona untouched.
        from prowl.browser.config import default_window_size  # noqa: PLC0415

        configured = default_window_size()
        if configured is not None:
            screen = {"width": configured[0], "height": configured[1]}

        self.options.add_argument(f"--window-size={screen['width']},{screen['height']}")
        self.options.add_argument(f"--fingerprint-screen-width={screen['width']}")
        self.options.add_argument(f"--fingerprint-screen-height={screen['height']}")

        self.options.add_argument("--fingerprint-storage-quota=1000")  # in MB
        self.options.add_argument("--fingerprint-noise=false")
        self.options.add_argument("--fingerprint-windows-font-metrics")
        self.options.add_argument("--fingerprint-allow-3p-cookies")

        # for docker with vnc
        self.options.add_argument("--use-gl=angle")
        self.options.add_argument("--use-angle=swiftshader")

        self.options.page_load_state = PageLoadState.COMPLETE
