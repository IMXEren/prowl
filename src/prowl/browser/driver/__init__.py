"""Concrete browser driver runtime state."""

from prowl.browser.driver.runtime import (
    BrowserRuntimeState,
    DriverRemoteAttachConfig,
    DriverStartupConfig,
    resolve_cdp_ws_url,
)

__all__ = ["BrowserRuntimeState", "DriverRemoteAttachConfig", "DriverStartupConfig", "resolve_cdp_ws_url"]
