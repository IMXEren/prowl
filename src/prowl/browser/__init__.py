"""Browser automation - process management, tab groups, cookies, and page loading."""

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.config import BrowserConfig
from prowl.browser.cookies import Cookies
from prowl.browser.driver import BrowserContextHandle
from prowl.browser.extensions import (
    EXTENSIONS_DIR_ENV,
    Extension,
    discover_extensions,
    extension_launch_arguments,
)
from prowl.browser.page_handler import PageHandler, PageResponse
from prowl.browser.policies import (
    MANAGED_POLICY_DIRS,
    POLICY_DIR_ENV,
    PolicyFile,
    apply_managed_policies,
    policy_files,
)
from prowl.browser.proxy.egress import (
    DEFAULT_EGRESS_IDLE_SECONDS,
    DEFAULT_EGRESS_NAME,
    EGRESS_IDLE_SECONDS_ENV,
    EGRESSES_ENV,
    EgressError,
    EgressPool,
    create_egress_browser,
    derive_egress_paths,
    parse_egress_spec,
)

__all__ = [
    "DEFAULT_EGRESS_IDLE_SECONDS",
    "DEFAULT_EGRESS_NAME",
    "EGRESSES_ENV",
    "EGRESS_IDLE_SECONDS_ENV",
    "EXTENSIONS_DIR_ENV",
    "MANAGED_POLICY_DIRS",
    "POLICY_DIR_ENV",
    "Browser",
    "BrowserConfig",
    "BrowserContextHandle",
    "Cookies",
    "EgressError",
    "EgressPool",
    "Extension",
    "PageHandler",
    "PageResponse",
    "PolicyFile",
    "TabGroup",
    "apply_managed_policies",
    "create_egress_browser",
    "derive_egress_paths",
    "discover_extensions",
    "extension_launch_arguments",
    "parse_egress_spec",
    "policy_files",
]
