"""Explicit launch configuration shared by the service and the browser core.

The browser is a process-wide singleton. Its egress proxy, its persistent profile
paths, and its unpacked extensions must be configured through one seam so the values a
caller supplies to the service are the values the launched browser actually uses,
instead of two independent reads of the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Local fallback paths. The archive lives outside the profile directory so
#: packing never recursively includes an older archive of itself.
DEFAULT_PROFILE_DIR = "/tmp/browser-profile"  # noqa: S108
DEFAULT_PROFILE_ARCHIVE = "browser-profile.zip"

PROFILE_DIR_ENV = "PROWL_PROFILE_DIR"
PROFILE_ARCHIVE_ENV = "PROWL_PROFILE_ARCHIVE"
PROXY_URL_ENV = "PROWL_PROXY_URL"
EXTENSIONS_DIR_ENV = "PROWL_EXTENSIONS_DIR"
POLICY_DIR_ENV = "PROWL_POLICY_DIR"
WINDOW_SIZE_ENV = "PROWL_WINDOW_SIZE"


def _env_or(env_name: str, default: str) -> str:
    """Return the non-empty environment value for *env_name*, else *default*."""
    return os.environ.get(env_name, "").strip() or default


def default_profile_dir() -> str:
    """Return the configured persistent profile directory or the local default."""
    return _env_or(PROFILE_DIR_ENV, DEFAULT_PROFILE_DIR)


def default_profile_archive() -> str:
    """Return the configured profile archive path or the local default."""
    return _env_or(PROFILE_ARCHIVE_ENV, DEFAULT_PROFILE_ARCHIVE)


def default_extensions_dir() -> str | None:
    """Return the configured unpacked-extension directory, or ``None`` when unset."""
    return os.environ.get(EXTENSIONS_DIR_ENV, "").strip() or None


def default_policy_dir() -> str | None:
    """Return the configured managed-policy directory, or ``None`` when unset."""
    return os.environ.get(POLICY_DIR_ENV, "").strip() or None


def default_window_size() -> tuple[int, int] | None:
    """Return the configured launch window size as (width, height), or ``None`` when unset.

    The size a window is launched with is otherwise the fingerprint screen, which is a persona
    property and can be larger than the display the browser is actually shown on. Setting this
    keeps the persona while the window fits the screen it is rendered to.

    :raises ValueError: when the value is not ``WIDTHxHEIGHT`` with positive integers.
    """
    raw = os.environ.get(WINDOW_SIZE_ENV, "").strip()
    if not raw:
        return None
    width, separator, height = raw.lower().partition("x")
    if not separator or not width.isdigit() or not height.isdigit():
        msg = f"{WINDOW_SIZE_ENV} must be WIDTHxHEIGHT in pixels, got {raw!r}"
        raise ValueError(msg)
    size = (int(width), int(height))
    if size[0] <= 0 or size[1] <= 0:
        msg = f"{WINDOW_SIZE_ENV} must be positive pixels, got {raw!r}"
        raise ValueError(msg)
    return size


@dataclass(slots=True)
class BrowserConfig:
    """Launch-level configuration for the single shared browser process."""

    proxy_url: str | None = None
    profile_dir: str = DEFAULT_PROFILE_DIR
    profile_archive: str = DEFAULT_PROFILE_ARCHIVE
    extensions_dir: str | None = None
    policy_dir: str | None = None
    window_size: tuple[int, int] | None = None

    @classmethod
    def from_env(cls, *, proxy_url: str | None = None) -> BrowserConfig:
        """Build configuration from environment defaults with an explicit proxy."""
        return cls(
            proxy_url=proxy_url,
            profile_dir=default_profile_dir(),
            profile_archive=default_profile_archive(),
            extensions_dir=default_extensions_dir(),
            policy_dir=default_policy_dir(),
            window_size=default_window_size(),
        )


__all__ = [
    "DEFAULT_PROFILE_ARCHIVE",
    "DEFAULT_PROFILE_DIR",
    "EXTENSIONS_DIR_ENV",
    "POLICY_DIR_ENV",
    "PROFILE_ARCHIVE_ENV",
    "PROFILE_DIR_ENV",
    "PROXY_URL_ENV",
    "WINDOW_SIZE_ENV",
    "BrowserConfig",
    "default_extensions_dir",
    "default_policy_dir",
    "default_profile_archive",
    "default_profile_dir",
    "default_window_size",
]
