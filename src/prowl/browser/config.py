"""Explicit launch configuration shared by the service and the browser core.

The browser is a process-wide singleton. Its egress proxy and its persistent
profile paths must be configured through one seam so the values a caller
supplies to the service are the values the launched browser actually uses,
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


def _env_or(env_name: str, default: str) -> str:
    """Return the non-empty environment value for *env_name*, else *default*."""
    return os.environ.get(env_name, "").strip() or default


def default_profile_dir() -> str:
    """Return the configured persistent profile directory or the local default."""
    return _env_or(PROFILE_DIR_ENV, DEFAULT_PROFILE_DIR)


def default_profile_archive() -> str:
    """Return the configured profile archive path or the local default."""
    return _env_or(PROFILE_ARCHIVE_ENV, DEFAULT_PROFILE_ARCHIVE)


@dataclass(slots=True)
class BrowserConfig:
    """Launch-level configuration for the single shared browser process."""

    proxy_url: str | None = None
    profile_dir: str = DEFAULT_PROFILE_DIR
    profile_archive: str = DEFAULT_PROFILE_ARCHIVE

    @classmethod
    def from_env(cls, *, proxy_url: str | None = None) -> BrowserConfig:
        """Build configuration from environment defaults with an explicit proxy."""
        return cls(
            proxy_url=proxy_url,
            profile_dir=default_profile_dir(),
            profile_archive=default_profile_archive(),
        )


__all__ = [
    "DEFAULT_PROFILE_ARCHIVE",
    "DEFAULT_PROFILE_DIR",
    "PROFILE_ARCHIVE_ENV",
    "PROFILE_DIR_ENV",
    "BrowserConfig",
    "default_profile_archive",
    "default_profile_dir",
]
