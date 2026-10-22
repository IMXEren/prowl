"""FlareSolverr-compatible HTTP service.

Public exports load on first access so browser proxy modules can use service
protocol and errors without importing the service app during package setup.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prowl.service.app import Service, ServiceConfig, create_app, run_server
    from prowl.service.backend import (
        DEFAULT_INTERACTIVE_IDLE_SECONDS,
        INTERACTIVE_IDLE_SECONDS_ENV,
        Backend,
        BrowserBackend,
        FetchRequest,
        FetchResult,
        InteractiveRequest,
        InteractiveTab,
    )
    from prowl.service.errors import CallerSafeError, SessionError, SessionLimitError, SessionNotFoundError
    from prowl.service.protocol import ProtocolError, parse_request
    from prowl.service.sessions import SessionCleanup, SessionInfo, SessionMode, SessionRegistry

__all__ = [
    "DEFAULT_INTERACTIVE_IDLE_SECONDS",
    "INTERACTIVE_IDLE_SECONDS_ENV",
    "Backend",
    "BrowserBackend",
    "CallerSafeError",
    "FetchRequest",
    "FetchResult",
    "InteractiveRequest",
    "InteractiveTab",
    "ProtocolError",
    "Service",
    "ServiceConfig",
    "SessionCleanup",
    "SessionError",
    "SessionInfo",
    "SessionLimitError",
    "SessionMode",
    "SessionNotFoundError",
    "SessionRegistry",
    "create_app",
    "parse_request",
    "run_server",
]

_EXPORT_MODULES = {
    "app": frozenset({"Service", "ServiceConfig", "create_app", "run_server"}),
    "backend": frozenset(
        {
            "DEFAULT_INTERACTIVE_IDLE_SECONDS",
            "INTERACTIVE_IDLE_SECONDS_ENV",
            "Backend",
            "BrowserBackend",
            "FetchRequest",
            "FetchResult",
            "InteractiveRequest",
            "InteractiveTab",
        }
    ),
    "errors": frozenset({"CallerSafeError", "SessionError", "SessionLimitError", "SessionNotFoundError"}),
    "protocol": frozenset({"ProtocolError", "parse_request"}),
    "sessions": frozenset({"SessionCleanup", "SessionInfo", "SessionMode", "SessionRegistry"}),
}


def __getattr__(name: str) -> Any:
    for module, exports in _EXPORT_MODULES.items():
        if name in exports:
            value = getattr(import_module(f"prowl.service.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)
