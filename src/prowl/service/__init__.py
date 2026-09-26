"""FlareSolverr-compatible HTTP service."""

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
from prowl.service.sessions import SessionRegistry

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
    "SessionError",
    "SessionLimitError",
    "SessionNotFoundError",
    "SessionRegistry",
    "create_app",
    "parse_request",
    "run_server",
]
