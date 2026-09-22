"""Caller-visible error types for the FlareSolverr-compatible service.

Only errors derived from :class:`CallerSafeError` have their message returned
to callers verbatim. Every other exception is reduced to a generic message so
browser internals, profile paths, and credentials can never leak through a
response.
"""

from __future__ import annotations


class CallerSafeError(Exception):
    """Base class for errors whose message is deliberately safe to return."""


class SessionError(CallerSafeError):
    """A caller-visible logical-session error."""


class SessionLimitError(SessionError):
    """Raised when no further logical sessions may be created."""


class SessionNotFoundError(SessionError):
    """Raised when a referenced logical session does not exist."""


class ProxyError(CallerSafeError):
    """Raised when a request proxy is not permitted by the service configuration."""


__all__ = [
    "CallerSafeError",
    "ProxyError",
    "SessionError",
    "SessionLimitError",
    "SessionNotFoundError",
]
