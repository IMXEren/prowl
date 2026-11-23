"""Low-cardinality CAPTCHA attempt outcomes and real provider-token contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

from prowl.vision.schemas import VisionError

type OutcomeStatus = Literal["handled", "unsupported", "needs_human", "timed_out", "error"]

OUTCOME_STATUSES: Final = ("handled", "unsupported", "needs_human", "timed_out", "error")
INVALID_OUTCOME: Final = "invalid_outcome"
LOW_CONFIDENCE_REASON: Final = "low_confidence"
TIMED_OUT_REASON: Final = "timed_out"
ERROR_REASON: Final = "error"
PAGE_ACCEPTED_REASON: Final = "page_accepted"


@dataclass(frozen=True, slots=True)
class VisionOutcome:
    """A terminal state, never a fabricated provider token."""

    status: OutcomeStatus
    reason: str

    def __post_init__(self) -> None:
        """Reject invalid status/reason pairs before they reach the service wire."""
        if self.status not in OUTCOME_STATUSES or not isinstance(self.reason, str) or not self.reason:
            raise VisionError(INVALID_OUTCOME)


@dataclass(frozen=True, slots=True)
class CaptchaAttemptResult:
    """A challenge outcome and an optional fresh provider-issued token."""

    outcome: VisionOutcome
    token: str | None = field(repr=False)


__all__ = [
    "ERROR_REASON",
    "INVALID_OUTCOME",
    "LOW_CONFIDENCE_REASON",
    "OUTCOME_STATUSES",
    "PAGE_ACCEPTED_REASON",
    "TIMED_OUT_REASON",
    "CaptchaAttemptResult",
    "OutcomeStatus",
    "VisionOutcome",
]
