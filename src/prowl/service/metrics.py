"""Fixed in-process Prometheus counters, gauges and cumulative timings."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_COUNTER_NAMES = (
    "requests_total",
    "http_fastpath_total",
    "http_fastpath_success_total",
    "browser_escalations_total",
    "challenge_detected_total",
    "context_created_total",
    "context_evicted_total",
    "browser_restart_total",
)

_GAUGE_NAMES = (
    "requests_active",
    "context_count",
    "tabgroups_active",
)

_SUMMARY_NAMES = (
    "request_duration_seconds",
    "browser_acquire_seconds",
)


@dataclass(slots=True)
class Timing:
    """Cumulative summary aggregate: observation count and summed seconds."""

    count: int = 0
    sum: float = 0.0

    def observe(self, seconds: float) -> None:
        """Add one observation of ``seconds`` to the cumulative aggregate."""
        self.count += 1
        self.sum += seconds


@dataclass(slots=True)
class Metrics:
    """Fixed-shape counters, gauges and timing summaries for one process."""

    requests_total: int = 0
    http_fastpath_total: int = 0
    http_fastpath_success_total: int = 0
    browser_escalations_total: int = 0
    challenge_detected_total: int = 0
    context_created_total: int = 0
    context_evicted_total: int = 0
    browser_restart_total: int = 0

    requests_active: int = 0
    context_count: int = 0
    tabgroups_active: int = 0

    request_duration_seconds: Timing = field(default_factory=Timing)
    browser_acquire_seconds: Timing = field(default_factory=Timing)

    @contextmanager
    def request(self) -> Iterator[None]:
        """Count and time a request, releasing the gauge on errors and cancellation."""
        self.requests_total += 1
        self.requests_active += 1
        start = time.perf_counter()
        try:
            yield None
        finally:
            self.requests_active -= 1
            self.request_duration_seconds.observe(time.perf_counter() - start)

    def render(self) -> str:
        """Render the standard Prometheus text exposition format."""
        lines: list[str] = []
        for name in _COUNTER_NAMES:
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {getattr(self, name)}")
        for name in _GAUGE_NAMES:
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {getattr(self, name)}")
        for name in _SUMMARY_NAMES:
            timing = getattr(self, name)
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count {timing.count}")
            lines.append(f"{name}_sum {timing.sum!r}")
        return "\n".join(lines) + "\n"
