import asyncio

import pytest

from prowl.service import metrics as metrics_module
from prowl.service.metrics import Metrics, Timing

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
_GAUGE_NAMES = ("requests_active", "context_count", "tabgroups_active")
_SUMMARY_NAMES = ("request_duration_seconds", "browser_acquire_seconds")


def _value_line(text: str, name: str) -> str:
    lines = [line for line in text.splitlines() if line.startswith(f"{name} ")]
    assert len(lines) == 1, f"expected exactly one sample line for {name}: {lines}"
    return lines[0]


def test_initial_render_exposes_all_goal_metric_names() -> None:
    text = Metrics().render()

    assert text.endswith("\n")
    for name in _COUNTER_NAMES:
        assert f"# TYPE {name} counter" in text
    for name in _GAUGE_NAMES:
        assert f"# TYPE {name} gauge" in text
    for name in _SUMMARY_NAMES:
        assert f"# TYPE {name} summary" in text
        assert _value_line(text, f"{name}_count") == f"{name}_count 0"
        assert _value_line(text, f"{name}_sum") == f"{name}_sum 0.0"


def test_counters_and_gauges_reflect_exact_values_and_types() -> None:
    metrics = Metrics()
    metrics.requests_total = 3
    metrics.http_fastpath_total = 2
    metrics.context_count = 1
    metrics.tabgroups_active = 4

    assert isinstance(metrics.requests_total, int)
    assert isinstance(metrics.context_count, int)

    text = metrics.render()
    assert _value_line(text, "requests_total") == "requests_total 3"
    assert _value_line(text, "http_fastpath_total") == "http_fastpath_total 2"
    assert _value_line(text, "context_count") == "context_count 1"
    assert _value_line(text, "tabgroups_active") == "tabgroups_active 4"


def test_timing_summaries_accumulate_observations() -> None:
    timing = Timing()
    assert timing.count == 0
    assert timing.sum == 0.0

    timing.observe(0.25)
    timing.observe(0.75)
    assert timing.count == 2
    assert timing.sum == pytest.approx(1.0)

    metrics = Metrics()
    metrics.browser_acquire_seconds.observe(2.0)
    metrics.browser_acquire_seconds.observe(4.0)
    text = metrics.render()
    assert _value_line(text, "browser_acquire_seconds_count") == "browser_acquire_seconds_count 2"
    assert _value_line(text, "browser_acquire_seconds_sum") == "browser_acquire_seconds_sum 6.0"


def test_request_success_releases_active_and_records_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = Metrics()
    ticks = iter([10.0, 10.5])
    monkeypatch.setattr(metrics_module.time, "perf_counter", lambda: next(ticks))

    with metrics.request() as yielded:
        assert yielded is None
        assert metrics.requests_active == 1
        assert metrics.requests_total == 1

    assert metrics.requests_active == 0
    assert metrics.requests_total == 1
    assert metrics.request_duration_seconds.count == 1
    assert metrics.request_duration_seconds.sum == pytest.approx(0.5)


def test_request_exception_releases_active_and_records_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = Metrics()
    ticks = iter([1.0, 1.75])
    monkeypatch.setattr(metrics_module.time, "perf_counter", lambda: next(ticks))
    message = "boom"
    active_samples: list[int] = []

    def raise_inside() -> None:
        with metrics.request():
            active_samples.append(metrics.requests_active)
            raise ValueError(message)

    with pytest.raises(ValueError, match=message):
        raise_inside()

    assert active_samples == [1]
    assert metrics.requests_active == 0
    assert metrics.requests_total == 1
    assert metrics.request_duration_seconds.count == 1
    assert metrics.request_duration_seconds.sum == pytest.approx(0.75)


def test_request_cancellation_releases_active() -> None:
    metrics = Metrics()

    async def scenario() -> None:
        async def work() -> None:
            with metrics.request():
                await asyncio.sleep(10)

        task = asyncio.create_task(work())
        await asyncio.sleep(0)
        assert metrics.requests_active == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert metrics.requests_active == 0
    assert metrics.requests_total == 1
    assert metrics.request_duration_seconds.count == 1


def test_nested_requests_peak_active_with_patched_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = Metrics()
    ticks = iter([1.0, 1.5, 3.0, 4.0])
    monkeypatch.setattr(metrics_module.time, "perf_counter", lambda: next(ticks))

    with metrics.request():
        assert metrics.requests_active == 1
        with metrics.request():
            assert metrics.requests_active == 2

    assert metrics.requests_active == 0
    assert metrics.requests_total == 2
    assert metrics.request_duration_seconds.count == 2
    assert metrics.request_duration_seconds.sum == pytest.approx(4.5)


def test_render_has_no_labels_quantiles_or_sample_storage() -> None:
    metrics = Metrics()
    with metrics.request():
        pass
    metrics.browser_acquire_seconds.observe(0.5)
    text = metrics.render()

    assert "{" not in text
    assert "}" not in text
    assert "quantile" not in text
    assert "# HELP" not in text
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            assert len(line.split()) == 4
        else:
            name, value = line.split()
            float(value)
            assert name == name.strip()
