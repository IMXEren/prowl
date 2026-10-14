"""Prometheus ``/metrics`` endpoint and native resource-metric aggregation.

The semantics pinned here: ``BrowserBackend.render_metrics`` sums the default browser's
native resource snapshot with the egress pool's, assigns only those five resource fields,
and leaves the request counters and timings it already records untouched. Repeated rendering
changes nothing. The app serves the text at ``GET /metrics`` with the Prometheus content
type, and a legacy backend that lacks the concrete API is told metrics are unavailable
instead of being handed zeroed ones. No browser is launched and no network is touched.
"""

from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from prowl.browser import Browser
from prowl.service import backend as backend_module
from prowl.service.app import ServiceConfig, create_app
from prowl.service.backend import (
    BrowserBackend,
    CookieQuery,
    FetchRequest,
    FetchResult,
    InteractiveOpenResult,
    InteractiveRequest,
    InteractiveTab,
)
from prowl.service.metrics import Metrics

_RESOURCE_FIELDS = (
    "context_count",
    "tabgroups_active",
    "context_created_total",
    "context_evicted_total",
    "browser_restart_total",
)


def _sample(text: str, name: str) -> str:
    """Return the single sample value for *name*, asserting exactly one line carries it."""
    lines = [line for line in text.splitlines() if line.startswith(f"{name} ")]
    assert len(lines) == 1, f"expected exactly one sample line for {name}: {lines}"
    return lines[0].split()[1]


class _LegacyBackend:
    """A custom backend predating the concrete resource-metric API."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        raise NotImplementedError

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        raise NotImplementedError

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        return []

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        return []

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        return []

    async def close_session(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True


class _RenderFixture(IsolatedAsyncioTestCase):
    """Builds a ``BrowserBackend`` whose browser class and pool snapshot are both stubbed."""

    def _backend(self) -> tuple[BrowserBackend, Mock, Mock]:
        browser = Mock(spec=Browser)
        browser.resource_metrics.return_value = {
            "context_count": 1,
            "tabgroups_active": 2,
            "context_created_total": 3,
            "context_evicted_total": 4,
            "browser_restart_total": 5,
        }
        browser.start = AsyncMock()
        browser.shutdown = AsyncMock()
        browser_patch = patch.object(backend_module, "Browser", browser)
        browser_patch.start()
        self.addCleanup(browser_patch.stop)

        backend = BrowserBackend(metrics=Metrics())
        pool_mock = Mock(
            return_value={
                "context_count": 10,
                "tabgroups_active": 20,
                "context_created_total": 30,
                "context_evicted_total": 40,
                "browser_restart_total": 50,
            },
        )
        pool_patch = patch.object(backend._pool, "resource_metrics", pool_mock)
        pool_patch.start()
        self.addCleanup(pool_patch.stop)
        return backend, browser, pool_mock


class RenderMetricsAggregationTests(_RenderFixture):
    """The five native resource fields are the default snapshot plus the pool's."""

    async def test_default_and_named_native_fields_are_summed(self) -> None:
        backend, browser, pool_mock = self._backend()
        native = browser.resource_metrics.return_value
        pooled = pool_mock.return_value

        text = backend.render_metrics()

        for name in _RESOURCE_FIELDS:
            self.assertEqual(_sample(text, name), str(native[name] + pooled[name]))
        await backend.aclose()

    async def test_repeated_render_is_stable_and_leaves_request_metrics_untouched(self) -> None:
        backend, _browser, _pool_mock = self._backend()
        metrics = backend.metrics
        metrics.requests_total = 7
        metrics.http_fastpath_total = 3
        metrics.http_fastpath_success_total = 2
        metrics.browser_escalations_total = 4
        metrics.challenge_detected_total = 5
        metrics.request_duration_seconds.observe(1.5)
        metrics.browser_acquire_seconds.observe(0.25)

        first = backend.render_metrics()
        second = backend.render_metrics()

        self.assertEqual(first, second)
        self.assertEqual(metrics.requests_total, 7)
        self.assertEqual(metrics.http_fastpath_total, 3)
        self.assertEqual(metrics.http_fastpath_success_total, 2)
        self.assertEqual(metrics.browser_escalations_total, 4)
        self.assertEqual(metrics.challenge_detected_total, 5)
        self.assertEqual(metrics.request_duration_seconds.count, 1)
        self.assertEqual(metrics.request_duration_seconds.sum, 1.5)
        self.assertEqual(metrics.browser_acquire_seconds.count, 1)
        self.assertEqual(metrics.browser_acquire_seconds.sum, 0.25)
        await backend.aclose()

    async def test_named_retirement_keeps_cumulative_counts_and_drops_gauges(self) -> None:
        backend, browser, pool_mock = self._backend()
        native = browser.resource_metrics.return_value
        pool_mock.return_value = {
            "context_count": 7,
            "tabgroups_active": 8,
            "context_created_total": 9,
            "context_evicted_total": 6,
            "browser_restart_total": 2,
        }
        live = backend.render_metrics()

        pool_mock.return_value = {
            "context_count": 0,
            "tabgroups_active": 0,
            "context_created_total": 9,
            "context_evicted_total": 6,
            "browser_restart_total": 2,
        }
        retired = backend.render_metrics()

        self.assertEqual(_sample(live, "context_count"), str(native["context_count"] + 7))
        self.assertEqual(_sample(live, "context_created_total"), str(native["context_created_total"] + 9))
        self.assertEqual(_sample(retired, "context_count"), str(native["context_count"]))
        self.assertEqual(_sample(retired, "tabgroups_active"), str(native["tabgroups_active"]))
        self.assertEqual(_sample(retired, "context_created_total"), str(native["context_created_total"] + 9))
        self.assertEqual(_sample(retired, "context_evicted_total"), str(native["context_evicted_total"] + 6))
        self.assertEqual(_sample(retired, "browser_restart_total"), str(native["browser_restart_total"] + 2))
        await backend.aclose()


class MetricsEndpointTests(IsolatedAsyncioTestCase):
    """The app serves the backend's Prometheus text at ``GET /metrics``."""

    async def test_get_metrics_serves_prometheus_text_for_a_browser_backend(self) -> None:
        browser = Mock(spec=Browser)
        browser.resource_metrics.return_value = {
            "context_count": 1,
            "tabgroups_active": 2,
            "context_created_total": 3,
            "context_evicted_total": 4,
            "browser_restart_total": 5,
        }
        browser.start = AsyncMock()
        browser.shutdown = AsyncMock()

        with patch.object(backend_module, "Browser", browser):
            backend = BrowserBackend()
            client = TestClient(TestServer(create_app(ServiceConfig(max_concurrency=2), backend)))
            await client.start_server()
            try:
                response = await client.get("/metrics")
                status = response.status
                content_type = response.headers["Content-Type"]
                text = await response.text()
            finally:
                await client.close()

        self.assertEqual(status, 200)
        self.assertEqual(content_type, "text/plain; version=0.0.4; charset=utf-8")
        self.assertIn("# TYPE context_count gauge", text)
        for name, value in (
            ("context_count", 1),
            ("tabgroups_active", 2),
            ("context_created_total", 3),
            ("context_evicted_total", 4),
            ("browser_restart_total", 5),
        ):
            self.assertEqual(_sample(text, name), str(value))
        browser.start.assert_awaited()
        browser.shutdown.assert_awaited()


class LegacyBackendMetricsTests(IsolatedAsyncioTestCase):
    """A legacy backend gets a clear 501 and keeps its health endpoints working."""

    async def test_legacy_backend_is_told_metrics_are_unavailable(self) -> None:
        backend = _LegacyBackend()
        client = TestClient(TestServer(create_app(ServiceConfig(max_concurrency=2), backend)))
        await client.start_server()
        try:
            metrics = await client.get("/metrics")
            metrics_status = metrics.status
            metrics_text = await metrics.text()
            health = await client.get("/healthz")
            ready = await client.get("/readyz")
            ready_body = await ready.json()
        finally:
            await client.close()

        self.assertEqual(metrics_status, 501)
        self.assertIn("not available", metrics_text)
        self.assertEqual(health.status, 200)
        self.assertEqual(ready.status, 200)
        self.assertEqual(ready_body["status"], "ok")
        self.assertTrue(backend.started)
        self.assertTrue(backend.closed)
