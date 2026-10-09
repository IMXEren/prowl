"""A pool's resource snapshot: cumulative counters plus live-owner gauges."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.browser.browser import Browser
from prowl.browser.config import BrowserConfig
from prowl.browser.proxy.egress import EgressPool

if TYPE_CHECKING:
    from collections.abc import Mapping

_COUNTERS = ("context_created_total", "context_evicted_total", "browser_restart_total")
_GAUGES = ("context_count", "tabgroups_active")
_SNAPSHOT_KEYS = frozenset(_COUNTERS + _GAUGES)


def _metrics(**overrides: int) -> dict[str, int]:
    """Return a full snapshot with every metric zeroed, overridden as asked."""
    snapshot: dict[str, int] = dict.fromkeys(_COUNTERS + _GAUGES, 0)
    snapshot.update(overrides)
    return snapshot


class EgressResourceMetricsTests(IsolatedAsyncioTestCase):
    def _pool(
        self,
        metrics: Mapping[str, list[dict[str, int]]],
        *,
        fail_first: Mapping[str, int] | None = None,
        legacy: frozenset[str] = frozenset(),
    ) -> tuple[EgressPool, dict[str, list[Mock]]]:
        """Build a pool whose owners are mocks handing out the given snapshots.

        Every created owner is recorded in creation order. A name in *legacy* gets an
        owner with no ``resource_metrics`` at all, standing in for a pre-counters one.
        *fail_first* makes that name's first N close attempts raise.
        """
        created: dict[str, list[Mock]] = {}
        fail_first = fail_first or {}

        def create(**kwargs: object) -> Mock:
            name = str(kwargs["name"])
            owner = Mock(spec=Browser)
            if name in legacy:
                del owner.resource_metrics
            else:
                calls = {"n": 0}

                async def shutdown() -> None:
                    calls["n"] += 1
                    if calls["n"] <= fail_first.get(name, 0):
                        message = f"shutdown {name} attempt {calls['n']} failed"
                        raise RuntimeError(message)

                owner.shutdown = AsyncMock(side_effect=shutdown)
                owner.retry_shutdown = AsyncMock(side_effect=shutdown)
                owner.resource_metrics.return_value = metrics[name][len(created.get(name, []))]
            created.setdefault(name, []).append(owner)
            return owner

        pool = EgressPool(
            BrowserConfig(profile_dir="profile", profile_archive="profile.zip"),
            {"a": "socks5://127.0.0.1:10001", "b": "socks5://127.0.0.1:10002"},
            idle_seconds=0,
        )
        patcher = patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return pool, created

    async def _retire(self, pool: EgressPool, name: str) -> None:
        """Release *name* and wait for its tracked close to finish, whatever its outcome."""
        entry = pool._live[name]
        await pool.release(name)
        await asyncio.wait_for(entry.shutdown_done.wait(), timeout=1)

    async def test_aggregates_gauges_and_counters_across_live_owners(self) -> None:
        pool, _created = self._pool(
            {
                "a": [
                    _metrics(
                        context_count=2,
                        tabgroups_active=1,
                        context_created_total=5,
                        context_evicted_total=1,
                        browser_restart_total=1,
                    ),
                ],
                "b": [_metrics(context_count=1, tabgroups_active=2, context_created_total=3)],
            },
        )
        await pool.acquire("a")
        await pool.acquire("b")
        snapshot = pool.resource_metrics()
        self.assertEqual(set(snapshot), set(_SNAPSHOT_KEYS))
        self.assertEqual(
            snapshot,
            _metrics(
                context_count=3,
                tabgroups_active=3,
                context_created_total=8,
                context_evicted_total=1,
                browser_restart_total=1,
            ),
        )
        await pool.aclose()

    async def test_successful_idle_retirement_banks_counters_and_drops_gauges(self) -> None:
        pool, created = self._pool(
            {
                "a": [
                    _metrics(
                        context_count=3,
                        tabgroups_active=2,
                        context_created_total=4,
                        context_evicted_total=2,
                        browser_restart_total=1,
                    ),
                ],
            },
        )
        await pool.acquire("a")
        await self._retire(pool, "a")
        self.assertEqual(pool.live_names(), ())
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=0,
                tabgroups_active=0,
                context_created_total=4,
                context_evicted_total=2,
                browser_restart_total=1,
            ),
        )
        created["a"][0].shutdown.assert_awaited_once()
        self.assertEqual(pool.resource_metrics()["context_created_total"], 4)

    async def test_failed_close_retains_gauge_and_banks_on_successful_retry(self) -> None:
        pool, created = self._pool(
            {
                "a": [
                    _metrics(
                        context_count=3,
                        tabgroups_active=2,
                        context_created_total=4,
                        context_evicted_total=2,
                        browser_restart_total=1,
                    ),
                ],
            },
            fail_first={"a": 1},
        )
        await pool.acquire("a")
        await self._retire(pool, "a")
        self.assertEqual(pool.live_names(), ("a",))
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=3,
                tabgroups_active=2,
                context_created_total=4,
                context_evicted_total=2,
                browser_restart_total=1,
            ),
        )
        await pool.aclose()
        self.assertEqual(pool.live_names(), ())
        created["a"][0].retry_shutdown.assert_awaited_once()
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=0,
                tabgroups_active=0,
                context_created_total=4,
                context_evicted_total=2,
                browser_restart_total=1,
            ),
        )

    async def test_reacquisition_accumulates_old_and_new_totals(self) -> None:
        pool, _created = self._pool(
            {
                "a": [
                    _metrics(context_created_total=2, browser_restart_total=1),
                    _metrics(
                        context_count=1,
                        tabgroups_active=1,
                        context_created_total=3,
                        context_evicted_total=1,
                    ),
                ],
            },
        )
        await pool.acquire("a")
        await self._retire(pool, "a")
        self.assertEqual(pool.live_names(), ())
        await pool.acquire("a")
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=1,
                tabgroups_active=1,
                context_created_total=5,
                context_evicted_total=1,
                browser_restart_total=1,
            ),
        )
        await pool.aclose()
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=0,
                tabgroups_active=0,
                context_created_total=5,
                context_evicted_total=1,
                browser_restart_total=1,
            ),
        )

    async def test_pool_close_preserves_counters_and_zeroes_gauges(self) -> None:
        pool, _created = self._pool(
            {
                "a": [
                    _metrics(
                        context_count=4,
                        tabgroups_active=1,
                        context_created_total=7,
                        context_evicted_total=2,
                        browser_restart_total=3,
                    ),
                ],
            },
        )
        await pool.acquire("a")
        await pool.aclose()
        self.assertEqual(
            pool.resource_metrics(),
            _metrics(
                context_count=0,
                tabgroups_active=0,
                context_created_total=7,
                context_evicted_total=2,
                browser_restart_total=3,
            ),
        )

    async def test_legacy_owner_without_metrics_contributes_nothing(self) -> None:
        pool, _created = self._pool({}, legacy=frozenset({"a"}))
        await pool.acquire("a")
        self.assertEqual(pool.resource_metrics(), _metrics())
        await self._retire(pool, "a")
        self.assertEqual(pool.live_names(), ())
        self.assertEqual(pool.resource_metrics(), _metrics())
