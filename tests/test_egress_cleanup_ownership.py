"""Named egress ownership survives failed or cancelled native cleanup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.browser.browser import Browser
from prowl.browser.config import BrowserConfig
from prowl.browser.proxy.egress import EgressError, EgressPool

if TYPE_CHECKING:
    from collections.abc import Callable


class _ShutdownStub:
    def __init__(self, *, fail_first: int = 0, gate: asyncio.Event | None = None) -> None:
        self.calls = 0
        self.fail_first = fail_first
        self.gate = gate
        self.started = asyncio.Event()

    async def shutdown(self) -> None:
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.calls <= self.fail_first:
            message = f"shutdown attempt {self.calls} failed"
            raise RuntimeError(message)


class EgressCleanupOwnershipTests(IsolatedAsyncioTestCase):
    def _pool(
        self,
        factory: Callable[[str, int], _ShutdownStub] | None = None,
    ) -> tuple[EgressPool, list[str], dict[str, list[_ShutdownStub]]]:
        created: list[str] = []
        stubs: dict[str, list[_ShutdownStub]] = {}

        def create(**kwargs: object) -> Mock:
            name = str(kwargs["name"])
            stub = factory(name, len(stubs.get(name, []))) if factory else _ShutdownStub()
            stubs.setdefault(name, []).append(stub)
            created.append(name)
            owner = Mock(spec=Browser)
            owner.shutdown = AsyncMock(side_effect=stub.shutdown)
            owner.retry_shutdown = AsyncMock(side_effect=stub.shutdown)
            owner.resource_metrics.return_value = dict.fromkeys(
                (
                    "context_count",
                    "context_created_total",
                    "context_evicted_total",
                    "browser_restart_total",
                    "tabgroups_active",
                ),
                0,
            )
            return owner

        pool = EgressPool(
            BrowserConfig(profile_dir="profile", profile_archive="profile.zip"),
            {"a": "socks5://127.0.0.1:10001", "b": "socks5://127.0.0.1:10002"},
            idle_seconds=0,
        )
        patcher = patch("prowl.browser.proxy.egress.create_egress_browser", side_effect=create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return pool, created, stubs

    async def _close_task(self, pool: EgressPool, stub: _ShutdownStub) -> asyncio.Task[BaseException | None]:
        await asyncio.wait_for(stub.started.wait(), timeout=1)
        task = pool._live["a"].close_task
        self.assertIsNotNone(task)
        assert task is not None
        return task

    async def test_failed_idle_close_retains_owner_without_replacement(self) -> None:
        pool, created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(fail_first=1))
        await pool.acquire("a")
        await pool.release("a")
        task = await self._close_task(pool, stubs["a"][0])
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
        with self.assertRaises(EgressError):
            await pool.acquire("a")
        self.assertEqual(created, ["a"])
        self.assertEqual(pool.live_names(), ("a",))
        await pool.aclose()

    async def test_aclose_retries_failed_idle_close(self) -> None:
        pool, created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(fail_first=1))
        await pool.acquire("a")
        await pool.release("a")
        task = await self._close_task(pool, stubs["a"][0])
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
        await pool.aclose()
        self.assertEqual(stubs["a"][0].calls, 2)
        self.assertEqual(pool.live_names(), ())
        self.assertEqual(created, ["a"])

    async def test_second_aclose_retries_repeated_failure(self) -> None:
        pool, _created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(fail_first=2))
        await pool.acquire("a")
        with self.assertRaises(EgressError):
            await pool.aclose()
        with self.assertRaises(EgressError):
            await pool.aclose()
        self.assertEqual(pool.live_names(), ("a",))
        await pool.aclose()
        self.assertEqual(stubs["a"][0].calls, 3)
        self.assertEqual(pool.live_names(), ())

    async def test_aclose_joins_pending_close(self) -> None:
        gate = asyncio.Event()
        pool, _created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(gate=gate))
        await pool.acquire("a")
        await pool.release("a")
        await self._close_task(pool, stubs["a"][0])
        closer = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        self.assertFalse(closer.done())
        self.assertEqual(pool.live_names(), ("a",))
        gate.set()
        await asyncio.wait_for(closer, timeout=1)
        self.assertEqual(stubs["a"][0].calls, 1)
        self.assertEqual(pool.live_names(), ())

    async def test_failure_still_drains_other_browser(self) -> None:
        gate = asyncio.Event()
        pool, _created, stubs = self._pool(
            lambda name, _attempt: _ShutdownStub(
                fail_first=1 if name == "a" else 0, gate=gate if name == "b" else None
            ),
        )
        await pool.acquire("a")
        await pool.acquire("b")
        closer = asyncio.create_task(pool.aclose())
        await asyncio.wait_for(stubs["b"][0].started.wait(), timeout=1)
        self.assertFalse(closer.done())
        gate.set()
        with self.assertRaises(EgressError):
            await asyncio.wait_for(closer, timeout=1)
        self.assertEqual(stubs["a"][0].calls, 1)
        self.assertEqual(stubs["b"][0].calls, 1)
        self.assertEqual(pool.live_names(), ("a",))
        await pool.aclose()

    async def test_cancelled_waiter_preserves_owned_close(self) -> None:
        gate = asyncio.Event()
        pool, _created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(gate=gate))
        await pool.acquire("a")
        closer = asyncio.create_task(pool.aclose())
        task = await self._close_task(pool, stubs["a"][0])
        closer.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closer
        self.assertEqual(pool.live_names(), ("a",))
        self.assertFalse(task.cancelled())
        gate.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
        self.assertEqual(stubs["a"][0].calls, 1)
        self.assertEqual(pool.live_names(), ())

    async def test_concurrent_shutdown_deduplicates_close(self) -> None:
        gate = asyncio.Event()
        pool, _created, stubs = self._pool(lambda _name, _attempt: _ShutdownStub(gate=gate))
        await pool.acquire("a")
        first = asyncio.create_task(pool.aclose())
        second = asyncio.create_task(pool.aclose())
        await self._close_task(pool, stubs["a"][0])
        self.assertEqual(stubs["a"][0].calls, 1)
        gate.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(stubs["a"][0].calls, 1)
        self.assertEqual(pool.live_names(), ())

    async def test_acquisition_fenced_after_shutdown(self) -> None:
        pool, created, _stubs = self._pool()
        await pool.acquire("a")
        await pool.aclose()
        with self.assertRaises(EgressError):
            await pool.acquire("a")
        self.assertEqual(created, ["a"])

    async def test_cancelled_close_before_start_does_not_strand_acquisition(self) -> None:
        pool, created, _stubs = self._pool()
        await pool.acquire("a")
        async with pool._lock:
            entry = pool._live["a"]
            entry.closing = True
            task = pool._start_close_locked("a", entry)
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        with self.assertRaises(EgressError):
            await asyncio.wait_for(pool.acquire("a"), timeout=1)
        self.assertEqual(created, ["a"])
        self.assertEqual(pool.live_names(), ("a",))
        await pool.aclose()
        self.assertEqual(pool.live_names(), ())
