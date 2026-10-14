"""Identity-scoped tab pressure: one egress's busy work never disturbs another's tabs.

The backend keeps a live request count per egress and selects tab takeovers only among the tabs
that leave through the same egress, so a saturated identity cannot drive the pressure check or the
oldest-tab choice for a quiet one. These tests pin that scoping and the count lifecycle (kept across
success, error and cancellation) with in-memory ``_LiveTab`` entries and small native-spec mocks.
No browser is launched and no network is touched.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import BrowserContext as PWBrowserContext

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.lifecycle.startup import BrowserLifecycle
from prowl.browser.proxy.egress import DEFAULT_EGRESS_NAME
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, FetchRequest, FetchResult, InteractiveTab
from prowl.service.protocol import BROWSER_MODE, HTTP_MODE

_LiveTab = backend_module._LiveTab

_URL = "https://example.com/"
_PAGE = "<html><body>ok</body></html>"


class _FakeOwner(Browser):
    _runtime = BrowserRuntimeState(max_groups=1)
    _lifecycle = BrowserLifecycle(driver=_runtime)

    @classmethod
    async def start(cls) -> None:
        message = "unexpected native startup"
        raise AssertionError(message)

    @classmethod
    async def shutdown(cls) -> None:
        cls._runtime.contexts.reset()


def _request(**fields: Any) -> FetchRequest:
    values: dict[str, Any] = {"url": _URL}
    values.update(fields)
    return FetchRequest(**values)


def _result() -> FetchResult:
    return FetchResult(
        url=_URL,
        status_code=200,
        headers={"content-type": "text/html"},
        response=_PAGE,
        cookies=[],
        user_agent="Mozilla/5.0 (Browser)",
    )


def _entry(tab_id: str, *, egress: str, last_used: float) -> backend_module._LiveTab:
    tab = InteractiveTab(tab_id=tab_id, url=_URL, title="", status_code=200, requested_url=_URL, egress=egress)
    return _LiveTab(tab=tab, group=Mock(spec=TabGroup), last_used=last_used)


class _PressureFixture(IsolatedAsyncioTestCase):
    """Builds a backend whose browser class is a stub, so no process is launched."""

    def _backend(self, *, max_concurrency: int = 0, steal_least_recent: bool = True) -> BrowserBackend:
        backend = BrowserBackend(steal_least_recent=steal_least_recent, max_concurrency=max_concurrency)
        _FakeOwner._runtime = BrowserRuntimeState(max_groups=1)
        _FakeOwner._lifecycle = BrowserLifecycle(driver=_FakeOwner._runtime)
        _FakeOwner._runtime.contexts.bind_shared(Mock(spec=PWBrowserContext))
        patcher = patch.object(backend_module, "Browser", _FakeOwner)
        patcher.start()
        self.addCleanup(patcher.stop)
        return backend

    def _record_close(self, backend: BrowserBackend) -> list[str]:
        """Replace ``close_interactive`` with a recorder that also removes the chosen tab."""
        closed: list[str] = []

        async def close(tab_id: str | None) -> list[str]:
            if tab_id is None:
                ids = sorted(backend._tabs)
                backend._tabs.clear()
                return ids
            if backend._tabs.pop(tab_id, None) is not None:
                closed.append(tab_id)
                return [tab_id]
            return []

        backend.close_interactive = close
        return closed


class IdentityScopedSelectionTests(_PressureFixture):
    """Takeover selection and the pressure check are scoped to the request's egress."""

    async def test_a_quiet_identity_feels_no_pressure_while_another_is_saturated(self) -> None:
        backend = self._backend(max_concurrency=2)
        backend._tabs["d1"] = _entry("d1", egress="decodo", last_used=1.0)
        backend._tabs["d2"] = _entry("d2", egress="decodo", last_used=2.0)
        backend._in_flight_by_egress["decodo"] = 2
        closed = self._record_close(backend)

        # The default identity has no tabs and no work: nothing to take, nothing disturbed.
        self.assertIsNone(await backend._steal_least_recent(DEFAULT_EGRESS_NAME))
        self.assertEqual(closed, [])
        self.assertEqual(sorted(backend._tabs), ["d1", "d2"])

        # The saturated identity itself is eligible, and takes its own oldest tab.
        self.assertEqual(await backend._steal_least_recent("decodo"), "d1")
        self.assertEqual(closed, ["d1"])
        await backend.aclose()

    async def test_pressure_takes_the_oldest_matching_tab_even_when_a_foreign_tab_is_older(self) -> None:
        backend = self._backend(max_concurrency=2)
        backend._tabs["foreign"] = _entry("foreign", egress="decodo", last_used=0.5)
        backend._tabs["old"] = _entry("old", egress=DEFAULT_EGRESS_NAME, last_used=1.0)
        backend._tabs["new"] = _entry("new", egress=DEFAULT_EGRESS_NAME, last_used=2.0)
        self._record_close(backend)

        # The global oldest tab belongs to another identity; the default identity takes its own.
        self.assertEqual(await backend._steal_least_recent(DEFAULT_EGRESS_NAME), "old")
        self.assertIn("foreign", backend._tabs)
        self.assertIn("new", backend._tabs)
        await backend.aclose()

    async def test_a_lone_matching_tab_is_protected_despite_foreign_tabs(self) -> None:
        backend = self._backend(max_concurrency=2)
        backend._tabs["only"] = _entry("only", egress=DEFAULT_EGRESS_NAME, last_used=1.0)
        for index in range(3):
            backend._tabs[f"f{index}"] = _entry(f"f{index}", egress="decodo", last_used=0.1 + index)
        backend._in_flight_by_egress[DEFAULT_EGRESS_NAME] = 5
        closed = self._record_close(backend)

        # Its own single tab is never taken, however many foreign tabs there are.
        self.assertIsNone(await backend._steal_least_recent(DEFAULT_EGRESS_NAME))
        self.assertEqual(closed, [])
        self.assertIn("only", backend._tabs)
        await backend.aclose()


class IdentityScopedCountTests(_PressureFixture):
    """The per-egress in-flight count tracks only live requests and is cleared on every outcome."""

    async def test_counts_stay_separate_across_egresses_and_clear_on_success(self) -> None:
        backend = self._backend()
        release = asyncio.Event()
        both = asyncio.Event()
        started = 0

        async def gated(request: FetchRequest) -> FetchResult:
            nonlocal started
            self.assertIn(request.egress, (DEFAULT_EGRESS_NAME, "decodo"))
            started += 1
            if started == 2:
                both.set()
            await release.wait()
            return _result()

        backend._fetch = gated
        first = asyncio.create_task(backend.fetch(None, _request(egress=DEFAULT_EGRESS_NAME)))
        second = asyncio.create_task(backend.fetch(None, _request(egress="decodo")))
        await asyncio.wait_for(both.wait(), timeout=1)
        self.assertEqual(backend._in_flight_by_egress, {DEFAULT_EGRESS_NAME: 1, "decodo": 1})
        self.assertEqual(backend._in_flight, 2)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(backend._in_flight_by_egress, {})
        self.assertEqual(backend._in_flight, 0)
        await backend.aclose()

    async def test_counts_clear_on_error_and_cancellation(self) -> None:
        backend = self._backend()

        async def boom(request: FetchRequest) -> FetchResult:
            self.assertEqual(request.egress, "decodo")
            msg = "boom"
            raise RuntimeError(msg)

        backend._fetch = boom
        with self.assertRaises(RuntimeError):
            await backend.fetch(None, _request(egress="decodo"))
        self.assertEqual(backend._in_flight_by_egress, {})
        self.assertEqual(backend._in_flight, 0)

        entered = asyncio.Event()

        async def blocked(request: FetchRequest) -> FetchResult:
            self.assertEqual(request.egress, "decodo")
            entered.set()
            await asyncio.Event().wait()
            return _result()

        backend._fetch = blocked
        task = asyncio.create_task(backend.fetch(None, _request(egress="decodo")))
        await asyncio.wait_for(entered.wait(), timeout=1)
        self.assertEqual(backend._in_flight_by_egress, {"decodo": 1})
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(backend._in_flight_by_egress, {})
        self.assertEqual(backend._in_flight, 0)
        await backend.aclose()


class EgressForwardingTests(_PressureFixture):
    """Both take-over call sites pass the request's actual egress to the selection."""

    async def test_browser_branch_and_auto_stage_forward_the_request_egress(self) -> None:
        backend = self._backend()
        recorded: list[str] = []

        async def record(egress: str = DEFAULT_EGRESS_NAME) -> None:
            recorded.append(egress)

        backend._steal_least_recent = record
        backend._fetch = AsyncMock(return_value=_result())
        await backend.fetch(None, _request(mode=BROWSER_MODE, egress="decodo"))
        self.assertEqual(recorded, ["decodo"])

        # The browser fallback stage is the call site an auto fetch reaches its browser through.
        backend._create_group = AsyncMock(return_value=Mock(spec=TabGroup))
        backend._fetch_in_group = AsyncMock(return_value=_result())
        backend._quit_group = AsyncMock()
        handle = _FakeOwner._runtime.contexts.shared()
        assert handle is not None
        await backend._browser_stage(
            _FakeOwner,
            handle,
            _request(egress="decodo"),
            time.monotonic() + 5,
            apply_cookies=False,
        )
        self.assertEqual(recorded, ["decodo", "decodo"])
        await backend.aclose()

    async def test_a_successful_http_request_never_calls_takeover(self) -> None:
        backend = self._backend()
        result = _result()
        result = replace(result, mode=HTTP_MODE)
        backend._fetch = AsyncMock(return_value=result)
        backend._steal_least_recent = AsyncMock()

        result = await backend.fetch(None, _request(mode=HTTP_MODE, egress=DEFAULT_EGRESS_NAME))

        backend._steal_least_recent.assert_not_awaited()
        self.assertEqual(result.mode, HTTP_MODE)
        await backend.aclose()
