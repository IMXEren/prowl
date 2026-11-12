"""BrowserHttpClients regressions: probe isolation, per-generation identity reuse and cleanup.

Every test drives native-spec mocks (``Mock(spec=BrowserContext)`` and friends) wrapping lightweight
stand-ins, so the manager is exercised through the real native interface without launching a
browser. Nothing external is contacted: the probe page's requests are answered in process.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from playwright.async_api import Browser, BrowserContext, Page

from prowl.service.browser_http import SYNTHETIC_ORIGIN, BrowserHttpClients, observe_identity
from prowl.service.http_transport import HttpTransportError, UnsupportedIdentityError

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
)
_BRANDS = (("Not?A_Brand", "99"), ("Chromium", "150"), ("Google Chrome", "150"))
_FULL_VERSIONS = (
    ("Not?A_Brand", "99.0.0.0"),
    ("Chromium", "150.0.0.0"),
    ("Google Chrome", "150.0.0.0"),
)
_WIRE_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Not?A_Brand";v="99", "Chromium";v="150", "Google Chrome";v="150"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Accept": "text/html,application/xhtml+xml",
    "sec-fetch-mode": "navigate",
}


def _reported(**overrides: Any) -> dict[str, Any]:
    """Return a fresh navigator report with *overrides* applied."""
    values: dict[str, Any] = {
        "userAgent": _USER_AGENT,
        "languages": ["en-US", "en"],
        "platform": "Windows",
        "mobile": False,
        "brands": [[brand, version] for brand, version in _BRANDS],
        "architecture": "x86",
        "bitness": "64",
        "platformVersion": "15.0.0",
        "fullVersionList": [[brand, version] for brand, version in _FULL_VERSIONS],
    }
    values.update(overrides)
    return values


class _FakeRoute:
    """Records what a route did, so a test can tell a fulfilled request from an aborted one."""

    def __init__(self) -> None:
        self.fulfilled: list[tuple[int, str, str]] = []
        self.aborted = False

    async def fulfill(self, *, status: int, content_type: str, body: str) -> None:
        self.fulfilled.append((status, content_type, body))

    async def abort(self) -> None:
        self.aborted = True


class _FakeRequest:
    """A request as a route handler sees it: its url and the headers it would send."""

    def __init__(self, url: str, headers: dict[str, str], all_headers: dict[str, str] | None = None) -> None:
        self.url = url
        self.headers = headers
        self._all_headers = headers if all_headers is None else all_headers

    async def all_headers(self) -> dict[str, str]:
        return dict(self._all_headers)


class _FakeBrowser:
    """A native browser generation, with the identity it reports and the headers it sends."""

    def __init__(self, *, reported: Any = None, wire_headers: dict[str, str] | None = None) -> None:
        self.reported = _reported() if reported is None else reported
        self.wire_headers = dict(_WIRE_HEADERS) if wire_headers is None else wire_headers
        self.complete_headers = dict(self.wire_headers)
        self.goto_error: Exception | None = None
        self.goto_gate: asyncio.Event | None = None
        self.pages_created = 0
        self.connected = True
        self.page_opened = asyncio.Event()
        self.goto_started = asyncio.Event()
        self.listeners: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> None:
        self.listeners.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Any) -> None:
        handlers = self.listeners.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, *args: object) -> None:
        if event == "disconnected":
            self.connected = False
        for handler in list(self.listeners.get(event, [])):
            handler(*args)


class _FakePage:
    """A native page that hands its navigation to the registered route and reports identity."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.routes: list[tuple[str, Any]] = []
        self.goto_calls = 0
        self.goto_route: _FakeRoute | None = None
        self.closed = False

    async def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    async def goto(self, url: str, *, wait_until: str | None = None) -> _FakeRoute:
        self.goto_calls += 1
        self._browser.goto_started.set()
        if self._browser.goto_error is not None:
            raise self._browser.goto_error
        if self._browser.goto_gate is not None:
            await self._browser.goto_gate.wait()
        route = _FakeRoute()
        request = _FakeRequest(url, self._browser.wire_headers, self._browser.complete_headers)
        await self.routes[-1][1](route, request)
        self.goto_route = route
        return route

    async def evaluate(self, script: str) -> Any:
        return self._browser.reported

    async def close(self) -> None:
        self.closed = True


class _FakeContext:
    """A native browser context stand-in, with its own pages and native events."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self.browser = browser
        self.pages: list[_FakePage] = []
        self.closed = False
        self.listeners: dict[str, list[Any]] = {}

    async def new_page(self) -> Mock:
        self.browser.pages_created += 1
        page = _FakePage(self.browser)
        self.pages.append(page)
        self.browser.page_opened.set()
        return _native_page(page)

    def on(self, event: str, handler: Any) -> None:
        self.listeners.setdefault(event, []).append(handler)

    def remove_listener(self, event: str, handler: Any) -> None:
        handlers = self.listeners.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, event: str, *args: object) -> None:
        if event == "close":
            self.closed = True
        for handler in list(self.listeners.get(event, [])):
            handler(*args)


def _native_page(fake: _FakePage) -> Mock:
    """Return a native-spec page mock whose async methods drive *fake*."""
    page = Mock(spec=Page)
    page.route = AsyncMock(side_effect=fake.route)
    page.goto = AsyncMock(side_effect=fake.goto)
    page.evaluate = AsyncMock(side_effect=fake.evaluate)
    page.close = AsyncMock(side_effect=fake.close)
    return page


def _native_browser(fake: _FakeBrowser) -> Mock:
    """Return a native-spec browser mock whose callbacks and liveness drive *fake*."""
    browser = Mock(spec=Browser)
    browser.on = Mock(side_effect=fake.on)
    browser.remove_listener = Mock(side_effect=fake.remove_listener)
    browser.is_connected = Mock(side_effect=lambda: fake.connected)
    return browser


def _native_context(fake: _FakeContext, browser: Mock) -> Mock:
    """Return a native-spec context mock wrapping *fake* and reporting *browser*."""
    context = Mock(spec=BrowserContext)
    context.browser = browser
    context.new_page = AsyncMock(side_effect=fake.new_page)
    context.on = Mock(side_effect=fake.on)
    context.remove_listener = Mock(side_effect=fake.remove_listener)
    context.is_closed = Mock(side_effect=lambda: fake.closed)
    return context


def _build(
    *,
    reported: Any = None,
    wire_headers: dict[str, str] | None = None,
) -> tuple[Mock, Mock, _FakeBrowser, _FakeContext]:
    """Build one native browser, one native context and the stand-ins behind them."""
    fake_browser = _FakeBrowser(reported=reported, wire_headers=wire_headers)
    native_browser = _native_browser(fake_browser)
    fake_context = _FakeContext(fake_browser)
    native_context = _native_context(fake_context, native_browser)
    return native_browser, native_context, fake_browser, fake_context


async def _settle(manager: BrowserHttpClients) -> None:
    """Let every callback task the manager spawned run to completion."""
    for _ in range(20):
        if not manager._tasks:
            return
        await asyncio.gather(*list(manager._tasks), return_exceptions=True)
    msg = "callback tasks did not settle"
    raise AssertionError(msg)


async def _fail_close() -> None:
    """A client close that always fails, for retirement and shutdown regressions."""
    msg = "close failed"
    raise RuntimeError(msg)


class _ObservationTests(IsolatedAsyncioTestCase):
    """The probe page's isolation and the identity it reports."""

    async def test_observe_maps_navigator_metadata_to_native_identity_headers(self) -> None:
        wire = dict(_WIRE_HEADERS)
        wire.pop("Accept-Language")
        _, context, fake_browser, _ = _build(wire_headers=wire)
        fake_browser.complete_headers = dict(_WIRE_HEADERS)
        identity = await observe_identity(context)
        self.assertEqual(identity.user_agent, _USER_AGENT)
        self.assertEqual(identity.platform, "Windows")
        self.assertEqual(identity.chrome_major, 150)
        self.assertEqual(identity.impersonate, "chrome150")
        headers = dict(identity.headers)
        self.assertEqual(headers["user-agent"], _USER_AGENT)
        self.assertEqual(headers["accept-language"], "en-US,en;q=0.9")
        self.assertEqual(headers["sec-ch-ua"], _WIRE_HEADERS["sec-ch-ua"])
        self.assertEqual(headers["sec-ch-ua-platform"], '"Windows"')
        self.assertNotIn("sec-ch-ua-arch", headers)
        self.assertNotIn("accept", headers)
        self.assertNotIn("sec-fetch-mode", headers)
        self.assertEqual(fake_browser.pages_created, 1)

    async def test_absent_default_english_header_uses_the_verified_wire_value(self) -> None:
        wire = dict(_WIRE_HEADERS)
        wire.pop("Accept-Language")
        _, context, _, _ = _build(wire_headers=wire)
        identity = await observe_identity(context)
        self.assertEqual(dict(identity.headers)["accept-language"], "en-US,en;q=0.9")

    async def test_absent_non_english_header_refuses_the_http_identity(self) -> None:
        wire = dict(_WIRE_HEADERS)
        wire.pop("Accept-Language")
        _, context, _, fake_context = _build(wire_headers=wire, reported=_reported(languages=["fr-FR"]))
        with self.assertRaisesRegex(UnsupportedIdentityError, "Accept-Language"):
            await observe_identity(context)
        self.assertTrue(fake_context.pages[0].closed)

    async def test_probe_fulfills_only_the_synthetic_origin_and_aborts_everything_else(self) -> None:
        _, context, _, fake_context = _build()
        await observe_identity(context)
        page = fake_context.pages[0]
        self.assertEqual(page.routes[0][0], "**/*")
        route = page.goto_route
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.fulfilled[0][0], 200)
        self.assertEqual(route.fulfilled[0][1], "text/html")
        self.assertFalse(route.aborted)
        other = _FakeRoute()
        await page.routes[0][1](other, _FakeRequest(f"{SYNTHETIC_ORIGIN}/asset.js", {}))
        self.assertTrue(other.aborted)
        self.assertEqual(other.fulfilled, [])

    async def test_wire_user_agent_must_agree_with_the_reported_one(self) -> None:
        wire = dict(_WIRE_HEADERS)
        wire["User-Agent"] = "Other/1.0"
        _, context, _, _ = _build(wire_headers=wire)
        with self.assertRaises(UnsupportedIdentityError):
            await observe_identity(context)

    async def test_probe_page_is_closed_even_when_navigation_fails(self) -> None:
        _, context, fake_browser, fake_context = _build()
        fake_browser.goto_error = RuntimeError("navigation failed")
        with self.assertRaises(RuntimeError):
            await observe_identity(context)
        self.assertTrue(fake_context.pages[0].closed)


class _ClientTests(IsolatedAsyncioTestCase):
    """Which client a context gets, and how identity, proxy and shutdown relate to it."""

    async def test_client_reuses_identity_per_generation_without_an_extra_probe(self) -> None:
        _, one, fake_browser, _ = _build()
        two = _native_context(_FakeContext(fake_browser), one.browser)
        manager = BrowserHttpClients()
        try:
            first = await manager.client(one, None)
            second = await manager.client(two, None)
        finally:
            await manager.aclose()
        self.assertEqual(fake_browser.pages_created, 1)
        self.assertIsNot(first, second)
        self.assertIs(first.identity, second.identity)
        self.assertIs(first._context, one)
        self.assertIs(second._context, two)

    async def test_ready_path_returns_the_same_client_without_probing_again(self) -> None:
        _, context, fake_browser, _ = _build()
        manager = BrowserHttpClients()
        try:
            first = await manager.client(context, None)
            again = await manager.client(context, None)
        finally:
            await manager.aclose()
        self.assertIs(first, again)
        self.assertEqual(fake_browser.pages_created, 1)

    async def test_a_context_client_keeps_the_proxy_it_was_created_with(self) -> None:
        _, context, _, _ = _build()
        manager = BrowserHttpClients()
        try:
            await manager.client(context, "http://proxy-a:8080")
            with self.assertRaises(HttpTransportError):
                await manager.client(context, "http://proxy-b:8080")
            with self.assertRaises(HttpTransportError):
                await manager.client(context, None)
        finally:
            await manager.aclose()

    async def test_unsupported_persona_is_cached_for_the_generation(self) -> None:
        _, one, fake_browser, _ = _build(reported=_reported(platform="Android", mobile=True))
        two = _native_context(_FakeContext(fake_browser), one.browser)
        manager = BrowserHttpClients()
        try:
            with self.assertRaises(UnsupportedIdentityError):
                await manager.client(one, None)
            with self.assertRaises(UnsupportedIdentityError):
                await manager.client(two, None)
        finally:
            await manager.aclose()
        self.assertEqual(fake_browser.pages_created, 1)

    async def test_transient_probe_failure_is_not_cached(self) -> None:
        _, context, fake_browser, fake_context = _build()
        fake_browser.goto_error = RuntimeError("flaky probe")
        manager = BrowserHttpClients()
        try:
            with self.assertRaises(RuntimeError):
                await manager.client(context, None)
            self.assertTrue(fake_context.pages[0].closed)
            fake_browser.goto_error = None
            client = await manager.client(context, None)
        finally:
            await manager.aclose()
        self.assertEqual(fake_browser.pages_created, 2)
        self.assertIs(client._context, context)

    async def test_cancelled_probe_closes_its_page_and_is_retried(self) -> None:
        _, context, fake_browser, fake_context = _build()
        fake_browser.goto_gate = asyncio.Event()
        manager = BrowserHttpClients()
        task = asyncio.ensure_future(manager.client(context, None))
        await asyncio.wait_for(fake_browser.goto_started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(fake_context.pages[0].closed)
        fake_browser.goto_gate = None
        try:
            client = await manager.client(context, None)
        finally:
            await manager.aclose()
        self.assertEqual(fake_browser.pages_created, 2)
        self.assertIs(client._context, context)

    async def test_a_closed_context_is_not_reopened(self) -> None:
        _, context, _, fake_context = _build()
        manager = BrowserHttpClients()
        first = await manager.client(context, None)
        fake_context.emit("close")
        await _settle(manager)
        self.assertTrue(first._closed)
        self.assertNotIn(context, manager._states)
        self.assertFalse(fake_context.listeners.get("close"))
        with self.assertRaises(HttpTransportError):
            await manager.client(context, None)
        await manager.aclose()

    async def test_browser_disconnect_retires_the_generation_and_its_clients(self) -> None:
        native_browser, one, fake_browser, _ = _build()
        two = _native_context(_FakeContext(fake_browser), native_browser)
        manager = BrowserHttpClients()
        first = await manager.client(one, None)
        second = await manager.client(two, None)
        fake_browser.emit("disconnected")
        await _settle(manager)
        self.assertTrue(first._closed)
        self.assertTrue(second._closed)
        self.assertFalse(fake_browser.listeners.get("disconnected"))
        with self.assertRaises(HttpTransportError):
            await manager.client(one, None)
        await manager.aclose()

    async def test_retirement_without_a_client_releases_state_and_listeners(self) -> None:
        _, context, _, fake_context = _build(reported=_reported(platform="Android", mobile=True))
        manager = BrowserHttpClients()
        with self.assertRaises(UnsupportedIdentityError):
            await manager.client(context, None)
        self.assertIn(context, manager._states)
        fake_context.emit("close")
        await _settle(manager)
        self.assertNotIn(context, manager._states)
        self.assertFalse(fake_context.listeners.get("close"))
        await manager.aclose()

    async def test_native_event_arguments_are_accepted(self) -> None:
        _, context, _, fake_context = _build()
        manager = BrowserHttpClients()
        client = await manager.client(context, None)
        fake_context.emit("close", "unexpected")
        await _settle(manager)
        self.assertTrue(client._closed)

    async def test_generation_retirement_attempts_every_client_before_raising(self) -> None:
        native_browser, one, fake_browser, _ = _build()
        two = _native_context(_FakeContext(fake_browser), native_browser)
        manager = BrowserHttpClients()
        first = await manager.client(one, None)
        second = await manager.client(two, None)
        original = first.aclose
        first.aclose = _fail_close
        fake_browser.emit("disconnected")
        await _settle(manager)
        self.assertTrue(second._closed)
        self.assertFalse(first._closed)
        self.assertIs(manager._states[one].client, first)
        first.aclose = original
        await manager.aclose()
        self.assertNotIn(one, manager._states)

    async def test_a_failed_close_retains_ownership_and_fences_new_clients(self) -> None:
        _, context, _, _ = _build()
        manager = BrowserHttpClients()
        client = await manager.client(context, None)
        original = client.aclose
        client.aclose = _fail_close
        with self.assertRaises(RuntimeError):
            await manager.close_context(context)
        self.assertIs(manager._states[context].client, client)
        with self.assertRaises(HttpTransportError):
            await manager.client(context, None)
        client.aclose = original
        await manager.close_context(context)
        self.assertNotIn(context, manager._states)
        await manager.aclose()

    async def test_shutdown_fences_new_clients_and_closes_owned_sessions(self) -> None:
        native_browser, one, fake_browser, _ = _build()
        two = _native_context(_FakeContext(fake_browser), native_browser)
        manager = BrowserHttpClients()
        first = await manager.client(one, None)
        second = await manager.client(two, None)
        await manager.aclose()
        self.assertTrue(first._closed)
        self.assertTrue(second._closed)
        with self.assertRaises(HttpTransportError):
            await manager.client(one, None)

    async def test_shutdown_attempts_every_client_before_raising(self) -> None:
        native_browser, one, fake_browser, _ = _build()
        two = _native_context(_FakeContext(fake_browser), native_browser)
        manager = BrowserHttpClients()
        first = await manager.client(one, None)
        second = await manager.client(two, None)
        original = first.aclose
        first.aclose = _fail_close
        with self.assertRaises(RuntimeError):
            await manager.aclose()
        self.assertTrue(second._closed)
        first.aclose = original
        await manager.close_context(one)

    async def test_shutdown_fences_a_pending_initialization(self) -> None:
        _, context, fake_browser, fake_context = _build()
        fake_browser.goto_gate = asyncio.Event()
        manager = BrowserHttpClients()
        task = asyncio.ensure_future(manager.client(context, None))
        await asyncio.wait_for(fake_browser.page_opened.wait(), 2)
        closing = asyncio.ensure_future(manager.aclose())
        await asyncio.sleep(0)
        fake_browser.goto_gate.set()
        with self.assertRaises(HttpTransportError):
            await task
        await closing
        self.assertTrue(fake_context.pages[0].closed)
