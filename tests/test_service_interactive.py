"""Tests for interactive browser tabs: the command surface and the backend lifetime.

A tab opened with ``browser.open`` outlives its command, is displayed on the shared X display
for a person to drive, and must share the egress's browser with ordinary fetches so a clearance
earned by clicking through a challenge is the one the fetches then use. The pool is exercised
with stub browsers and the service with an in-memory backend, so no process is launched.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.egress import DEFAULT_EGRESS_NAME
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import DEFAULT_INTERACTIVE_IDLE_SECONDS

_EGRESS_URL = "socks5://127.0.0.1:10001"
_OTHER_URL = "socks5://127.0.0.1:10002"
_PAGE = "<html><head><title>Example Domain</title></head><body>ok</body></html>"


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-test-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-test-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


class FakeBackend:
    """Backend double that keeps interactive tabs in memory."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self.requests: list[tuple[str | None, Any]] = []
        self.opened: list[Any] = []
        self.closed: list[str | None] = []
        self.listed_tabs: list[str | None] = []
        self.tabs: dict[str, Any] = {}
        self.active = 0
        self.max_active = 0
        self._delay = delay
        self._counter = 0

    def result(self) -> Any:
        from prowl.service.backend import FetchResult  # noqa: PLC0415

        return FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            response=_PAGE,
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def start(self) -> None:
        """No-op for the double."""

    async def close_session(self, session_id: str) -> None:
        """No-op for the double."""

    async def fetch(self, session_id: str | None, request: Any) -> Any:
        self.requests.append((session_id, request))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return self.result()
        finally:
            self.active -= 1

    async def open_interactive(self, request: Any) -> Any:
        from prowl.service.backend import InteractiveOpenResult, InteractiveTab  # noqa: PLC0415

        self.opened.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            self._counter += 1
            tab = InteractiveTab(
                tab_id=f"tab-{self._counter}",
                url=request.url,
                title="Example Domain",
                status_code=200,
                requested_url=request.url,
                egress=request.egress,
            )
            self.tabs[tab.tab_id] = tab
            return InteractiveOpenResult(tab=tab)
        finally:
            self.active -= 1

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        from prowl.service.backend import InteractiveTab  # noqa: PLC0415

        self.closed.append(tab_id)
        if tab_id is None:
            closed = sorted(self.tabs)
            self.tabs.clear()
            return closed
        _ = InteractiveTab
        return [tab_id] if self.tabs.pop(tab_id, None) is not None else []

    async def list_interactive(self, tab_id: str | None = None) -> list[Any]:
        self.listed_tabs.append(tab_id)
        return [self.tabs[key] for key in sorted(self.tabs)]

    async def aclose(self) -> None:
        """No-op for the double."""


def _service(**overrides: Any) -> tuple[Service, FakeBackend]:
    backend = FakeBackend()
    return Service(_config(**overrides), backend), backend


async def _open(service: Service, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"cmd": "browser.open", "url": "https://example.com/"}
    payload.update(fields)
    _status, body = await service.handle(payload)
    return body


class BrowserCommandParsingTests(TestCase):
    """The new commands are validated like the existing ones."""

    def test_the_interactive_commands_are_supported(self) -> None:
        from prowl.service.protocol import SUPPORTED_COMMANDS  # noqa: PLC0415

        self.assertLessEqual(
            {"browser.open", "browser.close", "browser.list"},
            SUPPORTED_COMMANDS,
        )

    def test_open_is_parsed_with_defaults(self) -> None:
        from prowl.service.protocol import DEFAULT_TIMEOUT_MS, BrowserOpenCommand, parse_request  # noqa: PLC0415

        command = parse_request({"cmd": "browser.open", "url": "https://example.com/"})
        self.assertIsInstance(command, BrowserOpenCommand)
        self.assertEqual(command.url, "https://example.com/")
        self.assertEqual(command.timeout_ms, DEFAULT_TIMEOUT_MS)
        self.assertIsNone(command.session)
        self.assertIsNone(command.proxy)

    def test_open_requires_a_url(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.open"})

    def test_open_rejects_a_non_http_url(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.open", "url": "file:///etc/passwd"})

    def test_open_rejects_unknown_fields(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.open", "url": "https://example.com/", "postData": "x"})

    def test_open_accepts_a_session_and_an_egress_name(self) -> None:
        from prowl.service.protocol import parse_request  # noqa: PLC0415

        command = parse_request(
            {
                "cmd": "browser.open",
                "url": "https://example.com/",
                "session": "reading",
                "proxy": {"name": "decodo"},
            },
        )
        self.assertEqual(command.session, "reading")
        self.assertIsNotNone(command.proxy)
        self.assertEqual(command.proxy.name, "decodo")

    def test_close_defaults_to_every_tab(self) -> None:
        from prowl.service.protocol import BrowserCloseCommand, parse_request  # noqa: PLC0415

        command = parse_request({"cmd": "browser.close"})
        self.assertIsInstance(command, BrowserCloseCommand)
        self.assertIsNone(command.tab)

    def test_close_accepts_a_tab_id(self) -> None:
        from prowl.service.protocol import parse_request  # noqa: PLC0415

        command = parse_request({"cmd": "browser.close", "tab": "tab-3"})
        self.assertEqual(command.tab, "tab-3")

    def test_close_rejects_an_unsafe_tab_id(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        for tab in ("", "   ", "../tab", "tab with space"):
            with self.assertRaises(ProtocolError, msg=tab):
                parse_request({"cmd": "browser.close", "tab": tab})

    def test_close_rejects_unknown_fields(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.close", "session": "reading"})

    def test_list_defaults_to_no_tab(self) -> None:
        from prowl.service.protocol import BrowserListCommand, ProtocolError, parse_request  # noqa: PLC0415

        command = parse_request({"cmd": "browser.list"})
        self.assertIsInstance(command, BrowserListCommand)
        self.assertIsNone(command.tab)
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.list", "url": "https://example.com/"})

    def test_list_accepts_a_tab_to_refresh_it(self) -> None:
        from prowl.service.protocol import BrowserListCommand, parse_request  # noqa: PLC0415

        command = parse_request({"cmd": "browser.list", "tab": "tab-7"})
        self.assertIsInstance(command, BrowserListCommand)
        self.assertEqual(command.tab, "tab-7")

    def test_list_rejects_an_unsafe_tab_id(self) -> None:
        from prowl.service.protocol import ProtocolError, parse_request  # noqa: PLC0415

        for tab in ("", "   ", "../tab", "tab with space"):
            with self.assertRaises(ProtocolError, msg=tab):
                parse_request({"cmd": "browser.list", "tab": tab})

    def test_tab_payload_never_exposes_the_egress(self) -> None:
        from prowl.service.backend import InteractiveTab  # noqa: PLC0415
        from prowl.service.protocol import tab_payload  # noqa: PLC0415

        tab = InteractiveTab(
            tab_id="tab-1",
            url="https://example.com/",
            title="Example",
            status_code=200,
            egress=DEFAULT_EGRESS_NAME,
        )
        payload = tab_payload(tab)
        self.assertEqual(sorted(payload), ["id", "status", "title", "url"])
        self.assertNotIn(_EGRESS_URL, str(payload))


class InteractiveCommandTests(IsolatedAsyncioTestCase):
    """The service drives interactive tabs through the backend."""

    async def test_open_reports_the_new_tab(self) -> None:
        service, backend = _service(egresses={"decodo": _EGRESS_URL})
        body = await _open(service, proxy={"name": "decodo"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["tab"]["id"], "tab-1")
        self.assertEqual(body["tab"]["url"], "https://example.com/")
        self.assertEqual(body["tab"]["status"], 200)
        self.assertEqual(backend.opened[0].egress, "decodo")

    async def test_open_leaves_the_tab_open_for_the_next_command(self) -> None:
        service, backend = _service()
        opened = await _open(service)
        self.assertEqual(backend.closed, [])
        listed = await service.handle({"cmd": "browser.list"})
        self.assertEqual([tab["id"] for tab in listed[1]["tabs"]], [opened["tab"]["id"]])

    async def test_list_reports_every_open_tab(self) -> None:
        service, _backend = _service()
        await _open(service)
        await _open(service, url="https://example.org/")
        _status, body = await service.handle({"cmd": "browser.list"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(body["tabs"]), 2)
        self.assertEqual(body["tabs"][1]["url"], "https://example.org/")

    async def test_list_without_a_tab_is_a_read_only_enumeration(self) -> None:
        service, backend = _service()
        await _open(service)
        _status, body = await service.handle({"cmd": "browser.list"})
        self.assertEqual(len(body["tabs"]), 1)
        self.assertIsNone(backend.listed_tabs[-1])

    async def test_list_passes_the_named_tab_to_the_backend(self) -> None:
        service, backend = _service()
        opened = await _open(service)
        _status, body = await service.handle({"cmd": "browser.list", "tab": opened["tab"]["id"]})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.listed_tabs[-1], opened["tab"]["id"])

    async def test_close_named_tab_leaves_the_other_open(self) -> None:
        service, _backend = _service()
        first = await _open(service)
        await _open(service, url="https://example.org/")
        _status, body = await service.handle({"cmd": "browser.close", "tab": first["tab"]["id"]})
        self.assertEqual(body["closed"], [first["tab"]["id"]])
        _status, listed = await service.handle({"cmd": "browser.list"})
        self.assertEqual(len(listed["tabs"]), 1)

    async def test_close_without_a_tab_closes_them_all(self) -> None:
        service, _backend = _service()
        await _open(service)
        await _open(service, url="https://example.org/")
        _status, body = await service.handle({"cmd": "browser.close"})
        self.assertEqual(len(body["closed"]), 2)
        _status, listed = await service.handle({"cmd": "browser.list"})
        self.assertEqual(listed["tabs"], [])

    async def test_closing_an_unknown_tab_is_idempotent(self) -> None:
        service, _backend = _service()
        _status, body = await service.handle({"cmd": "browser.close", "tab": "tab-99"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["closed"], [])

    async def test_open_binds_a_session_to_its_egress(self) -> None:
        service, _backend = _service(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL})
        first = await _open(service, session="reading", proxy={"name": "decodo"})
        self.assertEqual(first["status"], "ok")
        _status, disagreeing = await service.handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "session": "reading",
                "proxy": {"name": "warp"},
            },
        )
        self.assertEqual(disagreeing["status"], "error")
        self.assertIn("egress", disagreeing["message"])

    async def test_an_unknown_egress_is_rejected_without_opening_a_tab(self) -> None:
        service, backend = _service(egresses={"decodo": _EGRESS_URL})
        body = await _open(service, proxy={"name": "nope"})
        self.assertEqual(body["status"], "error")
        self.assertNotIn(_EGRESS_URL, body["message"])
        self.assertEqual(backend.opened, [])

    async def test_an_open_does_not_disturb_an_ordinary_fetch(self) -> None:
        service, backend = _service(egresses={"decodo": _EGRESS_URL})
        await _open(service, proxy={"name": "decodo"})
        _status, fetched = await service.handle(
            {"cmd": "request.get", "url": "https://example.com/", "proxy": {"name": "decodo"}},
        )
        self.assertEqual(fetched["status"], "ok")
        # The fetch ran, and the interactive tab is still open.
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(sorted(backend.tabs), ["tab-1"])

    async def test_an_ordinary_fetch_does_not_close_an_open_tab(self) -> None:
        service, backend = _service()
        opened = await _open(service)
        await service.handle({"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(backend.closed, [])
        self.assertIn(opened["tab"]["id"], backend.tabs)

    async def test_anonymous_opens_on_one_egress_serialize(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(egresses={"decodo": _EGRESS_URL}, max_concurrency=2), backend)
        await asyncio.gather(
            _open(service, proxy={"name": "decodo"}),
            _open(service, proxy={"name": "decodo"}),
        )
        self.assertEqual(backend.max_active, 1)

    async def test_anonymous_opens_on_different_egresses_overlap(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(
            _config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}, max_concurrency=2),
            backend,
        )
        await asyncio.gather(
            _open(service, proxy={"name": "decodo"}),
            _open(service, proxy={"name": "warp"}),
        )
        self.assertEqual(backend.max_active, 2)


class InteractiveConfigTests(TestCase):
    """The idle delay is read from the environment."""

    def test_absent_idle_delay_disables_it(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ServiceConfig.from_env().interactive_idle_seconds, DEFAULT_INTERACTIVE_IDLE_SECONDS)

    def test_idle_delay_is_read(self) -> None:
        with patch.dict(os.environ, {"PROWL_INTERACTIVE_IDLE_SECONDS": "42.5"}, clear=True):
            self.assertEqual(ServiceConfig.from_env().interactive_idle_seconds, 42.5)

    def test_zero_idle_delay_keeps_tabs_open(self) -> None:
        with patch.dict(os.environ, {"PROWL_INTERACTIVE_IDLE_SECONDS": "0"}, clear=True):
            self.assertEqual(ServiceConfig.from_env().interactive_idle_seconds, 0.0)

    def test_a_negative_idle_delay_is_clamped(self) -> None:
        with patch.dict(os.environ, {"PROWL_INTERACTIVE_IDLE_SECONDS": "-9"}, clear=True):
            self.assertEqual(ServiceConfig.from_env().interactive_idle_seconds, 0.0)

    def test_a_non_numeric_idle_delay_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"PROWL_INTERACTIVE_IDLE_SECONDS": "later"}, clear=True),
            self.assertRaises(ValueError) as ctx,
        ):
            ServiceConfig.from_env()
        self.assertIn("PROWL_INTERACTIVE_IDLE_SECONDS", str(ctx.exception))


class StubGroup:
    """Tab group double that records whether it was closed."""

    class _StubPage:
        url = "https://example.com/from-page"

    def __init__(self) -> None:
        self.quit_calls = 0
        self.ppage = StubGroup._StubPage()

    @property
    async def ptab(self) -> Any:
        """Return a parent tab double."""

        class _Tab:
            async def set_cookies(self, cookies: Any) -> None:
                """Accept cookies, as the real tab does."""

        return _Tab()

    def pd(self) -> Any:
        """Return a cookie reader."""

        class _Pd:
            async def get_cookies(self) -> list[Any]:
                """Report no cookies."""
                return []

        return _Pd()

    async def quit(self) -> None:
        """Record the close."""
        self.quit_calls += 1


class _StubSite:
    """Site double returning a fixed page."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
        from prowl.browser.site import Source  # noqa: PLC0415

        if self._fail:
            msg = "navigation failed"
            raise RuntimeError(msg)
        return Source(
            source=_PAGE,
            status_code=200,
            headers={},
            user_agent="Mozilla/5.0 (Test)",
            url=url,
        )


class InteractiveBackendTests(IsolatedAsyncioTestCase):
    """The backend keeps a tab open on the egress browser and eventually releases it."""

    def _backend(  # noqa: PLR0913
        self,
        *,
        idle_seconds: float = 0.0,
        egress_idle_seconds: float = 60.0,
        created: list[str] | None = None,
        shutdown_log: list[str] | None = None,
        max_concurrency: int = 0,
        steal_least_recent: bool = False,
        start_error: BaseException | None = None,
        create_error: BaseException | None = None,
        create_entered: asyncio.Event | None = None,
        create_gate: asyncio.Event | None = None,
    ) -> tuple[Any, list[StubGroup]]:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        groups: list[StubGroup] = []

        def _create_group() -> StubGroup:
            group = StubGroup()
            groups.append(group)
            return group

        def _create_browser(**kwargs: Any) -> type[Any]:
            name = str(kwargs["name"])
            if created is not None:
                created.append(name)

            class FakeEgressBrowser:
                @classmethod
                async def start(cls) -> None:
                    if start_error is not None:
                        raise start_error

                @classmethod
                async def create(cls) -> StubGroup:
                    if create_entered is not None:
                        create_entered.set()
                    if create_gate is not None:
                        await create_gate.wait()
                    if create_error is not None:
                        raise create_error
                    return _create_group()

                @classmethod
                async def shutdown(cls) -> None:
                    if shutdown_log is not None:
                        shutdown_log.append(name)

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(
            BrowserConfig(profile_dir="/tmp/p", profile_archive="/tmp/p.zip"),  # noqa: S108
            egresses={"decodo": _EGRESS_URL},
            egress_idle_seconds=egress_idle_seconds,
            interactive_idle_seconds=idle_seconds,
            steal_least_recent=steal_least_recent,
            max_concurrency=max_concurrency,
        )

        class FakeDefaultBrowser:
            """Double for the process-wide browser, so no real Chrome is launched."""

            @classmethod
            async def start(cls) -> None:
                """No-op."""

            @classmethod
            async def create(cls) -> StubGroup:
                return _create_group()

            @classmethod
            async def shutdown(cls) -> None:
                """No-op."""

        patcher = patch("prowl.browser.egress.create_egress_browser", side_effect=_create_browser)
        patcher.start()
        self.addCleanup(patcher.stop)
        default_patcher = patch.object(backend_module, "Browser", FakeDefaultBrowser)
        default_patcher.start()
        self.addCleanup(default_patcher.stop)
        return backend, groups

    async def test_open_returns_the_page_title_and_leaves_the_group_open(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend()
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/"))).tab
        self.assertEqual(tab.title, "Example Domain")
        self.assertEqual(tab.status_code, 200)
        self.assertEqual(groups[0].quit_calls, 0)
        self.assertEqual([entry.tab_id for entry in await backend.list_interactive()], [tab.tab_id])
        await backend.aclose()

    async def test_an_interactive_tab_and_a_fetch_share_one_browser(self) -> None:
        """The invariant that makes a clearance earned in the tab usable by a fetch."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import FetchRequest, InteractiveRequest  # noqa: PLC0415

        created: list[str] = []
        backend, _groups = self._backend(created=created)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))
        # One browser was created for the egress, so the tab and the fetch share a profile.
        self.assertEqual(created, ["decodo"])
        await backend.aclose()

    async def test_close_quits_the_group_and_releases_the_egress(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, groups = self._backend(egress_idle_seconds=0.01, shutdown_log=shutdown_log)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))).tab
        closed = await backend.close_interactive(tab.tab_id)
        self.assertEqual(closed, [tab.tab_id])
        self.assertEqual(groups[0].quit_calls, 1)
        # Releasing the claim lets the idle pool shut the browser down again.
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])

    async def test_closing_an_unknown_tab_reports_nothing(self) -> None:
        backend, _groups = self._backend()
        self.assertEqual(await backend.close_interactive("tab-404"), [])
        await backend.aclose()

    async def test_a_hold_on_the_egress_survives_an_idle_pool_delay(self) -> None:
        """An open tab must not be closed by the egress idling out."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, groups = self._backend(egress_idle_seconds=0.01, shutdown_log=shutdown_log)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))).tab
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, [])
        self.assertEqual(groups[0].quit_calls, 0)
        self.assertEqual([entry.tab_id for entry in await backend.list_interactive()], [tab.tab_id])
        await backend.aclose()

    async def test_the_idle_timeout_closes_a_forgotten_tab(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, groups = self._backend(idle_seconds=0.01, egress_idle_seconds=0.01, shutdown_log=shutdown_log)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
        await asyncio.sleep(0.15)
        self.assertEqual(await backend.list_interactive(), [])
        self.assertEqual(groups[0].quit_calls, 1)
        # The idle close has to release the egress too, or the browser is held forever.
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_polling_the_named_tab_keeps_it_alive(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, _groups = self._backend(idle_seconds=0.05)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/"))).tab
        # Poll the tab the caller is displaying, as a WebView does, so its countdown restarts.
        for _ in range(4):
            await asyncio.sleep(0.02)
            await backend.list_interactive(tab.tab_id)
        self.assertEqual(len(await backend.list_interactive()), 1)
        await backend.aclose()

    async def test_listing_without_a_tab_does_not_keep_it_alive(self) -> None:
        """A read-only enumeration must not refresh any countdown, so orphans still reap."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend(idle_seconds=0.05)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        for _ in range(4):
            await asyncio.sleep(0.02)
            await backend.list_interactive()
        self.assertEqual(await backend.list_interactive(), [])
        self.assertEqual(groups[0].quit_calls, 1)
        await backend.aclose()

    async def test_polling_one_tab_does_not_keep_another_alive(self) -> None:
        """The fix for one displayed view keeping every forgotten tab open forever."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend(idle_seconds=0.06)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            watched = (await backend.open_interactive(InteractiveRequest(url="https://watched.example/"))).tab
            await backend.open_interactive(InteractiveRequest(url="https://forgotten.example/"))
            for _ in range(6):
                await asyncio.sleep(0.02)
                await backend.list_interactive(watched.tab_id)
        remaining = [entry.tab_id for entry in await backend.list_interactive()]
        self.assertEqual(remaining, [watched.tab_id])
        self.assertEqual(groups[1].quit_calls, 1)
        await backend.aclose()

    async def test_polling_a_tab_makes_it_the_one_protected_from_takeover(self) -> None:
        """Touching one tab must move it to the most recent position the LRU steal protects."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend(max_concurrency=1, steal_least_recent=True)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            watched = (await backend.open_interactive(InteractiveRequest(url="https://watched.example/"))).tab
            await asyncio.sleep(0.01)
            forgotten = (await backend.open_interactive(InteractiveRequest(url="https://forgotten.example/"))).tab
            await asyncio.sleep(0.01)
            await backend.list_interactive(watched.tab_id)
            # A saturated caller opens a third tab, which takes over the least recently used one.
            await backend.open_interactive(InteractiveRequest(url="https://new.example/"))
        remaining = [entry.tab_id for entry in await backend.list_interactive()]
        self.assertIn(watched.tab_id, remaining)
        self.assertNotIn(forgotten.tab_id, remaining)
        self.assertEqual(groups[1].quit_calls, 1)
        self.assertEqual(groups[0].quit_calls, 0)
        await backend.aclose()

    async def test_idling_a_tab_does_not_touch_a_fetch_group(self) -> None:
        """A fetch in flight keeps its own group while an interactive tab idles out."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import FetchRequest, InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend(idle_seconds=0.01)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/"))).tab
            await backend.fetch(None, FetchRequest(url="https://example.com/"))
        await asyncio.sleep(0.15)
        self.assertEqual(await backend.list_interactive(), [])
        fetch_group = next(group for group in groups if group is not groups[0])
        # The fetch closed its own group; the interactive group was closed by the idle timer.
        self.assertEqual(fetch_group.quit_calls, 1)
        self.assertEqual(len(groups), 2)
        _ = tab
        await backend.aclose()

    async def test_the_list_reports_the_live_location(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, _groups = self._backend()
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        tabs = await backend.list_interactive()
        self.assertEqual(tabs[0].url, "https://example.com/from-page")
        await backend.aclose()

    async def test_a_failed_navigation_closes_the_group_and_releases_the_egress(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, groups = self._backend(egress_idle_seconds=0.01, shutdown_log=shutdown_log)
        with (
            patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite(fail=True)),
            self.assertRaises(RuntimeError),
        ):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
        self.assertEqual(groups[0].quit_calls, 1)
        self.assertEqual(await backend.list_interactive(), [])
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_closing_everything_releases_the_egress(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend()
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            await backend.open_interactive(InteractiveRequest(url="https://example.org/"))
        closed = await backend.close_interactive(None)
        self.assertEqual(len(closed), 2)
        self.assertEqual([group.quit_calls for group in groups], [1, 1])
        await backend.aclose()

    async def test_aclose_closes_open_tabs(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend()
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        await backend.aclose()
        self.assertEqual(groups[0].quit_calls, 1)
        self.assertEqual(await backend.list_interactive(), [])

    async def test_a_lone_open_tab_is_never_stolen_at_saturation(self) -> None:
        """A saturated fetch leaves the only tab alone: it is the one a person is looking at."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import FetchRequest, InteractiveRequest  # noqa: PLC0415

        backend, groups = self._backend(max_concurrency=1, steal_least_recent=True)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite()):
            tab = (await backend.open_interactive(InteractiveRequest(url="https://example.com/"))).tab
            await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual(groups[0].quit_calls, 0)
        self.assertEqual([entry.tab_id for entry in await backend.list_interactive()], [tab.tab_id])
        await backend.aclose()

    async def test_no_open_tabs_with_saturated_in_flight_does_not_crash(self) -> None:
        """A fetch must not crash when every slot is busy and there is no tab to take over."""
        from prowl.service import backend as backend_module  # noqa: PLC0415
        from prowl.service.backend import FetchRequest  # noqa: PLC0415

        entered = asyncio.Event()
        release = asyncio.Event()

        class GatedSite:
            def __init__(self) -> None:
                self.calls = 0

            async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
                from prowl.browser.site import Source  # noqa: PLC0415

                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    await release.wait()
                return Source(source=_PAGE, status_code=200, headers={}, user_agent="Mozilla/5.0 (Test)", url=url)

        site = GatedSite()
        backend, _groups = self._backend(max_concurrency=1, steal_least_recent=True)
        with patch.object(backend_module, "resolve_site", lambda _group, _url: site):
            first = asyncio.create_task(backend.fetch(None, FetchRequest(url="https://example.com/one")))
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            # One fetch is already in flight with no tab open, which is the saturated state the
            # takeover used to raise IndexError on.
            second = await backend.fetch(None, FetchRequest(url="https://example.com/two"))
            release.set()
            await first
        self.assertEqual(second.status_code, 200)
        await backend.aclose()

    async def test_open_releases_the_egress_claim_when_start_fails(self) -> None:
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, _groups = self._backend(
            egress_idle_seconds=0.01,
            shutdown_log=shutdown_log,
            start_error=RuntimeError("start failed"),
        )
        with self.assertRaises(RuntimeError):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
        # The claim taken before the failure has to come back, or the egress never idles out.
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_open_releases_the_egress_claim_when_create_fails(self) -> None:
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        backend, _groups = self._backend(
            egress_idle_seconds=0.01,
            shutdown_log=shutdown_log,
            create_error=RuntimeError("create failed"),
        )
        with self.assertRaises(RuntimeError):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_open_releases_the_egress_claim_when_create_is_cancelled(self) -> None:
        from prowl.service.backend import InteractiveRequest  # noqa: PLC0415

        shutdown_log: list[str] = []
        entered = asyncio.Event()
        gate = asyncio.Event()
        backend, _groups = self._backend(
            egress_idle_seconds=0.01,
            shutdown_log=shutdown_log,
            create_entered=entered,
            create_gate=gate,
        )
        task = asyncio.create_task(
            backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo")),
        )
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()
