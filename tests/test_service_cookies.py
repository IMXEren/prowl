"""Cookie access and tab reuse: the command surface, the backend, and the lifetime rules.

An external WebView is only useful to the app driving it if the app can carry the session a
person establishes there into its own requests, so the profile's cookies have to be readable.
Seeding works the other way, through ``browser.open``, which also reuses a tab that already
shows the requested url: a caller that never learned the first tab's id, because its own request
timed out, must not leave a tab nobody can close.

The service is exercised with an in-memory backend and the backend with stub browsers, so no
process is launched.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.egress import DEFAULT_EGRESS_NAME
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import (
    CookieQuery,
    FetchRequest,
    FetchResult,
    InteractiveOpenResult,
    InteractiveRequest,
    InteractiveTab,
    _normalize_cookies,
)
from prowl.service.protocol import (
    CMD_COOKIES_LIST,
    SUPPORTED_COMMANDS,
    CookiesListCommand,
    ProtocolError,
    cookies_payload,
    parse_request,
)

_EGRESS_URL = "socks5://127.0.0.1:10001"
_OTHER_URL = "socks5://127.0.0.1:10002"
_PAGE = "<html><head><title>Example Domain</title></head><body>ok</body></html>"

#: Every field a cookie carries in a fetch reply, which is the shape this command reuses.
_COOKIE_FIELDS = ("name", "value", "domain", "path", "expires", "secure", "httpOnly", "sameSite")

#: A profile holding a login cookie, a cookie for another site, and an expired one. The login
#: cookie carries the browser's session-cookie expiry, which must not read as an expiry in the
#: past, or exactly the cookie this feature exists for would be filtered out.
_PROFILE_COOKIES: list[dict[str, Any]] = [
    {
        "name": "session",
        "value": "abc",
        "domain": ".example.com",
        "path": "/",
        "expires": -1,
        "secure": True,
        "httpOnly": True,
        "sameSite": "None",
    },
    {
        "name": "theme",
        "value": "dark",
        "domain": "other.example",
        "path": "/",
        "expires": time.time() + 3600,
        "secure": False,
        "httpOnly": False,
    },
    {
        "name": "legacy",
        "value": "old",
        "domain": ".example.com",
        "path": "/",
        "expires": 1,
        "secure": False,
        "httpOnly": False,
    },
]


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
    """Backend double holding cookies in memory and one tab per egress and url."""

    def __init__(self, cookies: list[dict[str, Any]] | None = None) -> None:
        self.cookies = list(cookies or [])
        self.queries: list[CookieQuery] = []
        self.opened: list[InteractiveRequest] = []
        self.tabs: dict[str, InteractiveTab] = {}
        self._counter = 0

    async def start(self) -> None:
        """No-op for the double."""

    async def close_session(self, session_id: str) -> None:
        """No-op for the double."""

    async def fetch(self, session_id: str | None, request: Any) -> Any:
        """Report a fixed page, unused by these tests."""
        return FetchResult(
            url=request.url,
            status_code=200,
            headers={"content-type": "text/html"},
            response=_PAGE,
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        """Reuse the tab already showing this url on this egress, as the real backend does."""
        self.opened.append(request)
        if not request.new_tab:
            for tab in self.tabs.values():
                if tab.egress == request.egress and request.url in (tab.requested_url, tab.url):
                    return InteractiveOpenResult(tab=tab, reused=True)
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

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        """Close one tab, or every one."""
        if tab_id is None:
            closed = sorted(self.tabs)
            self.tabs.clear()
            return closed
        return [tab_id] if self.tabs.pop(tab_id, None) is not None else []

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        """Return the open tabs."""
        return [self.tabs[key] for key in sorted(self.tabs)]

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        """Report the cookies this double holds."""
        self.queries.append(query)
        return list(self.cookies)

    async def aclose(self) -> None:
        """No-op for the double."""


def _service(cookies: list[dict[str, Any]] | None = None, **overrides: Any) -> tuple[Service, FakeBackend]:
    backend = FakeBackend(cookies)
    return Service(_config(**overrides), backend), backend


class CookieCommandParsingTests(TestCase):
    """The new command and field are validated like the existing ones."""

    def test_cookies_list_is_a_supported_command(self) -> None:
        self.assertIn(CMD_COOKIES_LIST, SUPPORTED_COMMANDS)

    def test_cookies_list_is_parsed_with_defaults(self) -> None:
        command = parse_request({"cmd": "cookies.list"})
        self.assertIsInstance(command, CookiesListCommand)
        self.assertIsNone(command.url)
        self.assertIsNone(command.proxy)

    def test_cookies_list_accepts_a_url_and_an_egress_name(self) -> None:
        command = parse_request(
            {"cmd": "cookies.list", "url": "https://example.com/path", "proxy": {"name": "decodo"}},
        )
        self.assertEqual(command.url, "https://example.com/path")
        self.assertIsNotNone(command.proxy)
        self.assertEqual(command.proxy.name, "decodo")

    def test_cookies_list_rejects_a_non_http_url(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "cookies.list", "url": "file:///etc/passwd"})

    def test_cookies_list_rejects_an_unknown_field(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "cookies.list", "tab": "tab-1"})

    def test_open_accepts_a_cookie_list(self) -> None:
        cookies = [{"name": "session", "value": "abc", "domain": ".example.com", "path": "/"}]
        command = parse_request(
            {"cmd": "browser.open", "url": "https://example.com/", "cookies": cookies},
        )
        self.assertEqual(command.cookies, cookies)
        self.assertFalse(command.new_tab)

    def test_open_defaults_to_reusing_a_tab(self) -> None:
        command = parse_request({"cmd": "browser.open", "url": "https://example.com/"})
        self.assertEqual(command.cookies, [])
        self.assertFalse(command.new_tab)

    def test_open_can_force_a_new_tab(self) -> None:
        command = parse_request({"cmd": "browser.open", "url": "https://example.com/", "newTab": True})
        self.assertTrue(command.new_tab)

    def test_open_rejects_a_non_boolean_new_tab(self) -> None:
        with self.assertRaises(ProtocolError):
            parse_request({"cmd": "browser.open", "url": "https://example.com/", "newTab": "yes"})

    def test_open_rejects_a_malformed_cookie_list(self) -> None:
        bad_payloads = [
            "session=abc",
            [{"name": "x"}],
            [{"value": "y"}],
            ["session=abc"],
            [{"name": 1, "value": "y"}],
        ]
        for cookies in bad_payloads:
            with self.assertRaises(ProtocolError, msg=cookies):
                parse_request({"cmd": "browser.open", "url": "https://example.com/", "cookies": cookies})

    def test_the_cookie_reply_uses_the_field_names_a_fetch_reports(self) -> None:
        payload = cookies_payload([dict(_PROFILE_COOKIES[0])])
        self.assertEqual(sorted(payload), ["cookies"])
        self.assertEqual(sorted(payload["cookies"][0]), sorted(_COOKIE_FIELDS))


class CookieCommandTests(IsolatedAsyncioTestCase):
    """The service reports the profile's cookies without holding a tab."""

    async def test_list_reports_the_profile_cookies(self) -> None:
        service, _backend = _service(_PROFILE_COOKIES)
        status, body = await service.handle({"cmd": "cookies.list"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual([cookie["name"] for cookie in body["cookies"]], ["session", "theme", "legacy"])

    async def test_list_forwards_the_url_to_the_backend(self) -> None:
        service, backend = _service(_PROFILE_COOKIES)
        await service.handle({"cmd": "cookies.list", "url": "https://sub.example.com/path"})
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual(backend.queries[0].url, "https://sub.example.com/path")
        self.assertEqual(backend.queries[0].egress, DEFAULT_EGRESS_NAME)

    async def test_list_forwards_the_egress_name(self) -> None:
        service, backend = _service(egresses={"decodo": _EGRESS_URL})
        await service.handle({"cmd": "cookies.list", "proxy": {"name": "decodo"}})
        self.assertEqual(backend.queries[0].egress, "decodo")

    async def test_an_empty_profile_reports_nothing(self) -> None:
        service, _backend = _service()
        _status, body = await service.handle({"cmd": "cookies.list"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["cookies"], [])

    async def test_an_unknown_egress_is_rejected_without_reading_cookies(self) -> None:
        service, backend = _service(egresses={"decodo": _EGRESS_URL})
        _status, body = await service.handle({"cmd": "cookies.list", "proxy": {"name": "nope"}})
        self.assertEqual(body["status"], "error")
        self.assertNotIn(_EGRESS_URL, body["message"])
        self.assertEqual(backend.queries, [])

    async def test_the_reply_never_exposes_the_egress_url(self) -> None:
        service, _backend = _service(_PROFILE_COOKIES, egresses={"decodo": _EGRESS_URL})
        _status, body = await service.handle({"cmd": "cookies.list", "proxy": {"name": "decodo"}})
        self.assertNotIn(_EGRESS_URL, str(body))

    async def test_list_does_not_open_a_tab(self) -> None:
        service, backend = _service(_PROFILE_COOKIES)
        await service.handle({"cmd": "cookies.list"})
        self.assertEqual(backend.opened, [])
        self.assertEqual(backend.tabs, {})

    async def test_open_reports_a_reused_tab(self) -> None:
        service, _backend = _service()
        first_status, first = await service.handle({"cmd": "browser.open", "url": "https://example.com/"})
        self.assertEqual(first_status, 200)
        self.assertFalse(first["reused"])
        self.assertEqual(first["message"], "Tab opened")

        _status, second = await service.handle({"cmd": "browser.open", "url": "https://example.com/"})
        self.assertTrue(second["reused"])
        self.assertEqual(second["message"], "Tab reused")
        self.assertEqual(second["tab"]["id"], first["tab"]["id"])

        _status, listed = await service.handle({"cmd": "browser.list"})
        self.assertEqual(len(listed["tabs"]), 1)

    async def test_open_forwards_the_cookies_and_the_new_tab_flag(self) -> None:
        service, backend = _service()
        await service.handle(
            {
                "cmd": "browser.open",
                "url": "https://example.com/",
                "cookies": [{"name": "session", "value": "abc", "domain": ".example.com"}],
                "newTab": True,
            },
        )
        self.assertEqual(backend.opened[0].cookies, [{"name": "session", "value": "abc", "domain": ".example.com"}])
        self.assertTrue(backend.opened[0].new_tab)


class _CookieReader:
    """Stands in for the shared pydoll connection's cookie reads."""

    def __init__(self, cookies: list[dict[str, Any]]) -> None:
        self._cookies = cookies
        self.reads = 0

    async def get_cookies(self) -> list[dict[str, Any]]:
        """Report the cookies this browser holds."""
        self.reads += 1
        return list(self._cookies)


class _StubTab:
    """Parent tab double that records an installed cookie set."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.cookies: list[dict[str, Any]] | None = None

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        """Record the cookies a caller seeded."""
        self.events.append("cookies")
        self.cookies = list(cookies)


class StubGroup:
    """Tab group double recording cookie installs, closes, and the order of both."""

    class _StubPage:
        url = "https://example.com/from-page"

    def __init__(self, reader: _CookieReader, events: list[str]) -> None:
        self.quit_calls = 0
        self.ppage = StubGroup._StubPage()
        self.events = events
        self.reader = reader
        self.tab = _StubTab(events)

    @property
    async def ptab(self) -> Any:
        """Return the parent tab double."""
        return self.tab

    def pd(self) -> Any:
        """Return the cookie reader."""
        return self.reader

    async def quit(self) -> None:
        """Record the close."""
        self.events.append("quit")
        self.quit_calls += 1


class _StubSite:
    """Site double returning a fixed page and recording the navigation."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
        from prowl.browser.site import Source  # noqa: PLC0415

        self.events.append("navigate")
        return Source(
            source=_PAGE,
            status_code=200,
            headers={},
            user_agent="Mozilla/5.0 (Test)",
            url=url,
        )


class _BackendFixture(IsolatedAsyncioTestCase):
    """Shared construction of a backend whose browsers are stubs."""

    def _backend(  # noqa: PLR0913
        self,
        *,
        cookies: list[dict[str, Any]] | None = None,
        idle_seconds: float = 0.0,
        egress_idle_seconds: float = 60.0,
        created: list[str] | None = None,
        shutdown_log: list[str] | None = None,
        start_error: BaseException | None = None,
    ) -> tuple[Any, list[StubGroup], _CookieReader, list[str]]:
        """Build a backend with stub browsers, returning it and everything it recorded."""
        from prowl.service import backend as backend_module  # noqa: PLC0415

        events: list[str] = []
        reader = _CookieReader(list(cookies or []))
        groups: list[StubGroup] = []

        def _create_group() -> StubGroup:
            group = StubGroup(reader, events)
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
                    return _create_group()

                @classmethod
                async def shutdown(cls) -> None:
                    if shutdown_log is not None:
                        shutdown_log.append(name)

                @classmethod
                def pd(cls) -> Any:
                    return reader

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(
            BrowserConfig(profile_dir="/tmp/p", profile_archive="/tmp/p.zip"),  # noqa: S108
            egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL},
            egress_idle_seconds=egress_idle_seconds,
            interactive_idle_seconds=idle_seconds,
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
                if shutdown_log is not None:
                    shutdown_log.append(DEFAULT_EGRESS_NAME)

            @classmethod
            def pd(cls) -> Any:
                return reader

        patcher = patch("prowl.browser.egress.create_egress_browser", side_effect=_create_browser)
        patcher.start()
        self.addCleanup(patcher.stop)
        default_patcher = patch.object(backend_module, "Browser", FakeDefaultBrowser)
        default_patcher.start()
        self.addCleanup(default_patcher.stop)
        return backend, groups, reader, events

    def _patch_site(self, events: list[str]) -> Any:
        """Patch site resolution so navigation is recorded rather than performed."""
        from prowl.service import backend as backend_module  # noqa: PLC0415

        return patch.object(backend_module, "resolve_site", lambda _group, _url: _StubSite(events))


class CookieBackendTests(_BackendFixture):
    """The backend reads a profile's cookies and applies them where they belong."""

    async def test_every_cookie_is_reported_when_no_url_is_given(self) -> None:
        backend, _groups, _reader, _events = self._backend(cookies=_PROFILE_COOKIES)
        listed = await backend.list_cookies(CookieQuery())
        self.assertEqual([cookie["name"] for cookie in listed], ["session", "theme", "legacy"])
        await backend.aclose()

    async def test_the_shape_matches_what_a_fetch_reports(self) -> None:
        backend, _groups, _reader, _events = self._backend(cookies=_PROFILE_COOKIES)
        listed = await backend.list_cookies(CookieQuery())
        self.assertEqual(listed, _normalize_cookies(_PROFILE_COOKIES))
        self.assertEqual(sorted(listed[0]), sorted(_COOKIE_FIELDS))
        await backend.aclose()

    async def test_a_url_scopes_the_reply_to_the_cookies_it_would_receive(self) -> None:
        backend, _groups, _reader, _events = self._backend(cookies=_PROFILE_COOKIES)
        # The secure session cookie reaches the https subdomain, while the cookie for another
        # site and the expired one are not sent there at all.
        listed = await backend.list_cookies(CookieQuery(url="https://sub.example.com/path"))
        self.assertEqual([cookie["name"] for cookie in listed], ["session"])
        await backend.aclose()

    async def test_a_secure_cookie_is_not_reported_for_http(self) -> None:
        backend, _groups, _reader, _events = self._backend(cookies=_PROFILE_COOKIES)
        listed = await backend.list_cookies(CookieQuery(url="http://sub.example.com/"))
        self.assertEqual(listed, [])
        await backend.aclose()

    async def test_a_cookie_path_is_respected(self) -> None:
        cookies = [{"name": "admin", "value": "1", "domain": "example.com", "path": "/admin"}]
        backend, _groups, _reader, _events = self._backend(cookies=cookies)
        self.assertEqual(await backend.list_cookies(CookieQuery(url="https://example.com/admin/x")), cookies)
        self.assertEqual(await backend.list_cookies(CookieQuery(url="https://example.com/other")), [])
        await backend.aclose()

    async def test_incomplete_cookies_are_not_reported(self) -> None:
        cookies = [{"name": "novalue", "domain": "example.com"}, {"value": "noname", "domain": "example.com"}]
        backend, _groups, _reader, _events = self._backend(cookies=cookies)
        self.assertEqual(await backend.list_cookies(CookieQuery()), [])
        await backend.aclose()

    async def test_a_cookie_without_a_domain_is_not_attributed_to_a_url(self) -> None:
        cookies = [{"name": "wild", "value": "1"}]
        backend, _groups, _reader, _events = self._backend(cookies=cookies)
        self.assertEqual(await backend.list_cookies(CookieQuery(url="https://example.com/")), [])
        await backend.aclose()

    async def test_listing_reads_the_browser(self) -> None:
        backend, _groups, reader, _events = self._backend(cookies=_PROFILE_COOKIES)
        await backend.list_cookies(CookieQuery())
        self.assertEqual(reader.reads, 1)
        await backend.aclose()

    async def test_listing_does_not_disturb_an_open_tab(self) -> None:
        backend, groups, _reader, events = self._backend(cookies=_PROFILE_COOKIES)
        with self._patch_site(events):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        await backend.list_cookies(CookieQuery(url="https://example.com/"))
        self.assertEqual(groups[0].quit_calls, 0)
        self.assertEqual(len(await backend.list_interactive()), 1)
        await backend.aclose()

    async def test_listing_releases_the_egress_claim(self) -> None:
        shutdown_log: list[str] = []
        backend, _groups, _reader, _events = self._backend(
            cookies=_PROFILE_COOKIES,
            egress_idle_seconds=0.01,
            shutdown_log=shutdown_log,
        )
        await backend.list_cookies(CookieQuery(egress="decodo"))
        # Releasing the claim lets the idle pool shut the egress browser down again.
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_a_failed_start_releases_the_egress_claim(self) -> None:
        """A start that fails before the read still owes the egress claim back."""
        shutdown_log: list[str] = []
        backend, _groups, _reader, _events = self._backend(
            cookies=_PROFILE_COOKIES,
            egress_idle_seconds=0.01,
            shutdown_log=shutdown_log,
            start_error=RuntimeError("start failed"),
        )
        with self.assertRaises(RuntimeError):
            await backend.list_cookies(CookieQuery(egress="decodo"))
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_seeded_cookies_are_installed_before_the_navigation(self) -> None:
        cookies = [{"name": "session", "value": "abc", "domain": ".example.com"}]
        backend, _groups, _reader, events = self._backend()
        with self._patch_site(events):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", cookies=cookies))
        self.assertEqual(events, ["cookies", "navigate"])
        await backend.aclose()

    async def test_a_reused_tab_is_left_alone(self) -> None:
        backend, groups, _reader, events = self._backend()
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            second = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        self.assertEqual(second.tab.tab_id, first.tab.tab_id)
        self.assertTrue(second.reused)
        self.assertFalse(first.reused)
        # Nothing was created and nothing was closed, so the page a person is looking at stands.
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].quit_calls, 0)
        self.assertEqual(events, ["navigate"])
        await backend.aclose()

    async def test_a_reused_tab_is_the_only_open_tab(self) -> None:
        backend, _groups, _reader, events = self._backend()
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        self.assertEqual([tab.tab_id for tab in await backend.list_interactive()], [first.tab.tab_id])
        await backend.aclose()

    async def test_a_reused_tab_refreshes_its_idle_countdown(self) -> None:
        backend, groups, _reader, events = self._backend(idle_seconds=0.12)
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            await asyncio.sleep(0.08)
            reused = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        # Past the original deadline, and inside the refreshed one.
        await asyncio.sleep(0.08)
        tabs = await backend.list_interactive()
        self.assertEqual(reused.tab.tab_id, first.tab.tab_id)
        self.assertEqual([tab.tab_id for tab in tabs], [first.tab.tab_id])
        self.assertEqual(groups[0].quit_calls, 0)
        await backend.aclose()

    async def test_a_repeat_request_for_the_page_the_tab_landed_on_reuses_it(self) -> None:
        backend, groups, _reader, events = self._backend()
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
        # Reading the list refreshes a tab's location from the live page, which is the url a
        # caller sees after a redirect rather than the one it asked for.
        landed = (await backend.list_interactive())[0].url
        self.assertNotEqual(landed, "https://example.com/")
        with self._patch_site(events):
            second = await backend.open_interactive(InteractiveRequest(url=landed))
        self.assertEqual(second.tab.tab_id, first.tab.tab_id)
        self.assertTrue(second.reused)
        self.assertEqual(len(groups), 1)
        await backend.aclose()

    async def test_a_different_url_opens_another_tab(self) -> None:
        backend, groups, _reader, events = self._backend()
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            second = await backend.open_interactive(InteractiveRequest(url="https://example.org/"))
        self.assertNotEqual(second.tab.tab_id, first.tab.tab_id)
        self.assertFalse(second.reused)
        self.assertEqual(len(groups), 2)
        await backend.aclose()

    async def test_new_tab_forces_a_second_tab_for_the_same_url(self) -> None:
        backend, groups, _reader, events = self._backend()
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/"))
            forced = await backend.open_interactive(InteractiveRequest(url="https://example.com/", new_tab=True))
        self.assertNotEqual(forced.tab.tab_id, first.tab.tab_id)
        self.assertFalse(forced.reused)
        self.assertEqual(len(groups), 2)
        await backend.aclose()

    async def test_reuse_is_scoped_to_the_egress(self) -> None:
        created: list[str] = []
        backend, groups, _reader, events = self._backend(created=created)
        with self._patch_site(events):
            first = await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
            other = await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="warp"))
        # Two egresses have two profiles and two browsers, so a tab open on one is not open on
        # the other.
        self.assertNotEqual(other.tab.tab_id, first.tab.tab_id)
        self.assertEqual(created, ["decodo", "warp"])
        self.assertEqual(len(groups), 2)
        await backend.aclose()

    async def test_a_reused_tab_and_a_fetch_share_one_browser(self) -> None:
        created: list[str] = []
        backend, _groups, _reader, events = self._backend(created=created)
        with self._patch_site(events):
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
            await backend.open_interactive(InteractiveRequest(url="https://example.com/", egress="decodo"))
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))
        self.assertEqual(created, ["decodo"])
        await backend.aclose()
