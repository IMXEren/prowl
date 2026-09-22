"""Protocol, dispatch, and HTTP contract tests for the FlareSolverr service.

A fake backend stands in for the browser so every command, error, timeout,
cookie, POST, and session path is exercised without a live browser or
network dependency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from prowl.browser.exceptions import BrowserError
from prowl.service import app as app_module
from prowl.service.app import Service, ServiceConfig, create_app
from prowl.service.backend import FetchRequest, FetchResult
from prowl.service.errors import ProxyError, SessionLimitError, SessionNotFoundError
from prowl.service.protocol import MAX_TTL_MINUTES, VERSION
from prowl.service.sessions import SessionRegistry

CF_COOKIE = {
    "name": "cf_clearance",
    "value": "solved-token",
    "domain": ".example.com",
    "path": "/",
    "expires": -1,
    "secure": True,
    "httpOnly": True,
    "sameSite": "None",
}

#: The exact payload a FlareSolverr client sends for a browser GET.
FLARESOLVERR_GET = {
    "cmd": "request.get",
    "url": "https://example.com/api/graphql",
    "maxTimeout": 60000,
    "session": "example",
    "session_ttl_minutes": 10,
    "cookies": [{"name": "session", "value": "abc", "domain": ".example.com", "path": "/"}],
    "returnOnlyCookies": False,
    "proxy": {"url": "socks5://127.0.0.1:1080"},
}

#: A proxied FlareSolverr payload: the client clears session/TTL when a proxy is set.
FLARESOLVERR_PROXIED_GET = {
    "cmd": "request.get",
    "url": "https://example.com/api/graphql",
    "maxTimeout": 60000,
    "returnOnlyCookies": False,
    "proxy": {"url": "socks5://127.0.0.1:1080"},
}


async def _fixture_handler(_request: web.Request) -> web.Response:
    return web.Response(text="<html><title>example</title></html>")


class FakeBackend:
    """In-memory backend double recording every request it receives."""

    def __init__(
        self,
        *,
        result: FetchResult | None = None,
        error: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self.started = False
        self.closed = False
        self.closed_sessions: list[str] = []
        self.requests: list[tuple[str | None, FetchRequest]] = []
        self.active = 0
        self.max_active = 0
        self._result = result
        self._error = error
        self._delay = delay

    def result(self) -> FetchResult:
        if self._result is not None:
            return self._result
        return FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            response="<html><body>ok</body></html>",
            cookies=[CF_COOKIE],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def start(self) -> None:
        self.started = True

    async def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        self.requests.append((session_id, request))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._error is not None:
                raise self._error
            return self.result()
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {"max_concurrency": 2}
    values.update(overrides)
    return ServiceConfig(**values)


class ProtocolDispatchTests(IsolatedAsyncioTestCase):
    """Service dispatch for every supported and rejected command path."""

    async def _handle(self, payload: object, backend: FakeBackend | None = None) -> tuple[int, dict]:
        service = Service(_config(), backend or FakeBackend())
        return await service.handle(payload)

    async def test_request_get_returns_solution(self) -> None:
        status, body = await self._handle({"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["solution"]["status"], 200)
        self.assertEqual(body["solution"]["url"], "https://example.com/")
        self.assertIn("response", body["solution"])
        for key in ("startTimestamp", "endTimestamp", "version", "message"):
            self.assertIn(key, body)

    async def test_request_get_returns_cf_clearance(self) -> None:
        _, body = await self._handle({"cmd": "request.get", "url": "https://example.com/"})
        names = [cookie["name"] for cookie in body["solution"]["cookies"]]
        self.assertIn("cf_clearance", names)
        self.assertEqual(body["solution"]["userAgent"], "Mozilla/5.0 (Test)")

    async def test_request_post_really_posts(self) -> None:
        backend = FakeBackend()
        await self._handle(
            {
                "cmd": "request.post",
                "url": "https://example.com/search",
                "postData": "q=test",
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            },
            backend,
        )
        ((_, request),) = backend.requests
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.post_data, "q=test")
        self.assertEqual(request.headers["content-type"], "application/x-www-form-urlencoded")

    async def test_request_post_accepts_json_object_body(self) -> None:
        backend = FakeBackend()
        await self._handle({"cmd": "request.post", "url": "https://example.com/api", "postData": {"q": "x"}}, backend)
        ((_, request),) = backend.requests
        self.assertEqual(request.post_data, '{"q": "x"}')

    async def test_return_only_cookies_blanks_response(self) -> None:
        _, body = await self._handle(
            {"cmd": "request.get", "url": "https://example.com/", "returnOnlyCookies": True},
        )
        self.assertEqual(body["solution"]["response"], "")
        self.assertTrue(body["solution"]["cookies"])

    async def test_request_get_auto_creates_named_session(self) -> None:
        backend = FakeBackend()
        service = Service(_config(), backend)
        await service.handle({"cmd": "request.get", "url": "https://example.com/", "session": "example"})
        self.assertEqual(await service.sessions.list_sessions(), ["example"])
        self.assertEqual(backend.requests[0][0], "example")

    async def test_max_timeout_is_clamped_and_forwarded(self) -> None:
        backend = FakeBackend()
        await self._handle({"cmd": "request.get", "url": "https://example.com/", "maxTimeout": 120000}, backend)
        self.assertEqual(backend.requests[0][1].timeout_seconds, 120)

    async def test_cookies_are_forwarded_to_backend(self) -> None:
        backend = FakeBackend()
        await self._handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "cookies": [{"name": "a", "value": "b", "domain": ".example.com"}],
            },
            backend,
        )
        self.assertEqual(backend.requests[0][1].cookies, [{"name": "a", "value": "b", "domain": ".example.com"}])

    async def test_sessions_create_list_destroy(self) -> None:
        backend = FakeBackend()
        service = Service(_config(), backend)
        _, created = await service.handle({"cmd": "sessions.create"})
        session_id = created["session"]
        _, listed = await service.handle({"cmd": "sessions.list"})
        self.assertEqual(listed["sessions"], [session_id])
        _, destroyed = await service.handle({"cmd": "sessions.destroy", "session": session_id})
        self.assertEqual(destroyed["status"], "ok")
        self.assertEqual(await service.sessions.list_sessions(), [])
        self.assertEqual(backend.closed_sessions, [session_id])

    async def test_sessions_destroy_unknown_is_error(self) -> None:
        _, body = await self._handle({"cmd": "sessions.destroy", "session": "nope"})
        self.assertEqual(body["status"], "error")
        self.assertIn("unknown session", body["message"])

    async def test_sessions_create_duplicate_is_error(self) -> None:
        service = Service(_config(), FakeBackend())
        await service.handle({"cmd": "sessions.create", "session": "dup"})
        _, body = await service.handle({"cmd": "sessions.create", "session": "dup"})
        self.assertEqual(body["status"], "error")

    async def test_proxied_payload_without_session_or_ttl(self) -> None:
        """The exact proxied payload (session/TTL cleared by the client) parses and runs."""
        backend = FakeBackend()
        config = _config(proxy_url="socks5://127.0.0.1:1080")
        status, body = await Service(config, backend).handle(FLARESOLVERR_PROXIED_GET)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        session, request = backend.requests[0]
        self.assertIsNone(session)
        self.assertEqual(request.url, "https://example.com/api/graphql")

    async def test_exact_flaresolverr_request_and_response_shape(self) -> None:
        """The exact FlareSolverr payload parses and yields the FlareSolverr solution shape."""
        backend = FakeBackend()
        config = _config(proxy_url="socks5://127.0.0.1:1080")
        service = Service(config, backend)
        status, body = await service.handle(FLARESOLVERR_GET)

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["version"], VERSION)
        session, request = backend.requests[0]
        self.assertEqual(session, "example")
        self.assertEqual(request.url, "https://example.com/api/graphql")
        self.assertEqual(request.timeout_seconds, 60)
        self.assertEqual(request.headers, {})
        self.assertEqual(request.cookies, [{"name": "session", "value": "abc", "domain": ".example.com", "path": "/"}])

        solution = body["solution"]
        for key in ("url", "status", "headers", "response", "cookies", "userAgent"):
            self.assertIn(key, solution)
        cookie = solution["cookies"][0]
        for key in ("name", "value", "domain", "path", "expires", "secure", "httpOnly", "sameSite"):
            self.assertIn(key, cookie)


class ProtocolValidationTests(IsolatedAsyncioTestCase):
    """Malformed, unknown, and security-sensitive inputs are rejected."""

    async def _error(self, payload: object, config: ServiceConfig | None = None) -> tuple[int, dict]:
        service = Service(config or _config(), FakeBackend())
        return await service.handle(payload)

    async def test_non_object_body_rejected(self) -> None:
        status, body = await self._error("not-an-object")
        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "error")

    async def test_missing_cmd_rejected(self) -> None:
        status, _ = await self._error({"url": "https://example.com/"})
        self.assertEqual(status, 400)

    async def test_unknown_cmd_rejected(self) -> None:
        status, body = await self._error({"cmd": "request.delete", "url": "https://example.com/"})
        self.assertEqual(status, 400)
        self.assertIn("unsupported cmd", body["message"])

    async def test_unknown_field_rejected_not_ignored(self) -> None:
        status, body = await self._error({"cmd": "request.get", "url": "https://example.com/", "totally_new": 1})
        self.assertEqual(status, 400)
        self.assertIn("unknown field", body["message"])

    async def test_sessions_list_rejects_extra_fields(self) -> None:
        status, _ = await self._error({"cmd": "sessions.list", "session": "x"})
        self.assertEqual(status, 400)

    async def test_unsupported_scheme_rejected(self) -> None:
        for url in ("file:///etc/passwd", "ftp://example.com", "gopher://x", "javascript:alert(1)"):
            status, body = await self._error({"cmd": "request.get", "url": url})
            self.assertEqual(status, 400, url)
            self.assertIn("scheme", body["message"])

    async def test_missing_url_rejected(self) -> None:
        status, _ = await self._error({"cmd": "request.get"})
        self.assertEqual(status, 400)

    async def test_get_headers_require_an_explicit_scope(self) -> None:
        status, body = await self._error(
            {"cmd": "request.get", "url": "https://example.com/", "headers": {"Authorization": "token"}},
        )
        self.assertEqual(status, 400)
        self.assertIn("headerScope", body["message"])

    async def test_get_scoped_custom_headers_accepted(self) -> None:
        for scope in ("document", "origin"):
            with self.subTest(scope=scope):
                backend = FakeBackend()
                await Service(_config(), backend).handle(
                    {
                        "cmd": "request.get",
                        "url": "https://example.com/",
                        "headers": {"Authorization": "Bearer token", "X-API-Key": "secret"},
                        "headerScope": scope,
                    },
                )
                request = backend.requests[0][1]
                self.assertEqual(request.headers, {"authorization": "Bearer token", "x-api-key": "secret"})
                self.assertEqual(request.header_scope, scope)

    async def test_get_browser_controlled_headers_rejected_even_when_scoped(self) -> None:
        for name in ("Accept", "Accept-Language", "User-Agent", "Sec-Fetch-Site", "sec-ch-ua-platform"):
            status, body = await self._error(
                {
                    "cmd": "request.get",
                    "url": "https://example.com/",
                    "headers": {name: "x"},
                    "headerScope": "origin",
                },
            )
            self.assertEqual(status, 400, name)
            self.assertIn("cannot be set", body["message"])

    async def test_header_scope_validation(self) -> None:
        for value in ("", "page", "same-site", 1, True):
            status, body = await self._error(
                {
                    "cmd": "request.get",
                    "url": "https://example.com/",
                    "headers": {"Authorization": "token"},
                    "headerScope": value,
                },
            )
            self.assertEqual(status, 400, value)
            self.assertIn("headerScope", body["message"])

        status, body = await self._error(
            {"cmd": "request.get", "url": "https://example.com/", "headerScope": "document"},
        )
        self.assertEqual(status, 400)
        self.assertIn("requires", body["message"])

        status, body = await self._error(
            {
                "cmd": "request.post",
                "url": "https://example.com/",
                "headers": {"Content-Type": "application/json"},
                "headerScope": "origin",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("request.get", body["message"])

    async def test_post_browser_controlled_headers_rejected(self) -> None:
        for name in ("Host", "Cookie", "Content-Length", "Proxy-Authorization", "Connection", "Sec-Fetch-Dest"):
            status, body = await self._error(
                {"cmd": "request.post", "url": "https://example.com/", "headers": {name: "x"}},
            )
            self.assertEqual(status, 400, name)
            self.assertIn("cannot be set", body["message"])

    async def test_post_rejects_headers_beyond_content_type(self) -> None:
        status, body = await self._error(
            {"cmd": "request.post", "url": "https://example.com/", "headers": {"x-test": "1"}},
        )
        self.assertEqual(status, 400)
        self.assertIn("not accepted", body["message"])

    async def test_post_content_type_accepted(self) -> None:
        backend = FakeBackend()
        await Service(_config(), backend).handle(
            {
                "cmd": "request.post",
                "url": "https://example.com/",
                "headers": {"Content-Type": "application/json"},
                "postData": "{}",
            },
        )
        self.assertEqual(backend.requests[0][1].headers, {"content-type": "application/json"})

    async def test_post_data_on_get_rejected(self) -> None:
        status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "postData": "a=b"})
        self.assertEqual(status, 400)

    async def test_malformed_headers_rejected(self) -> None:
        status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "headers": {"a": 1}})
        self.assertEqual(status, 400)

    async def test_malformed_cookies_rejected(self) -> None:
        status, _ = await self._error(
            {"cmd": "request.get", "url": "https://example.com/", "cookies": [{"name": "x"}]},
        )
        self.assertEqual(status, 400)

    async def test_bad_timeout_rejected(self) -> None:
        status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "maxTimeout": "soon"})
        self.assertEqual(status, 400)

    async def test_bad_ttl_rejected(self) -> None:
        for value in (0, -5, "ten", True):
            status, _ = await self._error(
                {"cmd": "sessions.create", "session": "x", "session_ttl_minutes": value},
            )
            self.assertEqual(status, 400, value)

    async def test_malformed_proxy_rejected(self) -> None:
        for value in ("socks5://p:1", {"host": "x"}, {"url": ""}):
            status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "proxy": value})
            self.assertEqual(status, 400, value)

    async def test_return_only_cookies_must_be_boolean(self) -> None:
        for value in ("false", "true", 0, 1, 1.0, []):
            status, _ = await self._error(
                {"cmd": "request.get", "url": "https://example.com/", "returnOnlyCookies": value},
            )
            self.assertEqual(status, 400, value)

    async def test_boolean_return_only_cookies_accepted(self) -> None:
        status, body = await self._error(
            {"cmd": "request.get", "url": "https://example.com/", "returnOnlyCookies": True},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    async def test_ttl_without_session_is_rejected_on_fetch(self) -> None:
        status, body = await self._error(
            {"cmd": "request.get", "url": "https://example.com/", "session_ttl_minutes": 10},
        )
        self.assertEqual(status, 400)
        self.assertIn("session", body["message"])

    async def test_ttl_upper_bound_is_enforced(self) -> None:
        status, _ = await self._error(
            {"cmd": "sessions.create", "session": "x", "session_ttl_minutes": MAX_TTL_MINUTES + 1},
        )
        self.assertEqual(status, 400)

    async def test_non_finite_timeout_and_ttl_rejected(self) -> None:
        for value in (float("inf"), float("nan"), float("-inf")):
            status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "maxTimeout": value})
            self.assertEqual(status, 400, value)
            status, _ = await self._error(
                {"cmd": "sessions.create", "session": "x", "session_ttl_minutes": value},
            )
            self.assertEqual(status, 400, value)

    async def test_url_with_userinfo_rejected(self) -> None:
        status, body = await self._error({"cmd": "request.get", "url": "https://user:pass@example.com/"})
        self.assertEqual(status, 400)
        self.assertNotIn("pass", body["message"])
        self.assertNotIn("user", body["message"])

    async def test_url_missing_host_rejected(self) -> None:
        for url in ("http:///path", "https://", "http://"):
            status, _ = await self._error({"cmd": "request.get", "url": url})
            self.assertEqual(status, 400, url)

    async def test_header_names_must_be_tokens(self) -> None:
        for name in ("bad name", "bad:name", "bad\nname", ""):
            status, _ = await self._error(
                {"cmd": "request.post", "url": "https://example.com/", "headers": {name: "x"}},
            )
            self.assertEqual(status, 400, name)

    async def test_header_values_reject_crlf(self) -> None:
        for value in ("a\r\nInjected: x", "a\nb"):
            status, _ = await self._error(
                {"cmd": "request.post", "url": "https://example.com/", "headers": {"x-test": value}},
            )
            self.assertEqual(status, 400, value)

    async def test_session_id_length_and_charset_enforced(self) -> None:
        status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "session": "x" * 129})
        self.assertEqual(status, 400)
        for bad in ("has space", "bad/slash", "bad?query"):
            status, _ = await self._error({"cmd": "request.get", "url": "https://example.com/", "session": bad})
            self.assertEqual(status, 400, bad)

    async def test_safe_session_id_accepted(self) -> None:
        backend = FakeBackend()
        await Service(_config(), backend).handle(
            {"cmd": "request.get", "url": "https://example.com/", "session": "example:1_test.2"},
        )
        self.assertEqual(backend.requests[0][0], "example:1_test.2")


class ProxyPolicyTests(IsolatedAsyncioTestCase):
    """A request proxy is honored only when it matches the process egress."""

    async def test_matching_proxy_is_accepted(self) -> None:
        backend = FakeBackend()
        config = _config(proxy_url="socks5://127.0.0.1:1080")
        _, body = await Service(config, backend).handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "proxy": {"url": "socks5://127.0.0.1:1080"},
            },
        )
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(backend.requests), 1)

    async def test_mismatched_proxy_is_rejected(self) -> None:
        config = _config(proxy_url="socks5://127.0.0.1:1080")
        _, body = await Service(config, FakeBackend()).handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "proxy": {"url": "socks5://10.0.0.1:9999"},
            },
        )
        self.assertEqual(body["status"], "error")
        self.assertNotIn("9999", body["message"])
        self.assertNotIn("10.0.0.1", body["message"])

    async def test_proxy_without_process_config_is_rejected(self) -> None:
        service = Service(_config(proxy_url=None), FakeBackend())
        _, body = await service.handle(
            {"cmd": "request.get", "url": "https://example.com/", "proxy": {"url": "socks5://127.0.0.1:1080"}},
        )
        self.assertEqual(body["status"], "error")
        self.assertIn("not permitted", body["message"])

    async def test_proxy_with_embedded_credentials_is_rejected(self) -> None:
        _, body = await Service(_config(proxy_url=None), FakeBackend()).handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "proxy": {"url": "socks5://user:secret@127.0.0.1:1080"},
            },
        )
        self.assertEqual(body["status"], "error")
        self.assertNotIn("secret", body["message"])
        self.assertNotIn("user", body["message"])

    async def test_proxy_with_unsupported_scheme_is_rejected(self) -> None:
        _, body = await Service(_config(proxy_url=None), FakeBackend()).handle(
            {"cmd": "request.get", "url": "https://example.com/", "proxy": {"url": "ftp://127.0.0.1:1080"}},
        )
        self.assertEqual(body["status"], "error")
        self.assertIn("scheme", body["message"])

    async def test_configured_proxy_with_credentials_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Service(_config(proxy_url="socks5://user:secret@127.0.0.1:1080"), FakeBackend())

    def test_proxy_error_is_caller_safe(self) -> None:
        self.assertTrue(issubclass(ProxyError, Exception))


class SessionRegistryTests(IsolatedAsyncioTestCase):
    """TTL expiry, capacity, and concurrency semantics of logical sessions."""

    async def test_ttl_expiry_hides_session_from_list(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("short", 5)
        registry._entries["short"].expires_at = time.monotonic() - 1
        self.assertEqual(await registry.list_sessions(), [])

    async def test_use_refreshes_ttl(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.create("s", 5)
        registry._entries["s"].expires_at = time.monotonic() - 1
        await registry.ensure("s", 5)
        self.assertEqual(await registry.list_sessions(), ["s"])

    async def test_capacity_limit_is_enforced(self) -> None:
        registry = SessionRegistry(max_sessions=1)
        await registry.create("a", None)
        with self.assertRaises(SessionLimitError):
            await registry.create("b", None)

    async def test_destroy_missing_session_raises_safe_error(self) -> None:
        registry = SessionRegistry(max_sessions=2)
        with self.assertRaises(SessionNotFoundError):
            await registry.destroy("missing")

    async def test_concurrent_ensure_creates_once(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await asyncio.gather(*(registry.ensure("shared", None) for _ in range(20)))
        self.assertEqual(await registry.list_sessions(), ["shared"])


class SessionLeaseRaceTests(IsolatedAsyncioTestCase):
    """Real asyncio races over the session lease, destroy, and expiry."""

    async def test_lease_serializes_the_same_session(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        order: list[str] = []

        async def worker(tag: str, delay: float) -> None:
            async with registry.lease("s"):
                order.append(f"{tag}-enter")
                await asyncio.sleep(delay)
                order.append(f"{tag}-exit")

        await asyncio.gather(worker("a", 0.05), worker("b", 0.01))
        self.assertEqual(order, ["a-enter", "a-exit", "b-enter", "b-exit"])

    async def test_destroy_fences_active_lease(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with registry.lease("s"):
                entered.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await entered.wait()
        destroy = asyncio.create_task(registry.destroy("s"))
        await asyncio.sleep(0.01)
        self.assertFalse(destroy.done())
        self.assertEqual(await registry.list_sessions(), ["s"])
        release.set()
        await holder
        await destroy
        self.assertEqual(await registry.list_sessions(), [])

    async def test_expiry_does_not_remove_active_lease(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", 5)
        async with registry.lease("s"):
            registry._entries["s"].expires_at = time.monotonic() - 1
            self.assertEqual(await registry.list_sessions(), ["s"])
        self.assertEqual(await registry.list_sessions(), [])

    async def test_recreate_does_not_overlap_older_lease(self) -> None:
        registry = SessionRegistry(max_sessions=4)
        await registry.ensure("s", None)
        entered = asyncio.Event()
        release = asyncio.Event()
        active = 0
        overlaps: list[int] = []

        async def run() -> None:
            nonlocal active
            async with registry.lease("s"):
                active += 1
                if active > 1:
                    overlaps.append(active)
                entered.set()
                await release.wait()
                active -= 1

        async def recreate() -> None:
            await registry.ensure("s", None)
            await run()

        first = asyncio.create_task(run())
        await entered.wait()
        destroy = asyncio.create_task(registry.destroy("s"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(recreate())
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(first, second, destroy)
        self.assertEqual(overlaps, [])
        self.assertEqual(await registry.list_sessions(), ["s"])

    async def test_waiting_recreate_makes_bounded_progress(self) -> None:
        """A recreate waiting out a destroy is woken once, then proceeds without spin.

        Repeated to stress the final-lease handoff: the destroying entry is removed
        under the registry lock before waiters are released, so a waiter never
        re-observes a dead entry (which would busy-spin the event loop).
        """
        registry = SessionRegistry(max_sessions=4)
        state = {"active": 0}
        overlaps: list[int] = []

        async def run_once(entered: asyncio.Event, release: asyncio.Event) -> None:
            async with registry.lease("s"):
                state["active"] += 1
                if state["active"] > 1:
                    overlaps.append(state["active"])
                entered.set()
                await release.wait()
                state["active"] -= 1

        for _ in range(25):
            await registry.ensure("s", None)
            entered = asyncio.Event()
            release = asyncio.Event()
            holder = asyncio.create_task(run_once(entered, release))
            await entered.wait()
            destroy = asyncio.create_task(registry.destroy("s"))
            await asyncio.sleep(0)
            recreate = asyncio.create_task(registry.ensure("s", None))
            await asyncio.sleep(0)
            release.set()
            # A post-drain busy-spin would never yield here and would hang this await.
            await asyncio.wait_for(asyncio.gather(holder, destroy, recreate), timeout=2.0)
            self.assertEqual(overlaps, [])
        self.assertEqual(await registry.list_sessions(), ["s"])

    async def test_no_unbounded_lock_map_after_destroy(self) -> None:
        registry = SessionRegistry(max_sessions=64)
        for index in range(50):
            session_id = f"s{index}"
            await registry.ensure(session_id, None)
            async with registry.lease(session_id):
                pass
            await registry.destroy(session_id)
        self.assertEqual(len(registry._entries), 0)


class ErrorSanitizationTests(IsolatedAsyncioTestCase):
    """Backend failures surface safe messages, never internals."""

    async def test_unexpected_browser_error_is_generic(self) -> None:
        backend = FakeBackend(error=BrowserError("secret /home/user/.profile password=hunter2"))
        _, body = await Service(_config(), backend).handle({"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "internal error while executing the command")
        self.assertNotIn("hunter2", body["message"])
        self.assertNotIn(".profile", body["message"])

    async def test_unexpected_error_is_generic(self) -> None:
        backend = FakeBackend(error=RuntimeError("secret /home/user/.profile password=hunter2"))
        _, body = await Service(_config(), backend).handle({"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "internal error while executing the command")

    async def test_timeout_is_bounded_and_reported(self) -> None:
        backend = FakeBackend(delay=5.0)
        service = Service(_config(), backend)
        original = app_module._TIMEOUT_SLACK_SECONDS
        app_module._TIMEOUT_SLACK_SECONDS = 0.05
        try:
            _, body = await service.handle(
                {"cmd": "request.get", "url": "https://example.com/", "maxTimeout": 1000},
            )
        finally:
            app_module._TIMEOUT_SLACK_SECONDS = original
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "request timed out")

    async def test_anonymous_requests_are_serialized_by_service(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(max_concurrency=2), backend)
        await asyncio.gather(
            service.handle({"cmd": "request.get", "url": "https://example.com/a"}),
            service.handle({"cmd": "request.get", "url": "https://example.com/b"}),
        )
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(backend.max_active, 1)

    async def test_distinct_named_sessions_use_configured_concurrency(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(max_concurrency=2), backend)
        await asyncio.gather(
            service.handle({"cmd": "request.get", "url": "https://example.com/a", "session": "a"}),
            service.handle({"cmd": "request.get", "url": "https://example.com/b", "session": "b"}),
        )
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(backend.max_active, 2)

    async def test_concurrency_bound_serializes_requests(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(max_concurrency=1), backend)
        await asyncio.gather(
            service.handle({"cmd": "request.get", "url": "https://example.com/a"}),
            service.handle({"cmd": "request.get", "url": "https://example.com/b"}),
        )
        self.assertEqual(len(backend.requests), 2)

    async def test_concurrent_named_session_auto_create_is_race_safe(self) -> None:
        backend = FakeBackend(delay=0.02)
        service = Service(_config(max_sessions=1), backend)
        results = await asyncio.gather(
            *(service.handle({"cmd": "request.get", "url": "https://example.com/", "session": "s"}) for _ in range(10)),
        )
        self.assertTrue(all(body["status"] == "ok" for _, body in results))
        self.assertEqual(await service.sessions.list_sessions(), ["s"])


class HttpContractTests(IsolatedAsyncioTestCase):
    """HTTP transport, health, readiness, and local-server integration."""

    async def asyncSetUp(self) -> None:
        self.backend = FakeBackend()
        self.client = TestClient(TestServer(create_app(_config(), self.backend)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_post_v1_round_trip(self) -> None:
        response = await self.client.post("/v1", json={"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["solution"]["status"], 200)

    async def test_malformed_json_body_is_400(self) -> None:
        response = await self.client.post("/v1", data="{not json")
        self.assertEqual(response.status, 400)

    async def test_healthz_and_readyz(self) -> None:
        health = await self.client.get("/healthz")
        self.assertEqual(health.status, 200)
        ready = await self.client.get("/readyz")
        self.assertEqual(ready.status, 200)
        self.assertEqual((await ready.json())["status"], "ok")
        self.assertTrue(self.backend.started)

    async def test_cleanup_shuts_backend_down(self) -> None:
        await self.client.close()
        self.assertTrue(self.backend.closed)
        self.client = TestClient(TestServer(create_app(_config(), FakeBackend())))
        await self.client.start_server()

    async def test_local_http_fixture_end_to_end(self) -> None:
        """Service + fake browser fetching a deterministic local HTTP server."""
        local = FakeBackend()
        fixture_app = web.Application()
        fixture_app.router.add_get("/", _fixture_handler)
        client = TestClient(TestServer(fixture_app))
        await client.start_server()
        try:
            url = str(client.make_url("/"))
            local._result = FetchResult(
                url=url,
                status_code=200,
                headers={"content-type": "text/html"},
                response="<html><title>example</title></html>",
                cookies=[],
                user_agent="Mozilla/5.0 (Test)",
            )
            _, body = await Service(_config(), local).handle({"cmd": "request.get", "url": url})
            self.assertEqual(body["solution"]["response"], "<html><title>example</title></html>")
            self.assertEqual(body["solution"]["status"], 200)
        finally:
            await client.close()


class VersionTests(IsolatedAsyncioTestCase):
    """Protocol version tracks the installed package version."""

    async def test_version_matches_distribution(self) -> None:
        from importlib.metadata import version  # noqa: PLC0415

        self.assertEqual(VERSION, f"prowl/{version('prowl')}")


class ConfigTests(IsolatedAsyncioTestCase):
    """Environment-driven configuration defaults and overrides."""

    async def test_default_concurrency_is_one(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.max_concurrency, 1)

    async def test_environment_overrides_are_read(self) -> None:
        env = {
            "PROWL_MAX_CONCURRENCY": "3",
            "PROWL_MAX_SESSIONS": "7",
            "PROWL_PROXY_URL": "socks5://127.0.0.1:1080",
        }
        with patch.dict("os.environ", env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.max_concurrency, 3)
        self.assertEqual(config.max_sessions, 7)
        self.assertEqual(config.proxy_url, "socks5://127.0.0.1:1080")
