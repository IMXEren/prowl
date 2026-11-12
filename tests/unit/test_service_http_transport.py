"""HTTP fast-path regressions: identity matching, cookie mirroring and the client lifecycle.

Every test here talks to a loopback HTTP/1.1 server and a stand-in browser context. No browser is
launched, nothing external is contacted, and the only environment touched is a patched proxy
variable.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from curl_cffi.requests.impersonate import BrowserTypeLiteral
from playwright.async_api import BrowserContext

from prowl.browser.headers import HEADER_SCOPE_DOCUMENT, HEADER_SCOPE_ORIGIN
from prowl.service.classification import UNKNOWN_BLOCK
from prowl.service.http_transport import (
    CLASSIFICATION_BODY_BYTES,
    SUPPORTED_CHROME_MAJORS,
    HttpClient,
    HttpNetworkError,
    HttpTransportError,
    IdentityHeaderConflictError,
    ObservedBrowserIdentity,
    UnsupportedCookieError,
    UnsupportedIdentityError,
    chrome_major_from_user_agent,
    resolve_http_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from playwright._impl._api_structures import SetCookieParam

_REASONS = {
    200: "OK",
    302: "Found",
    403: "Forbidden",
    404: "Not Found",
    429: "Too Many Requests",
    500: "Internal Server Error",
}

_HOST = "127.0.0.1"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
)
_BRANDS = (("Not?A_Brand", "99"), ("Chromium", "150"), ("Google Chrome", "150"))
_FULL_VERSIONS = (
    ("Not?A_Brand", "99.0.0.0"),
    ("Chromium", "150.0.0.0"),
    ("Google Chrome", "150.0.0.0"),
)


@dataclass
class _Received:
    """One request as the loopback server read it off the wire."""

    method: str
    target: str
    headers: dict[str, str]
    body: bytes


_Reply = tuple[int, list[tuple[str, str]], bytes]


class _Loopback:
    """A scripted HTTP/1.1 server on the loopback interface.

    The handler is either a plain callable or a coroutine function, and answers with a status, a
    list of header pairs (so duplicates survive) and a body.
    """

    def __init__(self, handler: Callable[[_Received], Any], *, host: str = _HOST) -> None:
        self._handler = handler
        self.received: list[_Received] = []
        self.host = host
        self.port = 0
        self._server: asyncio.AbstractServer | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    @property
    def base_url(self) -> str:
        """The server's root url."""
        return f"http://{self.host}:{self.port}"

    async def start(self) -> _Loopback:
        """Listen on an ephemeral port and return this server."""
        self._server = await asyncio.start_server(self._serve, self.host, 0, family=socket.AF_INET)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        """Stop listening and drop every connection still held open.

        A keep-alive connection leaves its handler waiting for another request, and
        ``Server.wait_closed`` waits for that handler, so the writers are closed first.
        """
        if self._server is not None:
            self._server.close()
            for writer in list(self._writers):
                writer.close()
            await self._server.wait_closed()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            while True:
                request = await self._read_request(reader)
                if request is None:
                    return
                self.received.append(request)
                reply = self._handler(request)
                if inspect.isawaitable(reply):
                    reply = await reply
                status, headers, body = reply
                writer.write(_encode(status, headers, body))
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            return
        finally:
            self._writers.discard(writer)
            writer.close()

    @staticmethod
    async def _read_request(reader: asyncio.StreamReader) -> _Received | None:
        line = await reader.readline()
        if not line:
            return None
        method, target, _ = line.decode("latin-1").split(" ", 2)
        headers: dict[str, str] = {}
        while True:
            raw = await reader.readline()
            if raw in (b"\r\n", b"\n", b""):
                break
            name, _, value = raw.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length") or 0)
        body = await reader.readexactly(length) if length else b""
        return _Received(method=method, target=target, headers=headers, body=body)


def _encode(status: int, headers: list[tuple[str, str]], body: bytes) -> bytes:
    lines = [f"HTTP/1.1 {status} {_REASONS.get(status, 'Status')}"]
    lines.extend(f"{name}: {value}" for name, value in headers)
    lines.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


def _routes(routes: dict[str, _Reply]) -> Callable[[_Received], _Reply]:
    def handle(request: _Received) -> _Reply:
        return routes.get(request.target, (404, [], b"missing"))

    return handle


class _ContextStub:
    """A stand-in for a live Playwright context's cookie surface."""

    def __init__(self, cookies: list[dict[str, Any]] | None = None) -> None:
        self.held: list[dict[str, Any]] = [dict(cookie) for cookie in cookies or []]
        self.added: list[dict[str, Any]] = []
        self.cleared: list[dict[str, Any]] = []

    async def cookies(self) -> list[dict[str, Any]]:
        return [dict(cookie) for cookie in self.held]

    async def add_cookies(self, cookies: Sequence[SetCookieParam]) -> None:
        for cookie in cookies:
            self.added.append(dict(cookie))
            self.held = [held for held in self.held if _identity(held) != _identity(cookie)]
            self.held.append(dict(cookie))

    async def clear_cookies(
        self,
        *,
        name: str | None = None,
        domain: str | None = None,
        path: str | None = None,
    ) -> None:
        self.cleared.append({"name": name, "domain": domain, "path": path})
        self.held = [
            held
            for held in self.held
            if not (held.get("name") == name and held.get("domain") == domain and held.get("path") == path)
        ]

    def names(self) -> set[str]:
        return {str(cookie.get("name")) for cookie in self.held}

    def get(self, name: str) -> dict[str, Any]:
        for cookie in self.held:
            if cookie.get("name") == name:
                return cookie
        msg = f"{name} is not held"
        raise AssertionError(msg)


def _identity(cookie: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(cookie.get("name")), str(cookie.get("domain")), str(cookie.get("path")))


def _cookie(  # noqa: PLR0913 - one cookie's fields, each defaulted for the cases that need it
    name: str,
    value: str,
    *,
    domain: str = _HOST,
    path: str = "/",
    secure: bool = False,
    http_only: bool = False,
    expires: float = -1,
    same_site: str | None = None,
) -> dict[str, Any]:
    cookie: dict[str, Any] = {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "secure": secure,
        "httpOnly": http_only,
        "expires": expires,
    }
    if same_site is not None:
        cookie["sameSite"] = same_site
    return cookie


def _observed(**overrides: Any) -> ObservedBrowserIdentity:
    values: dict[str, Any] = {
        "user_agent": _USER_AGENT,
        "platform": "Windows",
        "brands": _BRANDS,
        "full_version_list": _FULL_VERSIONS,
        "architecture": "x86",
        "bitness": "64",
        "platform_version": "15.0.0",
    }
    values.update(overrides)
    return ObservedBrowserIdentity(**values)


class IdentityResolutionTests(TestCase):
    """The HTTP identity is matched to the browser, never adjusted to fit a profile."""

    def test_supported_majors_are_installed_impersonation_profiles(self) -> None:
        installed = {
            int(name[len("chrome") :])
            for name in get_args(BrowserTypeLiteral)
            if name.startswith("chrome") and name[len("chrome") :].isdigit()
        }

        self.assertTrue(installed >= SUPPORTED_CHROME_MAJORS, sorted(installed))

    def test_identity_carries_the_browser_user_agent_and_client_hints(self) -> None:
        identity = resolve_http_identity(_observed())
        headers = identity.request_headers()

        self.assertEqual(identity.impersonate, "chrome150")
        self.assertEqual(identity.chrome_major, 150)
        self.assertEqual(headers["user-agent"], _USER_AGENT)
        self.assertEqual(
            headers["sec-ch-ua"],
            '"Not?A_Brand";v="99", "Chromium";v="150", "Google Chrome";v="150"',
        )
        self.assertEqual(
            headers["sec-ch-ua-full-version-list"],
            '"Not?A_Brand";v="99.0.0.0", "Chromium";v="150.0.0.0", "Google Chrome";v="150.0.0.0"',
        )
        self.assertEqual(headers["sec-ch-ua-platform"], '"Windows"')
        self.assertEqual(headers["sec-ch-ua-arch"], '"x86"')
        self.assertEqual(headers["sec-ch-ua-bitness"], '"64"')
        self.assertEqual(headers["sec-ch-ua-platform-version"], '"15.0.0"')
        self.assertEqual(headers["sec-ch-ua-mobile"], "?0")

    def test_chrome_major_is_read_from_the_user_agent(self) -> None:
        self.assertEqual(chrome_major_from_user_agent(_USER_AGENT), 150)
        self.assertEqual(chrome_major_from_user_agent("Mozilla/5.0 HeadlessChrome/146.0.0.0"), 146)
        self.assertIsNone(chrome_major_from_user_agent("Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/147.0"))

    def test_browser_major_without_a_profile_is_refused(self) -> None:
        observed = _observed(user_agent=_USER_AGENT.replace("Chrome/150", "Chrome/149"))

        with self.assertRaises(UnsupportedIdentityError) as caught:
            resolve_http_identity(observed)

        self.assertIn("149", str(caught.exception))

    def test_non_chrome_persona_is_refused(self) -> None:
        observed = _observed(user_agent="Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/147.0")

        with self.assertRaises(UnsupportedIdentityError):
            resolve_http_identity(observed)

    def test_mobile_persona_is_refused(self) -> None:
        with self.assertRaises(UnsupportedIdentityError):
            resolve_http_identity(_observed(mobile=True))

    def test_unsupported_platform_is_refused(self) -> None:
        with self.assertRaises(UnsupportedIdentityError):
            resolve_http_identity(_observed(platform="Android"))

    def test_unsupported_architecture_and_bitness_are_refused(self) -> None:
        for overrides in ({"architecture": "arm64"}, {"bitness": "128"}):
            with self.subTest(overrides=overrides), self.assertRaises(UnsupportedIdentityError):
                resolve_http_identity(_observed(**overrides))

    def test_brands_that_contradict_the_user_agent_are_refused(self) -> None:
        with self.assertRaises(UnsupportedIdentityError):
            resolve_http_identity(_observed(brands=(("Chromium", "149"), ("Google Chrome", "149"))))

    def test_missing_brand_list_is_refused(self) -> None:
        with self.assertRaises(UnsupportedIdentityError):
            resolve_http_identity(_observed(brands=(), full_version_list=()))

    def test_playwright_context_satisfies_the_cookie_protocol(self) -> None:
        # The transport drives the installed Playwright context directly; this is the seam.
        for name in ("cookies", "add_cookies", "clear_cookies"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(BrowserContext, name, None)))


class _TransportTestCase(IsolatedAsyncioTestCase):
    """Shared client construction for the loopback tests."""

    context: _ContextStub

    def make_client(self, **kwargs: Any) -> HttpClient:
        self.context = kwargs.pop("context", _ContextStub())
        return HttpClient(context=self.context, identity=resolve_http_identity(_observed()), **kwargs)

    async def start(self, handler: Callable[[_Received], Any], *, host: str = _HOST) -> _Loopback:
        server = await _Loopback(handler, host=host).start()
        self.addAsyncCleanup(server.stop)
        return server


class RequestTests(_TransportTestCase):
    """The request itself: identity headers, body, redirects and the result shape."""

    async def test_identity_headers_replace_the_profile_defaults(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        received = server.received[0]
        self.assertEqual(received.headers["user-agent"], _USER_AGENT)
        self.assertEqual(received.headers["sec-ch-ua-platform"], '"Windows"')
        self.assertIn('"Chromium";v="150"', received.headers["sec-ch-ua"])

    async def test_result_preserves_status_headers_body_and_final_url(self) -> None:
        reply: _Reply = (
            200,
            [
                ("Content-Type", "text/html"),
                ("X-Trace", "a"),
                ("X-Trace", "b"),
                ("Set-Cookie", "a=1; Path=/"),
                ("Set-Cookie", "b=2; Path=/"),
            ],
            b"<html>ok</html>",
        )
        server = await self.start(_routes({"/page": reply, "/next": (200, [], b"final")}))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.body, "<html>ok</html>")
        self.assertEqual(result.url, server.base_url + "/page")
        self.assertEqual(result.headers["content-type"], "text/html")
        self.assertEqual(result.headers["x-trace"], "a, b")
        self.assertEqual(result.headers["set-cookie"], "a=1; Path=/")
        self.assertEqual(result.set_cookie_headers, ("a=1; Path=/", "b=2; Path=/"))

    async def test_post_body_reaches_the_server(self) -> None:
        server = await self.start(_routes({"/submit": (200, [], b"done")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/submit", method="POST", content="a=1", deadline_seconds=10)

        self.assertEqual(server.received[0].method, "POST")
        self.assertEqual(server.received[0].body, b"a=1")

    async def test_classification_scans_only_the_bounded_prefix(self) -> None:
        marker = b'<form id="challenge-form"></form>'
        reply: _Reply = (403, [], b"a" * (CLASSIFICATION_BODY_BYTES + 512) + marker)
        server = await self.start(_routes({"/page": reply}))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(result.classify().category, UNKNOWN_BLOCK)

    async def test_conflicting_identity_headers_are_refused(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        for headers in ({"User-Agent": "curl/8"}, {"sec-ch-ua": '"Chromium";v="120"'}, {"Cookie": "a=1"}):
            with self.subTest(headers=headers), self.assertRaises(IdentityHeaderConflictError):
                await client.fetch(server.base_url + "/page", headers=headers, deadline_seconds=10)

        self.assertEqual(server.received, [])

    async def test_caller_headers_are_sent_lowercased(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", headers={"X-Api-Key": "k"}, deadline_seconds=10)

        self.assertEqual(server.received[0].headers["x-api-key"], "k")

    async def test_redirect_to_another_host_drops_the_authorization_header(self) -> None:
        # The second origin answers on a different host name, which is what libcurl compares.
        other = await self.start(_routes({"/end": (200, [], b"done")}), host="localhost")
        start = await self.start(_routes({"/start": (302, [("Location", f"{other.base_url}/end")], b"")}))
        client = self.make_client()

        result = await client.fetch(
            start.base_url + "/start",
            headers={"Authorization": "Bearer secret"},
            deadline_seconds=10,
        )

        self.assertEqual(result.body, "done")
        self.assertEqual(start.received[0].headers.get("authorization"), "Bearer secret")
        self.assertNotIn("authorization", other.received[0].headers)


class CookieTests(_TransportTestCase):
    """Cookies the context owns are sent, and only the exchange's own changes come back."""

    async def test_context_cookies_are_sent_and_only_changes_are_mirrored(self) -> None:
        context = _ContextStub(
            [
                _cookie("sid", "abc", http_only=True, same_site="Strict"),
                _cookie("theme", "dark"),
            ],
        )
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "fresh=1; Path=/")], b"ok")}))
        client = self.make_client(context=context)

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        sent = server.received[0].headers["cookie"]
        self.assertIn("sid=abc", sent)
        self.assertIn("theme=dark", sent)
        self.assertEqual([cookie["name"] for cookie in context.added], ["fresh"])
        self.assertEqual(context.cleared, [])
        self.assertEqual(context.get("fresh")["domain"], _HOST)
        self.assertEqual({cookie["name"] for cookie in result.cookies}, {"sid", "theme", "fresh"})
        sid = next(cookie for cookie in result.cookies if cookie["name"] == "sid")
        self.assertEqual(sid["sameSite"], "Strict")
        self.assertTrue(sid["httpOnly"])

    async def test_unchanged_cookie_is_not_written_back(self) -> None:
        context = _ContextStub([_cookie("sid", "abc")])
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "sid=abc; Path=/")], b"ok")}))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.added, [])
        self.assertEqual(context.cleared, [])

    async def test_same_name_on_two_paths_survives_as_two_cookies(self) -> None:
        headers = [("Set-Cookie", "sid=root; Path=/"), ("Set-Cookie", "sid=nested; Path=/x")]
        server = await self.start(_routes({"/page": (200, headers, b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        stored = {cookie["path"]: cookie["value"] for cookie in self.context.held}
        self.assertEqual(stored, {"/": "root", "/x": "nested"})

    async def test_deletion_clears_exactly_the_cookie_the_response_deleted(self) -> None:
        context = _ContextStub([_cookie("sid", "abc"), _cookie("keep", "1", path="/keep")])
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "sid=; Max-Age=0; Path=/")], b"ok")}))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.cleared, [{"name": "sid", "domain": _HOST, "path": "/"}])
        self.assertEqual(context.names(), {"keep"})

    async def test_domain_declaration_keeps_the_scope_the_jar_accepted(self) -> None:
        # A loopback server can only set cookies for its own host, and the accepted jar decides the
        # scope: nothing is invented from the header, so neither cookie gains a domain form.
        headers = [("Set-Cookie", f"wide=1; Domain={_HOST}; Path=/"), ("Set-Cookie", "narrow=2; Path=/")]
        server = await self.start(_routes({"/page": (200, headers, b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(self.context.get("wide")["domain"], _HOST)
        self.assertEqual(self.context.get("narrow")["domain"], _HOST)

    async def test_rejected_domain_cannot_move_or_delete_an_existing_cookie(self) -> None:
        context = _ContextStub([_cookie("sid", "kept", http_only=True)])
        headers = [("Set-Cookie", "sid=evil; Domain=other.prowl.test; Path=/")]
        server = await self.start(_routes({"/page": (200, headers, b"ok")}))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.added, [])
        self.assertEqual(context.cleared, [])
        self.assertEqual(context.get("sid")["value"], "kept")

    async def test_same_name_on_two_paths_keeps_its_own_metadata(self) -> None:
        headers = [
            ("Set-Cookie", "sid=root; Path=/; SameSite=Lax; HttpOnly"),
            ("Set-Cookie", "sid=nested; Path=/x; SameSite=Strict"),
        ]
        server = await self.start(_routes({"/page": (200, headers, b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        by_path = {cookie["path"]: cookie for cookie in self.context.held}
        self.assertEqual(by_path["/"]["sameSite"], "Lax")
        self.assertTrue(by_path["/"]["httpOnly"])
        self.assertEqual(by_path["/x"]["sameSite"], "Strict")
        self.assertFalse(by_path["/x"]["httpOnly"])

    async def test_same_site_declaration_is_mirrored(self) -> None:
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "sid=1; Path=/; SameSite=Strict")], b"ok")}))
        client = self.make_client()

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(self.context.get("sid")["sameSite"], "Strict")

    async def test_metadata_only_change_is_mirrored(self) -> None:
        context = _ContextStub([_cookie("sid", "abc", same_site="Strict")])
        headers = [("Set-Cookie", "sid=abc; Path=/; SameSite=Lax; HttpOnly")]
        server = await self.start(_routes({"/page": (200, headers, b"ok")}))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.get("sid")["value"], "abc")
        self.assertEqual(context.get("sid")["sameSite"], "Lax")
        self.assertTrue(context.get("sid")["httpOnly"])

    async def test_declaration_without_same_site_resets_it(self) -> None:
        context = _ContextStub([_cookie("sid", "abc", same_site="Strict", http_only=True)])
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "sid=def; Path=/")], b"ok")}))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.get("sid")["value"], "def")
        self.assertNotIn("sameSite", context.get("sid"))
        self.assertFalse(context.get("sid")["httpOnly"])

    async def test_untouched_cookie_metadata_is_preserved_in_the_result(self) -> None:
        context = _ContextStub([_cookie("imported", "1", http_only=True, same_site="Strict")])
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "fresh=1; Path=/")], b"ok")}))
        client = self.make_client(context=context)

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        imported = next(cookie for cookie in result.cookies if cookie["name"] == "imported")
        self.assertEqual(imported["sameSite"], "Strict")
        self.assertTrue(imported["httpOnly"])
        self.assertEqual([cookie["name"] for cookie in context.added], ["fresh"])

    async def test_redirect_declaration_metadata_reaches_the_cookie(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/end"), ("Set-Cookie", "stage=one; Path=/; SameSite=Strict")], b""),
            "/end": (200, [], b"final"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual(self.context.get("stage")["value"], "one")
        self.assertEqual(self.context.get("stage")["sameSite"], "Strict")

    async def test_rejected_domain_cannot_change_foreign_cookie_metadata(self) -> None:
        context = _ContextStub([_cookie("sid", "original", domain="other.prowl.test", same_site="Strict")])
        server = await self.start(
            _routes(
                {"/page": (200, [("Set-Cookie", "sid=evil; Domain=other.prowl.test; Path=/; SameSite=Lax")], b"ok")}
            )
        )
        client = self.make_client(context=context)

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(context.added, [])
        self.assertEqual(context.cleared, [])
        self.assertEqual(result.cookies[0]["sameSite"], "Strict")
        self.assertEqual(result.cookies[0]["value"], "original")

    async def test_each_request_starts_from_the_context_jar(self) -> None:
        routes = {"/one": (200, [("Set-Cookie", "stale=1; Path=/")], b"ok"), "/two": (200, [], b"ok")}
        context = _ContextStub()
        server = await self.start(_routes(routes))
        client = self.make_client(context=context)

        await client.fetch(server.base_url + "/one", deadline_seconds=10)
        self.assertEqual(context.names(), {"stale"})
        context.held = []
        await client.fetch(server.base_url + "/two", deadline_seconds=10)

        self.assertNotIn("stale=1", server.received[1].headers.get("cookie", ""))

    async def test_redirect_cookie_changes_are_mirrored(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/end"), ("Set-Cookie", "stage=one; Path=/")], b""),
            "/end": (200, [("Set-Cookie", "stage=two; Path=/")], b"final"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual(result.url, server.base_url + "/end")
        self.assertEqual(result.body, "final")
        self.assertEqual(self.context.get("stage")["value"], "two")
        self.assertEqual(len(server.received), 2)

    async def test_caller_cookies_are_request_scoped_and_not_written_back(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        await client.fetch(
            server.base_url + "/page",
            cookies=[_cookie("supplied", "1")],
            deadline_seconds=10,
        )

        self.assertIn("supplied=1", server.received[0].headers["cookie"])
        self.assertEqual(self.context.added, [])

    async def test_caller_cookie_the_site_replaces_is_mirrored(self) -> None:
        server = await self.start(_routes({"/page": (200, [("Set-Cookie", "supplied=2; Path=/")], b"ok")}))
        client = self.make_client()

        await client.fetch(
            server.base_url + "/page",
            cookies=[_cookie("supplied", "1")],
            deadline_seconds=10,
        )

        self.assertEqual(self.context.get("supplied")["value"], "2")


class UnsupportedCookieTests(_TransportTestCase):
    """State a jar cannot express is refused before anything is sent."""

    async def test_partitioned_context_cookie_is_refused_before_the_request(self) -> None:
        context = _ContextStub([{**_cookie("sid", "abc"), "partitionKey": "https://top.example"}])
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client(context=context)

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(server.received, [])

    async def test_partitioned_caller_cookie_is_refused(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(
                server.base_url + "/page",
                cookies=[{**_cookie("sid", "abc"), "partitionKey": "https://top.example"}],
                deadline_seconds=10,
            )

        self.assertEqual(server.received, [])

    async def test_cookie_without_a_domain_is_refused(self) -> None:
        context = _ContextStub([{**_cookie("sid", "abc"), "domain": ""}])
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client(context=context)

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/page", deadline_seconds=10)


class EgressTests(_TransportTestCase):
    """The fast path leaves through the browser's own egress and nothing else."""

    async def test_the_configured_proxy_carries_the_request(self) -> None:
        proxy = await self.start(_routes({"http://prowl.test/page": (200, [], b"proxied")}))
        client = self.make_client(proxy=proxy.base_url)

        result = await client.fetch("http://prowl.test/page", deadline_seconds=10)

        self.assertEqual(result.body, "proxied")
        self.assertEqual(proxy.received[0].target, "http://prowl.test/page")

    async def test_environment_proxies_are_not_used(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"direct")}))
        client = self.make_client()
        environment = {
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
        }

        with patch.dict(os.environ, environment):
            result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertEqual(result.body, "direct")
        self.assertEqual(len(server.received), 1)


class LifecycleTests(_TransportTestCase):
    """Deadlines, cancellation and closure leave the client usable or closed, never in between."""

    async def test_deadline_is_reported_as_a_network_error_with_its_cause(self) -> None:
        async def slow(_request: _Received) -> _Reply:
            await asyncio.sleep(1)
            return (200, [], b"late")

        server = await self.start(_routes({"/fast": (200, [], b"ok")}))
        slow_server = await self.start(slow)
        client = self.make_client()

        with self.assertRaises(HttpNetworkError) as caught:
            await client.fetch(slow_server.base_url + "/slow", deadline_seconds=0.2)

        self.assertIsNotNone(caught.exception.__cause__)
        result = await client.fetch(server.base_url + "/fast", deadline_seconds=10)
        self.assertEqual(result.body, "ok")

    async def test_budget_covers_waiting_for_the_client_lock(self) -> None:
        async def slow(_request: _Received) -> _Reply:
            await asyncio.sleep(0.4)
            return (200, [], b"slow")

        slow_server = await self.start(slow)
        server = await self.start(_routes({"/fast": (200, [], b"ok")}))
        client = self.make_client()

        held = asyncio.ensure_future(client.fetch(slow_server.base_url + "/slow", deadline_seconds=10))
        await asyncio.sleep(0.05)
        with self.assertRaises(HttpNetworkError):
            await client.fetch(server.base_url + "/fast", deadline_seconds=0.1)

        self.assertEqual((await held).body, "slow")
        self.assertEqual((await client.fetch(server.base_url + "/fast", deadline_seconds=10)).body, "ok")

    async def test_cancellation_leaves_the_client_usable(self) -> None:
        async def slow(_request: _Received) -> _Reply:
            await asyncio.sleep(1)
            return (200, [], b"late")

        server = await self.start(_routes({"/fast": (200, [], b"ok")}))
        slow_server = await self.start(slow)
        client = self.make_client()

        task = asyncio.ensure_future(client.fetch(slow_server.base_url + "/slow", deadline_seconds=30))
        await asyncio.sleep(0.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        result = await client.fetch(server.base_url + "/fast", deadline_seconds=10)
        self.assertEqual(result.body, "ok")

    async def test_close_is_idempotent_and_refuses_further_requests(self) -> None:
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        await client.aclose()
        await client.aclose()

        with self.assertRaises(HttpTransportError):
            await client.fetch(server.base_url + "/page", deadline_seconds=10)

    async def test_queued_fetch_is_refused_once_the_client_closed(self) -> None:
        async def slow(_request: _Received) -> _Reply:
            await asyncio.sleep(0.3)
            return (200, [], b"slow")

        slow_server = await self.start(slow)
        server = await self.start(_routes({"/page": (200, [], b"ok")}))
        client = self.make_client()

        held = asyncio.ensure_future(client.fetch(slow_server.base_url + "/slow", deadline_seconds=10))
        await asyncio.sleep(0.05)
        closing = asyncio.ensure_future(client.aclose())
        await asyncio.sleep(0)
        queued = asyncio.ensure_future(client.fetch(server.base_url + "/page", deadline_seconds=10))
        await asyncio.sleep(0.5)

        self.assertEqual((await held).body, "slow")
        await closing
        with self.assertRaises(HttpTransportError):
            await queued
        self.assertEqual(server.received, [])


class RedirectTests(_TransportTestCase):
    """Redirect hops: cookie semantics first, then scope, method and bounded following."""

    async def test_partitioned_cookie_in_a_redirect_is_refused_before_the_next_hop(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/end"), ("Set-Cookie", "p=1; Path=/; Partitioned")], b""),
            "/end": (200, [], b"end"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual([request.target for request in server.received], ["/start"])
        self.assertEqual(self.context.added, [])

    async def test_same_site_none_without_secure_in_a_redirect_is_refused(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/end"), ("Set-Cookie", 'p=1; Path=/; SameSite="None"')], b""),
            "/end": (200, [], b"end"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual([request.target for request in server.received], ["/start"])

    async def test_hop_cookie_survives_a_later_network_failure(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "http://127.0.0.1:1/end"), ("Set-Cookie", "kept=1; Path=/")], b""),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        with self.assertRaises(HttpNetworkError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual(self.context.get("kept")["value"], "1")

    async def test_a_scope_change_is_refused_without_rewriting_native_cookies(self) -> None:
        other = await self.start(
            _routes({"/page": (200, [("Set-Cookie", "sid=declared; Domain=localhost; Path=/; SameSite=Lax")], b"ok")}),
            host="localhost",
        )
        context = _ContextStub(
            [
                _cookie("sid", "host", domain="localhost", same_site="Lax"),
                _cookie("sid", "elsewhere", domain=".prowl.test", same_site="Strict"),
            ],
        )
        client = self.make_client(context=context)

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(other.base_url + "/page", deadline_seconds=10)
        self.assertEqual([entry["value"] for entry in context.held], ["host", "elsewhere"])
        self.assertEqual(context.added, [])
        self.assertEqual(context.cleared, [])

    async def test_overlapping_native_cookie_scopes_are_refused_before_network(self) -> None:
        server = await self.start(_routes({"/": (200, [], b"ok")}))
        context = _ContextStub(
            [
                _cookie("sid", "host", domain="localhost"),
                _cookie("sid", "domain", domain=".localhost"),
            ]
        )
        client = self.make_client(context=context)
        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/", deadline_seconds=10)
        self.assertEqual(server.received, [])
        self.assertEqual(context.added, [])
        self.assertEqual(context.cleared, [])

    async def test_cookie_named_partitioned_is_not_a_partitioned_attribute(self) -> None:
        server = await self.start(
            _routes(
                {
                    "/": (200, [("Set-Cookie", "partitioned=ordinary; Path=/")], b"ok"),
                }
            )
        )
        result = await self.make_client().fetch(server.base_url + "/", deadline_seconds=10)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.context.get("partitioned")["value"], "ordinary")

    async def test_document_scope_sends_caller_headers_only_on_the_initial_request(self) -> None:
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/end")], b""),
            "/end": (200, [], b"end"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        await client.fetch(
            server.base_url + "/start",
            headers={"X-Api-Key": "k"},
            header_scope=HEADER_SCOPE_DOCUMENT,
            deadline_seconds=10,
        )

        self.assertEqual(server.received[0].headers.get("x-api-key"), "k")
        self.assertNotIn("x-api-key", server.received[1].headers)

    async def test_origin_scope_drops_caller_headers_on_a_cross_origin_hop(self) -> None:
        first = await self.start(_routes({}), host=_HOST)
        second = await self.start(_routes({"/end": (200, [], b"final")}), host="localhost")
        routes: dict[str, _Reply] = {
            "/start": (302, [("Location", "/mid")], b""),
            "/mid": (302, [("Location", f"{second.base_url}/end")], b""),
        }
        server = await self.start(_routes(routes), host=_HOST)
        client = self.make_client()

        result = await client.fetch(
            server.base_url + "/start",
            headers={"X-Api-Key": "k"},
            header_scope=HEADER_SCOPE_ORIGIN,
            deadline_seconds=10,
        )

        self.assertEqual(result.body, "final")
        self.assertEqual(
            [(request.target, request.headers.get("x-api-key")) for request in server.received],
            [("/start", "k"), ("/mid", "k")],
        )
        self.assertIsNone(second.received[0].headers.get("x-api-key"))
        self.assertEqual(first.received, [])

    async def test_post_303_becomes_get_without_a_body(self) -> None:
        routes: dict[str, _Reply] = {
            "/submit": (303, [("Location", "/done")], b""),
            "/done": (200, [], b"done"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/submit", method="POST", content="a=1", deadline_seconds=10)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            [(request.method, request.body) for request in server.received],
            [("POST", b"a=1"), ("GET", b"")],
        )

    async def test_post_307_keeps_the_method_and_body(self) -> None:
        routes: dict[str, _Reply] = {
            "/submit": (307, [("Location", "/done")], b""),
            "/done": (200, [], b"done"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        await client.fetch(server.base_url + "/submit", method="POST", content="a=1", deadline_seconds=10)

        self.assertEqual(
            [(request.method, request.body) for request in server.received],
            [("POST", b"a=1"), ("POST", b"a=1")],
        )

    async def test_redirect_loop_is_bounded(self) -> None:
        server = await self.start(_routes({"/loop": (302, [("Location", "/loop")], b"")}))
        client = self.make_client()

        with self.assertRaises(HttpNetworkError) as caught:
            await client.fetch(server.base_url + "/loop", deadline_seconds=10)

        self.assertIn("redirects", str(caught.exception))
        self.assertEqual(len(server.received), 21)

    async def test_a_relative_location_is_resolved_against_the_hop(self) -> None:
        routes: dict[str, _Reply] = {
            "/deep/start": (302, [("Location", "next")], b""),
            "/deep/next": (200, [], b"resolved"),
        }
        server = await self.start(_routes(routes))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/deep/start", deadline_seconds=10)

        self.assertEqual(result.body, "resolved")
        self.assertEqual(result.url, server.base_url + "/deep/next")

    async def test_redirect_to_another_scheme_is_refused(self) -> None:
        server = await self.start(_routes({"/start": (302, [("Location", "ftp://127.0.0.1/x")], b"")}))
        client = self.make_client()

        with self.assertRaises(HttpNetworkError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)

    async def test_a_matching_strict_cookie_blocks_a_cross_host_redirect(self) -> None:
        context = _ContextStub([_cookie("sid", "1", domain="localhost", same_site="Strict")])
        other = await self.start(_routes({"/end": (200, [], b"end")}), host="localhost")
        server = await self.start(_routes({"/start": (302, [("Location", f"{other.base_url}/end")], b"")}))
        client = self.make_client(context=context)

        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual(other.received, [])

    async def test_an_unrelated_strict_cookie_does_not_block_a_cross_host_redirect(self) -> None:
        context = _ContextStub([_cookie("sid", "1", domain="elsewhere.prowl.test", same_site="Strict")])
        other = await self.start(_routes({"/end": (200, [], b"end")}), host="localhost")
        server = await self.start(_routes({"/start": (302, [("Location", f"{other.base_url}/end")], b"")}))
        client = self.make_client(context=context)

        result = await client.fetch(server.base_url + "/start", deadline_seconds=10)

        self.assertEqual(result.body, "end")
        self.assertEqual(len(other.received), 1)


class RedirectBoundaryTests(_TransportTestCase):
    async def test_cross_site_round_trip_does_not_send_a_strict_cookie_back(self) -> None:
        routes: dict[str, _Reply] = {}
        server = await self.start(_routes(routes), host="localhost")
        routes.update(
            {
                "/start": (302, [("Location", server.base_url.replace("localhost", "127.0.0.1") + "/middle")], b""),
                "/middle": (302, [("Location", server.base_url + "/end")], b""),
                "/end": (200, [], b"end"),
            }
        )
        context = _ContextStub([_cookie("sid", "native", domain="localhost", same_site="Strict")])
        client = self.make_client(context=context)
        with self.assertRaises(UnsupportedCookieError):
            await client.fetch(server.base_url + "/start", deadline_seconds=10)
        self.assertEqual([request.target for request in server.received], ["/start", "/middle"])

    async def test_deleting_another_scope_does_not_delete_native_host_cookie(self) -> None:
        server = await self.start(
            _routes(
                {
                    "/": (200, [("Set-Cookie", "sid=; Domain=localhost; Path=/; Max-Age=0")], b"ok"),
                }
            ),
            host="localhost",
        )
        context = _ContextStub([_cookie("sid", "native", domain="localhost", same_site="Strict")])
        client = self.make_client(context=context)
        await client.fetch(server.base_url + "/", deadline_seconds=10)
        self.assertEqual(context.held[0]["value"], "native")
        self.assertEqual(context.cleared, [])

    async def test_extended_secure_attribute_does_not_make_cookie_secure(self) -> None:
        server = await self.start(
            _routes(
                {
                    "/": (200, [("Set-Cookie", "sid=1; SameSite=None; secure*=utf-8''yes")], b"ok"),
                }
            )
        )
        with self.assertRaises(UnsupportedCookieError):
            await self.make_client().fetch(server.base_url + "/", deadline_seconds=10)
        self.assertEqual(self.context.added, [])
