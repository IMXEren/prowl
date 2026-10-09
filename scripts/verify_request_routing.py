r"""Explicit real-browser probe for the HTTP/browser request routing against a real local browser.

This is not part of the pytest suite. Run it explicitly on a machine that already has the
CloakBrowser binary and a display (or an X server):

    .\.venv\Scripts\python.exe -m scripts.verify_request_routing

It reuses the isolated-session probe's prerequisites, offline launch wrapper, ephemeral
certificate helpers and cleanup, then drives the real :class:`~prowl.service.backend.BrowserBackend`
in its ``http``, ``browser`` and ``auto`` modes. Everything the probe controls is loopback: two
HTTPS servers on 127.0.0.1 and 127.0.0.2, a fresh ``TemporaryDirectory`` profile, and the already
installed binary. It downloads nothing and adds no synthetic locale, timezone, user agent or proxy.

The HTTP fast path is a real ``curl_cffi`` session reaching the probe's self-signed loopback
certificate. Rather than weaken TLS, a script-local factory patches
``prowl.service.http_transport.AsyncSession`` so the request verifies against the generated
certificate; the production transport keeps ``verify=True`` and is never altered.

The probe is honest about what it cannot fake: the language/client-hint parity check compares the
headers the HTTP client actually sent on the wire with the headers the native browser actually
sent, so a default English persona whose Accept-Language the HTTP identity cannot reproduce is
reported as a failure instead of being papered over.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock
from urllib.parse import parse_qs

from prowl.browser import Browser, BrowserConfig
from prowl.service import http_transport
from prowl.service.backend import BrowserBackend, FetchRequest
from prowl.service.classification import SUCCESS
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from prowl.service.sessions import ISOLATED_MODE, SHARED_MODE

# The probe reuses the isolated-session probe's seams rather than duplicating them.
from scripts.verify_isolated_sessions import (
    ProbePrerequisiteError,
    _best_effort,
    _cleanup_servers,
    _cookie_present,
    _offline_launch_patches,
    _Probe,
    _resolve_binary_path,
    _Result,
    _run_checks_sequence,
    _write_ephemeral_tls,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence

#: The two loopback hosts that act as distinct sites for the routing probe.
_SITE_A_HOST = "127.0.0.1"
_SITE_B_HOST = "127.0.0.2"

_ISOLATED_ONE = "routing-iso-1"
_ISOLATED_TWO = "routing-iso-2"
_SHARED_ONE = "routing-shared-1"
_SHARED_TWO = "routing-shared-2"

_INDEX_PATH = "/"
_JS_PATH = "/js"
_SET_PATH = "/set-cookie"
_DELETE_PATH = "/delete-cookie"
_REPORT_PATH = "/cookie-report"
_ECHO_PATH = "/echo"
_POST_PATH = "/post"

#: The cookie the JS-required page writes, and the one the HTTP fast path sets and deletes.
_JS_COOKIE = "jsmark"
_ROUTING_COOKIE = "routing"

_AUTO_POST_BODY = "auto-payload"
_HTTP_POST_BODY = "http-payload"

#: The statuses the HTTP server answers every probe request with.
_OK = 200

_NOT_FOUND = "<!doctype html><html><body>not found</body></html>"

#: The text the JS-required page's inline script writes, which only a rendering browser produces.
_JS_RENDER_TEXT = "rendered by script"


class _ProbePage:
    """A page whose empty root is replaced by an inline script, with a JavaScript requirement."""

    INDEX = "<!doctype html><html><body><h1>routing probe</h1></body></html>"

    SET = "<!doctype html><html><body><h1>cookie set</h1></body></html>"

    POST = "<!doctype html><html><body><h1>posted</h1></body></html>"

    #: The HTTP fast path classifies this as JAVASCRIPT_REQUIRED; a browser renders the root text.
    JS_REQUIRED = (
        "<!doctype html><html><head></head><body>"
        "<noscript>enable JavaScript</noscript>"
        '<div id="root"></div>'
        "<script>"
        f"document.cookie = '{_JS_COOKIE}=1; path=/';"
        "document.getElementById('root').textContent = "
        f"'{_JS_RENDER_TEXT}';"
        "</script>"
        "</body></html>"
    )


# -- Loopback servers --------------------------------------------------------------


@dataclass(slots=True)
class _SiteState:
    """Pages one loopback site serves and the requests it observed."""

    pages: dict[str, str] = field(default_factory=dict)
    response_headers: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    #: The complete header set of the last request per ``source`` query value.
    echo: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Every POST body the site received, in arrival order.
    post_bodies: list[str] = field(default_factory=list)


def _handler_class(state: _SiteState) -> type[BaseHTTPRequestHandler]:
    """Return a request handler for *state* that serves pages, records echoes and POST bodies."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            params = parse_qs(query)
            if path == _ECHO_PATH:
                source = params.get("source", [""])[0]
                state.echo[source] = {name.lower(): value for name, value in self.headers.items()}
                self._respond("ok", "text/plain; charset=utf-8")
                return
            if path == _REPORT_PATH:
                name = params.get("name", [""])[0]
                seen = _cookie_present(self.headers.get("Cookie", ""), name)
                self._respond(
                    f'<!doctype html><html><body data-cookie-present="{int(seen)}">cookie report</body></html>',
                    "text/html; charset=utf-8",
                )
                return
            self._respond(
                state.pages.get(path, _NOT_FOUND),
                "text/html; charset=utf-8",
                state.response_headers.get(path, ()),
            )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
            state.post_bodies.append(body)
            self._respond(state.pages.get(_POST_PATH, _NOT_FOUND), "text/html; charset=utf-8")

        def _respond(
            self,
            body: str,
            content_type: str,
            extra: Sequence[tuple[str, str]] = (),
        ) -> None:
            data = body.encode("utf-8")
            self.send_response(_OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args: Any, **_kwargs: Any) -> None:
            """Silence the default access log."""

    return _Handler


def _server_tls(cert_path: Path, key_path: Path) -> ssl.SSLContext:
    """Return a server TLS context over the same certificate pair the HTTP client pins."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


def _serve(host: str, tls: ssl.SSLContext) -> tuple[ThreadingHTTPServer, _SiteState, str]:
    """Start a loopback HTTPS server on *host* and return it with its state and base URL."""
    state = _SiteState()
    server = ThreadingHTTPServer((host, 0), _handler_class(state))
    server.daemon_threads = True
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, state, f"https://{host}:{port}"


def _fill_pages(state_a: _SiteState, state_b: _SiteState) -> None:
    """Populate both sites' page maps and response headers once both ports are known."""
    state_a.pages[_INDEX_PATH] = _ProbePage.INDEX
    state_a.pages[_JS_PATH] = _ProbePage.JS_REQUIRED
    state_a.pages[_SET_PATH] = _ProbePage.SET
    state_a.pages[_DELETE_PATH] = _ProbePage.INDEX
    state_a.pages[_POST_PATH] = _ProbePage.POST
    state_a.response_headers[_SET_PATH] = (("Set-Cookie", f"{_ROUTING_COOKIE}=1; path=/"),)
    state_a.response_headers[_DELETE_PATH] = (("Set-Cookie", f"{_ROUTING_COOKIE}=; Max-Age=0; path=/"),)
    state_b.pages[_INDEX_PATH] = _ProbePage.INDEX


# -- Script-local seams ------------------------------------------------------------


@contextmanager
def _pinned_tls_patches(cert_path: Path) -> Iterator[None]:
    """Verify the HTTP fast path's TLS against *cert_path* without altering the transport.

    A caller's ``verify`` is replaced with the generated certificate for the duration, so the
    real session still verifies - against the probe's certificate rather than the public roots.
    The production ``HttpClient`` continues to construct its session with ``verify=True``.
    """
    real = http_transport.AsyncSession

    def factory(**options: Any) -> Any:
        options["verify"] = str(cert_path)
        return real(**options)

    with mock.patch.object(http_transport, "AsyncSession", factory):
        yield


@dataclass(slots=True)
class _Counters:
    """Forwarding counters for the two seams whose call counts the probe asserts on."""

    #: Every tab group ``Browser.create`` returned, in order.
    creates: list[Any] = field(default_factory=list)
    #: Every ``HttpClient`` instance that ran a fetch, in order.
    http_fetches: list[Any] = field(default_factory=list)


@contextmanager
def _counted_seams(counters: _Counters) -> Iterator[None]:
    """Count ``Browser.create`` groups and ``HttpClient.fetch`` clients while forwarding to them."""
    real_create = Browser.create
    real_fetch = http_transport.HttpClient.fetch

    async def counting_create(*args: Any, **kwargs: Any) -> Any:
        group = await real_create(*args, **kwargs)
        counters.creates.append(group)
        return group

    async def counting_fetch(self: Any, *args: Any, **kwargs: Any) -> Any:
        counters.http_fetches.append(self)
        return await real_fetch(self, *args, **kwargs)

    with (
        mock.patch.object(Browser, "create", counting_create),
        mock.patch.object(http_transport.HttpClient, "fetch", counting_fetch),
    ):
        yield


# -- Checks ------------------------------------------------------------------------


def _cookie_reported(body: str) -> bool:
    return 'data-cookie-present="1"' in body


def _client_for(counters: _Counters, context: Any) -> Any:
    """Return the HTTP client that last served *context*, or ``None`` when none did."""
    for client in reversed(counters.http_fetches):
        if getattr(client, "_context", None) is context:
            return client
    return None


async def _check_http_mode(probe: _Probe, counters: _Counters, backend: BrowserBackend, url: str) -> None:
    """An ordinary HTTP GET is answered from the fast path and creates no tab group."""
    before = len(counters.creates)
    result = await backend.fetch(None, FetchRequest(url=url, mode=HTTP_MODE))
    created = len(counters.creates) - before
    probe.record(
        _Result(
            "http mode answers without creating a tab group",
            ok=result.mode == HTTP_MODE and created == 0,
            detail=f"mode={result.mode} tab_groups={created} status={result.status_code}",
        ),
    )


async def _check_browser_mode(probe: _Probe, counters: _Counters, backend: BrowserBackend, url: str) -> None:
    """An explicit browser fetch never touches the HTTP client and keeps the legacy result shape."""
    before = len(counters.http_fetches)
    result = await backend.fetch(None, FetchRequest(url=url, mode=BROWSER_MODE))
    http_calls = len(counters.http_fetches) - before
    probe.record(
        _Result(
            "browser mode bypasses the HTTP client and carries no execution mode",
            ok=http_calls == 0 and result.mode is None and result.status_code == _OK,
            detail=f"http_client_calls={http_calls} mode={result.mode!r} status={result.status_code}",
        ),
    )


async def _check_auto_escalation(probe: _Probe, counters: _Counters, backend: BrowserBackend, a_url: str) -> None:
    """Auto tries HTTP, escalates a JS-required shell to the browser, and the browser cookie persists."""
    before_http = len(counters.http_fetches)
    before_creates = len(counters.creates)
    result = await backend.fetch(None, FetchRequest(url=f"{a_url}{_JS_PATH}", mode=AUTO_MODE))
    http_calls = len(counters.http_fetches) - before_http
    groups = len(counters.creates) - before_creates
    category = result.classification.category if result.classification is not None else None
    seen = await backend.fetch(
        None,
        FetchRequest(url=f"{a_url}{_REPORT_PATH}?name={_JS_COOKIE}", mode=HTTP_MODE),
    )
    ok = (
        http_calls == 1
        and groups == 1
        and result.mode == BROWSER_MODE
        and category == SUCCESS
        and _JS_RENDER_TEXT in result.response
        and _cookie_reported(seen.response)
    )
    probe.record(
        _Result(
            "auto escalates a JS shell to the browser and the browser cookie reaches the next HTTP request",
            ok=ok,
            detail=(
                f"http_calls={http_calls} groups={groups} mode={result.mode} category={category} "
                f"rendered={_JS_RENDER_TEXT in result.response} next_http_saw_cookie={seen.response.strip()!r}"
            ),
        ),
    )


async def _check_cookie_mirroring(
    probe: _Probe,
    backend: BrowserBackend,
    handle: Any,
    a_url: str,
) -> None:
    """An HTTP Set-Cookie lands in the native context, and Max-Age=0 removes it again."""
    await backend.fetch(None, FetchRequest(url=f"{a_url}{_SET_PATH}", mode=HTTP_MODE))
    after_set = await handle.context.cookies()
    native_set = await backend.fetch(
        None,
        FetchRequest(url=f"{a_url}{_REPORT_PATH}?name={_ROUTING_COOKIE}", mode=BROWSER_MODE),
    )
    await backend.fetch(None, FetchRequest(url=f"{a_url}{_DELETE_PATH}", mode=HTTP_MODE))
    after_delete = await handle.context.cookies()
    native_delete = await backend.fetch(
        None,
        FetchRequest(url=f"{a_url}{_REPORT_PATH}?name={_ROUTING_COOKIE}", mode=BROWSER_MODE),
    )
    present = any(cookie.get("name") == _ROUTING_COOKIE for cookie in after_set)
    absent = not any(cookie.get("name") == _ROUTING_COOKIE for cookie in after_delete)
    probe.record(
        _Result(
            "an HTTP Set-Cookie reaches the native context and Max-Age=0 removes it",
            ok=present
            and _cookie_reported(native_set.response)
            and absent
            and not _cookie_reported(native_delete.response),
            detail=(
                f"context_present={present} native_present={native_set.response.strip()!r} "
                f"context_absent={absent} native_absent={native_delete.response.strip()!r}"
            ),
        ),
    )


async def _check_isolation(probe: _Probe, counters: _Counters, backend: BrowserBackend, a_url: str) -> None:
    """Two isolated contexts share no cookie or client, and a shared id adds no context or client."""
    before = len(counters.http_fetches)
    await backend.fetch(
        _ISOLATED_ONE,
        FetchRequest(
            url=f"{a_url}{_SET_PATH}",
            mode=HTTP_MODE,
            session_id=_ISOLATED_ONE,
            session_mode=ISOLATED_MODE,
        ),
    )
    iso_one_client = counters.http_fetches[before]
    before = len(counters.http_fetches)
    seen = await backend.fetch(
        _ISOLATED_TWO,
        FetchRequest(
            url=f"{a_url}{_REPORT_PATH}?name={_ROUTING_COOKIE}",
            mode=HTTP_MODE,
            session_id=_ISOLATED_TWO,
            session_mode=ISOLATED_MODE,
        ),
    )
    iso_two_client = counters.http_fetches[before]

    main_handle = await Browser.get_context(None)
    before = len(counters.http_fetches)
    shared_one = await backend.fetch(
        _SHARED_ONE,
        FetchRequest(
            url=f"{a_url}{_REPORT_PATH}?name={_ROUTING_COOKIE}",
            mode=HTTP_MODE,
            session_id=_SHARED_ONE,
            session_mode=SHARED_MODE,
        ),
    )
    shared_one_client = counters.http_fetches[before]
    before = len(counters.http_fetches)
    await backend.fetch(
        _SHARED_TWO,
        FetchRequest(
            url=f"{a_url}{_REPORT_PATH}?name={_ROUTING_COOKIE}",
            mode=HTTP_MODE,
            session_id=_SHARED_TWO,
            session_mode=SHARED_MODE,
        ),
    )
    shared_two_client = counters.http_fetches[before]

    iso_one_context = (await Browser.get_context(_ISOLATED_ONE)).context
    iso_two_context = (await Browser.get_context(_ISOLATED_TWO)).context
    ok = (
        not _cookie_reported(seen.response)
        and not _cookie_reported(shared_one.response)
        and iso_one_client is not iso_two_client
        and iso_one_client._context is not iso_two_client._context  # noqa: SLF001
        and iso_one_context is not iso_two_context
        and shared_one_client is shared_two_client
        and shared_one_client._context is main_handle.context  # noqa: SLF001
        and iso_one_client._context is iso_one_context  # noqa: SLF001
        and iso_two_client._context is iso_two_context  # noqa: SLF001
    )
    probe.record(
        _Result(
            "isolated contexts share no cookie or client and a shared id adds neither",
            ok=ok,
            detail=(
                f"iso2_saw_isolated1_cookie={seen.response.strip()!r} "
                f"distinct_clients={iso_one_client is not iso_two_client} "
                f"distinct_contexts={iso_one_context is not iso_two_context} "
                f"shared_reused_client={shared_one_client is shared_two_client} "
                f"shared_client_on_main={shared_one_client._context is main_handle.context}"  # noqa: SLF001
            ),
        ),
    )


async def _check_post(
    probe: _Probe,
    counters: _Counters,
    backend: BrowserBackend,
    state_a: _SiteState,
    a_url: str,
) -> None:
    """Auto POST produces exactly one server effect and skips HTTP; explicit HTTP POST keeps its body."""
    before_http = len(counters.http_fetches)
    before_posts = len(state_a.post_bodies)
    auto = await backend.fetch(
        None,
        FetchRequest(
            url=f"{a_url}{_POST_PATH}",
            method="POST",
            post_data=_AUTO_POST_BODY,
            mode=AUTO_MODE,
        ),
    )
    auto_http = len(counters.http_fetches) - before_http
    auto_posts = state_a.post_bodies[before_posts:]

    before_http = len(counters.http_fetches)
    before_posts = len(state_a.post_bodies)
    explicit = await backend.fetch(
        None,
        FetchRequest(
            url=f"{a_url}{_POST_PATH}",
            method="POST",
            post_data=_HTTP_POST_BODY,
            mode=HTTP_MODE,
        ),
    )
    http_http = len(counters.http_fetches) - before_http
    http_posts = state_a.post_bodies[before_posts:]

    ok = (
        auto.mode == BROWSER_MODE
        and auto_http == 0
        and auto_posts == [_AUTO_POST_BODY]
        and explicit.mode == HTTP_MODE
        and http_http == 1
        and http_posts == [_HTTP_POST_BODY]
    )
    probe.record(
        _Result(
            "auto POST sends one browser effect and explicit HTTP POST preserves its body",
            ok=ok,
            detail=(
                f"auto_mode={auto.mode} auto_http_calls={auto_http} auto_bodies={auto_posts!r} "
                f"http_mode={explicit.mode} http_http_calls={http_http} http_bodies={http_posts!r}"
            ),
        ),
    )


async def _check_identity_parity(
    probe: _Probe,
    backend: BrowserBackend,
    state_b: _SiteState,
    b_url: str,
) -> None:
    """The HTTP identity's UA, client hints and Accept-Language match the native request on the wire."""
    await backend.fetch(None, FetchRequest(url=f"{b_url}{_ECHO_PATH}?source=native", mode=BROWSER_MODE))
    await backend.fetch(None, FetchRequest(url=f"{b_url}{_ECHO_PATH}?source=curl", mode=HTTP_MODE))
    native = state_b.echo.get("native", {})
    curl = state_b.echo.get("curl", {})
    compared = {
        "user-agent": (native.get("user-agent"), curl.get("user-agent")),
        "accept-language": (native.get("accept-language"), curl.get("accept-language")),
        "sec-ch-ua": (native.get("sec-ch-ua"), curl.get("sec-ch-ua")),
        "sec-ch-ua-mobile": (native.get("sec-ch-ua-mobile"), curl.get("sec-ch-ua-mobile")),
        "sec-ch-ua-platform": (native.get("sec-ch-ua-platform"), curl.get("sec-ch-ua-platform")),
    }
    mismatched = {
        name: {"native": pair[0], "curl": pair[1]}
        for name, pair in compared.items()
        if not pair[0] or pair[0] != pair[1]
    }
    probe.record(
        _Result(
            "the HTTP identity's UA, client hints and Accept-Language match the native request",
            ok=not mismatched,
            detail="parity held on every compared header" if not mismatched else f"mismatch: {mismatched}",
        ),
    )


async def _check_isolated_close(
    probe: _Probe,
    counters: _Counters,
    backend: BrowserBackend,
    handles: Mapping[str, Any],
) -> None:
    """Closing one isolated session closes its native context and HTTP client, and nothing else."""
    iso_one = handles["iso-1"]
    iso_two = handles["iso-2"]
    iso_one_client = _client_for(counters, iso_one.context)
    await backend.close_session(_ISOLATED_ONE)
    context_closed = iso_one.context.is_closed()
    client_closed = iso_one_client is not None and iso_one_client._closed  # noqa: SLF001
    other_open = not iso_two.context.is_closed()
    released = iso_one.context not in backend._http._states  # noqa: SLF001 - the ownership map is the point
    probe.record(
        _Result(
            "closing an isolated session closes its context and HTTP client only",
            ok=context_closed and client_closed and other_open and released,
            detail=(
                f"context_closed={context_closed} client_closed={client_closed} "
                f"other_context_open={other_open} client_released={released}"
            ),
        ),
    )


async def _check_metrics(
    probe: _Probe, counters: _Counters, backend: BrowserBackend, handles: Mapping[str, Any]
) -> None:
    observed_contexts = sum(not handle.context.is_closed() for handle in handles.values())
    first = backend.render_metrics()
    stable = first == backend.render_metrics()
    metrics = backend.metrics
    ok = (
        stable
        and metrics.requests_total > 0
        and metrics.requests_active == 0
        and metrics.http_fastpath_total == len(counters.http_fetches)
        and metrics.http_fastpath_success_total > 0
        and metrics.browser_escalations_total > 0
        and metrics.context_count == observed_contexts
        and metrics.context_created_total >= len(handles)
        and metrics.context_evicted_total == 0
        and metrics.tabgroups_active == 0
        and metrics.request_duration_seconds.count == metrics.requests_total
        and metrics.request_duration_seconds.sum > 0
        and metrics.browser_acquire_seconds.count == len(counters.creates)
        and metrics.browser_acquire_seconds.sum > 0
    )
    probe.record(
        _Result(
            "live routing metrics match transport/group observations",
            ok=ok,
            detail=(
                f"requests={metrics.requests_total} http_attempts={metrics.http_fastpath_total} "
                f"escalations={metrics.browser_escalations_total} contexts={metrics.context_count} "
                f"created={metrics.context_created_total} groups={metrics.tabgroups_active} stable={stable}"
            ),
        ),
    )


# -- Orchestration -----------------------------------------------------------------


async def _finalize_backend(probe: _Probe, counters: _Counters, backend: BrowserBackend) -> None:
    """Shut the backend down and confirm its HTTP clients and native groups are gone."""
    if not await _best_effort("backend shutdown", backend.aclose):
        probe.record(_Result("backend shutdown", ok=False, detail="see cleanup errors above"))
        return
    states = backend._http._states  # noqa: SLF001 - the ownership map is the point
    clients_closed = all(client._closed for client in counters.http_fetches)  # noqa: SLF001
    page_map = Browser._runtime.target_to_page_map  # noqa: SLF001 - the runtime page map is the point
    groups_gone = all(group.target_id not in page_map for group in counters.creates)
    running = Browser.is_running()
    probe.record(
        _Result(
            "HTTP clients and native tab groups are gone after backend shutdown",
            ok=not states and clients_closed and groups_gone and not running,
            detail=(
                f"http_client_states={len(states)} clients_closed={clients_closed} "
                f"native_groups_gone={groups_gone} browser_running={running}"
            ),
        ),
    )


async def _run_checks(
    probe: _Probe,
    counters: _Counters,
    backend: BrowserBackend,
    site_a: tuple[_SiteState, str],
    site_b: tuple[_SiteState, str],
) -> None:
    """Start the browser, then run every check in order against the real backend."""
    state_a, a_url = site_a
    state_b, b_url = site_b
    await backend.start()
    handles = {
        "shared": await Browser.get_context(None),
        "iso-1": await Browser.get_context(_ISOLATED_ONE),
        "iso-2": await Browser.get_context(_ISOLATED_TWO),
    }
    checks: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("http mode", partial(_check_http_mode, probe, counters, backend, f"{a_url}{_INDEX_PATH}")),
        ("browser mode", partial(_check_browser_mode, probe, counters, backend, f"{b_url}{_INDEX_PATH}")),
        ("auto escalation", partial(_check_auto_escalation, probe, counters, backend, a_url)),
        ("cookie mirroring", partial(_check_cookie_mirroring, probe, backend, handles["shared"], a_url)),
        ("context isolation", partial(_check_isolation, probe, counters, backend, a_url)),
        ("post routing", partial(_check_post, probe, counters, backend, state_a, a_url)),
        ("identity parity", partial(_check_identity_parity, probe, backend, state_b, b_url)),
        ("isolated close", partial(_check_isolated_close, probe, counters, backend, handles)),
        ("routing metrics", partial(_check_metrics, probe, counters, backend, handles)),
    ]
    await _run_checks_sequence(probe, checks)


async def _run(directory: Path, binary_path: str, probe: _Probe) -> None:
    """Bring up the loopback servers and the browser, run every check, and clean both up."""
    servers: list[ThreadingHTTPServer] = []
    counters = _Counters()
    backend: BrowserBackend | None = None
    try:
        cert_path, key_path = _write_ephemeral_tls(directory)
        tls = _server_tls(cert_path, key_path)
        site_a, state_a, a_url = _serve(_SITE_A_HOST, tls)
        servers.append(site_a)
        site_b, state_b, b_url = _serve(_SITE_B_HOST, tls)
        servers.append(site_b)
        _fill_pages(state_a, state_b)
        with (
            mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary_path}),
            _offline_launch_patches(),
            _pinned_tls_patches(cert_path),
            _counted_seams(counters),
        ):
            config = BrowserConfig(
                proxy_url=None,
                profile_dir=str(directory / "profile"),
                profile_archive=str(directory / "profile.zip"),
                extensions_dir=None,
                policy_dir=None,
            )
            backend = BrowserBackend(browser_config=config)
            await _run_checks(probe, counters, backend, (state_a, a_url), (state_b, b_url))
    finally:
        if backend is not None and not await _best_effort(
            "routing probe cleanup",
            partial(_finalize_backend, probe, counters, backend),
        ):
            probe.record(_Result("routing probe cleanup", ok=False, detail="see cleanup errors above"))
        if not _cleanup_servers(*servers):
            probe.record(_Result("server cleanup", ok=False, detail="see cleanup errors above"))


def main() -> int:
    """Run the probe, print each check as it completes, and return a nonzero status on failure.

    :return: ``0`` when every check passed, ``1`` when a check failed or the run aborted, and
        ``2`` when a local prerequisite (binary or dependency) is missing.
    """
    try:
        binary_path = _resolve_binary_path()
    except ProbePrerequisiteError as exc:
        sys.stderr.write(f"[PREREQ-FAIL] {exc}\n")
        sys.stderr.flush()
        return 2

    probe = _Probe()
    try:
        with tempfile.TemporaryDirectory(prefix="prowl-routing-probe-", ignore_cleanup_errors=True) as raw:
            asyncio.run(_run(Path(raw), binary_path, probe))
    except ProbePrerequisiteError as exc:
        sys.stderr.write(f"[PREREQ-FAIL] {exc}\n")
        sys.stderr.flush()
        return 2
    except BaseException as exc:  # noqa: BLE001 - report the abort rather than a traceback only
        sys.stderr.write(f"[ERROR] the probe aborted: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        return 1

    sys.stderr.write(
        f"{probe.total - probe.failures}/{probe.total} checks passed. "
        "This probe covers loopback traffic only and is not a host firewall.\n",
    )
    sys.stderr.flush()
    return 1 if probe.failures else 0


if __name__ == "__main__":
    sys.exit(main())
