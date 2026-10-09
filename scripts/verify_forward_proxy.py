"""Offline proxy verification with a temporary browser profile, origin and signing CA.

Run with ``python -m scripts.verify_forward_proxy``. Checks cover native routing and
cookie continuity, verified CONNECT, authority binding, exact POST effects, isolation
and shutdown. No trust is installed and no external origin or raw tunnel is used.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from unittest import mock
from urllib.parse import SplitResult, urlsplit

import h11
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prowl.browser import Browser, BrowserConfig
from prowl.browser.proxy.certificates import ProxyCertificates
from prowl.browser.proxy.http1 import ProxyRequest
from prowl.browser.proxy.response import ProxyResponse
from prowl.browser.proxy.server import ProxyServer
from prowl.service import ServiceConfig, create_app
from prowl.service import app as app_module
from prowl.service.app import _SERVICE_KEY
from prowl.service.backend import BrowserBackend
from prowl.service.sessions import ISOLATED_MODE
from scripts.verify_isolated_sessions import (
    _cleanup_servers,
    _offline_launch_patches,
    _resolve_binary_path,
    _write_ephemeral_tls,
)
from scripts.verify_request_routing import (
    _JS_RENDER_TEXT,
    _handler_class,
    _pinned_tls_patches,
    _ProbePage,
    _serve,
    _server_tls,
    _SiteState,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from http.server import BaseHTTPRequestHandler

EXCHANGE_TIMEOUT_SECONDS = 30.0
CLOSE_TIMEOUT_SECONDS = 5.0
OK_STATUS = 200
BAD_REQUEST = 400
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
_SNAPSHOT_PAGE = """<!DOCTYPE html><html><head><title>Snapshot</title></head><body>
<script>setTimeout(() => { document.body.dataset.snapshot = 'ready';
document.body.append('Late capture ready'); document.cookie = 'latecapture=ready; Path=/'; }, 5000);</script>
</body></html>"""
_POST_SNAPSHOT_PAGE = _SNAPSHOT_PAGE.replace("latecapture", "postcapture").replace("dataset.snapshot", "dataset.post")
_MEDIA_PAGE = """<!DOCTYPE html><html><head><title>Media probe</title>
<link rel="stylesheet" href="/probe-style?case=CASE"></head><body><h1>Media probe</h1>
<img src="/probe-image?case=CASE"><script>
new FontFace('probe', 'url("/probe-font?case=CASE")').load().catch(() => {});
</script></body></html>"""
_MEDIA_ASSETS = {
    "/probe-image": ("image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'),
    "/probe-style": ("text/css", b"body { background: white; }"),
    "/probe-font": ("font/woff", b"font-request-probe"),
}

#: A loopback address that is never the connected CONNECT authority, for the authority check.
_WRONG_AUTHORITY_HOST = "127.0.0.2"


class _ExchangeError(RuntimeError):
    """The proxy exchange did not complete within its bound."""


class _ProfileFenceError(RuntimeError):
    """The started browser does not own the declared probe profile directory."""


def _assert_profile_owned(webdata_path: Path | str, expected_profile: Path | str) -> None:
    """Fail before any request if the browser did not open the declared profile."""
    actual = Path(webdata_path).resolve()
    expected = Path(expected_profile).resolve()
    if actual != expected:
        msg = "browser profile fence: webdata path does not match the declared probe profile"
        raise _ProfileFenceError(msg)


def _header(response: ProxyResponse, name: str) -> str | None:
    lowered = name.lower()
    for header_name, value in response.headers:
        if header_name.lower() == lowered:
            return value
    return None


def _first_header(items: Iterable[tuple[bytes, bytes]], name: str) -> str | None:
    """Return the first value of *name* among raw header pairs, or ``None``."""
    wanted = name.lower().encode("ascii")
    for field, value in items:
        if field.lower() == wanted:
            return value.decode("latin-1")
    return None


def _write_proxy_ca(directory: Path) -> tuple[Path, Path]:
    """Write a throwaway ECDSA P-256 certificate authority for the probe's CONNECT tunnels.

    The authority lives under *directory* (a private temporary directory) and is distinct from the
    origin leaf; its key is an unencrypted PKCS#8 PEM. Nothing is installed into any trust store and
    nothing is tracked.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "prowl-forward-proxy-probe-ca")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(minutes=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    ca_path = directory / "proxy-ca.pem"
    key_path = directory / "proxy-ca-key.pem"
    ca_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return ca_path, key_path


async def _request(
    port: int,
    url: str,
    *,
    cookie_header: str | None = None,
    ca_cert: Path | None = None,
    request: ProxyRequest | None = None,
) -> ProxyResponse:
    """Send one GET or raw POST, optionally inside verified CONNECT."""
    parts = urlsplit(url)
    wire_target = request.target if request is not None else (_origin_form(parts) if ca_cert is not None else url)
    body = request.body if request is not None and request.method == "POST" else None
    headers = [(b"host", parts.netloc.encode("ascii"))]
    if request is not None:
        headers.extend((name.encode("ascii"), value.encode("latin-1")) for name, value in request.headers)
    if cookie_header is not None:
        headers.append((b"cookie", cookie_header.encode("ascii")))
    if body is not None:
        headers.append((b"content-length", str(len(body)).encode("ascii")))
        if not any(name.lower() == b"content-type" for name, _ in headers):
            headers.append((b"content-type", b"application/octet-stream"))
    method = request.method.encode("ascii") if request is not None else b"GET"

    async def _exchange() -> ProxyResponse:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            request_id = None
            if ca_cert is not None:
                request_id = await _upgrade_connect(reader, writer, parts, ca_cert)
            connection = h11.Connection(our_role=h11.CLIENT)
            writer.write(
                connection.send(h11.Request(method=method, target=wire_target.encode("ascii"), headers=headers))
            )
            if body is not None:
                writer.write(connection.send(h11.Data(data=body)))
            writer.write(connection.send(h11.EndOfMessage()))
            await writer.drain()
            response = await _read_response(reader, connection)
            if request_id is not None:
                _require_request_id(request_id, response)
            return response
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), CLOSE_TIMEOUT_SECONDS)

    return await asyncio.wait_for(_exchange(), EXCHANGE_TIMEOUT_SECONDS)


async def _upgrade_connect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    parts: SplitResult,
    ca_cert: Path,
) -> str:
    """Send CONNECT, verify the 200 handshake, and upgrade the transport; return its request id."""
    hostname = parts.hostname
    if hostname is None:
        msg = "CONNECT target has no hostname"
        raise _ExchangeError(msg)
    authority = parts.netloc.encode("ascii")
    connection = h11.Connection(our_role=h11.CLIENT)
    writer.write(
        connection.send(h11.Request(method=b"CONNECT", target=authority, headers=[(b"host", authority)]))
        + connection.send(h11.EndOfMessage())
    )
    await writer.drain()
    request_id = await _read_handshake(reader, connection)
    context = ssl.create_default_context(cafile=str(ca_cert))
    try:
        await writer.start_tls(context, server_hostname=hostname)
    except BaseException:
        writer.transport.abort()
        raise
    return request_id


async def _read_handshake(reader: asyncio.StreamReader, connection: h11.Connection) -> str:
    """Read the CONNECT 200 head and return its X-Request-ID."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError:
        msg = "CONNECT handshake closed before its response head"
        raise _ExchangeError(msg) from None
    connection.receive_data(head)
    event = connection.next_event()
    if not isinstance(event, h11.Response) or event.status_code != OK_STATUS:
        msg = "CONNECT handshake was not a 200 response"
        raise _ExchangeError(msg)
    ids = [value for name, value in event.headers if name.lower() == b"x-request-id"]
    request_id = _first_header(event.headers.raw_items(), "X-Request-ID")
    if len(ids) != 1 or request_id is None or _REQUEST_ID.fullmatch(request_id) is None:
        msg = "CONNECT handshake carried no X-Request-ID"
        raise _ExchangeError(msg)
    return request_id


def _require_request_id(handshake_id: str, response: ProxyResponse) -> None:
    """Fail unless the inner response carries the same X-Request-ID as the handshake."""
    ids = [value for name, value in response.headers if name.lower() == "x-request-id"]
    if ids != [handshake_id]:
        msg = "inner response X-Request-ID does not match the CONNECT handshake"
        raise _ExchangeError(msg)


def _origin_form(parts: SplitResult) -> str:
    """Return the origin-form target (path plus query) of *parts*, percent escapes preserved."""
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    return target


async def _read_response(reader: asyncio.StreamReader, connection: h11.Connection) -> ProxyResponse:
    status = 0
    headers: list[tuple[str, str]] = []
    body = bytearray()
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            msg = "proxy closed before EndOfMessage"
            raise _ExchangeError(msg)
        connection.receive_data(chunk)
        while True:
            event = connection.next_event()
            if event is h11.NEED_DATA:
                break
            if isinstance(event, h11.Response):
                status = event.status_code
                headers = [(name.decode("ascii"), value.decode("latin-1")) for name, value in event.headers]
            elif isinstance(event, h11.Data):
                body.extend(event.data)
            elif isinstance(event, h11.EndOfMessage):
                if await reader.read(1):
                    msg = "proxy sent data after the final response"
                    raise _ExchangeError(msg)
                return ProxyResponse(status, tuple(headers), bytes(body))


def _emit(text: str, *, stream: TextIO = sys.stdout) -> None:
    """Write one line to *stream*, flushing so a long run shows progress as it goes."""
    print(text, file=stream, flush=True)


def _report(results: list[tuple[str, bool]]) -> int:
    passed = 0
    for name, ok in results:
        _emit(("PASS  " if ok else "FAIL  ") + name)
        passed += int(bool(ok))
    _emit(f"{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


def _binary_handler(base: type[BaseHTTPRequestHandler], observed: list[bytes]) -> type[BaseHTTPRequestHandler]:
    class Handler(base):
        def do_POST(self) -> None:  # noqa: N802 - HTTP server callback name
            observed.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            html = _POST_SNAPSHOT_PAGE if self.path == "/snapshot-post" else _ProbePage.INDEX
            data = html.encode("utf-8")
            self.send_response(OK_STATUS)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


async def _check_posts(port: int, origin_url: str, ca_cert: Path, observed: list[bytes]) -> bool:
    for payload in (b"\x00\xff\x80raw\x00", b""):
        before = len(observed)
        request = ProxyRequest("POST", "/post", (), payload)
        response = await _request(port, origin_url + "/post", ca_cert=ca_cert, request=request)
        if (
            len(observed) != before + 1
            or observed[-1] != payload
            or response.status != OK_STATUS
            or response.body != _ProbePage.INDEX.encode("utf-8")
            or _header(response, "X-Prowl-Representation") != "origin"
        ):
            return False
    return True


def _valid_png(payload: object) -> bool:
    if not isinstance(payload, dict) or not isinstance(solution := payload.get("solution"), dict):
        return False
    screenshot = solution.get("screenshot")
    if not isinstance(screenshot, str):
        return False
    try:
        png = base64.b64decode(screenshot, validate=True)
    except ValueError:
        return False
    return (
        png.startswith(b"\x89PNG\r\n\x1a\n")
        and png[12:16] == b"IHDR"
        and int.from_bytes(png[16:20], "big") > 0
        and int.from_bytes(png[20:24], "big") > 0
    )


async def _check_snapshots(
    service: app_module.Service, backend: BrowserBackend, origin_url: str, observed: list[bytes]
) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    http_before = backend.metrics.http_fastpath_total
    for method, path, marker, cookie in (
        ("get", "/snapshot", 'data-snapshot="ready"', "latecapture"),
        ("post", "/snapshot-post", 'data-post="ready"', "postcapture"),
    ):
        before = len(observed)
        status, payload = await service.handle(
            {
                "cmd": "request." + method,
                "url": origin_url + path,
                "mode": "auto",
                "maxTimeout": 20000,
                "waitInSeconds": 6,
                "returnScreenshot": True,
                **({"postData": "capture=once"} if method == "post" else {}),
            }
        )
        solution = payload.get("solution", {})
        results.append(
            (
                "delayed " + method + " returns final DOM, late cookies and native PNG without replay",
                status == OK_STATUS
                and payload.get("status") == "ok"
                and marker in solution.get("response", "")
                and solution.get("url") == origin_url + path
                and any(
                    item.get("name") == cookie and item.get("value") == "ready" for item in solution.get("cookies", [])
                )
                and _valid_png(payload)
                and backend.metrics.http_fastpath_total == http_before
                and len(observed) == before + int(method == "post")
                and (method != "post" or observed[-1] == b"capture=once"),
            )
        )
    return results


async def _check_isolation(
    isolated_port: int, shared_port: int, origin_url: str, ca_cert: Path, state: _SiteState
) -> bool:
    await _request(isolated_port, origin_url + "/echo?source=isolated", ca_cert=ca_cert, cookie_header="sid=isolated")
    await _request(isolated_port, origin_url + "/echo?source=isolated-next", ca_cert=ca_cert)
    await _request(shared_port, origin_url + "/echo?source=shared-next", ca_cert=ca_cert)
    isolated = state.echo["isolated-next"].get("cookie", "").split("; ")
    shared = state.echo["shared-next"].get("cookie", "").split("; ")
    return (
        "sid=isolated" in isolated
        and "sid=from-client" not in isolated
        and "jsmark=1" not in isolated
        and "sid=from-client" in shared
        and "jsmark=1" in shared
        and "sid=isolated" not in shared
    )


async def _run_probe(
    directory: Path,
    origin_cert: Path,
    origin_url: str,
    state: _SiteState,
    observations: tuple[list[bytes], list[str]],
) -> list[tuple[str, bool]]:
    observed, media_observed = observations
    results: list[tuple[str, bool]] = []
    ca_cert, ca_key = _write_proxy_ca(directory)
    backend = BrowserBackend(
        BrowserConfig(profile_dir=str(directory / "profile"), profile_archive=str(directory / "profile.zip"))
    )
    with (
        _offline_launch_patches(),
        _pinned_tls_patches(origin_cert),
    ):
        app = create_app(
            ServiceConfig(
                host="127.0.0.1",
                forward_proxy_port=0,
                proxy_ca_cert=str(ca_cert),
                proxy_ca_key=str(ca_key),
            ),
            backend,
        )
        secondary: ProxyServer | None = None
        try:
            await app_module._on_startup(app)  # noqa: SLF001 - the probe drives the app's real lifecycle
            _assert_profile_owned(Browser._webdata_path().parent.parent, directory / "profile")  # noqa: SLF001 - the fence reads the native path
            proxy = app[app_module._PROXY_KEY]  # noqa: SLF001 - the listener handle is the point
            port = proxy.port

            index_response = await _request(port, origin_url + "/")
            results.append(
                (
                    "1 plain absolute GET returns origin INDEX bytes",
                    index_response.status == OK_STATUS
                    and index_response.body == _ProbePage.INDEX.encode("utf-8")
                    and _header(index_response, "X-Prowl-Representation") == "origin",
                )
            )

            escalations_before = backend.metrics.browser_escalations_total
            js_response = await _request(port, origin_url + "/js", cookie_header="sid=from-client")
            escalations_after = backend.metrics.browser_escalations_total
            results.append(
                (
                    "2 JS page renders native DOM and escalates once",
                    _header(js_response, "X-Prowl-Representation") == "rendered"
                    and _JS_RENDER_TEXT.encode("utf-8") in js_response.body
                    and escalations_after - escalations_before == 1,
                )
            )

            handle = await Browser.get_context(None)
            native_cookies = await handle.context.cookies([origin_url + "/"])
            client_cookie_present = any(
                cookie.get("name") == "sid" and cookie.get("value") == "from-client" for cookie in native_cookies
            )
            await _request(port, origin_url + "/echo?source=after")
            echo_cookie = state.echo["after"].get("cookie", "")
            results.append(
                (
                    "3 sid from client becomes native + headerless echo sees both cookies",
                    client_cookie_present and "sid=from-client" in echo_cookie and "jsmark" in echo_cookie,
                )
            )

            connect_response = await _request(port, origin_url + "/", ca_cert=ca_cert)
            results.append(
                (
                    "4 verified CONNECT returns origin INDEX bytes under the handshake id",
                    connect_response.status == OK_STATUS
                    and connect_response.body == _ProbePage.INDEX.encode("utf-8")
                    and _header(connect_response, "X-Prowl-Representation") == "origin",
                )
            )

            requests_before = backend.metrics.requests_total
            wrong_target = f"https://{_WRONG_AUTHORITY_HOST}:{urlsplit(origin_url).port}/"
            wrong_response = await _request(
                port, origin_url + "/", ca_cert=ca_cert, request=ProxyRequest("GET", wrong_target, (), b"")
            )
            results.append(
                (
                    "5 wrong-authority absolute HTTPS target is rejected with 400 before any fetch",
                    wrong_response.status == BAD_REQUEST and backend.metrics.requests_total == requests_before,
                )
            )
            escalations_before = backend.metrics.browser_escalations_total
            results.append(
                (
                    "binary and empty POST each have one exact origin effect",
                    await _check_posts(port, origin_url, ca_cert, observed)
                    and backend.metrics.browser_escalations_total == escalations_before,
                )
            )
            state.pages["/snapshot"] = _SNAPSHOT_PAGE
            results.extend(await _check_snapshots(app[_SERVICE_KEY], backend, origin_url, observed))
            results.append(
                (
                    "media resources load before/after filtering and never reach the filtered origin",
                    await _check_media(app[_SERVICE_KEY], origin_url, state, media_observed),
                )
            )
            results.append(
                (
                    "native verification extracts fresh tokens with one navigation and preserves cookie-only capture",
                    await _check_verification(app[_SERVICE_KEY], backend, origin_url, state, media_observed),
                )
            )
            results.append(
                (
                    "rendered diagnostics report success after HTTP shell escalation",
                    await _check_diagnostics(app[_SERVICE_KEY], backend, origin_url, state),
                )
            )
            secondary = ProxyServer(
                app[_SERVICE_KEY],
                host="127.0.0.1",
                port=0,
                session="proxy-probe-isolated",
                session_mode=ISOLATED_MODE,
                certificates=ProxyCertificates(ca_cert, ca_key),
            )
            await secondary.start()
            results.append(
                (
                    "isolated and shared cookie values stay separate on the wire",
                    await _check_isolation(secondary.port, port, origin_url, ca_cert, state),
                )
            )
        finally:
            try:
                if secondary is not None:
                    await secondary.aclose()
            finally:
                await app_module._on_cleanup(app)  # noqa: SLF001 - the probe drives the app's real lifecycle

        gauges = Browser.resource_metrics()
        results.append(
            (
                "6 context_count and tabgroups_active are zero after cleanup",
                gauges["context_count"] == 0 and gauges["tabgroups_active"] == 0,
            )
        )
    return results


async def _main() -> int:
    binary = _resolve_binary_path()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        origin_cert, origin_key = _write_ephemeral_tls(directory)
        observed: list[bytes] = []
        media_observed: list[str] = []
        original_factory = _handler_class

        def factory(state: _SiteState) -> type[BaseHTTPRequestHandler]:
            return _media_handler(_binary_handler(original_factory(state), observed), media_observed)

        with mock.patch("scripts.verify_request_routing._handler_class", factory):
            server, state, origin_url = _serve("127.0.0.1", _server_tls(origin_cert, origin_key))
        state.pages["/"] = _ProbePage.INDEX
        state.pages["/js"] = _ProbePage.JS_REQUIRED
        try:
            with mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary}):
                results = await _run_probe(directory, origin_cert, origin_url, state, (observed, media_observed))
        finally:
            servers_ok = _cleanup_servers(server)
    results.append(("7 loopback origin server shut down cleanly", servers_ok))
    return _report(results)


def _media_handler(base: type[BaseHTTPRequestHandler], observed: list[str]) -> type[BaseHTTPRequestHandler]:
    """Observe resource requests; the font fixture tests networking, not typography."""

    class Handler(base):
        def do_GET(self) -> None:  # noqa: N802 - HTTP server callback
            if urlsplit(self.path).path == "/widget":
                observed.append(self.path)
            resource = _MEDIA_ASSETS.get(urlsplit(self.path).path)
            if resource is None:
                getattr(base, f"do_{self.command}")(self)
                return
            observed.append(self.path)
            content_type, body = resource
            self.send_response(OK_STATUS)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


async def _check_media(service: app_module.Service, origin_url: str, state: _SiteState, observed: list[str]) -> bool:
    for label, enabled in (("before", False), ("filtered", True), ("after", False)):
        state.pages["/media"] = _MEDIA_PAGE.replace("CASE", label)
        status, payload = await service.handle(
            {
                "cmd": "request.get",
                "url": origin_url + "/media",
                "mode": "browser",
                "disableMedia": enabled,
                "waitInSeconds": 1,
                "maxTimeout": 30000,
            }
        )
        if status != OK_STATUS or payload.get("status") != "ok":
            return False
        expected = {path + "?case=" + label for path in _MEDIA_ASSETS}
        actual = set(observed) & expected
        if actual != (set() if enabled else expected):
            return False
    return True


_WIDGET_PAGE = """<!doctype html><html><head><title>Local verification</title></head><body>
<input name="cf-turnstile-response" type="hidden" value="old-value">
<label><input id="verify" type="checkbox">Verify locally</label><p id="result">Not verified</p>
<script>
const box = document.getElementById('verify');
box.addEventListener('change', () => {
    document.querySelector("input[name='cf-turnstile-response']").value = 'offline-widget-value';
    document.getElementById('result').textContent = 'Verified locally';
    document.cookie = 'widgetdone=1; Path=/; Secure';
});
if (location.search.includes('zero')) box.focus();
</script></body></html>"""


async def _check_verification(
    service: app_module.Service,
    backend: BrowserBackend,
    origin_url: str,
    state: _SiteState,
    observed: list[str],
) -> bool:
    state.pages["/widget"] = _WIDGET_PAGE
    fastpath_before = backend.metrics.http_fastpath_total
    for suffix, tabs, cookie_only in (("", 1, False), ("?zero", 0, True)):
        target = "/widget" + suffix
        before = observed.count(target)
        status, payload = await service.handle(
            {
                "cmd": "request.get",
                "url": origin_url + target,
                "mode": "auto",
                "tabs_till_verify": tabs,
                "returnScreenshot": True,
                "disableMedia": True,
                "returnOnlyCookies": cookie_only,
                "waitInSeconds": 0.1,
                "maxTimeout": 15000,
            }
        )
        solution = payload.get("solution", {})
        if not (
            status == OK_STATUS
            and payload.get("status") == "ok"
            and solution.get("turnstile_token") == "offline-widget-value"
            and _valid_png(payload)
            and solution.get("url") == origin_url + target
            and observed.count(target) - before == 1
            and any(
                cookie.get("name") == "widgetdone" and cookie.get("value") == "1"
                for cookie in solution.get("cookies", [])
            )
            and (solution.get("response") == "" if cookie_only else "Verified locally" in solution.get("response", ""))
        ):
            _emit(
                f"Verification fixture failed: tabs={tabs}, status={status}, "
                f"ok={payload.get('status') == 'ok'}, "
                f"fresh={solution.get('turnstile_token') == 'offline-widget-value'}, "
                f"png={_valid_png(payload)}, navigation_count={observed.count(target) - before}, "
                f"url_matches={solution.get('url') == origin_url + target}"
            )
            return False
    status, payload = await service.handle(
        {"cmd": "request.get", "url": origin_url + "/", "mode": "auto", "tabs_till_verify": 0, "maxTimeout": 15000}
    )
    return (
        status == OK_STATUS
        and payload.get("status") == "ok"
        and "turnstile_token" not in payload.get("solution", {})
        and backend.metrics.http_fastpath_total == fastpath_before
    )


async def _check_diagnostics(
    service: app_module.Service, backend: BrowserBackend, origin_url: str, state: _SiteState
) -> bool:
    state.pages["/diagnostics"] = (
        "<html><head><title>Diagnostics fixture</title></head><body>"
        '<noscript>enable javascript</noscript><main id="result"></main><script>/*'
        + "x" * 65536
        + '*/document.getElementById("result").textContent="Native diagnostics ready";</script></body></html>'
    )
    before = backend.metrics.browser_escalations_total
    status, payload = await service.handle(
        {"cmd": "request.get", "url": origin_url + "/diagnostics", "mode": "auto", "maxTimeout": 20000}
    )
    solution = payload.get("solution", {})
    execution = solution.get("execution", {})
    return (
        status == OK_STATUS
        and payload.get("status") == "ok"
        and "Native diagnostics ready" in solution.get("response", "")
        and execution.get("mode") == "browser"
        and execution.get("category") == "SUCCESS"
        and backend.metrics.browser_escalations_total == before + 1
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
