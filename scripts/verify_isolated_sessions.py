r"""Owner-run probe for isolated sessions against a real local browser.

This is not part of the pytest suite. Run it explicitly on a machine that already has the
CloakBrowser binary and a display (or an X server):

    .\.venv\Scripts\python.exe -m scripts.verify_isolated_sessions

It uses a fresh ``TemporaryDirectory`` profile and two loopback HTTPS servers on 127.0.0.1 and
127.0.0.2 only. It downloads and installs nothing. Its controlled requests are local, and it
never adds a synthetic locale, timezone, user agent or proxy. The offline claim covers the
traffic this probe controls: its own loopback servers, GeoIP resolution, and the binary
download/update path (which is pinned to the already-installed binary). It is not a host
firewall, so unrelated background traffic the operating system or the browser itself initiates
is outside its scope.

Launch goes through a script-local wrapper that forces ``geoip=False`` and appends process flags
that disable background networking/component updates and ignore the probe's ephemeral TLS
certificate errors. The real profile priming, managed-policy application and CloakBrowser
humanization still run.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import ssl
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock
from urllib.parse import parse_qs

from cloakbrowser import binary_info, launch_persistent_context_async

from prowl.browser import Browser, BrowserConfig
from prowl.browser.driver import runtime as driver_runtime
from prowl.browser.lifecycle import startup as lifecycle_startup

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
    from typing import TextIO

#: The two loopback hosts that act as distinct sites for the third-party cookie check.
_SITE_A_HOST = "127.0.0.1"
_SITE_B_HOST = "127.0.0.2"

_ISOLATED_ONE = "probe-iso-1"
_ISOLATED_TWO = "probe-iso-2"

_POPUP_PATH = "/popup"
_POPUP_TARGET_PATH = "/popup-target"
_THIRD_PARTY_PATH = "/third-party"
_CONTROL_PATH = "/control"
_FRAME_PATH = "/frame"
_REPORT_PATH = "/report"

_NOT_FOUND = "<!doctype html><html><body>not found</body></html>"

#: Playwright synchronization budgets (ms) and the popup-routing yield budget (s).
_SYNC_TIMEOUT_MS = 15_000
_ROUTING_TIMEOUT_SECONDS = 5.0
_ROUTING_POLL_SECONDS = 0.05

#: Flags appended to every launch so the probe cannot resolve GeoIP, fetch updates or trip on
#: its own self-signed loopback certificate. Locale, timezone, user agent and proxy are never
#: touched, so the browser persona is the one Prowl prepares.
_OFFLINE_PROCESS_FLAGS: tuple[str, ...] = (
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-domain-reliability",
    "--disable-sync",
    "--metrics-recording-only",
    "--no-default-browser-check",
    "--no-first-run",
    "--ignore-certificate-errors",
)

#: The installed CloakBrowser launch, captured so the wrapper always forwards to the real one.
_installed_launch = launch_persistent_context_async


class ProbePrerequisiteError(RuntimeError):
    """A required local prerequisite (binary, dependency, profile) is missing."""


# -- Inert and cookie-writing page fixtures ----------------------------------------


_SITE_A_INDEX = "<!doctype html><html><body><h1>site A</h1></body></html>"

_SITE_A_POPUP = """<!doctype html><html><body>
<button id="open" onclick="window.open('{A}/popup-target', '_blank')">open</button>
</body></html>"""

_SITE_A_POPUP_TARGET = "<!doctype html><html><body><h1>popup</h1></body></html>"

#: Site A embeds a site B iframe. The parent registers a message listener *before* the iframe
#: loads, so the frame can report its observed result back across origins without polling.
_SITE_A_THIRD_PARTY = """<!doctype html><html><body>
<script>
window.__probe = {{}};
window.addEventListener('message', (event) => {{
  if (event.data && event.data.probe) {{ window.__probe[event.data.probe] = event.data.seen; }}
}});
</script>
<iframe src="{B}/frame" width="220" height="120"></iframe>
</body></html>"""

_SITE_B_INDEX = "<!doctype html><html><body><h1>site B</h1></body></html>"

#: First-party positive control: B sets a Secure/SameSite=None cookie at its own origin, then
#: asks the B server whether that cookie came back on a same-origin request.
_SITE_B_CONTROL = """<!doctype html><html><body><h1>control</h1>
<script>
(async () => {
  document.cookie = 'b_control=1; Secure; SameSite=None; path=/';
  try {
    const resp = await fetch('/report?name=b_control', {credentials: 'include'});
    window.__probeControl = await resp.text();
  } catch (err) {
    window.__probeControl = 'error:' + err;
  }
})();
</script></body></html>"""

#: Cross-site iframe: B sets the same kind of cookie in a third-party frame, then reports the
#: server-observed send (on a B request made from inside an A page) to its parent.
_SITE_B_FRAME = """<!doctype html><html><body>
<script>
(async () => {
  document.cookie = 'b_third=1; Secure; SameSite=None; path=/';
  let seen = 'error';
  try {
    const resp = await fetch('/report?name=b_third', {credentials: 'include'});
    seen = await resp.text();
  } catch (err) {
    seen = 'error:' + err;
  }
  window.parent.postMessage({probe: 'third', seen: seen}, '*');
})();
</script></body></html>"""

#: Surfaces that must be identical between the shared and an isolated context, because the
#: fingerprint, proxy, locale and timezone are process-wide launch properties.
_FINGERPRINT_JS = """() => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.textBaseline = 'top';
  ctx.font = '14px Arial';
  ctx.fillText('prowl-probe', 2, 2);
  const gl = document.createElement('canvas').getContext('webgl');
  const dbg = gl && gl.getExtension('WEBGL_debug_renderer_info');
  return {
    userAgent: navigator.userAgent,
    languages: navigator.languages,
    language: navigator.language,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    platform: navigator.platform,
    screen: [screen.width, screen.height, screen.colorDepth],
    hardwareConcurrency: navigator.hardwareConcurrency,
    canvas: canvas.toDataURL(),
    webglVendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : null,
    webglRenderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null,
  };
}"""

#: Opens the state database, writes one real record and resolves only once the transaction
#: commits, so a reader that follows cannot observe a half-written value.
_IDB_WRITE_JS = """(marker) => new Promise((resolve, reject) => {
  const open = indexedDB.open('probe-db', 1);
  open.onupgradeneeded = () => { open.result.createObjectStore('records', { keyPath: 'id' }); };
  open.onerror = () => reject(open.error ? open.error.message : 'open failed');
  open.onsuccess = () => {
    const db = open.result;
    const tx = db.transaction('records', 'readwrite');
    tx.objectStore('records').put({ id: 'sentinel', value: marker });
    tx.oncomplete = () => { db.close(); resolve(marker); };
    tx.onerror = () => reject(tx.error ? tx.error.message : 'transaction failed');
    tx.onabort = () => reject('transaction aborted');
  };
})"""

#: Reads the state record, resolving ``null`` when the database or record is not visible.
_IDB_READ_JS = """() => new Promise((resolve, reject) => {
  const open = indexedDB.open('probe-db', 1);
  open.onupgradeneeded = () => { open.transaction.abort(); resolve(null); };
  open.onerror = () => reject(open.error ? open.error.message : 'open failed');
  open.onsuccess = () => {
    const db = open.result;
    if (!db.objectStoreNames.contains('records')) { db.close(); resolve(null); return; }
    const tx = db.transaction('records', 'readonly');
    const get = tx.objectStore('records').get('sentinel');
    get.onsuccess = () => { const record = get.result; db.close(); resolve(record ? record.value : null); };
    get.onerror = () => reject(get.error ? get.error.message : 'get failed');
  };
})"""

_EMPTY_STATE: dict[str, Any] = {"local": None, "cookie": None, "db": None}


# -- Prerequisites, offline launch wrapper, and TLS ---------------------------------


def _resolve_binary_path(
    environ: Mapping[str, str] | None = None,
    info: Callable[[], dict[str, Any]] | None = None,
) -> str:
    """Return the installed CloakBrowser binary path, refusing anything that would download.

    An existing ``CLOAKBROWSER_BINARY_PATH`` is honored as-is; otherwise the read-only
    ``cloakbrowser.binary_info`` installation probe is used. A missing binary is a hard refusal
    so the probe never triggers a download.

    :raises ProbePrerequisiteError: when the override is missing or no binary is installed.
    """
    env = os.environ if environ is None else environ
    override = env.get("CLOAKBROWSER_BINARY_PATH", "").strip()
    if override:
        if not Path(override).is_file():
            msg = f"CLOAKBROWSER_BINARY_PATH points at a missing binary: {override}"
            raise ProbePrerequisiteError(msg)
        return override

    installed = (info or binary_info)()
    if not installed.get("installed"):
        msg = (
            "No installed CloakBrowser binary was found. Install one or point "
            "CLOAKBROWSER_BINARY_PATH at an existing binary; this probe never downloads."
        )
        raise ProbePrerequisiteError(msg)
    path = Path(installed["binary_path"])
    if not path.is_file():
        msg = f"Installed browser binary is missing: {path}"
        raise ProbePrerequisiteError(msg)
    return str(path)


async def _offline_launch(*args: Any, **kwargs: Any) -> Any:
    """Forward to the installed CloakBrowser launch with GeoIP off and offline flags added.

    The caller's arguments are preserved; only ``geoip`` is forced to ``False`` and the offline
    process flags are appended. No locale, timezone, user agent or proxy is injected.
    """
    kwargs["geoip"] = False
    existing = list(kwargs.get("args") or [])
    kwargs["args"] = [*existing, *(flag for flag in _OFFLINE_PROCESS_FLAGS if flag not in existing)]
    return await _installed_launch(*args, **kwargs)


@contextmanager
def _offline_launch_patches() -> Iterator[None]:
    """Patch both Prowl launch seams to the offline wrapper, restoring them on exit."""
    with (
        mock.patch.object(lifecycle_startup, "launch_persistent_context_async", _offline_launch),
        mock.patch.object(driver_runtime, "launch_persistent_context_async", _offline_launch),
    ):
        yield


def _write_ephemeral_tls(directory: Path) -> tuple[Path, Path]:
    """Write a throwaway self-signed certificate and key covering both loopback hosts.

    Nothing is installed and nothing is tracked: the pair lives under *directory* (a temporary
    profile directory) and is deleted with it.

    :raises ProbePrerequisiteError: when the already-installed ``cryptography`` is unavailable.
    """
    try:
        from cryptography import x509  # noqa: PLC0415 - a clear prerequisite error needs a lazy import
        from cryptography.hazmat.primitives import hashes, serialization  # noqa: PLC0415
        from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: PLC0415
        from cryptography.x509.oid import NameOID  # noqa: PLC0415
    except ImportError as exc:
        msg = "The 'cryptography' package is required to generate the probe's loopback TLS certificate."
        raise ProbePrerequisiteError(msg) from exc

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "prowl-isolated-probe")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(_SITE_A_HOST)),
                    x509.IPAddress(ipaddress.ip_address(_SITE_B_HOST)),
                ],
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path = directory / "probe-cert.pem"
    key_path = directory / "probe-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    return cert_path, key_path


def _tls_context(directory: Path) -> ssl.SSLContext:
    """Return a server TLS context for the ephemeral loopback certificate."""
    cert_path, key_path = _write_ephemeral_tls(directory)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return context


# -- Loopback servers --------------------------------------------------------------


@dataclass(slots=True)
class _SiteState:
    """Pages one loopback site serves."""

    pages: dict[str, str] = field(default_factory=dict)


def _handler_class(state: _SiteState) -> type[BaseHTTPRequestHandler]:
    """Return a request handler for *state* that answers each GET, reporting cookie sends."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            cookie_header = self.headers.get("Cookie", "")
            path, _, query = self.path.partition("?")
            if path == _REPORT_PATH:
                name = parse_qs(query).get("name", [""])[0]
                self._respond("1" if _cookie_present(cookie_header, name) else "0", "text/plain; charset=utf-8")
                return
            self._respond(state.pages.get(path, _NOT_FOUND), "text/html; charset=utf-8")

        def _respond(self, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args: Any, **_kwargs: Any) -> None:
            """Silence the default access log."""

    return _Handler


def _serve(host: str, tls: ssl.SSLContext) -> tuple[ThreadingHTTPServer, _SiteState, str]:
    """Start a loopback HTTPS server on *host* and return it with its state and base URL."""
    state = _SiteState()
    server = ThreadingHTTPServer((host, 0), _handler_class(state))
    server.daemon_threads = True
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, state, f"https://{host}:{port}"


def _fill_pages(state_a: _SiteState, state_b: _SiteState, a_url: str, b_url: str) -> None:
    """Populate both servers' page maps once both ports are known."""
    state_a.pages["/"] = _SITE_A_INDEX
    state_a.pages[_POPUP_PATH] = _SITE_A_POPUP.format(A=a_url)
    state_a.pages[_POPUP_TARGET_PATH] = _SITE_A_POPUP_TARGET
    state_a.pages[_THIRD_PARTY_PATH] = _SITE_A_THIRD_PARTY.format(B=b_url)
    state_b.pages["/"] = _SITE_B_INDEX
    state_b.pages[_CONTROL_PATH] = _SITE_B_CONTROL
    state_b.pages[_FRAME_PATH] = _SITE_B_FRAME


def _cookie_present(cookie_header: str, name: str) -> bool:
    """Return whether the named cookie appears in a Cookie header."""
    if not name:
        return False
    for part in cookie_header.split(";"):
        key, separator, _value = part.strip().partition("=")
        if separator and key == name:
            return True
    return False


# -- Result collection -------------------------------------------------------------


def _emit(text: str, *, stream: TextIO) -> None:
    """Write one line to *stream*, flushing so a long run shows progress as it goes."""
    print(text, file=stream, flush=True)


@dataclass(frozen=True, slots=True)
class _Result:
    """Outcome of one probe check."""

    name: str
    ok: bool
    detail: str


class _Probe:
    """Collects check outcomes, printing each as it completes so progress is visible."""

    def __init__(self, stream: TextIO | None = None) -> None:
        """Create a probe writing to *stream* (stdout by default)."""
        self.total = 0
        self.failures = 0
        self._stream = stream if stream is not None else sys.stdout

    def record(self, result: _Result) -> None:
        """Print *result* immediately and count it."""
        self.total += 1
        if not result.ok:
            self.failures += 1
        marker = "PASS" if result.ok else "FAIL"
        _emit(f"[{marker}] {result.name}: {result.detail}", stream=self._stream)

    def error(self, name: str, exc: BaseException) -> None:
        """Record an unexpected exception as a failed check instead of aborting the run."""
        self.record(_Result(name, ok=False, detail=f"unexpected {type(exc).__name__}: {exc}"))


# -- Browser helpers ---------------------------------------------------------------


async def _target_info(page: Any) -> tuple[str, str | None]:
    """Return ``(targetId, browserContextId)`` for *page* from the real target info."""
    cdp = await page.context.new_cdp_session(page)
    try:
        result = await cdp.send("Target.getTargetInfo")
    finally:
        await cdp.detach()
    info = result["targetInfo"]
    return info["targetId"], info.get("browserContextId")


async def _pd_context_id(target_id: str) -> str | None:
    """Return the browserContextId the Pydoll tab for *target_id* was built with."""
    tab = await Browser.get_pd_tab(target_id)
    if tab is None:
        return None
    return tab._browser_context_id  # noqa: SLF001 - the attribute is the point of this check


async def _wait_until(predicate: Callable[[], bool], *, timeout: float, interval: float) -> bool:  # noqa: ASYNC109
    """Yield to the event loop until *predicate* holds or *timeout* elapses, with a deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return True
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(interval)


async def _with_group(handle: Any, url: str, action: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run *action* on a fresh group page in *handle*, then close the group."""
    group = await Browser.create(context=handle)
    try:
        page = group.ppage
        await page.goto(url)
        return await action(page)
    finally:
        await group.quit()


async def _set_state(page: Any, marker: str) -> None:
    """Write a distinct cookie, localStorage value and committed IndexedDB record for *marker*."""
    await page.evaluate("(marker) => { window.localStorage.setItem('probe-local', marker); }", marker)
    await page.evaluate(
        "(marker) => { document.cookie = 'probe-cookie=' + marker + '; path=/'; }",
        marker,
    )
    await page.evaluate(_IDB_WRITE_JS, marker)


async def _read_state(page: Any) -> dict[str, Any]:
    """Read the cookie, localStorage value and IndexedDB record visible on *page*."""
    local = await page.evaluate("() => window.localStorage.getItem('probe-local')")
    cookie = await page.evaluate(
        "() => { const m = document.cookie.match(/(?:^|; )probe-cookie=([^;]*)/); return m ? m[1] : null; }",
    )
    record = await page.evaluate(_IDB_READ_JS)
    return {"local": local, "cookie": cookie, "db": record}


# -- Checks ------------------------------------------------------------------------


async def _check_same_process(probe: _Probe, handles: Mapping[str, Any]) -> None:
    """The shared and isolated contexts belong to the one real Playwright browser."""
    browser = Browser.pw()
    same = {label: handle.context.browser is browser for label, handle in handles.items()}
    probe.record(
        _Result(
            "shared and isolated contexts share one browser process",
            ok=browser is not None and all(same.values()),
            detail=f"same_browser={same}",
        ),
    )


async def _check_persistence(probe: _Probe, a_url: str, shared: Any) -> None:
    """State written in one shared-context group is read back by the next, unchanged."""
    await _with_group(shared, a_url, partial(_set_state, marker="shared"))
    read = await _with_group(shared, a_url, _read_state)

    expected = {"local": "shared", "cookie": "shared", "db": "shared"}
    probe.record(
        _Result(
            "shared-context state survives across tab groups",
            ok=read == expected,
            detail=f"read={read!r} expected={expected!r}",
        ),
    )


async def _check_isolation(probe: _Probe, a_url: str, shared: Any, iso_one: Any, iso_two: Any) -> None:
    """Verify each reused context keeps its own state across groups."""
    reused = await Browser.get_context(_ISOLATED_ONE)
    await _with_group(iso_one, a_url, partial(_set_state, marker="iso1"))
    shared_read = await _with_group(shared, a_url, _read_state)
    initially_empty = await _with_group(iso_two, a_url, _read_state)
    await _with_group(iso_two, a_url, partial(_set_state, marker="iso2"))
    iso_one_read = await _with_group(reused, a_url, _read_state)
    iso_two_read = await _with_group(iso_two, a_url, _read_state)
    ok = (
        reused is iso_one
        and shared_read == {"local": "shared", "cookie": "shared", "db": "shared"}
        and initially_empty == _EMPTY_STATE
        and iso_one_read == {"local": "iso1", "cookie": "iso1", "db": "iso1"}
        and iso_two_read == {"local": "iso2", "cookie": "iso2", "db": "iso2"}
    )
    probe.record(
        _Result(
            "reused isolated contexts persist distinct state across groups",
            ok=ok,
            detail=(
                f"reused={reused is iso_one} shared={shared_read!r} "
                f"initial_iso2={initially_empty!r} iso1={iso_one_read!r} iso2={iso_two_read!r}"
            ),
        ),
    )


async def _check_popup(probe: _Probe, url: str, handle: Any, label: str) -> None:
    """Check auxiliary/popup routing and page-only group teardown."""
    group = await Browser.create(context=handle)
    try:
        page = group.ppage
        await page.goto(url)
        _, parent_context = await _target_info(page)
        await group.new_tab()
        auxiliary = Browser.get_pw_page(group.child_target_ids[-1])
        if auxiliary is None:
            msg = "auxiliary page was not registered"
            raise RuntimeError(msg)
        auxiliary_target, auxiliary_context = await _target_info(auxiliary)
        async with page.expect_popup() as popup_info:
            await page.click("#open")
        popup = await popup_info.value
        await popup.wait_for_load_state()
        target_id, popup_context = await _target_info(popup)
        routed = await _wait_until(
            lambda: target_id in group.child_target_ids,
            timeout=_ROUTING_TIMEOUT_SECONDS,
            interval=_ROUTING_POLL_SECONDS,
        )
        same_context = (
            popup.context is handle.context
            and auxiliary.context is handle.context
            and popup_context == auxiliary_context == parent_context
            and await _pd_context_id(target_id) == parent_context
            and await _pd_context_id(auxiliary_target) == parent_context
        )
        same_opener = await popup.opener() is page
        auxiliary_owned = auxiliary_target in group.child_target_ids
    finally:
        await group.quit()
    closed = all(p.is_closed() for p in (page, auxiliary, popup))
    context_open = not handle.context.is_closed()
    probe.record(
        _Result(
            f"{label}: auxiliary and popup ownership/cleanup",
            ok=routed and auxiliary_owned and same_context and same_opener and closed and context_open,
            detail=(
                f"routed={routed} auxiliary_owned={auxiliary_owned} same_context={same_context} "
                f"same_opener={same_opener} pages_closed={closed} context_open={context_open}"
            ),
        ),
    )


async def _page_context_ids(page: Any) -> tuple[str | None, str | None]:
    target_id, context_id = await _target_info(page)
    return context_id, await _pd_context_id(target_id)


async def _check_context_ids(probe: _Probe, url: str, handles: Mapping[str, Any]) -> None:
    """Compare actual CDP/Pydoll IDs for every selected context."""
    ids = {label: await _with_group(handle, url, _page_context_ids) for label, handle in handles.items()}
    shared_id = ids["shared"][0]
    isolated = [ids[label][0] for label in ids if label != "shared"]
    ok = (
        all(cdp == pd for cdp, pd in ids.values())
        and all(isinstance(value, str) and value and value != shared_id for value in isolated)
        and len(set(isolated)) == len(isolated)
    )
    probe.record(_Result("Pydoll matches distinct real context IDs", ok=ok, detail=f"cdp/pydoll={ids!r}"))


async def _fingerprint(url: str, handle: Any) -> dict[str, Any]:
    """Return the fingerprint surfaces a page in *handle* reports at the same origin."""
    return await _with_group(handle, url, lambda page: page.evaluate(_FINGERPRINT_JS))


async def _check_fingerprint(probe: _Probe, url: str, handles: Mapping[str, Any]) -> None:
    """The shared and both isolated contexts report identical fingerprint surfaces."""
    fingerprints = {label: await _fingerprint(url, handle) for label, handle in handles.items()}
    labels = list(fingerprints)
    base_label = labels[0]
    base = fingerprints[base_label]
    differing = {
        label: [key for key in base if fingerprints[label].get(key) != base.get(key)]
        for label in labels[1:]
        if fingerprints[label] != base
    }
    probe.record(
        _Result(
            "fingerprint surfaces match across shared and isolated contexts",
            ok=not differing,
            detail="identical" if not differing else f"differ from {base_label}: {differing}",
        ),
    )


@dataclass(frozen=True, slots=True)
class _CookiePolicy:
    """Whether the first-party control and the cross-site send were accepted."""

    control: bool
    third_party: bool


async def _cookie_policy(a_url: str, b_url: str, handle: Any) -> _CookiePolicy:
    """Return whether the first-party control and the third-party send were accepted in *handle*."""
    group = await Browser.create(context=handle)
    try:
        page = group.ppage
        await page.goto(f"{b_url}{_CONTROL_PATH}")
        await page.wait_for_function("() => window.__probeControl !== undefined", timeout=_SYNC_TIMEOUT_MS)
        control = await page.evaluate("() => window.__probeControl")
        await page.goto(f"{a_url}{_THIRD_PARTY_PATH}")
        await page.wait_for_function(
            "() => window.__probe && window.__probe.third !== undefined",
            timeout=_SYNC_TIMEOUT_MS,
        )
        third_party = await page.evaluate("() => window.__probe.third")
    finally:
        await group.quit()
    return _CookiePolicy(control=control == "1", third_party=third_party == "1")


async def _check_cookie_policy(probe: _Probe, a_url: str, b_url: str, handles: Mapping[str, Any]) -> None:
    """The prepared profile actually allows third-party cookies in shared and isolated contexts."""
    outcomes = {label: await _cookie_policy(a_url, b_url, handle) for label, handle in handles.items()}
    controls_ok = all(outcome.control for outcome in outcomes.values())
    accepted_ok = all(outcome.third_party for outcome in outcomes.values())
    detail = "; ".join(
        f"{label}: control={outcome.control} third_party={outcome.third_party}" for label, outcome in outcomes.items()
    )
    if not controls_ok:
        detail += " (first-party control failed, so the result is inconclusive)"
    probe.record(
        _Result(
            "prepared-profile third-party cookie policy holds in shared and isolated contexts",
            ok=controls_ok and accepted_ok,
            detail=detail,
        ),
    )


async def _check_isolated_close(probe: _Probe, shared: Any, iso_one: Any, iso_two: Any) -> None:
    """Closing one isolated context leaves the shared and the other isolated context open."""
    await Browser.close_context(_ISOLATED_ONE)
    shared_open = not shared.context.is_closed()
    iso_one_closed = iso_one.context.is_closed()
    iso_two_open = not iso_two.context.is_closed()
    probe.record(
        _Result(
            "closing one isolated context leaves shared and the other isolated context open",
            ok=shared_open and iso_one_closed and iso_two_open,
            detail=f"shared_open={shared_open} iso1_closed={iso_one_closed} iso2_open={iso_two_open}",
        ),
    )


# -- Cleanup and orchestration -----------------------------------------------------


async def _best_effort(label: str, step: Callable[[], Awaitable[None]]) -> bool:
    """Await one cleanup step, reporting but never propagating its failure."""
    try:
        await step()
    except BaseException as exc:  # noqa: BLE001 - cleanup must not mask a primary error
        _emit(f"[CLEANUP] {label} failed: {type(exc).__name__}: {exc}", stream=sys.stderr)
        return False
    return True


async def _cleanup_browser() -> bool:
    """Close the isolated contexts and shut the browser down; failures are reported, not raised."""
    ok = True
    for session_id in (_ISOLATED_ONE, _ISOLATED_TWO):
        closed = await _best_effort(f"close context {session_id}", partial(Browser.close_context, session_id))
        ok = closed and ok
    shutdown = await _best_effort("browser shutdown", Browser.shutdown)
    return shutdown and ok


def _cleanup_servers(*servers: ThreadingHTTPServer) -> bool:
    """Stop every loopback server and release its socket, never masking a primary error."""
    ok = True
    for server in servers:
        try:
            server.shutdown()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask a primary error
            ok = False
            _emit(f"[CLEANUP] server shutdown failed: {exc}", stream=sys.stderr)
        try:
            server.server_close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask a primary error
            ok = False
            _emit(f"[CLEANUP] server_close failed: {exc}", stream=sys.stderr)
    return ok


async def _run_checks_sequence(probe: _Probe, checks: Sequence[tuple[str, Callable[[], Awaitable[None]]]]) -> None:
    """Run every check in order, recording an unexpected exception as that check's failure."""
    for name, check in checks:
        try:
            await check()
        except Exception as exc:  # noqa: BLE001 - one failing check must not stop the rest
            probe.error(name, exc)


async def _run_checks(probe: _Probe, a_url: str, b_url: str) -> None:
    """Start the browser, run the ordered checks, and always clean the browser up."""
    try:
        await Browser.start()
        shared = await Browser.get_context(None)
        iso_one = await Browser.get_context(_ISOLATED_ONE)
        iso_two = await Browser.get_context(_ISOLATED_TWO)
        handles = {"shared": shared, "iso-1": iso_one, "iso-2": iso_two}
        checks: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("same browser process", partial(_check_same_process, probe, handles)),
            ("shared persistence", partial(_check_persistence, probe, a_url, shared)),
            ("context isolation", partial(_check_isolation, probe, a_url, shared, iso_one, iso_two)),
            ("shared popup routing", partial(_check_popup, probe, f"{a_url}{_POPUP_PATH}", shared, "shared")),
            ("isolated popup routing", partial(_check_popup, probe, f"{a_url}{_POPUP_PATH}", iso_one, "isolated")),
            ("CDP target context ids", partial(_check_context_ids, probe, a_url, handles)),
            ("fingerprint parity", partial(_check_fingerprint, probe, a_url, handles)),
            ("third-party cookie policy", partial(_check_cookie_policy, probe, a_url, b_url, handles)),
            (
                "isolated context close independence",
                partial(_check_isolated_close, probe, shared, iso_one, iso_two),
            ),
        ]
        await _run_checks_sequence(probe, checks)
    finally:
        if not await _cleanup_browser():
            probe.record(_Result("browser cleanup", ok=False, detail="see cleanup errors above"))


async def _run(directory: Path, binary_path: str, probe: _Probe) -> None:
    """Bring up the loopback servers and the browser, run every check, and clean both up."""
    servers: list[ThreadingHTTPServer] = []
    try:
        tls = _tls_context(directory)
        site_a, state_a, a_url = _serve(_SITE_A_HOST, tls)
        servers.append(site_a)
        site_b, state_b, b_url = _serve(_SITE_B_HOST, tls)
        servers.append(site_b)
        _fill_pages(state_a, state_b, a_url, b_url)
        with (
            mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary_path}),
            _offline_launch_patches(),
        ):
            config = BrowserConfig(
                profile_dir=str(directory / "profile"),
                profile_archive=str(directory / "profile.zip"),
            )
            Browser.configure(config)
            await _run_checks(probe, a_url, b_url)
    finally:
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
        _emit(f"[PREREQ-FAIL] {exc}", stream=sys.stderr)
        return 2

    probe = _Probe()
    try:
        with tempfile.TemporaryDirectory(prefix="prowl-isolated-probe-", ignore_cleanup_errors=True) as raw_directory:
            asyncio.run(_run(Path(raw_directory), binary_path, probe))
    except ProbePrerequisiteError as exc:
        _emit(f"[PREREQ-FAIL] {exc}", stream=sys.stderr)
        return 2
    except BaseException as exc:  # noqa: BLE001 - report the abort rather than a traceback only
        _emit(f"[ERROR] the probe aborted: {type(exc).__name__}: {exc}", stream=sys.stderr)
        return 1

    _emit(
        f"{probe.total - probe.failures}/{probe.total} checks passed. "
        "This probe covers loopback traffic only and is not a host firewall.",
        stream=sys.stderr,
    )
    return 1 if probe.failures else 0


if __name__ == "__main__":
    sys.exit(main())
