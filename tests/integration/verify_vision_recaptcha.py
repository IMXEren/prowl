"""Offline real-browser reCAPTCHA-shaped 4x4 and refreshing 3x3 vision gate."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
from urllib.parse import urlsplit

from tests.integration.verify_vision_hcaptcha import _launch_patches, _resolve_binary_path

from prowl.browser import Browser, BrowserConfig
from prowl.browser.solvers.vision.config import VisionSettings, build_cloud_vlm_settings
from prowl.service.backend import BrowserBackend, FetchRequest

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Route

_ANCHOR_URL = "https://www.google.com/recaptcha/enterprise/anchor"
_FRAME_URL = "https://www.google.com/recaptcha/enterprise/bframe"


def _main_page(layout: str) -> str:
    return f"""<!doctype html><html><body>
<textarea name="g-recaptcha-response" hidden>stale</textarea>
<iframe title="checkbox" src="{_ANCHOR_URL}?layout={layout}" width="310" height="82"></iframe>
<iframe id="challenge" title="image challenge" src="{_FRAME_URL}?layout={layout}"
  style="display:none;width:460px;height:520px;border:0"></iframe>
<script>
addEventListener('message', event => {{
  if (event.origin !== 'https://www.google.com') return;
  if (event.data === 'open') document.querySelector('#challenge').style.display = 'block';
  if (event.data === 'accepted') {{
    document.querySelector('[name="g-recaptcha-response"]').value = 'fixture-fresh';
    // The live provider navigates away as soon as it accepts an answer, which destroys the
    // execution context the token was read from.
    if ('{layout}' === 'navigate') setTimeout(() => {{ location.href = '/accepted'; }}, 900);
  }}
}});
</script></body></html>"""


def _svg(color: str) -> str:
    return (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'"
        " width='90' height='90'%3E%3Crect width='90' height='90'"
        f" fill='%23{color}'/%3E%3C/svg%3E"
    )


def _tile(index: int, rows: int) -> str:
    color = "f33" if index in ({10} if rows == 4 else {1, 4}) else "ddd"
    return f'<td class="rc-imageselect-tile" data-index="{index}"><img width="90" height="90" src="{_svg(color)}"></td>'


def _challenge_page(layout: str) -> str:
    rows = 4 if layout != "refresh" else 3
    instruction = (
        "Select all images with cars Click verify once there are none left"
        if layout == "refresh"
        else "Select all squares with traffic lights"
    )
    tiles = "".join(
        "<tr>" + "".join(_tile(row * rows + col, rows) for col in range(rows)) + "</tr>" for row in range(rows)
    )
    return f"""<!doctype html><html><head><style>
body {{ margin: 0; background: white }}
#rc-imageselect {{ width: 380px; padding: 12px; background: #0097a0 }}
.rc-imageselect-desc-wrapper {{ height: 45px; color: white }}
#rc-imageselect-target {{ border-collapse: collapse }}
td {{ padding: 0; width: 90px; height: 90px; border: 1px solid white; cursor: pointer }}
td.selected {{ outline: 4px solid blue; outline-offset: -4px }}
button {{ margin-top: 8px; padding: 8px 20px }}
</style></head><body><div id="rc-imageselect">
<div class="rc-imageselect-desc-wrapper">{instruction}</div>
<table id="rc-imageselect-target" class="rc-imageselect-table-{rows}{rows}">{tiles}</table>
<button id="recaptcha-verify-button">Verify</button></div>
<script>
const wanted = '{layout}' === 'refresh' ? [1, 4] : [10];
const clicked = new Set(), replaced = new Set();
for (const tile of document.querySelectorAll('td')) tile.addEventListener('click', () => {{
  const index = +tile.dataset.index;
  if (!wanted.includes(index)) return;
  clicked.add(index);
  if ('{layout}' === 'refresh') setTimeout(() => {{
    tile.querySelector('img').src = "{_svg("999")}";
    replaced.add(index);
  }}, index === 1 ? 400 : 800);
  else tile.classList.add('selected');
}});
document.querySelector('button').addEventListener('click', () => {{
  if (clicked.size === wanted.length && ('{layout}' !== 'refresh' || replaced.size === wanted.length))
    parent.postMessage('accepted', '*');
}});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self) -> None:
        if self.path == "/accepted":
            data = b"<!doctype html><html><body>accepted</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.startswith("/tile/"):
            data = _svg("f33" if self.path == "/tile/10" else "ddd").encode()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path not in ("/static", "/refresh", "/navigate"):
            self.send_error(404)
            return
        layout = "refresh" if self.path == "/refresh" else self.path[1:]
        data = _main_page(layout).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if self.path != "/v1/chat/completions" or not 0 < length < 2_000_000:
            self.send_error(400)
            return
        payload = json.loads(self.rfile.read(length))
        content = payload["messages"][1]["content"]
        if payload["model"] != "fixture-points" or not content[1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        ):
            self.send_error(400)
            return
        if self.headers.get("Cookie") or self.headers.get("Proxy-Authorization"):
            self.send_error(400)
            return
        # The live provider navigates away as soon as it accepts an answer, which destroys the
        # execution context the token was read from, so a token read must survive that transition.
        type(self).requests += 1
        instruction = content[0]["text"]
        points: list[dict[str, float]] = (
            [{"x": 0.625, "y": 0.625}]
            if "4x4" in instruction
            else ([{"x": 0.5, "y": 0.16}, {"x": 0.5, "y": 0.5}] if self.requests == 2 else [])
        )
        data = json.dumps({"choices": [{"message": {"content": json.dumps({"points": points})}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


async def _route(context: BrowserContext, port: int) -> None:
    async def block(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == port:
            await route.continue_()
        else:
            await route.abort()

    async def frame(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.path.endswith("/anchor"):
            html = """<!doctype html><html><body><button id="recaptcha-anchor"
              onclick="parent.postMessage('open','*')">I'm not a robot</button></body></html>"""
        else:
            requested = "static"
            for candidate in ("refresh", "navigate"):
                if f"layout={candidate}" in url.query:
                    requested = candidate
            html = _challenge_page(requested)
        await route.fulfill(status=200, body=html, content_type="text/html")

    await context.route("**/*", block)
    await context.route("https://www.google.com/recaptcha/enterprise/**", frame)


async def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with tempfile.TemporaryDirectory(prefix="prowl-recaptcha-fixture-") as tmp:
            root = Path(tmp)
            config = BrowserConfig(
                profile_dir=str(root / "profile"),
                profile_archive=str(root / "profile.zip"),
                extensions_dir=None,
                policy_dir=None,
            )
            settings = VisionSettings(
                vlm=build_cloud_vlm_settings(endpoint=f"{base}/v1/chat/completions", model="fixture-points"),
            )
            backend = BrowserBackend(config, vision_settings=settings)
            try:
                with (
                    mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": _resolve_binary_path()}),
                    _launch_patches(),
                ):
                    Browser.configure(config)
                    await backend.start()
                    context = (await Browser.get_context(None)).context
                    await _route(context, server.server_port)
                    ordinary = await backend.fetch(
                        None, FetchRequest(url=f"{base}/static", mode="browser", timeout_seconds=18)
                    )
                    if ordinary.captcha_token or _Handler.requests:
                        msg = "disabled solveCaptcha uploaded an image"
                        raise AssertionError(msg)
                    for layout, expected_calls in (("static", 1), ("refresh", 2), ("navigate", 1)):
                        before = _Handler.requests
                        result = await backend.fetch(
                            None,
                            FetchRequest(
                                url=f"{base}/{layout}", mode="browser", solve_captcha=True, timeout_seconds=35
                            ),
                        )
                        if result.captcha_provider != "recaptcha" or not result.captcha_token:
                            msg = f"{layout}: no fresh provider token; reason={result.vision_reason}"
                            raise AssertionError(msg)
                        if result.vision_status != "handled" or result.vision_reason != "page_accepted":
                            msg = f"{layout}: no accepted page outcome"
                            raise AssertionError(msg)
                        if _Handler.requests - before != expected_calls:
                            msg = f"{layout}: unexpected VLM request count"
                            raise AssertionError(msg)
                        sys.stdout.write(f"reCAPTCHA-shaped {layout}: token_present=True uploads={expected_calls}\n")
            finally:
                await backend.aclose()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
