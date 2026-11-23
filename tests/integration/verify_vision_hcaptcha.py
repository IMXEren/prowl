"""Offline browser gate for one opted-in hCaptcha-shaped grid or canvas round.

Run with ``python -m tests.integration.verify_vision_hcaptcha``. The challenge frame and
multimodal endpoint are synthetic; this does not test a live provider or model accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest import mock
from urllib.parse import urlsplit

from tests.integration.verify_isolated_sessions import (
    _OFFLINE_PROCESS_FLAGS,
    _installed_launch,
    _resolve_binary_path,
)

from prowl.browser import Browser, BrowserConfig
from prowl.browser.driver import runtime as driver_runtime
from prowl.browser.lifecycle import startup as lifecycle_startup
from prowl.browser.solvers.vision.config import VisionSettings, build_cloud_vlm_settings
from prowl.service.backend import BrowserBackend, FetchRequest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.async_api import BrowserContext, Page, Route

_CHALLENGE_URL = "https://newassets.hcaptcha.com/captcha/v1/fixture/static/hcaptcha.html#frame=challenge"
_PAGE = f"""<!doctype html><html><body>
<textarea name="h-captcha-response" hidden></textarea>
<div id="fixture-status" role="status">Fixture pending</div>
<iframe width="430" height="490" src="{_CHALLENGE_URL}"></iframe>
<script>
window.addEventListener('message', event => {{
  if (event.origin !== 'https://newassets.hcaptcha.com') return;
  document.body.dataset.accepted = event.data === 'selected:2' ? 'true' : 'false';
  if (event.data === 'selected:2') document.querySelector('textarea').value = 'fixture-fresh-token';
  document.querySelector('#fixture-status').textContent = event.data === 'selected:2'
    ? 'Fixture accepted (synthetic token)' : 'Fixture rejected';
}});
</script></body></html>"""
_PAGE_CANVAS = _PAGE.replace(_CHALLENGE_URL, _CHALLENGE_URL.replace("#", "?layout=canvas#")).replace(
    "selected:2", "selected:0,12,24"
)
_PAGE_DRAG = _PAGE.replace(_CHALLENGE_URL, _CHALLENGE_URL.replace("#", "?layout=drag#")).replace(
    "selected:2", "selected:dragged"
)
_PAGE_ROUNDS = _PAGE.replace(_CHALLENGE_URL, _CHALLENGE_URL.replace("#", "?layout=rounds#")).replace(
    "selected:2", "selected:rounds"
)
_PAGE_RETRY = _PAGE_ROUNDS.replace("layout=rounds", "layout=retry")
_FRAME = """<!doctype html><html><head><style>
body { margin: 0; } .challenge-view { width: 360px; padding: 12px; background: white; }
.grid { display: grid; grid-template-columns: repeat(3, 108px); gap: 6px; }
.task { width: 108px; height: 85px; background: #e6e6e6; cursor: pointer; }
.task:nth-child(2) { background: #e44; } .task.selected { outline: 3px solid blue; }
.button-submit { width: 100px; height: 32px; margin-top: 8px; cursor: pointer; }
</style></head><body><div class="challenge-view">
<p>Select the red tile.</p><div class="grid">
<div class="task" aria-label="Image 1"></div><div class="task" aria-label="Image 2"></div>
<div class="task" aria-label="Image 3"></div><div class="task" aria-label="Image 4"></div>
<div class="task" aria-label="Image 5"></div><div class="task" aria-label="Image 6"></div>
<div class="task" aria-label="Image 7"></div><div class="task" aria-label="Image 8"></div>
<div class="task" aria-label="Image 9"></div></div>
<div class="button-submit button" role="button" tabindex="0">Verify</div></div>
<script>
for (const task of document.querySelectorAll('.task'))
  task.addEventListener('click', () => task.classList.toggle('selected'));
document.querySelector('.button-submit').addEventListener('click', () => {
  const selected = [...document.querySelectorAll('.task.selected')]
    .map(task => task.getAttribute('aria-label').split(' ')[1]).join(',');
  window.parent.postMessage('selected:' + selected, '*');
});
</script></body></html>"""
_FRAME_CANVAS = """<!doctype html><html><head><style>
body { margin: 0; } .challenge-view { width: 280px; padding: 12px; background: white; }
canvas { width: 250px; height: 250px; cursor: pointer; }
.button-submit { width: 100px; height: 32px; margin-top: 8px; cursor: pointer; }
</style></head><body><div class="challenge-view">
<h2 class="prompt-text">Select the three red squares.</h2>
<canvas role="img" width="250" height="250"></canvas>
<div class="button-submit button" role="button" tabindex="0">Verify</div></div>
<script>
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
for (let row = 0; row < 5; row++) for (let col = 0; col < 5; col++) {
  const index = row * 5 + col;
  ctx.fillStyle = [0, 12, 24].includes(index) ? '#e22' : '#1762bc';
  ctx.fillRect(col * 50 + 3, row * 50 + 3, 44, 44);
}
const selected = new Set();
canvas.addEventListener('click', event => {
  const col = Math.floor(event.offsetX / 50), row = Math.floor(event.offsetY / 50);
  const index = row * 5 + col;
  selected.add(index);
  ctx.strokeStyle = '#00f'; ctx.lineWidth = 3;
  ctx.strokeRect(col * 50 + 1, row * 50 + 1, 48, 48);
});
document.querySelector('.button-submit').addEventListener('click', () => {
  window.parent.postMessage('selected:' + [...selected].sort((a, b) => a - b).join(','), '*');
});
</script></body></html>"""
_FRAME_ROUNDS = """<!doctype html><html><head><style>
body { margin: 0; } .challenge-view { width: 280px; padding: 12px; background: white; }
canvas { width: 250px; height: 250px; cursor: pointer; }
.button-submit { width: 100px; height: 32px; margin-top: 8px; cursor: pointer; }
</style></head><body><div class="challenge-view">
<h2 class="prompt-text">Select the three red squares.</h2>
<canvas role="img" width="250" height="250"></canvas>
<div class="challenge-breadcrumbs"><div class="Crumb" style="width:24px"></div>
<div class="Crumb" style="width:8px"></div></div>
<div class="button-submit button" role="button" tabindex="0">Verify</div></div>
<script>
const canvas = document.querySelector('canvas'), ctx = canvas.getContext('2d');
const selected = new Set();
let round = 0;
function paint() {
  for (let row = 0; row < 5; row++) for (let col = 0; col < 5; col++) {
    const index = row * 5 + col;
    ctx.fillStyle = [0, 12, 24].includes(index) ? '#e22' : round ? '#238654' : '#1762bc';
    ctx.fillRect(col * 50 + 3, row * 50 + 3, 44, 44);
  }
}
paint();
canvas.addEventListener('click', event => {
  const index = Math.floor(event.offsetY / 50) * 5 + Math.floor(event.offsetX / 50);
  selected.add(index);
  ctx.strokeStyle = '#00f'; ctx.lineWidth = 3;
  ctx.strokeRect(Math.floor(event.offsetX / 50) * 50 + 1,
    Math.floor(event.offsetY / 50) * 50 + 1, 48, 48);
});
document.querySelector('.button-submit').addEventListener('click', () => {
  if ([...selected].sort((a, b) => a - b).join(',') !== '0,12,24') return;
  if (round === 0) {
    round = 1;
    selected.clear();
    const crumbs = document.querySelectorAll('.Crumb');
    crumbs[0].style.width = '8px'; crumbs[1].style.width = '24px';
    paint();
  } else window.parent.postMessage('selected:rounds', '*');
});
</script></body></html>"""
_FRAME_RETRY = (
    _FRAME_ROUNDS.replace("Select the three red squares.", "Retry: Select the three red squares.")
    .replace("let round = 0;", "let round = 0, resets = 0;")
    .replace("round ? '#238654' : '#1762bc'", "round ? '#238654' : resets ? '#9432a8' : '#1762bc'")
    .replace(
        "if ([...selected].sort((a, b) => a - b).join(',') !== '0,12,24') return;",
        """if ([...selected].sort((a, b) => a - b).join(',') !== '0,12,24') {
    round = 0; resets++; selected.clear();
    const crumbs = document.querySelectorAll('.Crumb');
    crumbs[0].style.width = '24px'; crumbs[1].style.width = '8px';
    paint(); return;
  }""",
    )
)
_FRAME_DRAG = """<!doctype html><html><head><style>
body { margin: 0; } .challenge-view { width: 280px; padding: 12px; background: white; }
canvas { width: 250px; height: 250px; cursor: grab; }
.button-submit { width: 100px; height: 32px; margin-top: 8px; cursor: pointer; }
</style></head><body><div class="challenge-view">
<h2 class="prompt-text">Drag red circle to blue square.</h2>
<canvas role="img" width="250" height="250"></canvas>
<div class="button-submit button" role="button" tabindex="0">Verify</div></div>
<script>
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const piece = {x: 25, y: 125};
let grab = null, moved = false;
function paint() {
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 250, 250);
  ctx.fillStyle = '#17b'; ctx.fillRect(210, 110, 30, 30);
  ctx.fillStyle = '#c22'; ctx.beginPath(); ctx.arc(piece.x, piece.y, 15, 0, Math.PI * 2); ctx.fill();
}
paint();
canvas.addEventListener('pointerdown', event => {
  const dx = event.offsetX - piece.x, dy = event.offsetY - piece.y;
  if (dx * dx + dy * dy < 225) {
    grab = {dx, dy};
    canvas.setPointerCapture(event.pointerId);
  }
});
canvas.addEventListener('pointermove', event => {
  if (grab) {
    piece.x = event.offsetX - grab.dx;
    piece.y = event.offsetY - grab.dy;
    paint();
  }
});
canvas.addEventListener('pointerup', () => {
  moved = grab !== null && Math.abs(piece.x - 225) < 3 && Math.abs(piece.y - 125) < 3;
  grab = null;
});
document.querySelector('.button-submit').addEventListener('click', () => {
  window.parent.postMessage('selected:' + (moved ? 'dragged' : 'none'), '*');
});
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    requests = 0
    retry_calls = 0
    sessions: ClassVar[list[str]] = []
    cells: ClassVar[list[int]] = [1]
    clicks: ClassVar[list[dict[str, object]]] = [
        {"op": "click", "x": 0.1, "y": 0.1},
        {"op": "click", "x": 0.5, "y": 0.5},
        {"op": "click", "x": 0.9, "y": 0.9},
    ]
    drags: ClassVar[list[dict[str, object]]] = [
        {
            "op": "drag",
            "from": {"x": 0.14, "y": 0.5},
            "source_center": {"x": 0.1, "y": 0.5},
            "to": {"x": 0.9, "y": 0.5},
            "hold_ms": 100,
        },
    ]

    def do_GET(self) -> None:
        if self.path not in {"/", "/mixed", "/canvas", "/drag", "/rounds", "/retry"}:
            self.send_error(404)
            return
        body = (
            _PAGE_DRAG
            if self.path == "/drag"
            else _PAGE_ROUNDS
            if self.path == "/rounds"
            else _PAGE_RETRY
            if self.path == "/retry"
            else _PAGE_CANVAS
            if self.path == "/canvas"
            else _PAGE
        )
        if self.path == "/mixed":
            body = body.replace("<iframe", '<div class="frc-captcha"></div><iframe', 1)
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if self.path != "/v1/chat/completions" or not 0 < length <= 2_900_000:
            self.send_error(400)
            return
        session = self.headers.get("X-Request-Nonce", "")
        if self.headers.get("Cookie") or self.headers.get("Proxy-Authorization") or not session:
            self.send_error(400)
            return
        try:
            if uuid.UUID(session).version != 4:
                self.send_error(400)
                return
            payload = json.loads(self.rfile.read(length))
            content = payload["messages"][1]["content"]
            if payload["model"] != "fixture-grid" or not content[1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            ):
                self.send_error(400)
                return
        except (KeyError, IndexError, TypeError, ValueError):
            self.send_error(400)
            return
        type(self).requests += 1
        type(self).sessions.append(session)
        system = str(payload["messages"][0]["content"])
        instruction = str(content[0]["text"])
        if '"actions"' in system:
            if "Retry:" in instruction:
                type(self).retry_calls += 1
                actions = [{"op": "click", "x": 0.5, "y": 0.5}] if self.retry_calls == 2 else type(self).clicks
            else:
                actions = type(self).drags if "Drag" in instruction else type(self).clicks
            selection = {"actions": actions}
        else:
            selection = {"cells": type(self).cells}
        data = json.dumps({"choices": [{"message": {"content": json.dumps(selection)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


async def _offline_launch_with_tls(*args: Any, **kwargs: Any) -> Any:
    kwargs["geoip"] = False
    existing = list(kwargs.get("args") or [])
    kwargs["args"] = [
        *existing,
        *(flag for flag in _OFFLINE_PROCESS_FLAGS if flag != "--ignore-certificate-errors" and flag not in existing),
    ]
    return await _installed_launch(*args, **kwargs)


@contextmanager
def _launch_patches(record_video_dir: Path | None = None) -> Iterator[None]:
    async def launch(*args: Any, **kwargs: Any) -> Any:
        if record_video_dir is not None:
            kwargs["record_video_dir"] = str(record_video_dir)
            kwargs["record_video_size"] = {"width": 1280, "height": 720}
        return await _offline_launch_with_tls(*args, **kwargs)

    with (
        mock.patch.object(lifecycle_startup, "launch_persistent_context_async", launch),
        mock.patch.object(driver_runtime, "launch_persistent_context_async", launch),
    ):
        yield


async def _route_fixture(context: BrowserContext, port: int) -> None:
    async def block_remote(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.scheme == "http" and url.hostname == "127.0.0.1" and url.port == port:
            await route.continue_()
        else:
            await route.abort()

    async def challenge(route: Route) -> None:
        url = route.request.url
        body = (
            _FRAME_RETRY
            if "layout=retry" in url
            else _FRAME_DRAG
            if "layout=drag" in url
            else _FRAME_ROUNDS
            if "layout=rounds" in url
            else _FRAME_CANVAS
            if "layout=canvas" in url
            else _FRAME
        )
        await route.fulfill(status=200, body=body, content_type="text/html")

    await context.route("**/*", block_remote)
    await context.route("https://newassets.hcaptcha.com/**", challenge)


async def _verify(backend: BrowserBackend, base: str, recorded: list[Page] | None = None) -> Page | None:
    ordinary = await backend.fetch(None, FetchRequest(url=base, mode="browser", timeout_seconds=20))
    if _Handler.requests or ordinary.captcha_provider or ordinary.captcha_token:
        msg = "ordinary fetch invoked grid solving"
        raise AssertionError(msg)
    before = len(recorded) if recorded is not None else 0
    solved = await backend.fetch(None, FetchRequest(url=base, mode="browser", solve_captcha=True, timeout_seconds=25))
    accepted_page = recorded[before] if recorded is not None and len(recorded) > before else None
    if _Handler.requests != 1 or solved.captcha_provider != "hcaptcha" or not solved.captcha_token:
        msg = "opted-in grid did not produce one fresh hCaptcha-shaped fixture token"
        raise AssertionError(msg)
    if solved.vision_status != "handled" or solved.vision_reason != "page_accepted":
        msg = "grid result was not projected as verified acceptance"
        raise AssertionError(msg)
    if 'data-accepted="true"' not in solved.response:
        msg = "native grid selection was not accepted by the fixture"
        raise AssertionError(msg)
    _Handler.cells = []
    empty = await backend.fetch(None, FetchRequest(url=base, mode="browser", solve_captcha=True, timeout_seconds=25))
    if _Handler.requests != 2 or empty.captcha_provider or empty.captcha_token:
        msg = "empty VLM selection fabricated a provider result"
        raise AssertionError(msg)
    if empty.vision_status != "needs_human" or empty.vision_reason != "empty_selection":
        msg = "empty VLM selection was not reported"
        raise AssertionError(msg)
    mixed = await backend.fetch(
        None, FetchRequest(url=f"{base}/mixed", mode="browser", solve_captcha=True, timeout_seconds=25)
    )
    if _Handler.requests != 2 or mixed.captcha_provider or mixed.captcha_token:
        msg = "mixed Friendly Captcha and hCaptcha providers incorrectly invoked the model"
        raise AssertionError(msg)
    if len(set(_Handler.sessions)) != 2:
        msg = "VLM requests did not use distinct opaque session values"
        raise AssertionError(msg)
    sys.stdout.write("hCaptcha-shaped grid fixture: accepted=1 empty-refused=1 mixed-provider-refused=1 uploads=2\n")
    return accepted_page


async def _verify_canvas(backend: BrowserBackend, base: str) -> None:
    result = await backend.fetch(
        None, FetchRequest(url=f"{base}/canvas", mode="browser", solve_captcha=True, timeout_seconds=25)
    )
    if _Handler.requests != 3 or result.captcha_provider != "hcaptcha" or not result.captcha_token:
        msg = "canvas native click actions did not produce a fresh fixture token"
        raise AssertionError(msg)
    if result.vision_status != "handled" or result.vision_reason != "page_accepted":
        msg = "canvas result was not projected as accepted"
        raise AssertionError(msg)
    if 'data-accepted="true"' not in result.response:
        msg = "canvas click actions were not accepted"
        raise AssertionError(msg)
    if len(set(_Handler.sessions)) != 3:
        msg = "canvas VLM request reused an opaque session value"
        raise AssertionError(msg)
    sys.stdout.write("hCaptcha-shaped canvas fixture: accepted=1 clicks=3 uploads=1\n")


async def _verify_drag(backend: BrowserBackend, base: str) -> None:
    result = await backend.fetch(
        None, FetchRequest(url=f"{base}/drag", mode="browser", solve_captcha=True, timeout_seconds=25)
    )
    if _Handler.requests != 4 or result.captcha_provider != "hcaptcha" or not result.captcha_token:
        msg = "canvas native drag did not produce a fresh fixture token"
        raise AssertionError(msg)
    if result.vision_status != "handled" or result.vision_reason != "page_accepted":
        msg = "drag result was not projected as accepted"
        raise AssertionError(msg)
    if 'data-accepted="true"' not in result.response:
        msg = "canvas drag was not accepted"
        raise AssertionError(msg)
    if len(set(_Handler.sessions)) != 4:
        msg = "drag VLM request reused an opaque session value"
        raise AssertionError(msg)
    sys.stdout.write("hCaptcha-shaped drag fixture: accepted=1 drags=1 uploads=1\n")


async def _verify_rounds(backend: BrowserBackend, base: str) -> None:
    result = await backend.fetch(
        None, FetchRequest(url=f"{base}/rounds", mode="browser", solve_captcha=True, timeout_seconds=25)
    )
    if _Handler.requests != 6 or result.captcha_provider != "hcaptcha" or not result.captcha_token:
        msg = "two-round canvas did not receive a fresh fixture token after two uploads"
        raise AssertionError(msg)
    if result.vision_status != "handled" or result.vision_reason != "page_accepted":
        msg = "two-round canvas was not reported as accepted"
        raise AssertionError(msg)
    if 'data-accepted="true"' not in result.response or len(set(_Handler.sessions)) != 6:
        msg = "two-round canvas did not validate acceptance and distinct upload sessions"
        raise AssertionError(msg)
    sys.stdout.write("hCaptcha-shaped two-round fixture: accepted=1 clicks=6 uploads=2\n")


async def _verify_retry(backend: BrowserBackend, base: str) -> None:
    result = await backend.fetch(
        None, FetchRequest(url=f"{base}/retry", mode="browser", solve_captcha=True, timeout_seconds=25)
    )
    if _Handler.requests != 10 or _Handler.retry_calls != 4:
        msg = "reset challenge did not make exactly four canvas calls across both attempts"
        raise AssertionError(msg)
    if result.captcha_provider != "hcaptcha" or not result.captcha_token:
        msg = "second-round rejection did not recover to a fresh fixture token"
        raise AssertionError(msg)
    if result.vision_status != "handled" or result.vision_reason != "page_accepted":
        msg = "retried two-page challenge was not reported as accepted"
        raise AssertionError(msg)
    if 'data-accepted="true"' not in result.response or len(set(_Handler.sessions)) != 10:
        msg = "reset challenge did not validate acceptance and distinct upload sessions"
        raise AssertionError(msg)
    sys.stdout.write("hCaptcha-shaped reset fixture: rejected=1 retries=1 accepted=1 uploads=4\n")


async def _run(video_output: Path | None = None) -> None:
    binary = _resolve_binary_path()
    _Handler.requests, _Handler.retry_calls, _Handler.sessions, _Handler.cells = 0, 0, [], [1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with tempfile.TemporaryDirectory(prefix="prowl-hcaptcha-fixture-") as tmp:
            root = Path(tmp)
            profile = root / "profile"
            config = BrowserConfig(
                profile_dir=str(profile),
                profile_archive=str(root / "profile.zip"),
                extensions_dir=None,
                policy_dir=None,
            )
            settings = VisionSettings(
                vlm=build_cloud_vlm_settings(
                    endpoint=f"{base}/v1/chat/completions",
                    model="fixture-grid",
                    headers={"X-Request-Nonce": "$uuid"},
                ),
            )
            backend = BrowserBackend(config, vision_settings=settings)
            try:
                with (
                    mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary}),
                    _launch_patches(root / "recordings" if video_output is not None else None),
                ):
                    Browser.configure(config)
                    if Browser._webdata_path().parent.parent.resolve() != profile.resolve():
                        msg = "probe profile fence failed"
                        raise RuntimeError(msg)
                    await backend.start()
                    context = (await Browser.get_context(None)).context
                    await _route_fixture(context, server.server_port)
                    recorded: list[Page] | None = [] if video_output is not None else None
                    if recorded is not None:
                        captured_pages = recorded

                        def record_page(page: Page) -> None:
                            captured_pages.append(page)

                        context.on("page", record_page)
                    accepted_page = await _verify(backend, base, recorded)
                    await _verify_canvas(backend, base)
                    await _verify_drag(backend, base)
                    await _verify_rounds(backend, base)
                    await _verify_retry(backend, base)
                    if video_output is not None:
                        if accepted_page is None or accepted_page.video is None:
                            msg = "accepted fixture page had no recording"
                            raise RuntimeError(msg)
                        video_output.parent.mkdir(parents=True, exist_ok=True)
                        await accepted_page.video.save_as(str(video_output))
                        sys.stdout.write(f"Accepted fixture recording: {video_output}\n")
            finally:
                await backend.aclose()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Save the accepted fixture page as WebM")
    asyncio.run(_run(parser.parse_args().video))
