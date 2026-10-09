"""Offline hard-crash gate: python -m scripts.verify_browser_recovery.

Only the browser launched with this script's temporary profile is terminated.
Shared cookies must already be durable before the crash; unflushed writes are not guaranteed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from zipfile import ZipFile

from prowl.browser import Browser, BrowserConfig
from prowl.browser.exceptions import BrowserContextError
from prowl.service.backend import BrowserBackend, FetchRequest, InteractiveRequest
from prowl.service.http_transport import HttpTransportError
from scripts.verify_isolated_sessions import (
    _emit,
    _offline_launch_patches,
    _Probe,
    _resolve_binary_path,
    _Result,
    _write_ephemeral_tls,
)
from scripts.verify_request_routing import (
    _INDEX_PATH,
    _cookie_reported,
    _counted_seams,
    _Counters,
    _pinned_tls_patches,
    _ProbePage,
    _serve,
    _server_tls,
)


async def _browser_pid() -> int:
    native = Browser.pw_main_ctx().browser
    if native is None:
        message = "Owned browser has no native process projection"
        raise RuntimeError(message)
    session = await native.new_browser_cdp_session()
    try:
        result = await session.send("SystemInfo.getProcessInfo")
        return next(int(item["id"]) for item in result["processInfo"] if item["type"] == "browser")
    finally:
        await session.detach()


async def _wait_cookie_durable(profile: Path, name: str) -> None:
    deadline = time.monotonic() + 45
    last_error = "cookie row absent"
    while time.monotonic() < deadline:
        for path in (profile / "Default/Network/Cookies", profile / "Default/Cookies"):
            if not path.is_file():
                continue
            try:
                with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=0.1)) as db:
                    found = db.execute(
                        "SELECT 1 FROM cookies WHERE name=? AND host_key=? AND is_persistent=1",
                        (name, "127.0.0.1"),
                    ).fetchone()
                if found:
                    return
            except sqlite3.OperationalError as exc:
                last_error = str(exc)
                continue
        await asyncio.sleep(0.1)
    message = f"Persistent cookie not readable within 45s: {last_error}"
    raise TimeoutError(message)


async def _prepare_durable_profile(backend: BrowserBackend, url: str, root: Path) -> None:
    await backend.start()
    if Browser._webdata_path().parent.parent.resolve() != (root / "profile").resolve():  # noqa: SLF001
        message = "Refusing crash probe outside its temporary profile"
        raise RuntimeError(message)
    shared = await Browser.get_context()
    await shared.context.add_cookies(
        [
            {"name": "durable", "value": "1", "url": url, "expires": time.time() + 3600},
        ]
    )
    await Browser.shutdown()
    await _wait_cookie_durable(root / "profile", "durable")
    with ZipFile(root / "profile.zip", "w") as zipped:
        zipped.writestr("recovery-marker", "stale")
    await backend.start()


async def _checks(backend: BrowserBackend, url: str, root: Path, probe: _Probe, counters: _Counters) -> None:
    _emit("Preparing committed shared state before the hard-crash gate...", stream=sys.stdout)
    await _prepare_durable_profile(backend, url, root)
    await backend.fetch(None, FetchRequest(url=url, mode="http"))
    shared = await Browser.get_context()
    await backend.fetch(None, FetchRequest(url=url, mode="http", session_id="recover", session_mode="isolated"))
    isolated = await Browser.get_context("recover")
    await isolated.context.add_cookies([{"name": "ephemeral", "value": "1", "url": url}])
    old_client = counters.http_fetches[-1]
    tab = await backend.open_interactive(InteractiveRequest(url=url))
    native = shared.context.browser
    if native is None:
        message = "No owned native browser to terminate"
        raise RuntimeError(message)
    disconnected = asyncio.Event()
    native.once("disconnected", lambda _browser: disconnected.set())
    old_pid = await _browser_pid()
    os.kill(old_pid, signal.SIGTERM)
    await asyncio.wait_for(disconnected.wait(), timeout=15)
    probe.record(_Result("owned process hard-terminated", not Browser.is_running(), str(old_pid)))
    seen = await backend.fetch(None, FetchRequest(url=f"{url}/cookie-report?name=durable", mode="http"))
    new_shared = await Browser.get_context()
    new_pid = await _browser_pid()
    probe.record(
        _Result(
            "new process preserves durable shared cookie",
            _cookie_reported(seen.response) and new_pid != old_pid,
            str(new_pid),
        )
    )
    seen = await backend.fetch(
        None,
        FetchRequest(
            url=f"{url}/cookie-report?name=ephemeral",
            mode="http",
            session_id="recover",
            session_mode="isolated",
        ),
    )
    new_isolated = await Browser.get_context("recover")
    new_client = counters.http_fetches[-1]
    probe.record(
        _Result(
            "isolated context refreshed without old state",
            new_isolated is not isolated and not _cookie_reported(seen.response),
            "ephemeral state reset",
        )
    )
    try:
        await old_client.fetch(url, deadline_seconds=5)
    except HttpTransportError as exc:
        retired = str(exc) == "the HTTP client is closed"
    else:
        retired = False
    probe.record(
        _Result("old HTTP client retired", old_client is not new_client and retired, "client generation changed")
    )
    listed = await backend.list_interactive()
    probe.record(
        _Result(
            "dead interactive tab retired", all(item.tab_id != tab.tab.tab_id for item in listed), "no old tab listed"
        )
    )
    probe.record(
        _Result(
            "stale archive never overwrites live profile",
            (root / "profile/recovery-marker").read_text(encoding="utf-8") == "live",
            "live marker preserved",
        )
    )
    try:
        unexpected = await Browser.create(context=shared)
    except BrowserContextError:
        rejected = True
    else:
        await unexpected.quit()
        rejected = False
    probe.record(
        _Result("old-generation handle rejected", rejected and new_shared is not shared, "ownership epoch changed")
    )


async def run() -> int:
    """Run a temporary-profile recovery gate and return a failing exit code on any failed check."""
    binary = _resolve_binary_path()
    probe = _Probe()
    with TemporaryDirectory(prefix="prowl-recovery-") as temporary:
        root = Path(temporary)
        profile = root / "profile"
        profile.mkdir()
        (profile / "recovery-marker").write_text("live", encoding="utf-8")
        archive = root / "profile.zip"
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("recovery-marker", "stale")
        cert, key = _write_ephemeral_tls(root)
        server, state, url = _serve("127.0.0.1", _server_tls(cert, key))
        state.pages[_INDEX_PATH] = _ProbePage.INDEX
        backend = BrowserBackend(
            BrowserConfig(
                profile_dir=str(profile),
                profile_archive=str(archive),
                extensions_dir=None,
                policy_dir=None,
                proxy_url=None,
            )
        )
        counters = _Counters()
        with (
            mock.patch.dict(os.environ, {"CLOAKBROWSER_BINARY_PATH": binary}),
            _offline_launch_patches(),
            _pinned_tls_patches(cert),
            _counted_seams(counters),
        ):
            try:
                await _checks(backend, url, root, probe, counters)
            except Exception as exc:
                logging.getLogger(__name__).exception("Hard-crash gate failed")
                probe.error("hard-crash gate", exc)
            finally:
                try:
                    await backend.aclose()
                except Exception as exc:
                    logging.getLogger(__name__).exception("Recovery cleanup failed")
                    probe.error("backend cleanup", exc)
                finally:
                    await asyncio.to_thread(server.shutdown)
                    server.server_close()
    _emit(f"Recovery checks: {probe.total - probe.failures}/{probe.total} passed", stream=sys.stdout)
    return int(probe.failures > 0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
