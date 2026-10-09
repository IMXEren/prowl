"""Pure regressions for temporary-profile crash-probe startup and durability checks."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch
from zipfile import ZipFile

from playwright.async_api import BrowserContext
from scripts import verify_browser_recovery as probe

from prowl.browser import Browser, BrowserContextHandle
from prowl.service.backend import BrowserBackend


class RecoveryProbeTests(IsolatedAsyncioTestCase):
    async def test_configuration_precedes_cookie_write_and_flush_precedes_disk_check(self) -> None:
        events: list[str] = []
        backend = Mock(spec=BrowserBackend)
        backend.start.side_effect = lambda: events.append("start")
        context = Mock(spec=BrowserContext)
        context.add_cookies.side_effect = lambda _cookies: events.append("write")
        handle = Mock(spec=BrowserContextHandle)
        handle.context = context
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(Browser, "_webdata_path", return_value=root / "profile/Default/Web Data"),
                patch.object(Browser, "get_context", new=AsyncMock(return_value=handle)),
                patch.object(Browser, "shutdown", new=AsyncMock(side_effect=lambda: events.append("flush"))),
                patch.object(
                    probe, "_wait_cookie_durable", new=AsyncMock(side_effect=lambda *_args: events.append("disk"))
                ),
            ):
                await probe._prepare_durable_profile(backend, "https://127.0.0.1:12345", root)
            with ZipFile(root / "profile.zip") as archive:
                self.assertEqual(archive.read("recovery-marker"), b"stale")
        self.assertEqual(events, ["start", "write", "flush", "disk", "start"])

    async def test_wrong_profile_is_refused_before_cookie_or_shutdown(self) -> None:
        backend = Mock(spec=BrowserBackend)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(Browser, "_webdata_path", return_value=root / "wrong/Default/Web Data"),
                patch.object(Browser, "get_context", new=AsyncMock()) as get_context,
                patch.object(Browser, "shutdown", new=AsyncMock()) as shutdown,
            ):
                with self.assertRaisesRegex(RuntimeError, "outside its temporary profile"):
                    await probe._prepare_durable_profile(backend, "https://127.0.0.1:12345", root)
                get_context.assert_not_awaited()
                shutdown.assert_not_awaited()
        backend.start.assert_awaited_once()

    async def test_durability_reads_an_actual_persistent_cookie_row(self) -> None:
        with TemporaryDirectory() as temporary:
            profile = Path(temporary)
            path = profile / "Default/Network/Cookies"
            path.parent.mkdir(parents=True)
            with closing(sqlite3.connect(path)) as database:
                database.execute("CREATE TABLE cookies (name TEXT, host_key TEXT, is_persistent INTEGER)")
                database.execute("INSERT INTO cookies VALUES ('durable', '127.0.0.1', 1)")
                database.commit()
            await probe._wait_cookie_durable(profile, "durable")

    async def test_missing_binary_fails_before_server_or_launch(self) -> None:
        message = "missing existing binary"
        with (
            patch.object(probe, "_resolve_binary_path", side_effect=RuntimeError(message)),
            patch.object(probe, "_serve") as serve,
            patch.object(Browser, "start", new=AsyncMock()) as start,
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                await probe.run()
            serve.assert_not_called()
            start.assert_not_awaited()
