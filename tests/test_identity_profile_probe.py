"""No-browser regressions for temporary native profile relocation."""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import BrowserContext
from scripts import verify_identity_profiles as probe

from prowl.browser.driver.contexts import BrowserContextHandle
from prowl.browser.proxy.egress import EgressPool
from prowl.service.backend import BrowserBackend


class _RecordingProbe(probe._Probe):
    """A probe stand-in that keeps the results a check records."""

    def __init__(self) -> None:
        super().__init__(stream=io.StringIO())
        self.results: list[probe._Result] = []

    def record(self, result: probe._Result) -> None:
        super().record(result)
        self.results.append(result)


def _write_user_data_root(root: Path, cookie: bytes) -> None:
    """Create a synthetic Chromium user-data root carrying *cookie* bytes."""
    (root / "Default").mkdir(parents=True)
    (root / "Local State").write_bytes(b"local-state")
    (root / "Default" / "Cookies").write_bytes(cookie)


class FenceOrderTests(IsolatedAsyncioTestCase):
    """``_open_owned`` must fence the launched profile before it reads any native handle."""

    async def test_mismatched_profile_is_rejected_before_the_context_getter(self) -> None:
        """Error path: a wrong profile raises before ``get_context`` is awaited at all."""
        recorder = _RecordingProbe()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrong = root / "other" / "Default" / "Web Data"
            with (
                patch.object(probe.Browser, "_webdata_path", return_value=wrong),
                patch.object(probe.Browser, "start", new=AsyncMock()) as start,
                patch.object(probe.Browser, "get_context", new=AsyncMock()) as getter,
                self.assertRaises(probe._OwnershipError),
            ):
                await probe._open_owned(recorder, probe.Browser.start, probe.Browser, root / "profile")

        start.assert_not_awaited()
        getter.assert_not_awaited()
        self.assertEqual(recorder.failures, 1)

    async def test_matching_profile_returns_the_shared_context(self) -> None:
        """Happy path: a matching fence passes and the shared context is read afterwards."""
        recorder = _RecordingProbe()
        sentinel = BrowserContextHandle(None, Mock(spec=BrowserContext), object())
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "profile" / "Default" / "Web Data"
            with (
                patch.object(probe.Browser, "_webdata_path", return_value=data),
                patch.object(probe.Browser, "start", new=AsyncMock()),
                patch.object(probe.Browser, "get_context", new=AsyncMock(return_value=sentinel)) as getter,
            ):
                handle = await probe._open_owned(recorder, probe.Browser.start, probe.Browser, root / "profile")

        self.assertIs(handle, sentinel)
        getter.assert_awaited_once_with(None)
        self.assertEqual(recorder.failures, 0)


class ArchiveTransitionTests(TestCase):
    """The probe's real migration seam separates a nested named root from the default identity."""

    def test_transition_moves_the_nested_root_and_cleans_the_default_bundle(self) -> None:
        """Happy path: the synthetic nested root moves to its sibling and the bundle is cleaned."""
        with TemporaryDirectory() as raw:
            base = Path(raw).resolve()
            default_root = base / "profile"
            nested = default_root / "one"
            _write_user_data_root(nested, b"named-cookie")
            (default_root / "Default" / "Network").mkdir(parents=True)
            (default_root / "Local State").write_bytes(b"default-state")
            (default_root / "Default" / "Network" / "Cookies").write_bytes(b"main-cookie")
            standalone = base / "profile-one.zip"
            standalone.write_bytes(b"standalone-named-archive")
            archive = base / "profile.zip"

            result, original = probe._transition_named(default_root, archive, ["one"])

            sibling = base / "profile-one"
            self.assertTrue(sibling.is_dir())
            self.assertFalse(nested.exists())
            self.assertEqual((sibling / "Default" / "Cookies").read_bytes(), b"named-cookie")
            self.assertTrue(result.archive_rewritten)
            self.assertIsNotNone(result.backup)
            assert result.backup is not None
            self.assertTrue(result.backup.is_file())
            self.assertEqual(probe._archive_digest(result.backup), original)
            members = probe._archive_members(archive)
            self.assertIn("Default/Network/Cookies", members)
            self.assertFalse(any(name.startswith("one/") for name in members))
            self.assertEqual(standalone.read_bytes(), b"standalone-named-archive")


class CookieFixtureTests(TestCase):
    """The identity cookie fixture must be persistent and scoped to the synthetic origin."""

    def test_identity_cookie_is_persistent_and_url_scoped(self) -> None:
        """Happy path: the cookie carries a real future expiry and an https url scope, no domain."""
        before = time.time()
        cookie = probe._persistent_identity(probe._MAIN)
        after = time.time()

        self.assertEqual(cookie.get("name"), probe._COOKIE_NAME)
        self.assertEqual(cookie.get("value"), "main")
        self.assertEqual(cookie.get("url"), probe._PROFILE_URL)
        self.assertNotIn("path", cookie)
        self.assertTrue(cookie.get("secure"))
        self.assertTrue(cookie.get("httpOnly"))
        self.assertEqual(cookie.get("sameSite"), "Lax")
        self.assertNotIn("domain", cookie)
        expiry = cookie.get("expires")
        assert expiry is not None
        self.assertGreater(expiry, after)
        self.assertGreaterEqual(expiry, before + probe._COOKIE_LIFETIME_SECONDS)


class CleanupOrderTests(IsolatedAsyncioTestCase):
    """Cleanup releases the egress claim, drains the pool and closes the backend, in order."""

    async def test_release_then_pool_then_backend(self) -> None:
        """Happy path: the claim is released, then the pool and the backend are closed in order."""
        events: list[str] = []
        recorder = _RecordingProbe()
        pool = Mock(spec=EgressPool)
        pool.release = AsyncMock(side_effect=lambda name: events.append(f"release:{name}"))
        pool.aclose = AsyncMock(side_effect=lambda: events.append("pool.aclose"))
        backend = Mock(spec=BrowserBackend)
        backend.aclose = AsyncMock(side_effect=lambda: events.append("backend.aclose"))

        await probe._close_owners(pool, "one", backend, acquired=True)

        self.assertEqual(events, ["release:one", "pool.aclose", "backend.aclose"])
        self.assertEqual(recorder.failures, 0)

    async def test_a_failed_pool_close_still_closes_the_backend(self) -> None:
        """Error path: a failing pool close is recorded and the backend is still closed."""
        events: list[str] = []
        recorder = _RecordingProbe()
        pool = Mock(spec=EgressPool)
        pool.aclose = AsyncMock(side_effect=RuntimeError("pool boom"))
        backend = Mock(spec=BrowserBackend)
        backend.aclose = AsyncMock(side_effect=lambda: events.append("backend.aclose"))

        with self.assertRaisesRegex(RuntimeError, "pool boom"):
            await probe._close_owners(pool, "one", backend, acquired=False)

        pool.release.assert_not_awaited()
        self.assertEqual(events, ["backend.aclose"])
        self.assertEqual(recorder.failures, 0)
