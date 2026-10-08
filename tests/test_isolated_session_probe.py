"""Focused no-browser tests for the owner-run isolated-session probe.

These tests never launch a browser: they cover the probe's page fixtures, its cookie-report
parsing, its binary prerequisites, its offline launch wrapper, its check sequencing and its
cleanup guarantees. The real-browser policy/fingerprint/isolation gates are owner-run.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, mock

try:
    from cryptography import x509

    _HAVE_CRYPTOGRAPHY = True
except ImportError:
    x509 = None
    _HAVE_CRYPTOGRAPHY = False

from scripts import verify_isolated_sessions as probe


class _RecordingProbe(probe._Probe):
    """A probe stand-in that keeps the results a check records."""

    def __init__(self) -> None:
        super().__init__(stream=io.StringIO())
        self.results: list[probe._Result] = []

    def record(self, result: probe._Result) -> None:
        super().record(result)
        self.results.append(result)


class StateFixtureTests(TestCase):
    """The state fixtures must not write state themselves, or a read would reset the sentinel."""

    def test_state_fixtures_are_inert(self) -> None:
        """Happy path: navigable state pages never touch cookies, storage or IndexedDB."""
        for name in ("_SITE_A_INDEX", "_SITE_A_POPUP_TARGET", "_SITE_B_INDEX"):
            html = getattr(probe, name)
            self.assertNotIn("document.cookie", html, name)
            self.assertNotIn("localStorage", html, name)
            self.assertNotIn("indexedDB", html, name)

    def test_cross_site_page_only_listens_and_embeds(self) -> None:
        """The site A third-party page embeds the B frame and only listens for its report."""
        rendered = probe._SITE_A_THIRD_PARTY.format(B="https://127.0.0.2:9999")
        self.assertNotIn("document.cookie", rendered)
        self.assertIn("<iframe", rendered)
        self.assertIn("https://127.0.0.2:9999/frame", rendered)
        self.assertIn("'message'", rendered)


class CookieFixtureTests(TestCase):
    """The cookie fixtures must set Secure/SameSite=None cookies and report server-observed sends."""

    def test_site_b_control_sets_a_first_party_secure_cookie(self) -> None:
        """The first-party control writes a Secure/SameSite=None cookie and reports it."""
        html = probe._SITE_B_CONTROL
        self.assertIn("b_control=1", html)
        self.assertIn("Secure", html)
        self.assertIn("SameSite=None", html)
        self.assertIn("/report?name=b_control", html)
        self.assertIn("__probeControl", html)

    def test_site_b_frame_sets_a_secure_third_party_cookie_and_reports(self) -> None:
        """The cross-site iframe writes the same kind of cookie and posts the observed result."""
        html = probe._SITE_B_FRAME
        self.assertIn("b_third=1", html)
        self.assertIn("Secure", html)
        self.assertIn("SameSite=None", html)
        self.assertIn("/report?name=b_third", html)
        self.assertIn("postMessage", html)

    def test_cookie_present_matches_only_the_exact_name(self) -> None:
        """Legend: an exact cookie name is matched, a prefix and an empty name are not."""
        self.assertTrue(probe._cookie_present("a=1; b_third=1", "b_third"))
        self.assertTrue(probe._cookie_present("b_third=1", "b_third"))
        self.assertFalse(probe._cookie_present("b_thirdish=1", "b_third"))
        self.assertFalse(probe._cookie_present("b_third", "b_third"))
        self.assertFalse(probe._cookie_present("", "b_third"))
        self.assertFalse(probe._cookie_present("b_third=1", ""))


class BinaryPrerequisiteTests(TestCase):
    """The binary prerequisite must refuse anything that would trigger a download."""

    def test_existing_override_is_honored(self) -> None:
        """Happy path: an existing CLOAKBROWSER_BINARY_PATH is returned unchanged."""
        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "chrome.exe"
            binary.write_bytes(b"binary")
            resolved = probe._resolve_binary_path(
                environ={"CLOAKBROWSER_BINARY_PATH": str(binary)},
                info=lambda: {"installed": False, "binary_path": "ignored"},
            )
            self.assertEqual(resolved, str(binary))

    def test_missing_override_is_refused(self) -> None:
        """Error path: a configured-override path that does not exist is a hard refusal."""
        with tempfile.TemporaryDirectory() as raw:
            missing = str(Path(raw) / "absent.exe")
            with self.assertRaises(probe.ProbePrerequisiteError):
                probe._resolve_binary_path(
                    environ={"CLOAKBROWSER_BINARY_PATH": missing},
                    info=lambda: {"installed": True, "binary_path": "ignored"},
                )

    def test_absent_installed_binary_is_refused(self) -> None:
        """Error path: with no override and nothing installed the probe refuses to run."""
        with self.assertRaises(probe.ProbePrerequisiteError):
            probe._resolve_binary_path(environ={}, info=lambda: {"installed": False, "binary_path": "/x"})

    def test_installed_binary_is_used(self) -> None:
        """Input variation: the read-only binary_info installation path is used when installed."""
        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "chrome.exe"
            binary.write_bytes(b"binary")
            resolved = probe._resolve_binary_path(
                environ={},
                info=lambda: {"installed": True, "binary_path": str(binary)},
            )
            self.assertEqual(resolved, str(binary))


class OfflineWrapperTests(IsolatedAsyncioTestCase):
    """The offline launch wrapper must force GeoIP off without adding identity overrides."""

    async def test_wrapper_forces_geoip_off_and_preserves_arguments(self) -> None:
        """Happy path: caller arguments survive, geoip is off, offline flags are appended."""
        captured: dict[str, Any] = {}

        async def fake_launch(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "context"

        with mock.patch.object(probe, "_installed_launch", fake_launch):
            result = await probe._offline_launch(
                headless=False,
                args=["--existing"],
                viewport={"width": 800, "height": 600},
                locale="en-US",
            )

        self.assertEqual(result, "context")
        self.assertFalse(captured["geoip"])
        self.assertEqual(captured["args"][0], "--existing")
        self.assertEqual(captured["viewport"], {"width": 800, "height": 600})
        self.assertEqual(captured["locale"], "en-US")
        self.assertTrue(set(probe._OFFLINE_PROCESS_FLAGS).issubset(captured["args"]))

    async def test_wrapper_never_injects_identity_or_proxy_overrides(self) -> None:
        """Invariant: no synthetic user agent, timezone, locale or proxy is added."""
        captured: dict[str, Any] = {}

        async def fake_launch(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "context"

        with mock.patch.object(probe, "_installed_launch", fake_launch):
            await probe._offline_launch(args=["--existing"])

        self.assertNotIn("user_agent", captured)
        self.assertNotIn("timezone", captured)
        self.assertNotIn("timezone_id", captured)
        self.assertNotIn("proxy", captured)
        self.assertNotIn("--proxy-server", " ".join(captured["args"]))

    async def test_wrapper_does_not_duplicate_an_existing_flag(self) -> None:
        """Boundary: an offline flag already present in args is not added twice."""
        flag = probe._OFFLINE_PROCESS_FLAGS[0]
        captured: dict[str, Any] = {}

        async def fake_launch(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "context"

        with mock.patch.object(probe, "_installed_launch", fake_launch):
            await probe._offline_launch(args=[flag])

        self.assertEqual(captured["args"].count(flag), 1)
        self.assertIn("--disable-component-update", captured["args"])

    def test_patches_cover_both_launch_seams_and_restore(self) -> None:
        """Happy path: both Prowl launch modules are patched, then restored on exit."""
        startup = probe.lifecycle_startup
        runtime = probe.driver_runtime
        before_startup = startup.launch_persistent_context_async
        before_runtime = runtime.launch_persistent_context_async

        with probe._offline_launch_patches():
            self.assertIs(startup.launch_persistent_context_async, probe._offline_launch)
            self.assertIs(runtime.launch_persistent_context_async, probe._offline_launch)

        self.assertIs(startup.launch_persistent_context_async, before_startup)
        self.assertIs(runtime.launch_persistent_context_async, before_runtime)


class CookieGateTests(IsolatedAsyncioTestCase):
    """The third-party gate must fail unless both contexts actually accept the cookie."""

    async def _run_gate(self, outcome: probe._CookiePolicy) -> _RecordingProbe:
        async def fake_policy(_a_url: str, _b_url: str, _handle: object) -> probe._CookiePolicy:
            return outcome

        recorder = _RecordingProbe()
        with mock.patch.object(probe, "_cookie_policy", fake_policy):
            await probe._check_cookie_policy(
                recorder,
                "https://127.0.0.1:1",
                "https://127.0.0.2:1",
                {"shared": object(), "iso-1": object()},
            )
        return recorder

    async def test_false_false_parity_fails(self) -> None:
        """Error path: a shared/isolated false parity on third-party cookies fails the gate."""
        recorder = await self._run_gate(probe._CookiePolicy(control=True, third_party=False))
        self.assertEqual(len(recorder.results), 1)
        self.assertFalse(recorder.results[0].ok)

    async def test_accepted_in_both_contexts_passes(self) -> None:
        """Happy path: control and third-party acceptance in both contexts passes the gate."""
        recorder = await self._run_gate(probe._CookiePolicy(control=True, third_party=True))
        self.assertTrue(recorder.results[0].ok)

    async def test_failed_control_is_inconclusive_and_fails(self) -> None:
        """Error path: a failed first-party control fails the gate instead of passing silently."""
        recorder = await self._run_gate(probe._CookiePolicy(control=False, third_party=True))
        self.assertFalse(recorder.results[0].ok)
        self.assertIn("control failed", recorder.results[0].detail)


class CheckSequenceTests(IsolatedAsyncioTestCase):
    """A failing check is recorded and the remaining checks still run."""

    async def test_failure_is_recorded_and_iteration_continues(self) -> None:
        """Error path: an unexpected exception becomes that check's failure, not an abort."""
        recorder = probe._Probe(stream=io.StringIO())
        calls: list[str] = []

        async def bad() -> None:
            calls.append("bad")
            msg = "boom"
            raise RuntimeError(msg)

        async def good() -> None:
            calls.append("good")
            recorder.record(probe._Result("good", ok=True, detail="ok"))

        await probe._run_checks_sequence(recorder, [("bad", bad), ("good", good)])

        self.assertEqual(calls, ["bad", "good"])
        self.assertEqual(recorder.total, 2)
        self.assertEqual(recorder.failures, 1)


class CleanupTests(IsolatedAsyncioTestCase):
    """Cleanup reports its own failures but never masks a primary error or stops early."""

    async def test_best_effort_never_propagates_a_cleanup_failure(self) -> None:
        """Error path: a failing cleanup step is swallowed so callers keep going."""
        ran: list[str] = []

        async def bad() -> None:
            ran.append("bad")
            msg = "cleanup boom"
            raise RuntimeError(msg)

        async def good() -> None:
            ran.append("good")

        await probe._best_effort("bad", bad)
        await probe._best_effort("good", good)

        self.assertEqual(ran, ["bad", "good"])

    async def test_browser_cleanup_survives_failures(self) -> None:
        """Error path: browser cleanup failures do not raise out of the finally block."""

        async def boom(*_args: object, **_kwargs: object) -> None:
            msg = "cleanup failure"
            raise RuntimeError(msg)

        with (
            mock.patch.object(probe.Browser, "close_context", boom),
            mock.patch.object(probe.Browser, "shutdown", boom),
        ):
            await probe._cleanup_browser()

    def test_server_cleanup_stops_and_closes_every_server(self) -> None:
        """Happy path: each server is both shut down and socket-closed."""
        calls: list[str] = []

        class _FakeServer:
            def shutdown(self) -> None:
                calls.append("shutdown")

            def server_close(self) -> None:
                calls.append("server_close")

        probe._cleanup_servers(mock.Mock(wraps=_FakeServer()))
        self.assertEqual(calls, ["shutdown", "server_close"])

    def test_server_cleanup_survives_a_failing_shutdown(self) -> None:
        """Error path: a failing shutdown still attempts server_close and does not raise."""
        calls: list[str] = []

        class _FakeServer:
            def shutdown(self) -> None:
                calls.append("shutdown")
                msg = "boom"
                raise RuntimeError(msg)

            def server_close(self) -> None:
                calls.append("server_close")

        probe._cleanup_servers(mock.Mock(wraps=_FakeServer()))
        self.assertEqual(calls, ["shutdown", "server_close"])


@unittest.skipUnless(_HAVE_CRYPTOGRAPHY, "cryptography is not installed")
class TlsCertificateTests(TestCase):
    """The ephemeral loopback certificate must cover both probe hosts and stay temporary."""

    def test_certificate_covers_both_loopback_hosts(self) -> None:
        """Happy path: the self-signed certificate carries IP SANs for 127.0.0.1 and 127.0.0.2."""
        assert x509 is not None
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            cert_path, key_path = probe._write_ephemeral_tls(directory)

            self.assertTrue(cert_path.is_file())
            self.assertTrue(key_path.is_file())
            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            addresses = {str(address) for address in san.get_values_for_type(x509.IPAddress)}

        self.assertEqual(addresses, {"127.0.0.1", "127.0.0.2"})


class ProbeReviewRegressions(IsolatedAsyncioTestCase):
    async def test_isolation_preserves_each_contexts_existing_sentinel(self) -> None:
        from functools import partial  # noqa: PLC0415

        shared, one, two = object(), object(), object()
        jars = {
            shared: {"local": "shared", "cookie": "shared", "db": "shared"},
            one: dict(probe._EMPTY_STATE),
            two: dict(probe._EMPTY_STATE),
        }

        async def group_action(handle: object, _url: str, action: Any) -> Any:
            if isinstance(action, partial) and action.func is probe._set_state:
                marker = action.keywords["marker"]
                jars[handle] = dict.fromkeys(("local", "cookie", "db"), marker)
                return None
            self.assertIs(action, probe._read_state)
            return dict(jars[handle])

        recorder = _RecordingProbe()
        with (
            mock.patch.object(probe, "_with_group", group_action),
            mock.patch.object(probe.Browser, "get_context", mock.AsyncMock(return_value=one)) as get_context,
        ):
            await probe._check_isolation(recorder, "https://127.0.0.1:1", shared, one, two)
        get_context.assert_awaited_once_with(probe._ISOLATED_ONE)
        self.assertTrue(recorder.results[0].ok)

    def test_absent_database_read_aborts_creation(self) -> None:
        self.assertIn("open.transaction.abort()", probe._IDB_READ_JS)

    async def test_failed_browser_start_still_runs_cleanup(self) -> None:
        with (
            mock.patch.object(probe.Browser, "start", mock.AsyncMock(side_effect=RuntimeError("start failed"))),
            mock.patch.object(probe, "_cleanup_browser", mock.AsyncMock(return_value=True)) as cleanup,
            self.assertRaisesRegex(RuntimeError, "start failed"),
        ):
            await probe._run_checks(probe._Probe(stream=io.StringIO()), "https://127.0.0.1:1", "https://127.0.0.2:1")
        cleanup.assert_awaited_once()

    async def test_cleanup_failure_makes_a_successful_probe_fail(self) -> None:
        recorder = probe._Probe(stream=io.StringIO())
        with (
            mock.patch.object(probe.Browser, "start", mock.AsyncMock()),
            mock.patch.object(probe.Browser, "get_context", mock.AsyncMock(return_value=object())),
            mock.patch.object(probe, "_run_checks_sequence", mock.AsyncMock()),
            mock.patch.object(probe, "_cleanup_browser", mock.AsyncMock(return_value=False)),
        ):
            await probe._run_checks(recorder, "https://127.0.0.1:1", "https://127.0.0.2:1")
        self.assertEqual(recorder.failures, 1)

    def test_binary_directory_is_not_an_executable_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as raw, self.assertRaises(probe.ProbePrerequisiteError):
            probe._resolve_binary_path(environ={"CLOAKBROWSER_BINARY_PATH": raw})

    def test_stale_installed_binary_metadata_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw, self.assertRaises(probe.ProbePrerequisiteError):
            probe._resolve_binary_path(
                environ={},
                info=lambda: {"installed": True, "binary_path": str(Path(raw) / "absent")},
            )

    async def test_second_server_failure_closes_the_first_server(self) -> None:
        server = mock.Mock(spec=probe.ThreadingHTTPServer)
        recorder = probe._Probe(stream=io.StringIO())
        with (
            tempfile.TemporaryDirectory() as raw,
            mock.patch.object(probe, "_tls_context", return_value=mock.Mock()),
            mock.patch.object(
                probe,
                "_serve",
                side_effect=[(server, probe._SiteState(), "https://127.0.0.1:1"), RuntimeError("second server")],
            ),
            self.assertRaisesRegex(RuntimeError, "second server"),
        ):
            await probe._run(Path(raw), "unused-binary", recorder)
        server.shutdown.assert_called_once()
        server.server_close.assert_called_once()
