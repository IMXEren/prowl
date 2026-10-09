"""Focused no-browser tests for the real-browser request-routing probe.

These tests never launch a browser and never leave the loopback interface: they cover the probe's
page fixtures, its shell-to-rendered classification, its loopback request recording, its
certificate-pinning factory, its binary prerequisite and exit codes, and its cleanup orchestration
via honest mocks. The real HTTP/browser routing gate is real-browser.
"""

from __future__ import annotations

import io
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from playwright.async_api import BrowserContext as PWBrowserContext
from scripts import verify_request_routing as probe

from prowl.browser.driver.contexts import BrowserContextHandle
from prowl.service import http_transport
from prowl.service.backend import BrowserBackend
from prowl.service.classification import JAVASCRIPT_REQUIRED, SUCCESS, classify
from prowl.service.metrics import Metrics


class _RecordingProbe(probe._Probe):
    """A probe stand-in that keeps the results a check records."""

    def __init__(self) -> None:
        super().__init__(stream=io.StringIO())
        self.results: list[probe._Result] = []

    def record(self, result: probe._Result) -> None:
        super().record(result)
        self.results.append(result)


class PageFixtureTests(TestCase):
    """The JS-required fixture has to be a shell the HTTP path escalates and a browser renders."""

    def test_js_page_is_a_noscript_shell_with_an_inline_script(self) -> None:
        """The page states JavaScript is required, writes a cookie and replaces its empty root."""
        html = probe._ProbePage.JS_REQUIRED
        self.assertIn("<noscript>enable JavaScript</noscript>", html)
        self.assertIn('<div id="root"></div>', html)
        self.assertIn(f"document.cookie = '{probe._JS_COOKIE}=1; path=/'", html)
        self.assertIn(probe._JS_RENDER_TEXT, html)

    def test_shell_classifies_javascript_required_and_rendered_classifies_success(self) -> None:
        """Legend: the raw shell needs a browser, and the same page with rendered text is a success."""
        shell = classify(probe._OK, {}, probe._ProbePage.JS_REQUIRED)
        self.assertEqual(shell.category, JAVASCRIPT_REQUIRED)
        self.assertTrue(shell.browser_required)

        rendered = probe._ProbePage.JS_REQUIRED.replace(
            '<div id="root"></div>',
            f'<div id="root">{probe._JS_RENDER_TEXT}</div>',
        )
        self.assertEqual(classify(probe._OK, {}, rendered).category, SUCCESS)

    def test_index_page_is_an_ordinary_success(self) -> None:
        """The ordinary page carries no challenge signature."""
        self.assertEqual(classify(probe._OK, {}, probe._ProbePage.INDEX).category, SUCCESS)


class SiteStateRecordingTests(TestCase):
    """The loopback handler has to record POST bodies, echoed headers and cookie sends."""

    def _serve(self) -> tuple[ThreadingHTTPServer, probe._SiteState, str]:
        state = probe._SiteState()
        probe._fill_pages(state, probe._SiteState())
        server = ThreadingHTTPServer(("127.0.0.1", 0), probe._handler_class(state))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, state, f"http://127.0.0.1:{server.server_address[1]}"

    def test_post_body_echo_headers_and_cookie_report_are_recorded(self) -> None:
        """Happy path: one POST body, one echo header set, and an exact cookie presence report."""
        server, state, base = self._serve()
        try:
            post = urllib.request.Request(
                f"{base}{probe._POST_PATH}",
                data=probe._AUTO_POST_BODY.encode(),
                method="POST",
            )
            with urllib.request.urlopen(post, timeout=5) as response:
                self.assertEqual(response.status, probe._OK)
            self.assertEqual(state.post_bodies, [probe._AUTO_POST_BODY])

            echo = urllib.request.Request(
                f"{base}{probe._ECHO_PATH}?source=native",
                headers={"User-Agent": "probe-UA"},
            )
            with urllib.request.urlopen(echo, timeout=5):
                pass
            self.assertEqual(state.echo["native"]["user-agent"], "probe-UA")

            seen = urllib.request.Request(
                f"{base}{probe._REPORT_PATH}?name={probe._ROUTING_COOKIE}",
                headers={"Cookie": f"{probe._ROUTING_COOKIE}=1"},
            )
            with urllib.request.urlopen(seen, timeout=5) as response:
                self.assertTrue(probe._cookie_reported(response.read().decode()))

            absent = urllib.request.Request(
                f"{base}{probe._REPORT_PATH}?name={probe._ROUTING_COOKIE}",
                headers={"Cookie": "other=1"},
            )
            with urllib.request.urlopen(absent, timeout=5) as response:
                self.assertFalse(probe._cookie_reported(response.read().decode()))
        finally:
            server.shutdown()
            server.server_close()

    def test_set_and_delete_paths_declare_the_routing_cookie(self) -> None:
        """The Set-Cookie and Max-Age=0 declarations are wired to their paths."""
        state_a = probe._SiteState()
        probe._fill_pages(state_a, probe._SiteState())
        self.assertEqual(state_a.response_headers[probe._SET_PATH], (("Set-Cookie", "routing=1; path=/"),))
        self.assertEqual(
            state_a.response_headers[probe._DELETE_PATH],
            (("Set-Cookie", "routing=; Max-Age=0; path=/"),),
        )


class CertificatePinningTests(TestCase):
    """The factory must keep TLS verification on, pinned to the generated certificate."""

    def test_factory_forwards_with_the_generated_certificate(self) -> None:
        """Happy path: the real session is built with verify set to the certificate path."""
        cert = Path("probe-cert.pem")
        fake = mock.Mock(return_value="session")
        with mock.patch.object(http_transport, "AsyncSession", fake), probe._pinned_tls_patches(cert):
            result = http_transport.AsyncSession(impersonate="chrome142", verify=True, proxies={"all": ""})

        self.assertEqual(result, "session")
        fake.assert_called_once_with(impersonate="chrome142", verify=str(cert), proxies={"all": ""})

    def test_factory_replaces_and_restores_the_transport(self) -> None:
        """Invariant: the module attribute is swapped inside the patch and restored afterwards."""
        real = http_transport.AsyncSession
        with probe._pinned_tls_patches(Path("probe-cert.pem")):
            self.assertIsNot(http_transport.AsyncSession, real)
        self.assertIs(http_transport.AsyncSession, real)


class PrerequisiteTests(TestCase):
    """A missing binary is a hard refusal and never a download."""

    def test_missing_override_is_refused(self) -> None:
        """Error path: a configured override that does not exist is refused before any launch."""
        with tempfile.TemporaryDirectory() as raw:
            missing = str(Path(raw) / "absent.exe")
            with self.assertRaises(probe.ProbePrerequisiteError):
                probe._resolve_binary_path(
                    environ={"CLOAKBROWSER_BINARY_PATH": missing},
                    info=lambda: {"installed": False, "binary_path": "ignored"},
                )

    def test_main_returns_the_prerequisite_exit_code(self) -> None:
        """Error path: a failing prerequisite resolves to exit code 2, not a run."""
        with mock.patch.object(
            probe,
            "_resolve_binary_path",
            side_effect=probe.ProbePrerequisiteError("no binary"),
        ):
            self.assertEqual(probe.main(), 2)


class ExitCodeTests(TestCase):
    """The probe's exit code has to follow its recorded failures."""

    def _run_main(self, run_impl: Any) -> int:
        with (
            mock.patch.object(probe, "_resolve_binary_path", return_value="binary"),
            mock.patch.object(probe, "_run", run_impl),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("sys.stderr", io.StringIO()),
        ):
            return probe.main()

    def test_all_checks_passing_exits_zero(self) -> None:
        """Happy path: a run with no failures returns 0."""

        async def passing(_directory: Path, _binary: str, probe_obj: probe._Probe) -> None:
            probe_obj.record(probe._Result("check", ok=True, detail="ok"))

        self.assertEqual(self._run_main(passing), 0)

    def test_a_failing_check_exits_nonzero(self) -> None:
        """Error path: a single failed check returns 1."""

        async def failing(_directory: Path, _binary: str, probe_obj: probe._Probe) -> None:
            probe_obj.record(probe._Result("check", ok=False, detail="boom"))

        self.assertEqual(self._run_main(failing), 1)

    def test_an_abort_exits_nonzero(self) -> None:
        """Error path: an unexpected abort returns 1 rather than propagating."""

        async def aborting(_directory: Path, _binary: str, _probe_obj: probe._Probe) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        self.assertEqual(self._run_main(aborting), 1)


class ShutdownCleanupTests(IsolatedAsyncioTestCase):
    """Backend shutdown reports its own failures and never raises out of cleanup."""

    async def test_shutdown_failure_is_recorded_not_raised(self) -> None:
        """Error path: a failing aclose becomes a failed check, not an exception."""
        backend = mock.Mock()
        backend.aclose = mock.AsyncMock(side_effect=RuntimeError("boom"))
        recorder = _RecordingProbe()

        await probe._finalize_backend(recorder, probe._Counters(), backend)

        self.assertEqual(len(recorder.results), 1)
        self.assertFalse(recorder.results[0].ok)

    async def test_open_client_or_running_browser_fails_the_check(self) -> None:
        """Error path: a client still open or a running browser fails the shutdown check."""
        backend = mock.Mock()
        backend.aclose = mock.AsyncMock()
        backend._http._states = {}
        client = mock.Mock()
        client._closed = False
        recorder = _RecordingProbe()
        with mock.patch.object(probe.Browser, "is_running", return_value=True):
            await probe._finalize_backend(recorder, probe._Counters(http_fetches=[client]), backend)
        self.assertFalse(recorder.results[0].ok)

    async def test_clean_shutdown_passes(self) -> None:
        """Happy path: no owned client state, closed clients and a stopped browser pass."""
        backend = mock.Mock()
        backend.aclose = mock.AsyncMock()
        backend._http._states = {}
        client = mock.Mock()
        client._closed = True
        recorder = _RecordingProbe()
        with mock.patch.object(probe.Browser, "is_running", return_value=False):
            await probe._finalize_backend(recorder, probe._Counters(http_fetches=[client]), backend)
        self.assertTrue(recorder.results[0].ok)

    async def test_run_cleans_up_servers_and_backend_when_a_check_raises(self) -> None:
        """Error path: a failed run still shuts the backend down and closes both servers."""
        server = mock.Mock(spec=ThreadingHTTPServer)
        recorder = _RecordingProbe()
        with (
            tempfile.TemporaryDirectory() as raw,
            mock.patch.object(
                probe,
                "_write_ephemeral_tls",
                return_value=(Path(raw) / "cert.pem", Path(raw) / "key.pem"),
            ),
            mock.patch.object(probe, "_server_tls", return_value=mock.Mock()),
            mock.patch.object(
                probe,
                "_serve",
                side_effect=[
                    (server, probe._SiteState(), "https://127.0.0.1:1"),
                    (server, probe._SiteState(), "https://127.0.0.2:1"),
                ],
            ),
            mock.patch.object(probe, "_run_checks", mock.AsyncMock(side_effect=RuntimeError("boom"))),
            mock.patch.object(probe, "_finalize_backend", mock.AsyncMock()) as finalize,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            await probe._run(Path(raw), "unused-binary", recorder)

        finalize.assert_awaited_once()
        self.assertEqual(server.shutdown.call_count, 2)
        self.assertEqual(server.server_close.call_count, 2)


class RoutingMetricsCheckTests(IsolatedAsyncioTestCase):
    async def test_metrics_must_match_observations_and_release_active_gauges(self) -> None:
        metrics = Metrics(
            requests_total=3,
            http_fastpath_total=2,
            http_fastpath_success_total=2,
            browser_escalations_total=1,
            context_count=2,
            context_created_total=3,
        )
        for _ in range(3):
            metrics.request_duration_seconds.observe(0.1)
        metrics.browser_acquire_seconds.observe(0.1)
        backend = BrowserBackend(metrics=metrics)
        counters = probe._Counters(creates=[mock.Mock()], http_fetches=[mock.Mock(), mock.Mock()])
        recording = _RecordingProbe()
        handles = {
            str(index): mock.Mock(spec=BrowserContextHandle, context=mock.Mock(spec=PWBrowserContext))
            for index in range(3)
        }
        for index, closed in enumerate((False, True, False)):
            handles[str(index)].context.is_closed.return_value = closed
        with mock.patch.object(backend, "render_metrics", side_effect=metrics.render):
            await probe._check_metrics(recording, counters, backend, handles)
            self.assertTrue(recording.results[-1].ok)
            metrics.requests_active = 1
            await probe._check_metrics(recording, counters, backend, handles)
            self.assertFalse(recording.results[-1].ok)
            metrics.requests_active = 0
            metrics.http_fastpath_total = 1
            await probe._check_metrics(recording, counters, backend, handles)
            self.assertFalse(recording.results[-1].ok)
