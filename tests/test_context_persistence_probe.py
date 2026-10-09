"""Focused no-browser tests for the context-state persistence probe.

These tests never launch a browser: they cover the probe's import safety, its ownership fence
ordering, its exact state predicate, its inert state page and its cleanup/exit-code behavior.
Real-browser verification is separate from these pure tests.
"""

from __future__ import annotations

import ast
import io
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from scripts import verify_context_persistence as probe

from prowl.browser import Browser
from prowl.service.backend import BrowserBackend

_PROBE_SOURCE = Path(probe.__file__)

#: Calls that must never run at import time, only from ``main()`` under the module guard.
_LAUNCH_CALLS = frozenset({"main", "run"})


class _RecordingProbe(probe._Probe):
    """A probe stand-in that keeps the results a check records."""

    def __init__(self) -> None:
        super().__init__(stream=io.StringIO())
        self.results: list[probe._Result] = []

    def record(self, result: probe._Result) -> None:
        super().record(result)
        self.results.append(result)


def _called_name(node: ast.expr) -> str | None:
    """Return the bare or attribute name a call expression targets, else ``None``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class ImportSafetyTests(TestCase):
    """Importing the module must not launch anything."""

    def test_import_does_not_launch_the_browser(self) -> None:
        """Happy path: the module imports without starting the shared browser singleton."""
        self.assertFalse(Browser.is_running())

    def test_no_top_level_launch_call_outside_the_main_guard(self) -> None:
        """Invariant: only the ``__main__`` guard may call ``main``; nothing launches at import."""
        tree = ast.parse(_PROBE_SOURCE.read_text(encoding="utf-8"))
        top_level_calls: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If):  # the ``if __name__ == "__main__"`` guard
                continue
            top_level_calls.extend(
                _called_name(child.func) or ""
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and _called_name(child.func) in _LAUNCH_CALLS
            )
        self.assertEqual(top_level_calls, [])


class StartOwnedTests(IsolatedAsyncioTestCase):
    """``_start_owned`` must configure the browser before it trusts the profile."""

    async def test_start_precedes_the_profile_fence(self) -> None:
        """Happy path: ``backend.start`` runs before the web-data path is read, with no write."""
        events: list[str] = []
        backend = Mock(spec=BrowserBackend)
        backend.start.side_effect = lambda: events.append("start")
        recorder = _RecordingProbe()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "profile/Default/Web Data"
            with patch.object(Browser, "_webdata_path", side_effect=lambda: (events.append("webdata"), data)[1]):
                ok = await probe._start_owned(recorder, backend, root)

        self.assertTrue(ok)
        self.assertEqual(events, ["start", "webdata"])
        backend.start.assert_awaited_once()
        self.assertEqual(recorder.failures, 0)

    async def test_ownership_mismatch_rejects_and_never_shuts_the_unowned_browser(self) -> None:
        """Error path: a wrong profile fails before any state write and leaves the browser alone."""
        events: list[str] = []
        backend = Mock(spec=BrowserBackend)
        backend.start.side_effect = lambda: events.append("start")
        recorder = _RecordingProbe()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrong = root / "wrong/Default/Web Data"
            with patch.object(Browser, "_webdata_path", return_value=wrong), self.assertRaises(probe._OwnershipError):
                await probe._start_owned(recorder, backend, root)

        self.assertEqual(events, ["start"])
        self.assertEqual(recorder.failures, 1)

        await probe._shutdown_owned(recorder, backend, owned=False)
        backend.aclose.assert_not_awaited()

    async def test_cleanup_failure_is_recorded_not_raised(self) -> None:
        """Error path: an owned cleanup failure increments the failure count without raising."""
        backend = Mock(spec=BrowserBackend)
        backend.aclose = AsyncMock(side_effect=RuntimeError("boom"))
        recorder = _RecordingProbe()
        await probe._shutdown_owned(recorder, backend, owned=True)
        self.assertEqual(recorder.failures, 1)


class StatePredicateTests(TestCase):
    """A marker only matches when all three surfaces carry it, and the state page is inert."""

    def test_state_ok_requires_cookie_local_storage_and_indexeddb(self) -> None:
        """Legend: a missing or mismatched surface fails, a full match passes."""
        self.assertTrue(probe._state_ok({"local": "x", "cookie": "x", "db": "x"}, "x"))
        self.assertFalse(probe._state_ok({"local": "x", "cookie": "x", "db": None}, "x"))
        self.assertFalse(probe._state_ok({"local": None, "cookie": "x", "db": "x"}, "x"))
        self.assertFalse(probe._state_ok({"local": "x", "cookie": None, "db": "x"}, "x"))
        self.assertFalse(probe._state_ok({"local": "x", "cookie": "x", "db": "other"}, "x"))

    def test_state_page_writes_no_state(self) -> None:
        """Invariant: the navigable state page never touches cookies, storage or IndexedDB."""
        for token in ("document.cookie", "localStorage", "indexedDB", "<script"):
            self.assertNotIn(token, probe._STATE_PAGE)


class StatusTests(TestCase):
    """The process exit code is nonzero on any recorded failure."""

    def test_status_is_nonzero_on_failure(self) -> None:
        """Happy path then error path: no failure maps to ``0``, any failure to nonzero."""
        recorder = _RecordingProbe()
        self.assertEqual(probe._status(recorder), 0)
        recorder.record(probe._Result("check", ok=False, detail="failed"))
        self.assertEqual(probe._status(recorder), 1)

    def test_root_json_bytes_reads_every_stored_file(self) -> None:
        """Happy path: each ``*.json`` file under the root is returned by name, byte for byte."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a.json").write_bytes(b"one")
            (root / "b.json").write_bytes(b"two")
            (root / "ignore.txt").write_bytes(b"no")
            self.assertEqual(probe._root_json_bytes(root), {"a.json": b"one", "b.json": b"two"})
