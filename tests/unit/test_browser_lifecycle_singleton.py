"""A browser profile claim left by a process that is gone must not block the next launch.

Chromium refuses to start when it finds a claim it cannot disprove, so an unclean stop followed
by a restart failed until the profile was cleared by hand. These tests pin which claims are
cleared and, just as importantly, which are left alone.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Self
from unittest import TestCase

from prowl.browser.lifecycle.startup import clear_stale_singleton_files

SINGLETON_FILES = ("SingletonLock", "SingletonSocket", "SingletonCookie")


class SingletonClaimTests(TestCase):
    """Stale claims are cleared, live claims are kept, and an absent claim is a no-op."""

    def setUp(self: Self) -> None:
        """Give each test its own profile directory."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = Path(tmp.name)

    def _write_claim(self, owner: str) -> None:
        """Write a full set of claim files naming *owner*."""
        lock = self.profile / "SingletonLock"
        if lock.exists() or lock.is_symlink():
            lock.unlink()
        lock.symlink_to(owner)
        (self.profile / "SingletonSocket").write_text("socket", encoding="utf-8")
        (self.profile / "SingletonCookie").write_text("cookie", encoding="utf-8")

    def _remaining(self) -> list[str]:
        """Return the names of the entries left in the profile directory."""
        return sorted(entry.name for entry in self.profile.iterdir())

    def test_happy_path_claim_from_another_host_is_removed(self: Self) -> None:
        """Happy path: a claim naming a host that is not this one is stale, so it goes."""
        self._write_claim("another-container-4242")

        self.assertEqual(clear_stale_singleton_files(self.profile), list(SINGLETON_FILES))
        self.assertEqual(self._remaining(), [])

    def test_state_transition_claim_from_a_dead_local_process_is_removed(self: Self) -> None:
        """State transition: a claim naming this host and a finished process is stale."""
        child = subprocess.Popen([sys.executable, "-c", ""])
        child.wait(timeout=60)
        self._write_claim(f"{socket.gethostname()}-{child.pid}")

        self.assertEqual(clear_stale_singleton_files(self.profile), list(SINGLETON_FILES))
        self.assertEqual(self._remaining(), [])

    def test_invariant_claim_owned_by_a_live_process_is_left_alone(self: Self) -> None:
        """Invariant: a claim whose owner is running is a real browser, so nothing is removed."""
        self._write_claim(f"{socket.gethostname()}-{os.getpid()}")

        self.assertEqual(clear_stale_singleton_files(self.profile), [])
        self.assertEqual(self._remaining(), sorted(SINGLETON_FILES))

    def test_input_variation_unparsable_claim_is_treated_as_stale(self: Self) -> None:
        """Input variation: a claim that names no owner at all cannot be shown to be live."""
        self._write_claim("garbage")

        self.assertEqual(clear_stale_singleton_files(self.profile), list(SINGLETON_FILES))
        self.assertEqual(self._remaining(), [])

    def test_boundary_absent_claim_is_a_no_op(self: Self) -> None:
        """Boundary: an empty profile directory has nothing to clear."""
        self.assertEqual(clear_stale_singleton_files(self.profile), [])
        self.assertEqual(self._remaining(), [])

    def test_boundary_missing_profile_directory_is_a_no_op(self: Self) -> None:
        """Boundary: a profile directory that does not exist yet is not an error."""
        missing = self.profile / "not-created-yet"

        self.assertEqual(clear_stale_singleton_files(missing), [])
        self.assertEqual(self._remaining(), [])
