"""Basic lock sharing and context-manager cleanup."""

from unittest import TestCase

from prowl.structures import RWLock


class RWLockTests(TestCase):
    def test_readers_share_the_lock_until_the_last_release(self) -> None:
        lock = RWLock()
        lock.r_acquire()
        lock.r_acquire()
        self.assertEqual(lock.num_r, 2)
        self.assertFalse(lock.w_lock.acquire(blocking=False))
        lock.r_release()
        self.assertEqual(lock.num_r, 1)
        self.assertTrue(lock.w_lock.locked())
        lock.r_release()
        self.assertEqual(lock.num_r, 0)
        self.assertFalse(lock.w_lock.locked())

    def test_write_context_is_exclusive_and_releases(self) -> None:
        lock = RWLock()
        with lock.w_locked():
            self.assertTrue(lock.w_lock.locked())
            self.assertFalse(lock.w_lock.acquire(blocking=False))
        self.assertFalse(lock.w_lock.locked())

    def test_read_context_releases_after_an_exception(self) -> None:
        lock = RWLock()
        error = RuntimeError("fixture")
        with self.assertRaisesRegex(RuntimeError, "fixture"), lock.r_locked():
            raise error
        self.assertEqual(lock.num_r, 0)
        self.assertFalse(lock.w_lock.locked())
        self.assertFalse(lock.num_r_lock.locked())

    def test_write_context_releases_after_an_exception(self) -> None:
        lock = RWLock()
        error = RuntimeError("fixture")
        with self.assertRaisesRegex(RuntimeError, "fixture"), lock.w_locked():
            raise error
        self.assertFalse(lock.w_lock.locked())

    def test_read_release_requires_a_reader(self) -> None:
        lock = RWLock()
        with self.assertRaises(AssertionError):
            lock.r_release()
        self.assertFalse(lock.w_lock.locked())
        self.assertFalse(lock.num_r_lock.locked())
