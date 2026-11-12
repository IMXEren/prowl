"""Reader-writer lock built from two mutexes."""

# Ref: https://gist.githubusercontent.com/tylerneylon/a7ff6017b7a1f9a506cf75aa23eacfd6/raw/6c8829fa44671a87557a49167f7283401c2d55df/rwlock.py

from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock


class RWLock:
    """Allow concurrent readers or one exclusive writer."""

    def __init__(self) -> None:
        self.w_lock = Lock()
        self.num_r_lock = Lock()
        self.num_r = 0

    def r_acquire(self) -> None:
        """Acquire a shared read lock."""
        self.num_r_lock.acquire()
        self.num_r += 1
        if self.num_r == 1:
            self.w_lock.acquire()
        self.num_r_lock.release()

    def r_release(self) -> None:
        """Release a shared read lock."""
        assert self.num_r > 0  # noqa: S101
        self.num_r_lock.acquire()
        self.num_r -= 1
        if self.num_r == 0:
            self.w_lock.release()
        self.num_r_lock.release()

    @contextmanager
    def r_locked(self) -> Generator[None, None, None]:
        """Hold a shared read lock for the block."""
        try:
            self.r_acquire()
            yield
        finally:
            self.r_release()

    def w_acquire(self) -> None:
        """Acquire the exclusive write lock."""
        self.w_lock.acquire()

    def w_release(self) -> None:
        """Release the exclusive write lock."""
        self.w_lock.release()

    @contextmanager
    def w_locked(self) -> Generator[None, None, None]:
        """Hold the exclusive write lock for the block."""
        try:
            self.w_acquire()
            yield
        finally:
            self.w_release()
