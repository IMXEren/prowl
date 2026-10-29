"""Run Prowl's CAPTCHA solvers on an externally navigated reserved tab."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from loguru import logger

from prowl.browser.solvers.checkbox import _provider_for_frame
from prowl.browser.solvers.dispatch import solve_visible_captcha

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import Frame, Page

    from prowl.browser.browser import TabGroup
    from prowl.browser.page_handler import PageHandler

_RESERVATION_CAPTCHA_SECONDS = 30


class ReservedCaptcha:
    """Observe only a CDP reservation's page; never navigate it or transfer its tokens."""

    def __init__(self, group: TabGroup, handler: PageHandler) -> None:
        self.group = group
        self.page: Page = group.ppage
        self.handler = handler
        self._attempt: asyncio.Task[None] | None = None
        self._started = False
        self._listeners: list[tuple[str, Callable[..., None]]] = []

    async def start(self) -> None:
        """Register listeners without solving the blank tab."""
        self.handler.tab = await self.group.ptab
        self.handler.timeout = _RESERVATION_CAPTCHA_SECONDS
        self.handler.start = time.perf_counter()
        try:
            self._started = True
            for name, callback in (("domcontentloaded", self._on_loaded), ("framenavigated", self._on_frame)):
                self.page.on(name, callback)
                self._listeners.append((name, callback))
        except BaseException:
            await self.stop()
            raise

    def _on_loaded(self, *_args: object) -> None:
        if self._attempt is not None and not self._attempt.done():
            self._attempt.add_done_callback(lambda _task: self._schedule())
        else:
            self._schedule()

    def _on_frame(self, frame: Frame) -> None:
        if frame == self.page.main_frame:
            if self._attempt is not None:
                self._attempt.cancel()
            return
        host = urlsplit(frame.url).hostname
        if (
            host == "challenges.cloudflare.com"
            or _provider_for_frame(frame.url) is not None
            or (host is not None and host.endswith(".geetest.com"))
        ):
            self._schedule()

    def _schedule(self) -> None:
        if not self._started or not self.page.url.startswith(("http://", "https://")):
            return
        if self._attempt is None or self._attempt.done():
            self._attempt = asyncio.create_task(self._solve())

    async def _solve(self) -> None:
        self.handler.start = time.perf_counter()
        self.handler.timeout = _RESERVATION_CAPTCHA_SECONDS
        try:
            solved = await solve_visible_captcha(
                self.page,
                self.handler.get_time_left,
                provider_attempts=self.handler._provider_attempts(self.page),  # noqa: SLF001 - reuse fetch dispatch
            )
            if solved is not None:
                logger.debug("Reserved tab solved CAPTCHA ({})", solved[0])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - solver must not close a caller-owned tab
            logger.warning("Reserved tab CAPTCHA attempt failed ({})", type(exc).__name__)

    async def stop(self) -> None:
        """Remove this tab's listeners and stop any in-flight attempt before closing it."""
        self._started = False
        for name, callback in self._listeners:
            self.page.remove_listener(name, callback)
        self._listeners.clear()
        if self._attempt is not None:
            self._attempt.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._attempt
            self._attempt = None
