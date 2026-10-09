"""Deferred media suppression is browser-only and scopes the whole site fetch."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from prowl.service.backend import FetchRequest
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from test_service_browser_snapshot import WARM_ROOT, _SnapshotFixture, _source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from prowl.browser.page_handler import PageResponse

OPTION_ERROR = "browser response options require browser or auto mode"


def _recording_media_filter(site: Mock, events: list[str]) -> Mock:
    """Install a ``media_filter`` double that records entering and leaving its scope."""

    @contextlib.asynccontextmanager
    async def scope() -> AsyncIterator[None]:
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    site.media_filter = Mock(side_effect=scope)
    return site.media_filter


def _failing_media_filter(site: Mock, error: BaseException) -> Mock:
    """Install a ``media_filter`` double whose scope fails on entry."""

    @contextlib.asynccontextmanager
    async def scope() -> AsyncIterator[None]:
        raise error
        yield

    site.media_filter = Mock(side_effect=scope)
    return site.media_filter


class MediaFilterScopeTests(_SnapshotFixture):
    """A media request opens one page-local scope around the fetch, capture and cookie read."""

    async def test_default_request_never_opens_a_media_scope(self) -> None:
        backend, group, site, events = self._backend()
        media_filter = Mock()
        site.media_filter = media_filter

        await backend.fetch(None, FetchRequest(url=WARM_ROOT))

        media_filter.assert_not_called()
        self.assertEqual(events, ["get", "cookies"])
        self.assertEqual(group.quit.await_count, 1)

    async def test_scoped_get_runs_inside_the_media_scope(self) -> None:
        backend, group, site, events = self._backend()
        media_filter = _recording_media_filter(site, events)

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, disable_media=True))

        self.assertEqual(media_filter.call_count, 1)
        self.assertEqual(events, ["enter", "get", "cookies", "exit"])
        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.post.await_count, 0)
        self.assertEqual(result.response, _source().text)
        self.assertEqual(group.quit.await_count, 1)

    async def test_scoped_post_snapshot_orders_the_body_before_the_scope_closes(self) -> None:
        late = _source(body="<html><body>late</body></html>", screenshot="UE5H")
        backend, group, site, events = self._backend()

        async def _snapshot(_captured: PageResponse, **_kwargs: object) -> PageResponse:
            events.append("snapshot")
            return late

        site.snapshot = AsyncMock(side_effect=_snapshot)
        media_filter = _recording_media_filter(site, events)

        result = await backend.fetch(
            None,
            FetchRequest(
                url=WARM_ROOT,
                method="POST",
                post_data="q=1",
                wait_in_seconds=0.5,
                return_screenshot=True,
                disable_media=True,
            ),
        )

        self.assertEqual(media_filter.call_count, 1)
        self.assertEqual(events, ["enter", "post", "snapshot", "cookies", "exit"])
        self.assertEqual(site.post.await_count, 1)
        self.assertEqual(site.get.await_count, 0)
        self.assertEqual(site.snapshot.await_args.kwargs["post_response"], True)
        self.assertEqual(result.screenshot, "UE5H")
        self.assertEqual(group.quit.await_count, 1)


class MediaFilterFailureTests(_SnapshotFixture):
    """A failing or cancelled media scope propagates and the request group still closes."""

    async def test_media_scope_failure_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        _failing_media_filter(site, RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, disable_media=True))

        self.assertEqual(site.get.await_count, 0)
        self.assertEqual(group.quit.await_count, 1)

    async def test_media_scope_cancellation_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        _failing_media_filter(site, asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, disable_media=True))

        self.assertEqual(group.quit.await_count, 1)


class AutoMediaRoutingTests(_SnapshotFixture):
    """An auto media fetch goes straight to the route's browser context without an HTTP attempt."""

    async def test_auto_disable_media_skips_the_http_stage(self) -> None:
        backend, group, site, events = self._backend()
        _recording_media_filter(site, events)
        http_stage = AsyncMock()
        backend._http_stage = http_stage

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=AUTO_MODE, disable_media=True))

        http_stage.assert_not_awaited()
        self.assertEqual(backend.metrics.http_fastpath_total, 0)
        self.assertEqual(backend.metrics.browser_escalations_total, 0)
        self.owner.get_context.assert_awaited_once_with(None)
        self.owner.create.assert_awaited_once_with(context=self.handle)
        self.assertEqual(events, ["enter", "get", "cookies", "exit"])
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertIsNotNone(result.classification)
        self.assertEqual(group.quit.await_count, 1)


class ExplicitHttpMediaGuardTests(_SnapshotFixture):
    """An explicit http request that asks for media suppression is refused before acquisition."""

    async def test_http_disable_media_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        pool_acquire = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for
        backend._pool.acquire = pool_acquire

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=HTTP_MODE, disable_media=True))

        self.assertEqual(str(caught.exception), OPTION_ERROR)
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()
        pool_acquire.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)


class MediaFilterBodyFailureTests(_SnapshotFixture):
    async def test_failed_navigation_retires_filter_before_closing_group(self) -> None:
        backend, group, site, events = self._backend()
        _recording_media_filter(site, events)
        site.get = AsyncMock(side_effect=RuntimeError("navigation failed"))
        with self.assertRaises(RuntimeError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, disable_media=True))
        self.assertEqual(events, ["enter", "exit"])
        group.quit.assert_awaited_once()
        self.assertEqual(backend.metrics.requests_active, 0)

    async def test_cleanup_failure_propagates_after_cookies_and_group_close(self) -> None:
        backend, group, site, events = self._backend()

        @contextlib.asynccontextmanager
        async def fail_cleanup() -> AsyncIterator[None]:
            yield
            events.append("cleanup")
            message = "cleanup failed"
            raise RuntimeError(message)

        site.media_filter = Mock(side_effect=fail_cleanup)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, disable_media=True))
        self.assertEqual(events, ["get", "cookies", "cleanup"])
        group.quit.assert_awaited_once()
        self.assertEqual(backend.metrics.requests_active, 0)
