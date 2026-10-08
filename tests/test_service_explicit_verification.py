"""Explicit-verification seam: a GET may request native keyboard verification.

The option is carried on ``FetchRequest.tabs_till_verify`` and surfaced on
``FetchResult.turnstile_token``. These tests use the real ``PageResponse``/``FetchRequest``/
``FetchResult`` shapes and reuse the ``_SnapshotFixture`` mocks from
``test_service_browser_snapshot``; no browser is launched and nothing external is contacted.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from prowl.browser.page_handler import PageResponse
from prowl.service.backend import FetchRequest
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from test_service_browser_snapshot import WARM_ROOT, _SnapshotFixture

VERIFY_ERROR = "explicit verification requires browser or auto GET"
TOKEN = "turnstile-token"  # noqa: S105 - a test fixture value, not a credential
LATE_TOKEN = "late-token"  # noqa: S105
LIVE_URL = "https://example.com/verified"
SCREENSHOT = "UE5H"


def _verified(*, url: str = LIVE_URL, screenshot: str | None = None, token: str | None = TOKEN) -> PageResponse:
    return PageResponse(
        source="<html><body>verified</body></html>",
        status_code=200,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url=url,
        screenshot=screenshot,
        turnstile_token=token,
    )


class ExplicitVerificationForwardingTests(_SnapshotFixture):
    """The verification option reaches the native GET once, with the caller headers."""

    async def test_zero_tabs_and_headers_reach_site_get_once(self) -> None:
        backend, group, site, events = self._backend(post_source=_verified())

        result = await backend.fetch(
            None,
            FetchRequest(url=WARM_ROOT, headers={"X-Test": "1"}, tabs_till_verify=0),
        )

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.get.await_args.args, (WARM_ROOT, 60))
        self.assertEqual(
            site.get.await_args.kwargs,
            {"headers": {"X-Test": "1"}, "header_scope": None, "tabs_till_verify": 0},
        )
        self.assertEqual(events, ["get", "cookies"])
        self.assertEqual(result.turnstile_token, TOKEN)
        self.assertEqual(result.url, LIVE_URL)
        self.assertEqual(group.quit.await_count, 1)

    async def test_absent_option_keeps_the_legacy_call_and_shape(self) -> None:
        backend, _group, site, events = self._backend(post_source=_verified())

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT))

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.get.await_args.args, (WARM_ROOT, 60))
        self.assertEqual(site.get.await_args.kwargs, {})
        self.assertEqual(events, ["get", "cookies"])
        self.assertIsNone(result.turnstile_token)
        self.assertIsNone(result.screenshot)
        self.assertIsNone(result.body_bytes)
        self.assertIsNone(result.header_items)
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)


class ExplicitVerificationCaptureTests(_SnapshotFixture):
    """The post-snapshot source supplies the token, live URL and late cookies."""

    async def test_token_url_screenshot_and_cookies_propagate_across_the_snapshot(self) -> None:
        verified = _verified()
        late = _verified(url="https://example.com/final", screenshot=SCREENSHOT, token=LATE_TOKEN)
        backend, group, site, events = self._backend(post_source=verified)

        async def _snapshot(_source: PageResponse, **_kwargs: object) -> PageResponse:
            events.append("snapshot")
            return late

        site.snapshot = AsyncMock(side_effect=_snapshot)

        result = await backend.fetch(
            None,
            FetchRequest(url=WARM_ROOT, tabs_till_verify=2, wait_in_seconds=0.5, return_screenshot=True),
        )

        self.assertEqual(events, ["get", "snapshot", "cookies"])
        self.assertEqual(site.snapshot.await_args.args[0].turnstile_token, TOKEN)
        self.assertEqual(result.turnstile_token, LATE_TOKEN)
        self.assertEqual(result.url, "https://example.com/final")
        self.assertEqual(result.screenshot, SCREENSHOT)
        self.assertEqual([cookie["name"] for cookie in result.cookies], ["cf_clearance"])
        self.assertEqual(group.quit.await_count, 1)


class AutoVerificationRoutingTests(_SnapshotFixture):
    """An auto verification goes straight to its route context with no HTTP attempt."""

    async def test_auto_verification_uses_the_route_handle_with_no_http_stage(self) -> None:
        backend, group, site, events = self._backend(post_source=_verified())
        http_stage = AsyncMock()
        backend._http_stage = http_stage

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=AUTO_MODE, tabs_till_verify=1))

        http_stage.assert_not_awaited()
        self.assertEqual(backend.metrics.http_fastpath_total, 0)
        self.assertEqual(backend.metrics.browser_escalations_total, 0)
        self.owner.get_context.assert_awaited_once_with(None)
        self.owner.create.assert_awaited_once_with(context=self.handle)
        self.assertEqual(site.get.await_args.kwargs["tabs_till_verify"], 1)
        self.assertEqual(events, ["get", "cookies"])
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(result.turnstile_token, TOKEN)
        self.assertEqual(group.quit.await_count, 1)


class ExplicitVerificationGuardTests(_SnapshotFixture):
    """An http-mode or non-GET verification is refused before any acquisition."""

    async def test_http_mode_verification_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        pool_acquire = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for
        backend._pool.acquire = pool_acquire

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=HTTP_MODE, tabs_till_verify=1))

        self.assertEqual(str(caught.exception), VERIFY_ERROR)
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()
        pool_acquire.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)

    async def test_post_verification_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        backend._steal_least_recent = steal

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, method="POST", tabs_till_verify=1))

        self.assertEqual(str(caught.exception), VERIFY_ERROR)
        steal.assert_not_awaited()


class ExplicitVerificationCleanupTests(_SnapshotFixture):
    """A failed or cancelled verification propagates and the group still closes."""

    async def test_verification_failure_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        site.get = AsyncMock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, tabs_till_verify=1))

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(group.quit.await_count, 1)

    async def test_verification_cancellation_propagates_and_the_group_still_closes(self) -> None:
        backend, group, site, _events = self._backend()
        site.get = AsyncMock(side_effect=asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, tabs_till_verify=1))

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(group.quit.await_count, 1)
