"""Checkbox captcha seam: a GET may opt into one native checkbox solve.

The option is carried on ``FetchRequest.solve_captcha`` and surfaced on
``FetchResult.captcha_provider``/``captcha_token``. These tests use the real
``PageResponse``/``FetchRequest``/``FetchResult`` shapes and reuse the
``_SnapshotFixture`` mocks from ``test_service_browser_snapshot``; no browser is
launched and nothing external is contacted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from prowl.browser.page_handler import PageResponse
from prowl.service.backend import FetchRequest
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, HTTP_MODE
from test_service_browser_snapshot import WARM_ROOT, _SnapshotFixture

CAPTCHA_ERROR = "captcha solving requires browser or auto GET"
PROVIDER = "recaptcha"
TOKEN = "captcha-token"  # noqa: S105 - a test fixture value, not a credential


def _solved(*, provider: str | None = PROVIDER, token: str | None = TOKEN) -> PageResponse:
    return PageResponse(
        source="<html><body>solved</body></html>",
        status_code=200,
        headers={"content-type": "text/html"},
        user_agent="Mozilla/5.0 (Browser)",
        url=WARM_ROOT,
        captcha_provider=provider,
        captcha_token=token,
    )


class CheckboxCaptchaGuardTests(_SnapshotFixture):
    """An http-mode or non-GET solve is refused before any acquisition."""

    async def test_http_mode_solve_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        owner_for = AsyncMock()
        pool_acquire = AsyncMock()
        backend._steal_least_recent = steal
        backend._owner_for = owner_for
        backend._pool.acquire = pool_acquire

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=HTTP_MODE, solve_captcha=True))

        self.assertEqual(str(caught.exception), CAPTCHA_ERROR)
        steal.assert_not_awaited()
        owner_for.assert_not_awaited()
        pool_acquire.assert_not_awaited()
        self.assertEqual(backend.metrics.requests_active, 0)

    async def test_post_solve_is_refused_before_acquisition(self) -> None:
        backend, _group, _site, _events = self._backend()
        steal = AsyncMock()
        backend._steal_least_recent = steal

        with self.assertRaises(CallerSafeError) as caught:
            await backend.fetch(None, FetchRequest(url=WARM_ROOT, method="POST", solve_captcha=True))

        self.assertEqual(str(caught.exception), CAPTCHA_ERROR)
        steal.assert_not_awaited()


class AutoCheckboxCaptchaRoutingTests(_SnapshotFixture):
    """An auto solve goes straight to its route context with no HTTP attempt."""

    async def test_auto_solve_uses_the_route_handle_with_no_http_stage(self) -> None:
        backend, group, site, events = self._backend(post_source=_solved())
        http_stage = AsyncMock()
        backend._http_stage = http_stage

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=AUTO_MODE, solve_captcha=True))

        http_stage.assert_not_awaited()
        self.assertEqual(backend.metrics.http_fastpath_total, 0)
        self.assertEqual(backend.metrics.browser_escalations_total, 0)
        self.owner.get_context.assert_awaited_once_with(None)
        self.owner.create.assert_awaited_once_with(context=self.handle)
        self.assertEqual(site.get.await_args.kwargs, {"solve_captcha": True})
        self.assertEqual(events, ["get", "cookies"])
        self.assertEqual(result.mode, BROWSER_MODE)
        self.assertEqual(result.captcha_provider, PROVIDER)
        self.assertEqual(result.captcha_token, TOKEN)
        self.assertEqual(group.quit.await_count, 1)


class CheckboxCaptchaResultTests(_SnapshotFixture):
    """The solve is opted in once and its provider and token ride the result."""

    async def test_solve_result_carries_the_provider_and_token(self) -> None:
        backend, group, site, events = self._backend(post_source=_solved())

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, solve_captcha=True))

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.get.await_args.args, (WARM_ROOT, 60))
        self.assertEqual(site.get.await_args.kwargs, {"solve_captcha": True})
        self.assertEqual(events, ["get", "cookies"])
        self.assertIsNone(result.mode)
        self.assertEqual(result.captcha_provider, PROVIDER)
        self.assertEqual(result.captcha_token, TOKEN)
        self.assertEqual(group.quit.await_count, 1)

    async def test_absent_option_keeps_the_legacy_call_and_shape(self) -> None:
        backend, _group, site, events = self._backend(post_source=_solved())

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT))

        self.assertEqual(site.get.await_count, 1)
        self.assertEqual(site.get.await_args.args, (WARM_ROOT, 60))
        self.assertEqual(site.get.await_args.kwargs, {})
        self.assertEqual(events, ["get", "cookies"])
        self.assertIsNone(result.captcha_provider)
        self.assertIsNone(result.captcha_token)
