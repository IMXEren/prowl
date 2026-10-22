"""Project fixed CAPTCHA outcomes through the service without leaking backend text."""

from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from tests.unit.test_service_browser_snapshot import WARM_ROOT, _SnapshotFixture

from prowl.browser.solvers.vision.schemas import VisionOutcome
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchRequest, FetchResult
from prowl.service.protocol import (
    BROWSER_MODE,
    solution_payload,
    vision_payload,
)

PROVIDER = "recaptcha"
TOKEN = "captcha-token"  # noqa: S105 - a test fixture value, not a credential


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-vision-outcome-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-vision-outcome-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


def _result(**overrides: Any) -> FetchResult:
    values: dict[str, Any] = {
        "url": "https://example.com/",
        "status_code": 200,
        "headers": {"content-type": "text/html"},
        "response": "hello",
        "cookies": [],
        "user_agent": "Mozilla/5.0 (Test)",
    }
    values.update(overrides)
    return FetchResult(**values)


def _backend(result: FetchResult) -> Mock:
    backend = Mock(spec=Backend)
    backend.fetch = AsyncMock(return_value=result)
    backend.close_session = AsyncMock()
    backend.aclose = AsyncMock()
    return backend


class VisionResultProjectionTests(_SnapshotFixture):
    """A real outcome on a captcha-solving fetch is projected; everything else is untouched."""

    async def test_handled_outcome_is_projected(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = VisionOutcome(status="handled", reason="page_accepted")

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE, solve_captcha=True))

        self.assertEqual(result.vision_status, "handled")
        self.assertEqual(result.vision_reason, "page_accepted")

    async def test_needs_human_outcome_is_projected(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = VisionOutcome(status="needs_human", reason="page_not_accepted")

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE, solve_captcha=True))

        self.assertEqual(result.vision_status, "needs_human")
        self.assertEqual(result.vision_reason, "page_not_accepted")

    async def test_timed_out_outcome_is_projected(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = VisionOutcome(status="timed_out", reason="timed_out")

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE, solve_captcha=True))

        self.assertEqual(result.vision_status, "timed_out")
        self.assertEqual(result.vision_reason, "timed_out")

    async def test_ordinary_request_leaves_vision_absent(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = VisionOutcome(status="handled", reason="page_accepted")

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE))

        self.assertIsNone(result.vision_status)
        self.assertIsNone(result.vision_reason)

    async def test_solve_without_an_outcome_leaves_vision_absent(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = None

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE, solve_captcha=True))

        self.assertIsNone(result.vision_status)
        self.assertIsNone(result.vision_reason)

    async def test_a_mock_outcome_is_not_projected(self) -> None:
        backend, _group, site, _events = self._backend()
        site.vision_outcome = Mock()

        result = await backend.fetch(None, FetchRequest(url=WARM_ROOT, mode=BROWSER_MODE, solve_captcha=True))

        self.assertIsNone(result.vision_status)
        self.assertIsNone(result.vision_reason)


class VisionPayloadTests(TestCase):
    """Unknown status and reason values do not reach the response."""

    def test_a_known_status_and_reason_produce_the_object(self) -> None:
        self.assertEqual(vision_payload("handled", "page_accepted"), {"status": "handled", "reason": "page_accepted"})
        self.assertEqual(
            vision_payload("needs_human", "page_not_accepted"),
            {
                "status": "needs_human",
                "reason": "page_not_accepted",
            },
        )

    def test_live_provider_reasons_are_fixed_and_safe(self) -> None:
        pairs = (
            ("unsupported", "unsupported_layout"),
            ("needs_human", "challenge_unavailable"),
            ("needs_human", "empty_selection"),
            ("needs_human", "page_not_accepted"),
            ("needs_human", "replacement_unavailable"),
            ("needs_human", "ambiguous_selection"),
            ("needs_human", "replacement_limit"),
            ("needs_human", "page_limit"),
        )
        for status, reason in pairs:
            with self.subTest(reason=reason):
                self.assertEqual(vision_payload(status, reason), {"status": status, "reason": reason})

    def test_success_and_failure_reasons_cannot_be_swapped(self) -> None:
        self.assertEqual(vision_payload("handled", "page_accepted"), {"status": "handled", "reason": "page_accepted"})
        self.assertIsNone(vision_payload("handled", "error"))
        self.assertIsNone(vision_payload("unsupported", "page_accepted"))
        self.assertIsNone(vision_payload("timed_out", "error"))
        self.assertIsNone(vision_payload("error", "timed_out"))

    def test_a_missing_or_unknown_value_is_omitted(self) -> None:
        self.assertIsNone(vision_payload(None, "page_accepted"))
        self.assertIsNone(vision_payload("handled", None))
        self.assertIsNone(vision_payload("bogus", "page_accepted"))
        self.assertIsNone(vision_payload("handled", "raw exception: boom"))
        self.assertIsNone(vision_payload("handled", "<script>alert(1)</script>"))
        self.assertIsNone(vision_payload("needs_human", "candidate_requires_validation"))
        self.assertIsNone(vision_payload("unrecognized-status", "page_accepted"))

    def test_solution_payload_adds_vision_without_other_keys(self) -> None:
        base = {
            "url": WARM_ROOT,
            "status_code": 200,
            "headers": {"content-type": "text/html"},
            "response": "hello",
            "cookies": [],
            "user_agent": "Mozilla/5.0 (Test)",
        }

        without = solution_payload(**base)
        self.assertEqual(set(without), {"url", "status", "headers", "response", "cookies", "userAgent"})
        self.assertNotIn("vision", without)

        with_vision = solution_payload(
            **base,
            captcha_provider=PROVIDER,
            captcha_token=TOKEN,
            vision_status="needs_human",
            vision_reason="page_not_accepted",
        )
        self.assertEqual(with_vision["vision"], {"status": "needs_human", "reason": "page_not_accepted"})
        self.assertEqual(set(with_vision["vision"]), {"status", "reason"})
        self.assertNotIn(TOKEN, str(with_vision["vision"]))

    def test_an_unknown_reason_is_omitted_from_the_solution(self) -> None:
        solution = solution_payload(
            url=WARM_ROOT,
            status_code=200,
            headers={},
            response="",
            cookies=[],
            user_agent="ua",
            vision_status="handled",
            vision_reason="model-id-or-secret",
        )
        self.assertNotIn("vision", solution)


class ServiceVisionProjectionTests(IsolatedAsyncioTestCase):
    """The service projects a result's vision fields only under ``solveCaptcha`` and never opens a tab."""

    async def test_solve_captcha_projects_the_direct_result_fields(self) -> None:
        backend = _backend(_result(vision_status="handled", vision_reason="page_accepted"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({"cmd": "request.get", "url": WARM_ROOT, "solveCaptcha": True})

        self.assertEqual(body["solution"]["vision"], {"status": "handled", "reason": "page_accepted"})

    async def test_no_opt_in_omits_vision_fields_from_a_custom_backend(self) -> None:
        backend = _backend(_result(vision_status="handled", vision_reason="page_accepted"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({"cmd": "request.get", "url": WARM_ROOT})

        self.assertNotIn("vision", body["solution"])

    async def test_a_custom_backend_cannot_smuggle_arbitrary_strings(self) -> None:
        backend = _backend(_result(vision_status="handled", vision_reason="secret-token-abc"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({"cmd": "request.get", "url": WARM_ROOT, "solveCaptcha": True})

        self.assertNotIn("vision", body["solution"])
        self.assertNotIn("secret-token-abc", str(body))

    async def test_a_needs_human_result_never_opens_a_tab(self) -> None:
        backend = _backend(_result(vision_status="needs_human", vision_reason="page_not_accepted"))
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({"cmd": "request.get", "url": WARM_ROOT, "solveCaptcha": True})

        self.assertEqual(body["solution"]["vision"]["status"], "needs_human")
        backend.open_interactive.assert_not_awaited()
