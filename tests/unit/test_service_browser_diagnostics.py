"""Rendered DOM diagnostics skip HTTP shell routing but retain bounded challenge detection."""

from __future__ import annotations

from unittest import TestCase

from prowl.service.backend import FetchResult, _classified
from prowl.service.classification import (
    AUTH_REQUIRED,
    CAPTCHA,
    CLOUDFLARE_CHALLENGE,
    GEO_BLOCKED,
    JAVASCRIPT_REQUIRED,
    RATE_LIMITED,
    SUCCESS,
    UNKNOWN_BLOCK,
    classify,
)
from prowl.service.protocol import BROWSER_MODE, HTTP_MODE

_NOSCRIPT = "<noscript>enable javascript</noscript>"
#: The reproduced case: the noscript shell and its script sit at the start, so the HTTP prefix
#: bound cuts the rendered ``<main>`` content away and only the shell is seen.
_LARGE_RENDERED_DOM = _NOSCRIPT + "<script>" + "x" * 65536 + "</script><main>Rendered content</main>"
_JS_SHELL_BODY = (
    '<html><head><script src="/static/app.js"></script></head><body>'
    "<noscript>You need to enable JavaScript to run this app.</noscript>"
    '<div id="root"></div></body></html>'
)
_TURNSTILE_VALUE = "value-carried-through"


class RenderedDomShellTests(TestCase):
    """A rendered DOM never re-triggers the HTTP-only JavaScript-shell heuristic."""

    def test_large_rendered_dom_is_success_where_the_http_sample_is_a_shell(self) -> None:
        http = classify(200, {"content-type": "text/html"}, _LARGE_RENDERED_DOM)
        rendered = classify(200, {"content-type": "text/html"}, _LARGE_RENDERED_DOM, rendered=True)

        self.assertEqual(http.category, JAVASCRIPT_REQUIRED)
        self.assertTrue(http.browser_required)
        self.assertEqual(rendered.category, SUCCESS)
        self.assertFalse(rendered.browser_required)

    def test_rendered_shell_without_visible_content_skips_the_routing_heuristic(self) -> None:
        self.assertEqual(classify(200, {}, _NOSCRIPT).category, JAVASCRIPT_REQUIRED)
        self.assertEqual(classify(200, {}, _NOSCRIPT, rendered=True).category, SUCCESS)

    def test_small_rendered_dom_with_visible_content_is_success(self) -> None:
        body = _NOSCRIPT + "<main>Rendered content</main>"

        self.assertEqual(classify(200, {}, body, rendered=True).category, SUCCESS)


class RenderedDomChallengeTests(TestCase):
    """A rendered DOM still reports the real challenge and refusal signatures it carries."""

    def test_real_cloudflare_and_captcha_signatures_are_retained(self) -> None:
        cf_header = classify(403, {"cf-mitigated": "challenge"}, "", rendered=True)
        self.assertEqual(cf_header.category, CLOUDFLARE_CHALLENGE)
        self.assertTrue(cf_header.browser_required)

        challenge_body = (
            '<html><body><form id="challenge-form" '
            'action="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></form></body></html>'
        )
        self.assertEqual(classify(403, {}, challenge_body, rendered=True).category, CLOUDFLARE_CHALLENGE)

        captcha_body = '<html><body><div class="cf-turnstile" data-sitekey="0x4AAA"></div></body></html>'
        captcha = classify(403, {}, captcha_body, rendered=True)
        self.assertEqual(captcha.category, CAPTCHA)
        self.assertTrue(captcha.browser_required)

    def test_auth_rate_and_geo_verdicts_stay_honest(self) -> None:
        self.assertEqual(classify(401, {"www-authenticate": "Basic"}, "", rendered=True).category, AUTH_REQUIRED)
        self.assertEqual(classify(429, {}, "", rendered=True).category, RATE_LIMITED)
        self.assertEqual(classify(451, {}, "", rendered=True).category, GEO_BLOCKED)

    def test_unknown_and_missing_status_stay_unnamed_blocks(self) -> None:
        self.assertEqual(classify(500, {}, "<html><body>error</body></html>", rendered=True).category, UNKNOWN_BLOCK)

        result = _classified(FetchResult("https://example.com/", None, {}, "<html><body>error</body></html>", [], None))

        assert result.classification is not None
        self.assertEqual(result.classification.category, UNKNOWN_BLOCK)
        self.assertFalse(result.classification.browser_required)


class ClassifiedBrowserResultTests(TestCase):
    """``_classified`` forces browser mode and preserves every other field of the input DTO."""

    def test_classified_returns_a_new_result_carrying_the_browser_stage(self) -> None:
        original = FetchResult(
            "https://example.com/",
            200,
            {"content-type": "text/html"},
            _NOSCRIPT + "<main>Rendered content</main>",
            [{"name": "a", "value": "1"}],
            "Mozilla/5.0 (Test)",
            mode=HTTP_MODE,
            body_bytes=b"<html></html>",
            header_items=(("content-type", "text/html"),),
            screenshot="base64-shot",
            turnstile_token=_TURNSTILE_VALUE,
        )

        result = _classified(original)

        self.assertIsNot(result, original)
        self.assertEqual(result.mode, BROWSER_MODE)
        assert result.classification is not None
        self.assertEqual(result.classification.category, SUCCESS)
        self.assertEqual(result.url, original.url)
        self.assertEqual(result.status_code, original.status_code)
        self.assertEqual(result.headers, original.headers)
        self.assertEqual(result.response, original.response)
        self.assertEqual(result.cookies, original.cookies)
        self.assertEqual(result.user_agent, original.user_agent)
        self.assertEqual(result.body_bytes, b"<html></html>")
        self.assertEqual(result.header_items, (("content-type", "text/html"),))
        self.assertEqual(result.screenshot, "base64-shot")
        self.assertEqual(result.turnstile_token, _TURNSTILE_VALUE)
        self.assertEqual(original.mode, HTTP_MODE)
        self.assertIsNone(original.classification)


class HttpDefaultClassificationTests(TestCase):
    """The default HTTP reading, including its 64 KiB bound, is unchanged."""

    def test_default_call_keeps_the_shell_verdict_and_the_prefix_bound(self) -> None:
        self.assertEqual(classify(200, {}, _JS_SHELL_BODY).category, JAVASCRIPT_REQUIRED)

        beyond_prefix = "x" * 65536 + '<script src="/cdn-cgi/challenge-platform/test.js"></script>'
        self.assertEqual(classify(403, {}, beyond_prefix).category, UNKNOWN_BLOCK)
        self.assertEqual(classify(403, {}, beyond_prefix, rendered=True).category, UNKNOWN_BLOCK)
