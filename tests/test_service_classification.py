"""Classifier regressions: every category, and the false positives it must never report.

Every case is one literal exchange. Nothing here touches a network or starts a browser.
"""

from __future__ import annotations

from unittest import TestCase

from prowl.service.classification import (
    AUTH_REQUIRED,
    BROWSER_REQUIRED_CATEGORIES,
    CAPTCHA,
    CLOUDFLARE_CHALLENGE,
    GEO_BLOCKED,
    IP_BLOCKED,
    JAVASCRIPT_REQUIRED,
    NETWORK_ERROR,
    RATE_LIMITED,
    SUCCESS,
    UNKNOWN_BLOCK,
    classify,
    classify_network_error,
)

_CLOUDFLARE_FRONTED_HEADERS = {
    "cf-ray": "87f1c2d3e4f5-LHR",
    "cf-cache-status": "HIT",
    "server": "cloudflare",
    "content-type": "text/html; charset=UTF-8",
}

_CHALLENGE_BODY = (
    "<html><head><title>Just a moment...</title></head><body>"
    '<form id="challenge-form" action="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></form>'
    "</body></html>"
)

_CAPTCHA_BODY = '<html><body><div class="cf-turnstile" data-sitekey="0x4AAA"></div></body></html>'

_JS_SHELL_BODY = (
    '<html><head><script src="/static/app.js"></script></head><body>'
    "<noscript>You need to enable JavaScript to run this app.</noscript>"
    '<div id="root"></div></body></html>'
)


class OrdinaryResponseTests(TestCase):
    """A response without a signature stays an ordinary success or an unnamed block."""

    def test_plain_page_is_success(self) -> None:
        result = classify(200, {"content-type": "text/html"}, "<html><body>hello</body></html>")

        self.assertEqual(result.category, SUCCESS)
        self.assertFalse(result.browser_required)

    def test_cloudflare_fronted_page_with_scripts_is_success(self) -> None:
        body = '<html><head><script src="/app.js"></script></head><body>ok</body></html>'

        result = classify(200, _CLOUDFLARE_FRONTED_HEADERS, body)

        self.assertEqual(result.category, SUCCESS)
        self.assertFalse(result.browser_required)

    def test_page_that_merely_mentions_a_captcha_is_success(self) -> None:
        body = "<html><body><p>We use a captcha to protect this form.</p></body></html>"

        self.assertEqual(classify(200, {}, body).category, SUCCESS)

    def test_redirect_status_is_success(self) -> None:
        self.assertEqual(classify(302, {"location": "/next"}, "").category, SUCCESS)

    def test_plain_403_is_an_unnamed_block_that_does_not_need_a_browser(self) -> None:
        body = '<html><body><h1>Access Denied</h1><script src="/a.js"></script></body></html>'

        result = classify(403, {}, body)

        self.assertEqual(result.category, UNKNOWN_BLOCK)
        self.assertFalse(result.browser_required)

    def test_generic_error_statuses_are_not_success(self) -> None:
        for status in (404, 418, 500, 502):
            with self.subTest(status=status):
                result = classify(status, {}, "<html><body>error</body></html>")

                self.assertEqual(result.category, UNKNOWN_BLOCK)
                self.assertFalse(result.browser_required)

    def test_captcha_widget_on_a_served_page_is_not_a_block(self) -> None:
        # A login page can legitimately carry a widget; only a refusal makes it a challenge.
        self.assertEqual(classify(200, {}, _CAPTCHA_BODY).category, SUCCESS)


class ChallengeTests(TestCase):
    """Explicit challenge signatures are named, and only they require a browser."""

    def test_cf_mitigated_challenge_header_is_a_cloudflare_challenge(self) -> None:
        result = classify(403, {"cf-mitigated": "challenge"}, "")

        self.assertEqual(result.category, CLOUDFLARE_CHALLENGE)
        self.assertTrue(result.browser_required)

    def test_cf_mitigated_challenge_header_is_honoured_on_a_success_status(self) -> None:
        result = classify(200, {"cf-mitigated": "Challenge"}, "")

        self.assertEqual(result.category, CLOUDFLARE_CHALLENGE)

    def test_cf_mitigated_with_another_value_is_not_a_challenge(self) -> None:
        self.assertEqual(classify(403, {"cf-mitigated": "block"}, "").category, UNKNOWN_BLOCK)

    def test_challenge_platform_body_is_a_cloudflare_challenge(self) -> None:
        result = classify(503, {"server": "cloudflare"}, _CHALLENGE_BODY)

        self.assertEqual(result.category, CLOUDFLARE_CHALLENGE)
        self.assertTrue(result.browser_required)

    def test_cloudflare_interstitial_with_a_noscript_is_still_a_cloudflare_challenge(self) -> None:
        body = (
            "<html><body><noscript>Please enable JavaScript to continue.</noscript>"
            '<form id="challenge-form"></form></body></html>'
        )

        self.assertEqual(classify(403, {}, body).category, CLOUDFLARE_CHALLENGE)

    def test_turnstile_interstitial_is_a_captcha(self) -> None:
        result = classify(403, {}, _CAPTCHA_BODY)

        self.assertEqual(result.category, CAPTCHA)
        self.assertTrue(result.browser_required)

    def test_authentication_status_is_not_browser_required(self) -> None:
        for status in (401, 407):
            with self.subTest(status=status):
                result = classify(status, {"www-authenticate": "Basic"}, "")

                self.assertEqual(result.category, AUTH_REQUIRED)
                self.assertFalse(result.browser_required)

    def test_forbidden_with_an_authenticate_header_asks_for_authentication(self) -> None:
        self.assertEqual(classify(403, {"www-authenticate": 'Basic realm="x"'}, "").category, AUTH_REQUIRED)

    def test_rate_limit_is_not_browser_required(self) -> None:
        result = classify(429, {"retry-after": "60"}, "")

        self.assertEqual(result.category, RATE_LIMITED)
        self.assertFalse(result.browser_required)

    def test_legal_restriction_status_is_a_region_block(self) -> None:
        result = classify(451, {}, "")

        self.assertEqual(result.category, GEO_BLOCKED)
        self.assertFalse(result.browser_required)

    def test_refusal_body_that_names_the_region_is_a_region_block(self) -> None:
        body = "<html><body>This service is not available in your country.</body></html>"

        self.assertEqual(classify(403, {}, body).category, GEO_BLOCKED)

    def test_refusal_body_that_names_the_address_is_an_address_block(self) -> None:
        body = "<html><body>Your IP address has been banned.</body></html>"

        result = classify(403, {}, body)

        self.assertEqual(result.category, IP_BLOCKED)
        self.assertFalse(result.browser_required)

    def test_explicit_javascript_shell_is_browser_required(self) -> None:
        result = classify(200, {"content-type": "text/html"}, _JS_SHELL_BODY)

        self.assertEqual(result.category, JAVASCRIPT_REQUIRED)
        self.assertTrue(result.browser_required)

    def test_rendered_page_with_noscript_warning_is_not_a_shell(self) -> None:
        body = "<head><title>News</title></head><body><article>Readable news.</article>" + _JS_SHELL_BODY

        self.assertEqual(classify(200, {}, body).category, SUCCESS)

    def test_classification_does_not_scan_beyond_bounded_prefix(self) -> None:
        body = "x" * 65536 + '<script src="/cdn-cgi/challenge-platform/test.js"></script>'

        self.assertEqual(classify(403, {}, body).category, UNKNOWN_BLOCK)

    def test_visible_text_asking_to_enable_javascript_is_not_a_shell(self) -> None:
        body = "<html><body><p>Please enable JavaScript to use our search.</p></body></html>"

        self.assertEqual(classify(200, {}, body).category, SUCCESS)


class ClassificationShapeTests(TestCase):
    """The result is small, immutable and internally consistent."""

    def test_network_error_names_only_the_exception_type(self) -> None:
        result = classify_network_error(TimeoutError("http://user:secret@proxy.internal:8080"))

        self.assertEqual(result.category, NETWORK_ERROR)
        self.assertFalse(result.browser_required)
        self.assertIn("TimeoutError", result.reason)
        self.assertNotIn("secret", result.reason)

    def test_browser_required_follows_the_category(self) -> None:
        result = classify(200, {}, _JS_SHELL_BODY)

        self.assertEqual(result.browser_required, result.category in BROWSER_REQUIRED_CATEGORIES)

    def test_result_is_immutable(self) -> None:
        result = classify(200, {}, "")

        with self.assertRaises(AttributeError):
            result.category = SUCCESS  # type: ignore[misc]
