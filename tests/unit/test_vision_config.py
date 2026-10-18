"""Validation tests for the operator vision settings in ``browser.solvers.vision.config``.

These tests cover the opt-in flags and cloud VLM settings without a browser or network.
"""

from __future__ import annotations

import dataclasses
import unittest

from prowl.browser.solvers.vision.config import (
    VISION_ENABLED_ENV,
    VISION_VLM_ENABLED_ENV,
    VISION_VLM_ENDPOINT_ENV,
    VISION_VLM_HEADERS_ENV,
    VISION_VLM_REASONING_EFFORT_ENV,
    CloudVlmSettings,
    build_cloud_vlm_settings,
    build_vision_settings,
    parse_vision_enabled,
    parse_vision_vlm_enabled,
)


class VisionEnabledFlagTests(unittest.TestCase):
    """The opt-in flag accepts exactly true/false/1/0 and defaults to off."""

    def test_blank_and_unset_default_to_disabled(self) -> None:
        self.assertFalse(parse_vision_enabled(None))
        self.assertFalse(parse_vision_enabled(""))
        self.assertFalse(parse_vision_enabled("   "))

    def test_accepts_case_insensitive_true_and_false_spellings(self) -> None:
        for raw in ("true", "TRUE", "True", "1", " 1 "):
            with self.subTest(raw=raw):
                self.assertTrue(parse_vision_enabled(raw))
        for raw in ("false", "FALSE", "False", "0", " 0 "):
            with self.subTest(raw=raw):
                self.assertFalse(parse_vision_enabled(raw))

    def test_invalid_value_is_rejected_without_echoing_the_input(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_vision_enabled("secret-maybe")
        self.assertIn(VISION_ENABLED_ENV, str(caught.exception))
        self.assertNotIn("secret-maybe", str(caught.exception))


class VisionVlmFlagTests(unittest.TestCase):
    """The VLM opt-in flag accepts exactly true/false/1/0 and names its own variable."""

    def test_blank_and_unset_default_to_disabled(self) -> None:
        for raw in (None, "", "  "):
            with self.subTest(raw=raw):
                self.assertFalse(parse_vision_vlm_enabled(raw))

    def test_accepts_true_and_false_spellings(self) -> None:
        for raw in ("true", "TRUE", "1", " 1 "):
            with self.subTest(raw=raw):
                self.assertTrue(parse_vision_vlm_enabled(raw))
        for raw in ("false", "FALSE", "0", " 0 "):
            with self.subTest(raw=raw):
                self.assertFalse(parse_vision_vlm_enabled(raw))

    def test_invalid_value_names_the_vlm_variable_without_echoing_the_input(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_vision_vlm_enabled("secret-maybe")
        message = str(caught.exception)
        self.assertIn(VISION_VLM_ENABLED_ENV, message)
        self.assertNotIn(VISION_ENABLED_ENV, message)
        self.assertNotIn("secret-maybe", message)


class CloudVlmSettingsTests(unittest.TestCase):
    """The frozen cloud settings validate through the client and hide every field from repr."""

    def test_accepts_https_and_loopback_http_endpoints(self) -> None:
        endpoints = (
            "https://api.example.com/v1/chat/completions",
            "http://127.0.0.1:11434/v1/chat/completions",
            "http://localhost:9000/v1/chat/completions",
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                settings = CloudVlmSettings(endpoint=endpoint, model="any-model")
                self.assertEqual(settings.endpoint, endpoint)
                self.assertIsNone(settings.api_key)

    def test_rejects_remote_http_and_malformed_endpoints(self) -> None:
        endpoints = (
            "http://example.com/v1/chat/completions",
            "http://192.168.0.1:8080/v1",
            "https://user:pass@api.example.com/v1",
            "https://api.example.com/v1?key=1",
            "https://api.example.com/v1#frag",
            "",
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                CloudVlmSettings(endpoint=endpoint, model="m")

    def test_rejects_a_blank_model_and_an_empty_key(self) -> None:
        builders = (
            lambda: CloudVlmSettings(endpoint="https://api.example.com/v1", model="   "),
            lambda: CloudVlmSettings(endpoint="https://api.example.com/v1", model="m", api_key=""),
        )
        for build in builders:
            with self.assertRaises(ValueError):
                build()

    def test_settings_are_frozen(self) -> None:
        settings = CloudVlmSettings(endpoint="https://api.example.com/v1", model="m")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.__setattr__("model", "other")

    def test_repr_hides_the_endpoint_model_and_key(self) -> None:
        settings = CloudVlmSettings(
            endpoint="https://secret.example.com/v1/chat/completions",
            model="private-model-id",
            api_key="sk-secret-value",
        )
        rendered = repr(settings)
        self.assertNotIn("secret.example.com", rendered)
        self.assertNotIn("private-model-id", rendered)
        self.assertNotIn("sk-secret-value", rendered)

    def test_builder_accepts_a_key_and_rejects_unusable_values(self) -> None:
        settings = build_cloud_vlm_settings(
            endpoint="https://api.example.com/v1/chat/completions",
            model="m",
            api_key="sk-key",
        )
        self.assertEqual(settings.api_key, "sk-key")
        for endpoint in (None, "", "http://example.com/v1"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError) as caught:
                build_cloud_vlm_settings(endpoint=endpoint, model="private-model-id", api_key="sk-secret")
            message = str(caught.exception)
            self.assertIn(VISION_VLM_ENDPOINT_ENV, message)
            self.assertNotIn("private-model-id", message)
            self.assertNotIn("sk-secret", message)

    def test_explicit_headers_are_parsed_and_hidden(self) -> None:
        value = "external-1928"
        settings = build_cloud_vlm_settings(
            endpoint="https://api.example.com/v1/responses",
            model="m",
            headers='{"X-Request-Nonce":"' + value + '"}',
        )
        self.assertEqual(settings.headers, (("X-Request-Nonce", value),))
        self.assertNotIn(value, repr(settings))
        self.assertEqual(build_cloud_vlm_settings(endpoint="https://api.example.com/v1", model="m").headers, ())

    def test_bad_explicit_headers_fail_without_leaking_value(self) -> None:
        for headers in (
            '{"x-open-session":"secret',
            '["secret"]',
            '{"Authorization":"secret"}',
            '{"x-open-session":"secret\\r\\nInjected: yes"}',
        ):
            with self.subTest(headers=headers), self.assertRaises(ValueError) as caught:
                build_cloud_vlm_settings(endpoint="https://api.example.com/v1", model="m", headers=headers)
            self.assertIn(VISION_VLM_HEADERS_ENV, str(caught.exception))
            self.assertNotIn("secret", str(caught.exception))

    def test_optional_minimal_reasoning_requires_responses(self) -> None:
        settings = build_cloud_vlm_settings(
            endpoint="https://api.example.com/v1/responses",
            model="m",
            reasoning_effort="minimal",
        )
        self.assertEqual(settings.reasoning_effort, "minimal")
        self.assertNotIn("minimal", repr(settings))
        for endpoint, effort in (
            ("https://api.example.com/v1/chat/completions", "minimal"),
            ("https://api.example.com/v1/responses", "verbose"),
        ):
            with self.subTest(endpoint=endpoint, effort=effort), self.assertRaises(ValueError) as caught:
                build_cloud_vlm_settings(endpoint=endpoint, model="m", reasoning_effort=effort)
            self.assertIn(VISION_VLM_REASONING_EFFORT_ENV, str(caught.exception))
            self.assertNotIn("verbose", str(caught.exception))


class VisionSettingsTests(unittest.TestCase):
    def test_geetest_only_needs_no_model_or_vlm(self) -> None:
        settings = build_vision_settings()
        self.assertIsNone(settings.vlm)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            settings.__setattr__("vlm", None)

    def test_cloud_settings_are_retained_without_leaking_credentials(self) -> None:
        vlm = build_cloud_vlm_settings(endpoint="https://api.example.com/v1/responses", model="m", api_key="private")
        settings = build_vision_settings(vlm=vlm)
        self.assertIs(settings.vlm, vlm)
        self.assertNotIn("private", repr(settings))
