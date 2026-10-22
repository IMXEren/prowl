"""Service vision opt-in and remote VLM configuration without local model settings."""

from __future__ import annotations

import dataclasses
import os
import unittest
from unittest.mock import patch

from prowl.service.app import ServiceConfig


class VisionConfigTests(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        config = ServiceConfig()
        self.assertFalse(config.vision_enabled)
        self.assertFalse(config.vision_vlm_enabled)
        self.assertIsNone(config.vision_vlm_endpoint)

    def test_local_geetest_needs_only_opt_in(self) -> None:
        self.assertTrue(ServiceConfig(vision_enabled=True).vision_enabled)

    def test_remote_vlm_requires_the_vision_opt_in(self) -> None:
        with self.assertRaisesRegex(ValueError, "PROWL_VISION_ENABLED"):
            ServiceConfig(vision_vlm_enabled=True)

    def test_remote_vlm_requires_an_endpoint_and_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "PROWL_VISION_VLM_ENDPOINT"):
            ServiceConfig(vision_enabled=True, vision_vlm_enabled=True)

    def test_remote_vlm_settings_hide_credentials_in_repr(self) -> None:
        config = ServiceConfig(
            vision_enabled=True,
            vision_vlm_enabled=True,
            vision_vlm_endpoint="https://api.example.com/v1/responses",
            vision_vlm_model="private-model",
            vision_vlm_api_key="secret-key",
            vision_vlm_reasoning_effort="minimal",
        )
        self.assertNotIn("private-model", repr(config))
        self.assertNotIn("secret-key", repr(config))
        self.assertEqual(config.vision_vlm_reasoning_effort, "minimal")

    def test_field_names_do_not_expose_local_model_configuration(self) -> None:
        fields = {field.name for field in dataclasses.fields(ServiceConfig)}
        self.assertFalse({"vision_manifest", "vision_model_dir", "vision_allowed_origins"} & fields)

    def test_unset_env_keeps_vision_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = ServiceConfig.from_env()
        self.assertFalse(config.vision_enabled)
        self.assertFalse(config.vision_vlm_enabled)

    def test_env_enables_geetest_without_a_model(self) -> None:
        with patch.dict(os.environ, {"PROWL_VISION_ENABLED": "true"}, clear=True):
            config = ServiceConfig.from_env()
        self.assertTrue(config.vision_enabled)
        self.assertFalse(config.vision_vlm_enabled)

    def test_env_enables_remote_vlm_when_explicit(self) -> None:
        env = {
            "PROWL_VISION_ENABLED": "true",
            "PROWL_VISION_VLM_ENABLED": "true",
            "PROWL_VISION_VLM_ENDPOINT": "https://api.example.com/v1/responses",
            "PROWL_VISION_VLM_MODEL": "m",
            "PROWL_VISION_VLM_REASONING_EFFORT": "minimal",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ServiceConfig.from_env()
        self.assertTrue(config.vision_vlm_enabled)
        self.assertEqual(config.vision_vlm_reasoning_effort, "minimal")

    def test_explicit_headers_only_load_when_vlm_enabled(self) -> None:
        headers = '{"X-Request-Nonce":"external-997"}'
        env = {
            "PROWL_VISION_ENABLED": "true",
            "PROWL_VISION_VLM_ENABLED": "true",
            "PROWL_VISION_VLM_ENDPOINT": "https://api.example.com/v1/responses",
            "PROWL_VISION_VLM_MODEL": "m",
            "PROWL_VISION_VLM_HEADERS": headers,
        }
        with patch.dict(os.environ, env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.vision_vlm_headers, headers)
        self.assertNotIn("external-997", repr(config))
        with patch.dict(os.environ, {"PROWL_VISION_VLM_HEADERS": "invalid"}, clear=True):
            self.assertIsNone(ServiceConfig.from_env().vision_vlm_headers)

    def test_invalid_explicit_headers_do_not_echo_input(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ServiceConfig(
                vision_enabled=True,
                vision_vlm_enabled=True,
                vision_vlm_endpoint="https://api.example.com/v1/responses",
                vision_vlm_model="m",
                vision_vlm_headers='{"Authorization":"secret-997"}',
            )
        self.assertNotIn("secret-997", str(caught.exception))

    def test_invalid_flag_does_not_echo_input(self) -> None:
        with (
            patch.dict(os.environ, {"PROWL_VISION_ENABLED": "private"}, clear=True),
            self.assertRaises(ValueError) as caught,
        ):
            ServiceConfig.from_env()
        self.assertNotIn("private", str(caught.exception))
