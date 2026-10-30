"""Validated opt-in settings for local GeeTest and cloud VLM CAPTCHA adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final

#: Process-level opt-in for visual CAPTCHA handling.
VISION_ENABLED_ENV: Final = "PROWL_VISION_ENABLED"
VISION_VLM_ENABLED_ENV: Final = "PROWL_VISION_VLM_ENABLED"
VISION_VLM_ENDPOINT_ENV: Final = "PROWL_VISION_VLM_ENDPOINT"
VISION_VLM_MODEL_ENV: Final = "PROWL_VISION_VLM_MODEL"
VISION_VLM_API_KEY_ENV: Final = "PROWL_VISION_VLM_API_KEY"
VISION_VLM_HEADERS_ENV: Final = "PROWL_VISION_VLM_HEADERS"
VISION_VLM_REASONING_EFFORT_ENV: Final = "PROWL_VISION_VLM_REASONING_EFFORT"

_TRUE_VALUES: Final = frozenset({"true", "1"})
_FALSE_VALUES: Final = frozenset({"false", "0"})
_ENABLED_INVALID_MESSAGE: Final = f"{VISION_ENABLED_ENV} must be one of true, false, 1 or 0"
_VLM_ENABLED_INVALID_MESSAGE: Final = f"{VISION_VLM_ENABLED_ENV} must be one of true, false, 1 or 0"
_VLM_SETTINGS_INVALID_MESSAGE: Final = (
    f"{VISION_VLM_ENDPOINT_ENV}, {VISION_VLM_MODEL_ENV}, {VISION_VLM_API_KEY_ENV}, "
    f"{VISION_VLM_HEADERS_ENV} and {VISION_VLM_REASONING_EFFORT_ENV} must form an "
    "OpenAI-compatible HTTPS or loopback HTTP endpoint, model and valid optional settings"
)


def _parse_flag(raw: str | None, *, invalid_message: str) -> bool:
    value = (raw or "").strip().lower()
    if not value:
        return False
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(invalid_message)


def parse_vision_enabled(raw: str | None) -> bool:
    """Parse the operator's process-level opt-in, defaulting to disabled."""
    return _parse_flag(raw, invalid_message=_ENABLED_INVALID_MESSAGE)


def parse_vision_vlm_enabled(raw: str | None) -> bool:
    """Parse the remote VLM opt-in, defaulting to disabled."""
    return _parse_flag(raw, invalid_message=_VLM_ENABLED_INVALID_MESSAGE)


def build_vision_settings(*, vlm: CloudVlmSettings | None = None) -> VisionSettings:
    """Enable local GeeTest and optionally remote VLM CAPTCHA attempts."""
    return VisionSettings(vlm=vlm)


def build_cloud_vlm_settings(
    *,
    endpoint: object,
    model: object,
    api_key: object = None,
    reasoning_effort: object = None,
    headers: object = None,
) -> CloudVlmSettings:
    """Validate the remote VLM endpoint and optional credentials without connecting."""
    if (
        not isinstance(endpoint, str)
        or not isinstance(model, str)
        or (api_key is not None and not isinstance(api_key, str))
        or (reasoning_effort is not None and not isinstance(reasoning_effort, str))
    ):
        raise ValueError(_VLM_SETTINGS_INVALID_MESSAGE)
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except (ValueError, TypeError) as error:
            raise ValueError(_VLM_SETTINGS_INVALID_MESSAGE) from error
    if headers is not None and not isinstance(headers, dict):
        raise ValueError(_VLM_SETTINGS_INVALID_MESSAGE)
    return CloudVlmSettings(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        headers=tuple(headers.items()) if headers is not None else (),
    )


@dataclass(frozen=True, slots=True)
class CloudVlmSettings:
    """Validated remote VLM settings; no endpoint, model or key leaks through repr."""

    endpoint: str = field(repr=False)
    model: str = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    reasoning_effort: str | None = field(default=None, repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """Reject invalid settings before any browser or network work."""
        _validate_cloud_vlm(self.endpoint, self.model, self.api_key, self.reasoning_effort, self.headers)


def _validate_cloud_vlm(
    endpoint: str,
    model: str,
    api_key: str | None,
    reasoning_effort: str | None,
    headers: tuple[tuple[str, str], ...],
) -> None:
    from prowl.vision.schemas import VisionError  # noqa: PLC0415 - imported only on the enabled path
    from prowl.vision.vlm import CloudVlmClient  # noqa: PLC0415 - validated without opening a session

    try:
        checked_headers = dict(headers)
        if len(checked_headers) != len(headers):
            raise ValueError(_VLM_SETTINGS_INVALID_MESSAGE)
        CloudVlmClient(
            endpoint,
            model,
            api_key,
            reasoning_effort=reasoning_effort,
            headers=checked_headers,
        )
    except (VisionError, TypeError, ValueError) as error:
        raise ValueError(_VLM_SETTINGS_INVALID_MESSAGE) from error


@dataclass(frozen=True, slots=True)
class VisionSettings:
    """Enabled GeeTest adapter and optional remote VLM settings."""

    vlm: CloudVlmSettings | None = None
