"""Bounded OpenAI-compatible image client for Chat Completions and Responses."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import uuid
from dataclasses import dataclass
from typing import Any, Final, NoReturn, TypeGuard
from urllib.parse import urlsplit

import aiohttp

from prowl.vision.schemas import MAX_INPUT_PIXELS, TASK_KINDS, TaskKind, VisionError

MAX_IMAGE_BYTES: Final = 2_097_152
MAX_OPTIONS: Final = 64
MAX_INSTRUCTION_CHARS: Final = 2048
MAX_POINTS: Final = 16
MAX_CANVAS_ACTIONS: Final = 8
MAX_HOLD_MS: Final = 1000

ENDPOINT_INVALID: Final = "invalid_endpoint"
MODEL_INVALID: Final = "invalid_model"
API_KEY_INVALID: Final = "invalid_api_key"
IMAGE_INVALID: Final = "invalid_image"
IMAGE_TOO_LARGE: Final = "image_too_large"
INSTRUCTION_INVALID: Final = "invalid_instruction"
OPTIONS_INVALID: Final = "invalid_options"
BUDGET_INVALID: Final = "invalid_budget"
CLIENT_CLOSED: Final = "client_closed"
REQUEST_FAILED: Final = "request_failed"
REQUEST_TIMEOUT: Final = "request_timeout"
STATUS_UNEXPECTED: Final = "unexpected_status"
RESPONSE_TOO_LARGE: Final = "response_too_large"
RESPONSE_MALFORMED: Final = "malformed_response"
INTERPRETATION_INVALID: Final = "invalid_interpretation"
GRID_SELECTION_INVALID: Final = "invalid_grid_selection"
POINT_INVALID: Final = "invalid_point"
POINT_SELECTION_INVALID: Final = "invalid_point_selection"
CANVAS_CLICK_INVALID: Final = "invalid_canvas_click"
CANVAS_DRAG_INVALID: Final = "invalid_canvas_drag"
CANVAS_PLAN_INVALID: Final = "invalid_canvas_action_plan"
REASONING_EFFORT_INVALID: Final = "invalid_reasoning_effort"
HEADERS_INVALID: Final = "invalid_headers"
_UUID_HEADER_VALUE: Final = "$uuid"
_MAX_HEADERS: Final = 8
_MAX_HEADER_NAME_LENGTH: Final = 64
_MAX_HEADER_VALUE_LENGTH: Final = 256
_RESERVED_HEADERS: Final = frozenset(
    {
        "accept",
        "accept-encoding",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "proxy-authorization",
        "te",
        "transfer-encoding",
        "upgrade",
        "user-agent",
    }
)
_REASONING_EFFORTS: Final = frozenset({"minimal", "low", "medium", "high"})

_USER_AGENT: Final = "prowl-vision/1"
_OK_STATUS: Final = 200
_RESPONSE_LIMIT: Final = 65_536
_GRID_MAX_INDEX: Final = 8
_LOOPBACK_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})
_INTERPRETATION_KEYS: Final = frozenset({"kind", "labels"})
_GRID_KEYS: Final = frozenset({"cells"})
_POINT_KEYS: Final = frozenset({"x", "y"})
_POINTS_KEYS: Final = frozenset({"points"})
_CANVAS_KEYS: Final = frozenset({"actions"})
_CLICK_KEYS: Final = frozenset({"op", "x", "y"})
_DRAG_KEYS: Final = frozenset({"op", "from", "source_center", "to", "hold_ms"})
_OP_CLICK: Final = "click"
_OP_DRAG: Final = "drag"

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_TYPE: Final = b"IHDR"
_IHDR_TYPE_START: Final = 12
_IHDR_WIDTH_START: Final = 16
_IHDR_HEIGHT_START: Final = 20
_IHDR_MIN_LENGTH: Final = 24

_SYSTEM_PROMPT: Final = (
    "You interpret one visual challenge image. Reply with a single JSON object and no other text, "
    'shaped exactly as {"kind": "classification" or "detection", "labels": ["<label>", ...]}. '
    "Every label must be copied verbatim from the allowed labels and no other label may be used."
)

_GRID_SYSTEM_PROMPT: Final = (
    "You select cells of one 3x3 image challenge. Reply with a single JSON object and no other text, "
    'shaped exactly as {"cells": [<index>, ...]}. Use row-major indexes where 0 is the upper-left '
    "cell and 8 is the lower-right cell. List each selected cell index once, and reply with an empty "
    "list when no cell satisfies the instruction. Never add coordinates or any other key."
)

_POINTS_SYSTEM_PROMPT: Final = (
    "You select click targets in one challenge image. Reply with a single JSON object and no other "
    'text, shaped exactly as {"points": [{"x": 0.25, "y": 0.5}]}. Give every x and y as a normalized '
    "center coordinate strictly between 0 and 1 relative to the supplied image, where 0 is the left or "
    "top edge and 1 is the right or bottom edge. Do not assume any grid size, do not add buttons or "
    "margins, and reply with an empty list when nothing matches. Never add any other key."
)

_CANVAS_SYSTEM_PROMPT: Final = (
    "You plan canvas interactions for one challenge image. Reply with a single JSON object and no "
    'other text, shaped exactly as {"actions": [{"op": "click", "x": 0.25, "y": 0.5}, {"op": "drag", '
    '"from": {"x": 0.25, "y": 0.5}, "source_center": {"x": 0.2, "y": 0.5}, '
    '"to": {"x": 0.75, "y": 0.5}, "hold_ms": 500}]}. '
    "For a drag, from is a draggable grab point, source_center is the movable piece's current "
    "visual center, and to is where that center should land. The browser compensates for the "
    "grab offset. Choose the operations from the instruction and image. Give every x and y as a "
    "normalized canvas coordinate strictly between 0 and 1, where 0 is the left or top edge and 1 is "
    "the right or bottom edge. A click has exactly the keys op, x and y. A drag has exactly the keys "
    "op, from, source_center, to and hold_ms, where hold_ms is a whole number from 0 to 1000. Reply "
    "with an empty list when no action is valid. Never send selectors, URLs, key presses, scripts, or "
    "coordinates outside the canvas."
)


def _reject(code: str) -> NoReturn:
    raise VisionError(code)


def _is_safe_label(label: object) -> bool:
    if not isinstance(label, str) or not label or label != label.strip():
        return False
    return all(character >= " " and character != "\x7f" for character in label)


def _is_normalized_coordinate(value: object) -> TypeGuard[int | float]:
    """Return whether *value* is a finite number strictly between 0 and 1.

    A boolean is rejected even though it is an ``int``, and non-number values, ``NaN`` and both
    infinities fail the range or the finiteness check.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 < value < 1 and math.isfinite(value)


def _validate_endpoint(endpoint: object) -> str:
    """Return *endpoint* when it is an exact HTTPS or loopback HTTP POST URL.

    HTTPS accepts any host and rejects userinfo, a query and a fragment. HTTP is permitted
    only for the exact loopback hosts localhost, 127.0.0.1 and [::1], for an operator gateway
    or an offline test. A malformed URL, an unsupported scheme or a bad port is rejected.
    """
    if not isinstance(endpoint, str) or not endpoint:
        _reject(ENDPOINT_INVALID)
    try:
        parts = urlsplit(endpoint)
        host = parts.hostname
        _port = parts.port  # Property access rejects a non-numeric or out-of-range port.
    except ValueError:
        _reject(ENDPOINT_INVALID)
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        _reject(ENDPOINT_INVALID)
    if host is None or not host:
        _reject(ENDPOINT_INVALID)
    scheme = parts.scheme.lower()
    if scheme == "https" or (scheme == "http" and host.lower() in _LOOPBACK_HOSTS):
        return endpoint
    _reject(ENDPOINT_INVALID)


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        _reject(MODEL_INVALID)
    return model


def _validate_api_key(api_key: object) -> str | None:
    if api_key is None:
        return None
    if not isinstance(api_key, str) or not api_key:
        _reject(API_KEY_INVALID)
    return api_key


def _validate_reasoning_effort(value: object, *, responses: bool) -> str | None:
    if value is None:
        return None
    if not responses or not isinstance(value, str) or value not in _REASONING_EFFORTS:
        _reject(REASONING_EFFORT_INVALID)
    return value


def _validate_headers(headers: object) -> dict[str, str]:
    """Accept bounded operator headers without overriding transport or credential fields."""
    if headers is None:
        return {}
    if not isinstance(headers, dict) or len(headers) > _MAX_HEADERS:
        _reject(HEADERS_INVALID)
    checked: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= _MAX_HEADER_NAME_LENGTH
            or not name[0].isascii()
            or not name[0].isalpha()
            or not all(character.isascii() and (character.isalnum() or character == "-") for character in name)
        ):
            _reject(HEADERS_INVALID)
        folded = name.lower()
        if folded in _RESERVED_HEADERS or folded.startswith("proxy-") or folded in seen:
            _reject(HEADERS_INVALID)
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= _MAX_HEADER_VALUE_LENGTH
            or not value.isascii()
            or not value.isprintable()
        ):
            _reject(HEADERS_INVALID)
        seen.add(folded)
        checked[name] = value
    return checked


def _validate_instruction(instruction: object) -> str:
    if not isinstance(instruction, str) or not instruction or len(instruction) > MAX_INSTRUCTION_CHARS:
        _reject(INSTRUCTION_INVALID)
    return instruction


def _validate_options(options: object) -> tuple[str, ...]:
    if not isinstance(options, tuple) or not options or len(options) > MAX_OPTIONS:
        _reject(OPTIONS_INVALID)
    if not all(_is_safe_label(option) for option in options):
        _reject(OPTIONS_INVALID)
    if len(set(options)) != len(options):
        _reject(OPTIONS_INVALID)
    return options


def _validate_budget(seconds_remaining: object) -> float:
    if isinstance(seconds_remaining, bool) or not isinstance(seconds_remaining, (int, float)):
        _reject(BUDGET_INVALID)
    try:
        value = float(seconds_remaining)
    except OverflowError:
        _reject(BUDGET_INVALID)
    if not math.isfinite(value) or value <= 0:
        _reject(BUDGET_INVALID)
    return value


def _validate_image(image: object) -> bytes:
    if not isinstance(image, bytes) or not image:
        _reject(IMAGE_INVALID)
    if len(image) > MAX_IMAGE_BYTES:
        _reject(IMAGE_TOO_LARGE)
    if image[: len(_PNG_SIGNATURE)] != _PNG_SIGNATURE:
        _reject(IMAGE_INVALID)
    if len(image) < _IHDR_MIN_LENGTH or image[_IHDR_TYPE_START:_IHDR_WIDTH_START] != _PNG_IHDR_TYPE:
        _reject(IMAGE_INVALID)
    width = int.from_bytes(image[_IHDR_WIDTH_START : _IHDR_WIDTH_START + 4], "big")
    height = int.from_bytes(image[_IHDR_HEIGHT_START : _IHDR_HEIGHT_START + 4], "big")
    if width <= 0 or height <= 0:
        _reject(IMAGE_INVALID)
    if width * height > MAX_INPUT_PIXELS:
        _reject(IMAGE_TOO_LARGE)
    return image


class _DuplicateKeyError(ValueError):
    """Raised while decoding a JSON object that repeats a key."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _parse_json(text: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ValueError:
        _reject(RESPONSE_MALFORMED)


def _loads(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _reject(RESPONSE_MALFORMED)
    return _parse_json(text)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                _reject(RESPONSE_MALFORMED)
            parts.append(part["text"])
        return "".join(parts)
    _reject(RESPONSE_MALFORMED)


def _extract_content(document: object) -> str:
    if not isinstance(document, dict):
        _reject(RESPONSE_MALFORMED)
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices:
        _reject(RESPONSE_MALFORMED)
    first = choices[0]
    if not isinstance(first, dict):
        _reject(RESPONSE_MALFORMED)
    message = first.get("message")
    if not isinstance(message, dict):
        _reject(RESPONSE_MALFORMED)
    return _content_text(message.get("content"))


def _responses_text(content: object) -> str:
    if not isinstance(content, list) or not content:
        _reject(RESPONSE_MALFORMED)
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
            _reject(RESPONSE_MALFORMED)
        parts.append(part["text"])
    return "".join(parts)


def _extract_responses(document: object) -> str:
    if not isinstance(document, dict) or document.get("status") != "completed":
        _reject(RESPONSE_MALFORMED)
    output = document.get("output")
    if not isinstance(output, list):
        _reject(RESPONSE_MALFORMED)
    messages = []
    for item in output:
        if not isinstance(item, dict):
            _reject(RESPONSE_MALFORMED)
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            _reject(RESPONSE_MALFORMED)
        messages.append(item)
    if len(messages) != 1:
        _reject(RESPONSE_MALFORMED)
    return _responses_text(messages[0].get("content"))


def _interpretation_from_text(text: str, options: tuple[str, ...]) -> VlmInterpretation:
    document = _parse_json(text)
    if not isinstance(document, dict) or set(document) != _INTERPRETATION_KEYS:
        _reject(RESPONSE_MALFORMED)
    kind = document["kind"]
    labels = document["labels"]
    if kind not in TASK_KINDS:
        _reject(RESPONSE_MALFORMED)
    if not isinstance(labels, list) or not labels:
        _reject(RESPONSE_MALFORMED)
    if not all(_is_safe_label(label) for label in labels):
        _reject(RESPONSE_MALFORMED)
    if len(set(labels)) != len(labels):
        _reject(RESPONSE_MALFORMED)
    vocabulary = frozenset(options)
    if any(label not in vocabulary for label in labels):
        _reject(RESPONSE_MALFORMED)
    return VlmInterpretation(kind=kind, labels=tuple(labels))


def _normalized_point(x: object, y: object) -> Point:
    if not _is_normalized_coordinate(x) or not _is_normalized_coordinate(y):
        _reject(RESPONSE_MALFORMED)
    return Point(x=float(x), y=float(y))


def _point_from_object(value: object) -> Point:
    if not isinstance(value, dict) or set(value) != _POINT_KEYS:
        _reject(RESPONSE_MALFORMED)
    return _normalized_point(value["x"], value["y"])


def _points_from_text(text: str) -> PointSelection:
    document = _parse_json(text)
    if not isinstance(document, dict) or set(document) != _POINTS_KEYS:
        _reject(RESPONSE_MALFORMED)
    entries = document["points"]
    if not isinstance(entries, list) or len(entries) > MAX_POINTS:
        _reject(RESPONSE_MALFORMED)
    points = [_point_from_object(entry) for entry in entries]
    pairs = [(point.x, point.y) for point in points]
    if len(set(pairs)) != len(pairs):
        _reject(RESPONSE_MALFORMED)
    return PointSelection(points=tuple(points))


def _hold_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_HOLD_MS:
        _reject(RESPONSE_MALFORMED)
    return value


def _grid_from_text(text: str) -> GridSelection:
    document = _parse_json(text)
    if not isinstance(document, dict) or set(document) != _GRID_KEYS:
        _reject(RESPONSE_MALFORMED)
    cells = document["cells"]
    if not isinstance(cells, list):
        _reject(RESPONSE_MALFORMED)
    if any(isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= _GRID_MAX_INDEX for cell in cells):
        _reject(RESPONSE_MALFORMED)
    if len(set(cells)) != len(cells):
        _reject(RESPONSE_MALFORMED)
    return GridSelection(cells=tuple(sorted(cells)))


def _canvas_plan_from_text(text: str) -> CanvasActionPlan:
    document = _parse_json(text)
    if not isinstance(document, dict) or set(document) != _CANVAS_KEYS:
        _reject(RESPONSE_MALFORMED)
    entries = document["actions"]
    if not isinstance(entries, list) or len(entries) > MAX_CANVAS_ACTIONS:
        _reject(RESPONSE_MALFORMED)
    actions: list[CanvasClick | CanvasDrag] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _reject(RESPONSE_MALFORMED)
        operation = entry.get("op")
        if operation == _OP_CLICK:
            if set(entry) != _CLICK_KEYS:
                _reject(RESPONSE_MALFORMED)
            actions.append(CanvasClick(point=_normalized_point(entry["x"], entry["y"])))
        elif operation == _OP_DRAG:
            if set(entry) != _DRAG_KEYS:
                _reject(RESPONSE_MALFORMED)
            start = _point_from_object(entry["from"])
            center = _point_from_object(entry["source_center"])
            target = _point_from_object(entry["to"])
            hold_ms = _hold_ms(entry["hold_ms"])
            try:
                actions.append(CanvasDrag(start=start, source_center=center, target=target, hold_ms=hold_ms))
            except VisionError:
                _reject(RESPONSE_MALFORMED)
        else:
            _reject(RESPONSE_MALFORMED)
    return CanvasActionPlan(actions=tuple(actions))


@dataclass(frozen=True, slots=True)
class GridSelection:
    """The distinct 3x3 grid cells a VLM selected in ascending order; an empty tuple selects nothing."""

    cells: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate unique, in-range, ascending, non-boolean cell indexes."""
        if not isinstance(self.cells, tuple):
            _reject(GRID_SELECTION_INVALID)
        if any(
            isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= _GRID_MAX_INDEX
            for cell in self.cells
        ):
            _reject(GRID_SELECTION_INVALID)
        if any(later <= earlier for earlier, later in zip(self.cells, self.cells[1:], strict=False)):
            _reject(GRID_SELECTION_INVALID)


@dataclass(frozen=True, slots=True)
class Point:
    """One normalized click target; both coordinates are finite and strictly between 0 and 1."""

    x: float
    y: float

    def __post_init__(self) -> None:
        """Validate both normalized coordinates, rejecting booleans, non-numbers, non-finite values and edges."""
        if not _is_normalized_coordinate(self.x) or not _is_normalized_coordinate(self.y):
            _reject(POINT_INVALID)


@dataclass(frozen=True, slots=True)
class PointSelection:
    """The distinct normalized click targets a VLM selected; an empty tuple selects nothing."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        """Validate at most 16 distinct :class:`Point` values."""
        if not isinstance(self.points, tuple) or len(self.points) > MAX_POINTS:
            _reject(POINT_SELECTION_INVALID)
        if any(not isinstance(point, Point) for point in self.points):
            _reject(POINT_SELECTION_INVALID)
        pairs = [(point.x, point.y) for point in self.points]
        if len(set(pairs)) != len(pairs):
            _reject(POINT_SELECTION_INVALID)


@dataclass(frozen=True, slots=True)
class CanvasClick:
    """One normalized canvas click target."""

    point: Point

    def __post_init__(self) -> None:
        """Validate that the target is a :class:`Point`."""
        if not isinstance(self.point, Point):
            _reject(CANVAS_CLICK_INVALID)


@dataclass(frozen=True, slots=True)
class CanvasDrag:
    """Drag a piece from *start* until its center reaches *target*."""

    start: Point
    source_center: Point
    target: Point
    hold_ms: int

    def __post_init__(self) -> None:
        """Reject invalid or out-of-canvas cursor endpoints before any press."""
        if not all(isinstance(point, Point) for point in (self.start, self.source_center, self.target)):
            _reject(CANVAS_DRAG_INVALID)
        if isinstance(self.hold_ms, bool) or not isinstance(self.hold_ms, int):
            _reject(CANVAS_DRAG_INVALID)
        if not 0 <= self.hold_ms <= MAX_HOLD_MS:
            _reject(CANVAS_DRAG_INVALID)
        if not _is_normalized_coordinate(
            self.target.x + self.start.x - self.source_center.x
        ) or not _is_normalized_coordinate(self.target.y + self.start.y - self.source_center.y):
            _reject(CANVAS_DRAG_INVALID)

    @property
    def release(self) -> Point:
        """Compensate for the piece's offset from the grab point."""
        return Point(
            x=self.target.x + self.start.x - self.source_center.x,
            y=self.target.y + self.start.y - self.source_center.y,
        )


@dataclass(frozen=True, slots=True)
class CanvasActionPlan:
    """The clicks and drags a VLM planned in order; an empty tuple plans no action."""

    actions: tuple[CanvasClick | CanvasDrag, ...]

    def __post_init__(self) -> None:
        """Validate at most 8 click-or-drag actions."""
        if not isinstance(self.actions, tuple) or len(self.actions) > MAX_CANVAS_ACTIONS:
            _reject(CANVAS_PLAN_INVALID)
        if any(not isinstance(action, (CanvasClick, CanvasDrag)) for action in self.actions):
            _reject(CANVAS_PLAN_INVALID)


@dataclass(frozen=True, slots=True)
class VlmInterpretation:
    """A VLM-proposed task kind plus the label subset it selected from the given options."""

    kind: TaskKind
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the task kind and the unique, nonempty, safe label tuple."""
        if self.kind not in TASK_KINDS or not isinstance(self.labels, tuple) or not self.labels:
            _reject(INTERPRETATION_INVALID)
        if not all(_is_safe_label(label) for label in self.labels):
            _reject(INTERPRETATION_INVALID)
        if len(set(self.labels)) != len(self.labels):
            _reject(INTERPRETATION_INVALID)


class CloudVlmClient:
    """One multimodal client bound to a Chat Completions or Responses endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        *,
        reasoning_effort: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Validate the endpoint and operator-supplied request headers without connecting."""
        self._endpoint = _validate_endpoint(endpoint)
        self._responses = urlsplit(self._endpoint).path.rstrip("/").endswith("/responses")
        self._model = _validate_model(model)
        self._api_key = _validate_api_key(api_key)
        self._headers = _validate_headers(headers)
        self._reasoning_effort = _validate_reasoning_effort(reasoning_effort, responses=self._responses)
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    def __repr__(self) -> str:
        """Omit the API key and custom header values from the representation."""
        key_state = "set" if self._api_key is not None else "unset"
        return f"CloudVlmClient(endpoint=<configured>, model=<configured>, api_key={key_state})"

    def _build_payload(self, system_prompt: str, instruction: str, image: bytes) -> dict[str, object]:
        image_url = f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}"
        if self._responses:
            payload: dict[str, object] = {
                "model": self._model,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    },
                ],
            }
            if self._reasoning_effort is not None:
                payload["reasoning"] = {"effort": self._reasoning_effort}
            return payload
        return {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }

    async def interpret(
        self,
        image: bytes,
        instruction: str,
        options: tuple[str, ...],
        seconds_remaining: float,
    ) -> VlmInterpretation:
        """Interpret one bounded PNG against *instruction* and the allowed *options*.

        The request carries only the image data URL, the instruction and the allowed labels,
        over a session that ignores environment proxies, cookies and redirects. It is bounded
        by the caller's remaining deadline with no retry. A non-200 status, an
        oversized or malformed response, or any result outside the requested vocabulary is a
        fixed :class:`VisionError` category. Cancellation propagates unchanged.

        :raises VisionError: when validation fails or the request, status, or response violates
            the contract.
        """
        if self._closed:
            _reject(CLIENT_CLOSED)
        image_valid = _validate_image(image)
        instruction_valid = _validate_instruction(instruction)
        options_valid = _validate_options(options)
        timeout_seconds = _validate_budget(seconds_remaining)
        user_text = f"{instruction_valid}\n\nAllowed labels: {', '.join(options_valid)}"
        payload = self._build_payload(_SYSTEM_PROMPT, user_text, image_valid)
        body = await self._post_chat(payload, timeout_seconds)
        return _interpretation_from_text(self._response_text(body), options_valid)

    async def select_grid(
        self,
        image: bytes,
        instruction: str,
        seconds_remaining: float,
    ) -> GridSelection:
        """Select the cells of one bounded 3x3 grid PNG that satisfy *instruction*.

        The request carries only the image data URL and the instruction over the same bounded
        transport as :meth:`interpret`: environment proxies, cookies and redirects are ignored, the
        native timeout follows the caller's remaining deadline with no retry, and a non-200 or oversized response is
        a fixed :class:`VisionError` category. The response must be one JSON object shaped exactly
        as ``{"cells": [<index>, ...]}`` with unique, non-boolean row-major indexes in ``0`` to
        ``8``; accepted indexes are returned in ascending order and an empty list means no cell was
        selected. Cancellation propagates unchanged.

        :raises VisionError: when validation fails or the request, status, or response violates
            the contract.
        """
        if self._closed:
            _reject(CLIENT_CLOSED)
        image_valid = _validate_image(image)
        instruction_valid = _validate_instruction(instruction)
        timeout_seconds = _validate_budget(seconds_remaining)
        payload = self._build_payload(_GRID_SYSTEM_PROMPT, instruction_valid, image_valid)
        body = await self._post_chat(payload, timeout_seconds)
        return _grid_from_text(self._response_text(body))

    async def select_points(
        self,
        image: bytes,
        instruction: str,
        seconds_remaining: float,
    ) -> PointSelection:
        """Select normalized click targets in one bounded PNG that satisfy *instruction*.

        The request carries only the image data URL and the instruction over the same bounded
        transport as :meth:`interpret`: environment proxies, cookies and redirects are ignored, the
        native timeout follows the caller's remaining deadline with no retry, and a non-200 or oversized response is a
        fixed :class:`VisionError` category. The response must be one JSON object shaped exactly as
        ``{"points": [{"x": <x>, "y": <y>}, ...]}`` with at most 16 unique entries whose ``x`` and
        ``y`` are finite numbers strictly between ``0`` and ``1``; an empty list means nothing was
        selected. No grid size, click mode, model identifier or URL is taken from the response.
        Cancellation propagates unchanged.

        :raises VisionError: when validation fails or the request, status, or response violates
            the contract.
        """
        if self._closed:
            _reject(CLIENT_CLOSED)
        image_valid = _validate_image(image)
        instruction_valid = _validate_instruction(instruction)
        timeout_seconds = _validate_budget(seconds_remaining)
        payload = self._build_payload(_POINTS_SYSTEM_PROMPT, instruction_valid, image_valid)
        body = await self._post_chat(payload, timeout_seconds)
        return _points_from_text(self._response_text(body))

    async def plan_canvas(
        self,
        image: bytes,
        instruction: str,
        seconds_remaining: float,
    ) -> CanvasActionPlan:
        """Plan the canvas clicks and drags in one bounded PNG that satisfy *instruction*.

        The request carries only the image data URL and the instruction over the same bounded
        transport as :meth:`interpret`: environment proxies, cookies and redirects are ignored, the
        native timeout follows the caller's remaining deadline with no retry, and a non-200 or oversized response is a
        fixed :class:`VisionError` category. The response must be one JSON object shaped exactly as
        ``{"actions": [...]}`` with at most 8 entries, each either ``{"op": "click", "x": <x>,
        "y": <y>}`` or a drag with ``from``, ``source_center``, ``to`` and ``hold_ms``.
        Every coordinate is finite and strictly between ``0`` and ``1`` and ``hold_ms`` is a whole number
        from ``0`` to ``1000``. An empty list means no action is valid. No selector, URL, key press or
        script is taken from the response. Cancellation propagates unchanged.

        :raises VisionError: when validation fails or the request, status, or response violates
            the contract.
        """
        if self._closed:
            _reject(CLIENT_CLOSED)
        image_valid = _validate_image(image)
        instruction_valid = _validate_instruction(instruction)
        timeout_seconds = _validate_budget(seconds_remaining)
        payload = self._build_payload(_CANVAS_SYSTEM_PROMPT, instruction_valid, image_valid)
        body = await self._post_chat(payload, timeout_seconds)
        return _canvas_plan_from_text(self._response_text(body))

    def _response_text(self, body: bytes) -> str:
        document = _loads(body)
        return _extract_responses(document) if self._responses else _extract_content(document)

    async def _post_chat(self, payload: dict[str, object], timeout_seconds: float) -> bytes:
        """Post *payload* over the shared bounded transport and return the raw response body."""
        session = self._session
        if session is None:
            session = aiohttp.ClientSession(trust_env=False, cookie_jar=aiohttp.DummyCookieJar(), auto_decompress=False)
            self._session = session
        headers: dict[str, str] = {"Accept-Encoding": "identity", "User-Agent": _USER_AGENT}
        headers.update(
            (name, str(uuid.uuid4()) if value == _UUID_HEADER_VALUE else value) for name, value in self._headers.items()
        )
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with asyncio.timeout(timeout_seconds):
                async with session.post(
                    self._endpoint,
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                ) as response:
                    if response.status != _OK_STATUS:
                        _reject(STATUS_UNEXPECTED)
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.content.iter_chunked(8192):
                        size += len(chunk)
                        if size > _RESPONSE_LIMIT:
                            _reject(RESPONSE_TOO_LARGE)
                        chunks.append(chunk)
                    body = b"".join(chunks)
        except TimeoutError:
            _reject(REQUEST_TIMEOUT)
        except (aiohttp.ClientError, OSError):
            _reject(REQUEST_FAILED)
        return body

    async def aclose(self) -> None:
        """Close the underlying HTTP session; safe to call more than once."""
        self._closed = True
        session = self._session
        self._session = None
        if session is not None:
            await session.close()


__all__ = [
    "CanvasActionPlan",
    "CanvasClick",
    "CanvasDrag",
    "CloudVlmClient",
    "GridSelection",
    "Point",
    "PointSelection",
    "VlmInterpretation",
]
