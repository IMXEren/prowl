# pyright: reportArgumentType=false, reportPrivateUsage=false, reportOptionalMemberAccess=false
"""Offline tests for the OpenAI-compatible multimodal VLM client.

A loopback aiohttp fake endpoint captures what the client posts and returns controlled
bodies. Deliberately malformed arguments and private session state are inspected here,
which is why targeted file-level pyright exclusions are used. No external network, model,
weight or browser is touched.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from prowl.vision.schemas import VisionError
from prowl.vision.vlm import (
    CanvasActionPlan,
    CanvasClick,
    CanvasDrag,
    CloudVlmClient,
    GridSelection,
    Point,
    PointSelection,
    VlmInterpretation,
    _validate_budget,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_VALID_INTERPRETATION = '{"kind": "classification", "labels": ["alpha"]}'

# Centers of five cells of a 5x5 canvas pattern, described only in normalized image coordinates.
_CANVAS_PATTERN = (
    '{"points": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.5, "y": 0.5}, '
    '{"x": 0.1, "y": 0.9}, {"x": 0.9, "y": 0.9}]}'
)

_CLICK_PLAN = '{"actions": [{"op": "click", "x": 0.25, "y": 0.5}]}'

_DRAG_PLAN = (
    '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
    '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 500}]}'
)


def _points(count: int) -> str:
    """Return one response carrying *count* distinct normalized target points."""
    entries = ", ".join(f'{{"x": {0.05 * (index + 1):.2f}, "y": 0.5}}' for index in range(count))
    return '{"points": [' + entries + "]}"


def _actions(count: int) -> str:
    """Return one plan carrying *count* distinct click actions."""
    entries = ", ".join(f'{{"op": "click", "x": {0.05 * (index + 1):.2f}, "y": 0.5}}' for index in range(count))
    return '{"actions": [' + entries + "]}"


def _png_header(width: int, height: int) -> bytes:
    """Return a PNG signature plus one IHDR chunk carrying *width* and *height*."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr


_IMAGE = _png_header(2, 2)


def _envelope(content: object) -> web.Response:
    """Return an OpenAI-compatible chat response whose first message carries *content*."""
    return web.json_response({"choices": [{"message": {"role": "assistant", "content": content}}]})


def _responses_envelope(content: object) -> web.Response:
    return web.json_response(
        {
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]},
            ],
        }
    )


async def _default_responder(_request: web.Request) -> web.StreamResponse:
    return _envelope(_VALID_INTERPRETATION)


def _responder(content: object) -> Any:
    """Return an async fake-endpoint responder that answers with *content*."""

    async def respond(_request: web.Request) -> web.StreamResponse:
        return _envelope(content)

    return respond


class _Recorder:
    """Records every request a fake endpoint receives and answers with a chosen responder."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responder: Any = _default_responder

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        self.calls.append({"path": request.path, "headers": dict(request.headers), "body": body})
        return await self.responder(request)


class _ClientTestCase(IsolatedAsyncioTestCase):
    """Starts one loopback fake endpoint and builds clients bound to it."""

    async def asyncSetUp(self) -> None:
        self.recorder = _Recorder()
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.recorder.handle)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.endpoint = str(self.server.make_url("/v1/chat/completions"))

    def _client(
        self,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> CloudVlmClient:
        client = CloudVlmClient(self.endpoint, "arbitrary-model-id", api_key=api_key, headers=headers)
        self.addAsyncCleanup(client.aclose)
        return client

    def _header_values(self, name: str) -> list[str]:
        """Return every received value for header *name*, ignoring case."""
        return [
            value
            for call in self.recorder.calls
            for header, value in call["headers"].items()
            if header.lower() == name.lower()
        ]

    async def _interpret(self, client: CloudVlmClient, **overrides: Any) -> VlmInterpretation:
        arguments: dict[str, Any] = {
            "image": _IMAGE,
            "instruction": "Identify the target",
            "options": ("alpha", "beta"),
            "seconds_remaining": 5.0,
        }
        arguments.update(overrides)
        return await client.interpret(**arguments)

    async def _expect_code(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._interpret(client, **overrides)
        self.assertEqual(context.exception.code, code)

    async def _expect_validation(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._interpret(client, **overrides)
        self.assertEqual(context.exception.code, code)
        self.assertEqual(self.recorder.calls, [])

    async def _select_grid(self, client: CloudVlmClient, **overrides: Any) -> GridSelection:
        arguments: dict[str, Any] = {
            "image": _IMAGE,
            "instruction": "Select every cell with a cat",
            "seconds_remaining": 5.0,
        }
        arguments.update(overrides)
        return await client.select_grid(**arguments)

    async def _expect_grid_code(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._select_grid(client, **overrides)
        self.assertEqual(context.exception.code, code)

    async def _expect_grid_validation(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._select_grid(client, **overrides)
        self.assertEqual(context.exception.code, code)
        self.assertEqual(self.recorder.calls, [])

    async def _select_points(self, client: CloudVlmClient, **overrides: Any) -> PointSelection:
        arguments: dict[str, Any] = {
            "image": _IMAGE,
            "instruction": "Click every matching tile",
            "seconds_remaining": 5.0,
        }
        arguments.update(overrides)
        return await client.select_points(**arguments)

    async def _expect_points_code(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._select_points(client, **overrides)
        self.assertEqual(context.exception.code, code)

    async def _expect_points_validation(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._select_points(client, **overrides)
        self.assertEqual(context.exception.code, code)
        self.assertEqual(self.recorder.calls, [])

    async def _plan_canvas(self, client: CloudVlmClient, **overrides: Any) -> CanvasActionPlan:
        arguments: dict[str, Any] = {
            "image": _IMAGE,
            "instruction": "Drag the animal to the side of the scale that needs it",
            "seconds_remaining": 5.0,
        }
        arguments.update(overrides)
        return await client.plan_canvas(**arguments)

    async def _expect_canvas_code(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(client, **overrides)
        self.assertEqual(context.exception.code, code)

    async def _expect_canvas_validation(self, code: str, **overrides: Any) -> None:
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(client, **overrides)
        self.assertEqual(context.exception.code, code)
        self.assertEqual(self.recorder.calls, [])


class EndpointValidationTests(TestCase):
    """Endpoint, model, key and representation rules that need no server."""

    def test_accepts_https_endpoints(self) -> None:
        for endpoint in ("https://api.example.com/v1/chat/completions", "https://gateway.internal:8443/chat"):
            self.assertEqual(CloudVlmClient(endpoint, "model")._endpoint, endpoint)

    def test_accepts_http_loopback_hosts(self) -> None:
        for host in ("localhost", "127.0.0.1", "[::1]"):
            endpoint = f"http://{host}:11434/v1/chat/completions"
            self.assertEqual(CloudVlmClient(endpoint, "model")._endpoint, endpoint)

    def test_rejects_remote_http(self) -> None:
        for endpoint in ("http://example.com/v1", "http://192.168.0.1:8080/v1", "http://[2001:db8::1]/v1"):
            with self.assertRaises(VisionError) as context:
                CloudVlmClient(endpoint, "model")
            self.assertEqual(context.exception.code, "invalid_endpoint")

    def test_rejects_userinfo_query_and_fragment(self) -> None:
        endpoints = (
            "https://user:pass@api.example.com/v1",
            "https://api.example.com/v1?key=1",
            "https://api.example.com/v1#frag",
        )
        for endpoint in endpoints:
            with self.assertRaises(VisionError) as context:
                CloudVlmClient(endpoint, "model")
            self.assertEqual(context.exception.code, "invalid_endpoint")

    def test_rejects_other_schemes_and_unparseable_endpoints(self) -> None:
        endpoints: tuple[object, ...] = (
            "ftp://api.example.com/v1",
            "api.example.com/v1",
            "https://",
            "http://127.0.0.1:notaport/v1",
            "",
            123,
            None,
        )
        for endpoint in endpoints:
            with self.assertRaises(VisionError) as context:
                CloudVlmClient(endpoint, "model")
            self.assertEqual(context.exception.code, "invalid_endpoint")

    def test_accepts_arbitrary_model_identifiers(self) -> None:
        models = ("gpt-4o", "qwen2.5-vl-7b", "meta-llama/Llama-3.2-11B-Vision", "local.model:v1", "x")
        for model in models:
            self.assertEqual(CloudVlmClient("https://api.example.com/v1/chat/completions", model)._model, model)

    def test_rejects_blank_or_non_string_model(self) -> None:
        for model in ("", "   ", None, 5):
            with self.assertRaises(VisionError) as context:
                CloudVlmClient("https://api.example.com/v1/chat/completions", model)
            self.assertEqual(context.exception.code, "invalid_model")

    def test_rejects_blank_or_non_string_api_key(self) -> None:
        for api_key in ("", 5):
            with self.assertRaises(VisionError) as context:
                CloudVlmClient("https://api.example.com/v1/chat/completions", "model", api_key)
            self.assertEqual(context.exception.code, "invalid_api_key")

    def test_omitting_api_key_is_allowed(self) -> None:
        client = CloudVlmClient("https://api.example.com/v1/chat/completions", "model")
        self.assertIsNone(client._api_key)

    def test_construction_opens_no_session(self) -> None:
        client = CloudVlmClient("https://api.example.com/v1/chat/completions", "model", "sk-secret")
        self.assertIsNone(client._session)

    def test_repr_hides_the_api_key_value(self) -> None:
        client = CloudVlmClient("https://api.example.com/v1/chat/completions", "private-model-id", "sk-secret-value")
        text = repr(client)
        self.assertNotIn("sk-secret-value", text)
        self.assertNotIn("api.example.com", text)
        self.assertNotIn("private-model-id", text)
        self.assertIn("api_key=set", text)

    def test_repr_reports_an_unset_key(self) -> None:
        client = CloudVlmClient("https://api.example.com/v1/chat/completions", "model")
        self.assertIn("unset", repr(client))


class VlmInterpretationTests(TestCase):
    """The frozen interpretation schema validates kind and labels."""

    def test_accepts_a_valid_interpretation(self) -> None:
        result = VlmInterpretation(kind="detection", labels=("a", "b"))
        self.assertEqual(result.kind, "detection")
        self.assertEqual(result.labels, ("a", "b"))
        self.assertFalse(hasattr(result, "__dict__"))

    def test_rejects_invalid_kind_and_labels(self) -> None:
        builders = (
            lambda: VlmInterpretation(kind="segmentation", labels=("a",)),
            lambda: VlmInterpretation(kind="classification", labels=()),
            lambda: VlmInterpretation(kind="classification", labels=["a"]),
            lambda: VlmInterpretation(kind="classification", labels=("a", "a")),
            lambda: VlmInterpretation(kind="classification", labels=("a", "")),
        )
        for build in builders:
            with self.assertRaises(VisionError) as context:
                build()
            self.assertEqual(context.exception.code, "invalid_interpretation")


class RequestShapeTests(_ClientTestCase):
    """The request the client posts and the responses it accepts."""

    async def test_posts_the_openai_multimodal_shape(self) -> None:
        result = await self._interpret(self._client())
        self.assertEqual(result, VlmInterpretation(kind="classification", labels=("alpha",)))
        self.assertEqual(len(self.recorder.calls), 1)
        call = self.recorder.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        payload = json.loads(call["body"])
        self.assertEqual(payload["model"], "arbitrary-model-id")
        system = payload["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("kind", system["content"])
        self.assertIn("labels", system["content"])
        parts = payload["messages"][1]["content"]
        text_part = next(part for part in parts if part["type"] == "text")
        self.assertIn("Identify the target", text_part["text"])
        self.assertIn("alpha", text_part["text"])
        self.assertIn("beta", text_part["text"])
        image_part = next(part for part in parts if part["type"] == "image_url")
        expected = "data:image/png;base64," + base64.b64encode(_IMAGE).decode("ascii")
        self.assertEqual(image_part["image_url"]["url"], expected)

    async def test_transmits_no_cookie_or_proxy_headers(self) -> None:
        await self._interpret(self._client())
        headers = self.recorder.calls[0]["headers"]
        self.assertNotIn("Cookie", headers)
        self.assertFalse(any(name.lower().startswith("proxy") for name in headers))
        self.assertNotIn("Authorization", headers)

    async def test_sends_an_authorization_header_when_a_key_is_present(self) -> None:
        await self._interpret(self._client(api_key="sk-test"))
        self.assertEqual(self.recorder.calls[0]["headers"]["Authorization"], "Bearer sk-test")

    async def test_ignores_environment_proxy_variables(self) -> None:
        client = self._client()
        proxies = {
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
        }
        with patch.dict(os.environ, proxies):
            result = await self._interpret(client)
        self.assertEqual(result.labels, ("alpha",))

    async def test_refuses_to_follow_a_redirect(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(status=302, headers={"Location": "/elsewhere"})

        self.recorder.responder = respond
        await self._expect_code("unexpected_status")
        self.assertNotIn("/elsewhere", [call["path"] for call in self.recorder.calls])

    async def test_rejects_a_non_200_status(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(status=500, text="boom")

        self.recorder.responder = respond
        await self._expect_code("unexpected_status")

    async def test_rejects_an_oversized_response(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(text="x" * 70_000)

        self.recorder.responder = respond
        await self._expect_code("response_too_large")

    async def test_reads_all_chunks_of_a_bounded_response(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": _VALID_INTERPRETATION}}], "padding": "x" * 5000}
        ).encode()

        async def respond(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(body[:2048])
            await asyncio.sleep(0.05)
            await response.write(body[2048:])
            await response.write_eof()
            return response

        self.recorder.responder = respond
        result = await self._interpret(self._client())
        self.assertEqual(result.labels, ("alpha",))

    async def test_rejects_a_body_that_is_not_json(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(text="<html>nope</html>", content_type="application/json")

        self.recorder.responder = respond
        await self._expect_code("malformed_response")

    async def test_rejects_a_missing_choices_field(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.json_response({"error": "unavailable"})

        self.recorder.responder = respond
        await self._expect_code("malformed_response")

    async def test_rejects_non_string_content(self) -> None:
        self.recorder.responder = _responder(123)
        await self._expect_code("malformed_response")

    async def test_rejects_duplicate_keys(self) -> None:
        self.recorder.responder = _responder('{"kind": "classification", "labels": ["alpha"], "labels": ["beta"]}')
        await self._expect_code("malformed_response")

    async def test_rejects_duplicate_keys_in_the_envelope(self) -> None:
        body = '{"choices": [{"message": {"content": "x"}}], "choices": []}'

        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(text=body, content_type="application/json")

        self.recorder.responder = respond
        await self._expect_code("malformed_response")

    async def test_rejects_out_of_vocabulary_labels(self) -> None:
        self.recorder.responder = _responder('{"kind": "classification", "labels": ["gamma"]}')
        await self._expect_code("malformed_response")

    async def test_rejects_an_unknown_kind(self) -> None:
        self.recorder.responder = _responder('{"kind": "segmentation", "labels": ["alpha"]}')
        await self._expect_code("malformed_response")

    async def test_rejects_extra_keys(self) -> None:
        self.recorder.responder = _responder('{"kind": "classification", "labels": ["alpha"], "note": "x"}')
        await self._expect_code("malformed_response")

    async def test_rejects_a_missing_labels_field(self) -> None:
        self.recorder.responder = _responder('{"kind": "classification"}')
        await self._expect_code("malformed_response")

    async def test_accepts_list_text_parts(self) -> None:
        self.recorder.responder = _responder([{"type": "text", "text": _VALID_INTERPRETATION}])
        result = await self._interpret(self._client())
        self.assertEqual(result.labels, ("alpha",))

    async def test_rejects_a_non_text_content_part(self) -> None:
        self.recorder.responder = _responder([{"type": "image_url", "image_url": {}}])
        await self._expect_code("malformed_response")

    async def test_timeout_maps_to_a_fixed_code(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.3)
            return _envelope(_VALID_INTERPRETATION)

        self.recorder.responder = respond
        await self._expect_code("request_timeout", seconds_remaining=0.05)

    async def test_cancellation_propagates(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.5)
            return _envelope(_VALID_INTERPRETATION)

        self.recorder.responder = respond
        client = self._client()
        task = asyncio.create_task(self._interpret(client))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_network_failure_maps_to_a_fixed_code(self) -> None:
        client = CloudVlmClient("http://127.0.0.1:1/v1/chat/completions", "model")
        self.addAsyncCleanup(client.aclose)
        with self.assertRaises(VisionError) as context:
            await self._interpret(client)
        self.assertEqual(context.exception.code, "request_failed")

    async def test_aclose_is_idempotent_and_blocks_interpret(self) -> None:
        client = self._client()
        await client.aclose()
        await client.aclose()
        with self.assertRaises(VisionError) as context:
            await self._interpret(client)
        self.assertEqual(context.exception.code, "client_closed")


class ArgumentValidationTests(_ClientTestCase):
    """Input validation happens before any request is made."""

    async def test_rejects_a_non_png_image(self) -> None:
        await self._expect_validation("invalid_image", image=b"GIF89a" + b"\x00" * 40)

    async def test_rejects_a_truncated_png(self) -> None:
        await self._expect_validation("invalid_image", image=_PNG_SIGNATURE + b"\x00")

    async def test_rejects_an_invalid_ihdr(self) -> None:
        bad = _PNG_SIGNATURE + struct.pack(">I", 13) + b"JHDR" + b"\x00" * 13
        await self._expect_validation("invalid_image", image=bad)

    async def test_rejects_zero_dimensions(self) -> None:
        await self._expect_validation("invalid_image", image=_png_header(0, 4))

    async def test_rejects_a_non_bytes_image(self) -> None:
        await self._expect_validation("invalid_image", image="not-bytes")

    async def test_rejects_an_empty_image(self) -> None:
        await self._expect_validation("invalid_image", image=b"")

    async def test_rejects_an_image_over_two_mib(self) -> None:
        await self._expect_validation("image_too_large", image=_png_header(2, 2) + b"\x00" * (2 * 1024 * 1024))

    async def test_rejects_an_image_over_one_megapixel(self) -> None:
        await self._expect_validation("image_too_large", image=_png_header(2000, 2000))

    async def test_rejects_a_long_instruction(self) -> None:
        await self._expect_validation("invalid_instruction", instruction="a" * 2049)

    async def test_rejects_an_empty_instruction(self) -> None:
        await self._expect_validation("invalid_instruction", instruction="")

    async def test_rejects_a_non_string_instruction(self) -> None:
        await self._expect_validation("invalid_instruction", instruction=123)

    async def test_rejects_empty_options(self) -> None:
        await self._expect_validation("invalid_options", options=())

    async def test_rejects_options_that_are_not_a_tuple(self) -> None:
        await self._expect_validation("invalid_options", options=["alpha"])

    async def test_rejects_too_many_options(self) -> None:
        await self._expect_validation("invalid_options", options=tuple(f"label{index}" for index in range(65)))

    async def test_rejects_duplicate_options(self) -> None:
        await self._expect_validation("invalid_options", options=("alpha", "alpha"))

    async def test_rejects_unsafe_options(self) -> None:
        for option in ("al\npha", " alpha", "alpha ", ""):
            await self._expect_validation("invalid_options", options=(option,))

    async def test_rejects_a_non_string_option(self) -> None:
        await self._expect_validation("invalid_options", options=(1,))

    async def test_rejects_non_positive_or_non_finite_budgets(self) -> None:
        for seconds in (0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "5"):
            await self._expect_validation("invalid_budget", seconds_remaining=seconds)

    async def test_large_request_budget_is_capped_by_native_timeout(self) -> None:
        result = await self._interpret(self._client(), seconds_remaining=180.0)
        self.assertEqual(result, VlmInterpretation("classification", ("alpha",)))


class ImportBoundaryTests(TestCase):
    """Importing the client pulls no optional runtime and opens no session."""

    def test_import_does_not_pull_optional_runtimes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = (
            "import sys, prowl.vision.vlm; "
            "print(sorted({name.split('.')[0] for name in sys.modules} & {'numpy', 'PIL', 'onnxruntime'}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


class GridSelectionSchemaTests(TestCase):
    """The frozen grid-selection schema validates cell indexes without a server."""

    def test_accepts_a_valid_selection(self) -> None:
        result = GridSelection(cells=(0, 4, 8))
        self.assertEqual(result.cells, (0, 4, 8))
        self.assertFalse(hasattr(result, "__dict__"))

    def test_accepts_an_empty_selection(self) -> None:
        self.assertEqual(GridSelection(cells=()).cells, ())

    def test_rejects_invalid_cells(self) -> None:
        builders = (
            lambda: GridSelection(cells=[0]),
            lambda: GridSelection(cells=(1, 1)),
            lambda: GridSelection(cells=(4, 0)),
            lambda: GridSelection(cells=(-1,)),
            lambda: GridSelection(cells=(9,)),
            lambda: GridSelection(cells=(True,)),
            lambda: GridSelection(cells=(1.0,)),
            lambda: GridSelection(cells=("0",)),
        )
        for build in builders:
            with self.assertRaises(VisionError) as context:
                build()
            self.assertEqual(context.exception.code, "invalid_grid_selection")

    def test_repr_holds_only_cell_indexes(self) -> None:
        self.assertEqual(repr(GridSelection(cells=(0, 8))), "GridSelection(cells=(0, 8))")


class GridSelectionRequestTests(_ClientTestCase):
    """The grid-selection request the client posts and the responses it accepts."""

    async def test_posts_the_grid_shape(self) -> None:
        self.recorder.responder = _responder('{"cells": [0, 4, 8]}')
        result = await self._select_grid(self._client())
        self.assertEqual(result, GridSelection(cells=(0, 4, 8)))
        self.assertEqual(len(self.recorder.calls), 1)
        call = self.recorder.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        payload = json.loads(call["body"])
        self.assertEqual(payload["model"], "arbitrary-model-id")
        system = payload["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("cells", system["content"])
        parts = payload["messages"][1]["content"]
        text_part = next(part for part in parts if part["type"] == "text")
        self.assertIn("Select every cell with a cat", text_part["text"])
        image_part = next(part for part in parts if part["type"] == "image_url")
        expected = "data:image/png;base64," + base64.b64encode(_IMAGE).decode("ascii")
        self.assertEqual(image_part["image_url"]["url"], expected)

    async def test_sorts_cells_ascending(self) -> None:
        self.recorder.responder = _responder('{"cells": [8, 0, 4]}')
        result = await self._select_grid(self._client())
        self.assertEqual(result.cells, (0, 4, 8))

    async def test_accepts_an_empty_selection(self) -> None:
        self.recorder.responder = _responder('{"cells": []}')
        result = await self._select_grid(self._client())
        self.assertEqual(result.cells, ())

    async def test_transmits_no_cookie_or_proxy_headers(self) -> None:
        self.recorder.responder = _responder('{"cells": [0]}')
        await self._select_grid(self._client())
        headers = self.recorder.calls[0]["headers"]
        self.assertNotIn("Cookie", headers)
        self.assertFalse(any(name.lower().startswith("proxy") for name in headers))
        self.assertNotIn("Authorization", headers)

    async def test_ignores_environment_proxy_variables(self) -> None:
        self.recorder.responder = _responder('{"cells": [0]}')
        proxies = {
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
        }
        with patch.dict(os.environ, proxies):
            result = await self._select_grid(self._client())
        self.assertEqual(result.cells, (0,))

    async def test_sends_an_authorization_header_when_a_key_is_present(self) -> None:
        self.recorder.responder = _responder('{"cells": [0]}')
        await self._select_grid(self._client(api_key="sk-test"))
        self.assertEqual(self.recorder.calls[0]["headers"]["Authorization"], "Bearer sk-test")

    async def test_rejects_duplicate_cells(self) -> None:
        self.recorder.responder = _responder('{"cells": [1, 1]}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_out_of_range_cells(self) -> None:
        self.recorder.responder = _responder('{"cells": [9]}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_boolean_cells(self) -> None:
        self.recorder.responder = _responder('{"cells": [true]}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_float_cells(self) -> None:
        self.recorder.responder = _responder('{"cells": [1.0]}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_textual_coordinates(self) -> None:
        for content in ('{"cells": ["0,1"]}', '{"cells": ["0"]}'):
            self.recorder.responder = _responder(content)
            await self._expect_grid_code("malformed_response")

    async def test_rejects_extra_keys(self) -> None:
        self.recorder.responder = _responder('{"cells": [0], "note": "x"}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_a_missing_cells_field(self) -> None:
        self.recorder.responder = _responder('{"grid": []}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_non_list_cells(self) -> None:
        self.recorder.responder = _responder('{"cells": 0}')
        await self._expect_grid_code("malformed_response")

    async def test_rejects_a_provider_refusal(self) -> None:
        self.recorder.responder = _responder("I cannot help with that request.")
        await self._expect_grid_code("malformed_response")

    async def test_timeout_maps_to_a_fixed_code(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.3)
            return _envelope('{"cells": [0]}')

        self.recorder.responder = respond
        await self._expect_grid_code("request_timeout", seconds_remaining=0.05)

    async def test_cancellation_propagates(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.5)
            return _envelope('{"cells": [0]}')

        self.recorder.responder = respond
        client = self._client()
        task = asyncio.create_task(self._select_grid(client))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_rejects_a_non_200_status(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(status=429, text="slow down")

        self.recorder.responder = respond
        await self._expect_grid_code("unexpected_status")

    async def test_rejects_a_non_png_image_before_posting(self) -> None:
        await self._expect_grid_validation("invalid_image", image=b"GIF89a" + b"\x00" * 40)

    async def test_rejects_a_long_instruction_before_posting(self) -> None:
        await self._expect_grid_validation("invalid_instruction", instruction="a" * 2049)

    async def test_rejects_a_non_positive_budget_before_posting(self) -> None:
        await self._expect_grid_validation("invalid_budget", seconds_remaining=0.0)

    async def test_aclose_blocks_select_grid(self) -> None:
        client = self._client()
        await client.aclose()
        with self.assertRaises(VisionError) as context:
            await self._select_grid(client)
        self.assertEqual(context.exception.code, "client_closed")


class ExplicitHeaderRequestTests(_ClientTestCase):
    """An operator-supplied header value is forwarded, never minted or logged by Prowl."""

    async def test_external_value_reaches_each_request_unchanged(self) -> None:
        value = "external-1729"
        client = self._client(headers={"X-Request-Nonce": value})
        await self._interpret(client)
        await self._interpret(client)
        self.assertEqual(self._header_values("X-Request-Nonce"), [value, value])
        self.assertNotIn(value, repr(client))

    async def test_uuid_placeholder_generates_a_fresh_value_per_request(self) -> None:
        client = self._client(headers={"X-Request-Nonce": "$uuid"})
        await self._interpret(client)
        await self._interpret(client)
        values = self._header_values("X-Request-Nonce")
        self.assertEqual(len(values), 2)
        self.assertEqual([uuid.UUID(value).version for value in values], [4, 4])
        self.assertNotEqual(values[0], values[1])

    async def test_reserved_or_ambiguous_headers_fail_before_network(self) -> None:
        bad = (
            {"Authorization": "override"},
            {"Cookie": "secret"},
            {"User-Agent": "spoof"},
            {"X-Request-Nonce": "a\r\nb"},
            {"X-Request-Nonce": "a", "x-request-nonce": "b"},
        )
        for headers in bad:
            with self.subTest(headers=tuple(headers)):
                with self.assertRaises(VisionError) as caught:
                    self._client(headers=headers)
                self.assertEqual(caught.exception.code, "invalid_headers")
        self.assertEqual(self.recorder.calls, [])


class PointSchemaTests(TestCase):
    """The frozen normalized point schema validates both coordinates without a server."""

    def test_accepts_a_valid_point(self) -> None:
        point = Point(x=0.25, y=0.75)
        self.assertEqual((point.x, point.y), (0.25, 0.75))
        self.assertFalse(hasattr(point, "__dict__"))

    def test_rejects_booleans_strings_non_numbers_and_non_finite_values(self) -> None:
        for value in (True, False, "0.5", None, [0.5], 10**1000, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(VisionError) as context:
                Point(x=value, y=0.5)
            self.assertEqual(context.exception.code, "invalid_point")

    def test_rejects_coordinates_outside_the_open_unit_interval(self) -> None:
        for value in (0.0, 1.0, -0.0, -0.1, 1.1, -1.0, 2.0):
            with self.assertRaises(VisionError) as context:
                Point(x=0.5, y=value)
            self.assertEqual(context.exception.code, "invalid_point")

    def test_repr_holds_only_the_coordinates(self) -> None:
        self.assertEqual(repr(Point(x=0.25, y=0.5)), "Point(x=0.25, y=0.5)")


class PointSelectionSchemaTests(TestCase):
    """The frozen point-selection schema validates the distinct normalized targets."""

    def test_accepts_a_valid_selection(self) -> None:
        result = PointSelection(points=(Point(x=0.1, y=0.2), Point(x=0.3, y=0.4)))
        self.assertEqual(result.points, (Point(x=0.1, y=0.2), Point(x=0.3, y=0.4)))
        self.assertFalse(hasattr(result, "__dict__"))

    def test_accepts_an_empty_selection_and_the_sixteen_point_limit(self) -> None:
        self.assertEqual(PointSelection(points=()).points, ())
        full = tuple(Point(x=0.05 * (index + 1), y=0.5) for index in range(16))
        self.assertEqual(len(PointSelection(points=full).points), 16)

    def test_rejects_invalid_point_tuples(self) -> None:
        builders = (
            lambda: PointSelection(points=[Point(x=0.1, y=0.2)]),
            lambda: PointSelection(points=(Point(x=0.1, y=0.2), Point(x=0.1, y=0.2))),
            lambda: PointSelection(points=(Point(x=0.1, y=0.2), (0.3, 0.4))),
            lambda: PointSelection(points=tuple(Point(x=0.05 * (index + 1), y=0.5) for index in range(17))),
        )
        for build in builders:
            with self.assertRaises(VisionError) as context:
                build()
            self.assertEqual(context.exception.code, "invalid_point_selection")

    def test_repr_holds_only_the_points(self) -> None:
        self.assertEqual(
            repr(PointSelection(points=(Point(x=0.25, y=0.5),))),
            "PointSelection(points=(Point(x=0.25, y=0.5),))",
        )


class PointSelectionRequestTests(_ClientTestCase):
    """The normalized-point request the client posts and the responses it accepts."""

    async def test_posts_the_points_shape(self) -> None:
        self.recorder.responder = _responder(_CANVAS_PATTERN)
        result = await self._select_points(self._client())
        self.assertEqual(
            result.points,
            (
                Point(x=0.1, y=0.1),
                Point(x=0.9, y=0.1),
                Point(x=0.5, y=0.5),
                Point(x=0.1, y=0.9),
                Point(x=0.9, y=0.9),
            ),
        )
        self.assertEqual(len(self.recorder.calls), 1)
        call = self.recorder.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        payload = json.loads(call["body"])
        self.assertEqual(set(payload), {"model", "temperature", "messages"})
        system = payload["messages"][0]
        self.assertEqual(system["role"], "system")
        self.assertIn("points", system["content"])
        self.assertIn("normalized", system["content"])
        self.assertIn("empty list", system["content"])
        self.assertIn("grid", system["content"])
        self.assertNotIn("3x3", system["content"])
        parts = payload["messages"][1]["content"]
        text_part = next(part for part in parts if part["type"] == "text")
        self.assertIn("Click every matching tile", text_part["text"])
        image_part = next(part for part in parts if part["type"] == "image_url")
        expected = "data:image/png;base64," + base64.b64encode(_IMAGE).decode("ascii")
        self.assertEqual(image_part["image_url"]["url"], expected)

    async def test_accepts_an_empty_selection(self) -> None:
        self.recorder.responder = _responder('{"points": []}')
        result = await self._select_points(self._client())
        self.assertEqual(result.points, ())

    async def test_accepts_sixteen_distinct_points(self) -> None:
        self.recorder.responder = _responder(_points(16))
        result = await self._select_points(self._client())
        self.assertEqual(len(result.points), 16)

    async def test_rejects_seventeen_points(self) -> None:
        self.recorder.responder = _responder(_points(17))
        await self._expect_points_code("malformed_response")

    async def test_rejects_duplicate_points(self) -> None:
        self.recorder.responder = _responder('{"points": [{"x": 0.3, "y": 0.7}, {"x": 0.3, "y": 0.7}]}')
        await self._expect_points_code("malformed_response")

    async def test_rejects_boolean_coordinates(self) -> None:
        self.recorder.responder = _responder('{"points": [{"x": true, "y": 0.7}]}')
        await self._expect_points_code("malformed_response")

    async def test_rejects_textual_coordinates(self) -> None:
        self.recorder.responder = _responder('{"points": [{"x": "0.3", "y": "0.7"}]}')
        await self._expect_points_code("malformed_response")

    async def test_rejects_non_finite_and_edge_coordinates(self) -> None:
        for content in (
            '{"points": [{"x": NaN, "y": 0.7}]}',
            '{"points": [{"x": Infinity, "y": 0.7}]}',
            '{"points": [{"x": -Infinity, "y": 0.7}]}',
            '{"points": [{"x": 0.0, "y": 0.7}]}',
            '{"points": [{"x": 1.0, "y": 0.7}]}',
            '{"points": [{"x": 0.3, "y": -0.5}]}',
            '{"points": [{"x": 0.3, "y": 1.5}]}',
            '{"points": [{"x": ' + "9" * 1000 + ', "y": 0.7}]}',
        ):
            self.recorder.responder = _responder(content)
            await self._expect_points_code("malformed_response")

    async def test_rejects_extra_keys_at_either_level(self) -> None:
        for content in (
            '{"points": [{"x": 0.3, "y": 0.7}], "note": "x"}',
            '{"points": [{"x": 0.3, "y": 0.7, "z": 0.1}]}',
            '{"points": [{"x": 0.3, "y": 0.7}], "url": "https://evil.example/click"}',
            '{"points": [{"x": 0.3, "y": 0.7}], "mode": "browser"}',
        ):
            self.recorder.responder = _responder(content)
            await self._expect_points_code("malformed_response")

    async def test_rejects_duplicate_json_keys(self) -> None:
        for content in (
            '{"points": [{"x": 0.3, "x": 0.7, "y": 0.7}]}',
            '{"points": [], "points": [{"x": 0.3, "y": 0.7}]}',
        ):
            self.recorder.responder = _responder(content)
            await self._expect_points_code("malformed_response")

    async def test_rejects_missing_fields_and_wrong_shapes(self) -> None:
        for content in (
            '{"points": [{"x": 0.3}]}',
            '{"points": [{"y": 0.7}]}',
            '{"points": [{"x": 0.3, "y": [0.7]}]}',
            '{"points": [0.5]}',
            '{"points": 0.5}',
            '{"points": {}}',
            '{"scores": []}',
            '{"cells": [0]}',
            "[]",
        ):
            self.recorder.responder = _responder(content)
            await self._expect_points_code("malformed_response")

    async def test_rejects_a_provider_refusal_without_echoing_it(self) -> None:
        self.recorder.responder = _responder("I cannot help with that request.")
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._select_points(client)
        self.assertEqual(context.exception.code, "malformed_response")
        self.assertNotIn("cannot help", str(context.exception))

    async def test_timeout_maps_to_a_fixed_code(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.3)
            return _envelope(_CANVAS_PATTERN)

        self.recorder.responder = respond
        await self._expect_points_code("request_timeout", seconds_remaining=0.05)

    async def test_cancellation_propagates(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.5)
            return _envelope(_CANVAS_PATTERN)

        self.recorder.responder = respond
        client = self._client()
        task = asyncio.create_task(self._select_points(client))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_rejects_a_non_200_status(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(status=502, text="bad gateway")

        self.recorder.responder = respond
        await self._expect_points_code("unexpected_status")

    async def test_transmits_no_cookie_or_proxy_headers(self) -> None:
        self.recorder.responder = _responder(_CANVAS_PATTERN)
        await self._select_points(self._client())
        headers = self.recorder.calls[0]["headers"]
        self.assertNotIn("Cookie", headers)
        self.assertFalse(any(name.lower().startswith("proxy") for name in headers))
        self.assertNotIn("Authorization", headers)

    async def test_ignores_environment_proxy_variables(self) -> None:
        self.recorder.responder = _responder(_CANVAS_PATTERN)
        proxies = {
            "HTTP_PROXY": "http://127.0.0.1:1",
            "HTTPS_PROXY": "http://127.0.0.1:1",
            "ALL_PROXY": "http://127.0.0.1:1",
        }
        with patch.dict(os.environ, proxies):
            result = await self._select_points(self._client())
        self.assertEqual(len(result.points), 5)

    async def test_select_points_sends_a_fresh_uuid_v4(self) -> None:
        self.recorder.responder = _responder(_CANVAS_PATTERN)
        client = self._client(headers={"X-Request-Nonce": "$uuid"})
        await self._select_points(client)
        await self._select_points(client)
        values = self._header_values("X-Request-Nonce")
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertEqual(uuid.UUID(value).version, 4)
        self.assertNotEqual(values[0], values[1])
        self.assertNotIn(values[0], repr(client))

    async def test_rejects_invalid_arguments_before_posting(self) -> None:
        await self._expect_points_validation("invalid_image", image=b"GIF89a" + b"\x00" * 40)
        await self._expect_points_validation("invalid_instruction", instruction="a" * 2049)
        await self._expect_points_validation("invalid_budget", seconds_remaining=0.0)

    async def test_large_request_budget_is_capped_by_native_timeout(self) -> None:
        self.recorder.responder = _responder(_CANVAS_PATTERN)
        result = await self._select_points(self._client(), seconds_remaining=180.0)
        self.assertEqual(len(result.points), 5)

    async def test_aclose_blocks_select_points(self) -> None:
        client = self._client()
        await client.aclose()
        with self.assertRaises(VisionError) as context:
            await self._select_points(client)
        self.assertEqual(context.exception.code, "client_closed")


class CanvasActionSchemaTests(TestCase):
    """The frozen canvas click, drag and plan schemas validate without a server."""

    def test_accepts_a_click_and_a_drag(self) -> None:
        click = CanvasClick(point=Point(x=0.25, y=0.5))
        drag = CanvasDrag(
            start=Point(x=0.25, y=0.5), source_center=Point(x=0.25, y=0.5), target=Point(x=0.75, y=0.5), hold_ms=500
        )
        self.assertEqual(click.point, Point(x=0.25, y=0.5))
        self.assertEqual(drag.start, Point(x=0.25, y=0.5))
        self.assertEqual(drag.source_center, Point(x=0.25, y=0.5))
        self.assertEqual(drag.target, Point(x=0.75, y=0.5))
        self.assertEqual(drag.release, Point(x=0.75, y=0.5))
        self.assertEqual(drag.hold_ms, 500)
        self.assertFalse(hasattr(click, "__dict__"))
        self.assertFalse(hasattr(drag, "__dict__"))

    def test_compensates_for_an_off_center_grab(self) -> None:
        drag = CanvasDrag(
            start=Point(0.19, 0.39),
            source_center=Point(0.15, 0.38),
            target=Point(0.84, 0.5),
            hold_ms=500,
        )
        self.assertAlmostEqual(drag.release.x, 0.88)
        self.assertAlmostEqual(drag.release.y, 0.51)
        self.assertEqual(drag.target, Point(0.84, 0.5))

    def test_rejects_an_out_of_canvas_release(self) -> None:
        for start, center, target in (
            (Point(0.8, 0.5), Point(0.1, 0.5), Point(0.9, 0.5)),
            (Point(0.1, 0.5), Point(0.9, 0.5), Point(0.1, 0.5)),
            (Point(0.5, 0.8), Point(0.5, 0.1), Point(0.5, 0.9)),
            (Point(0.5, 0.1), Point(0.5, 0.9), Point(0.5, 0.1)),
        ):
            with self.assertRaises(VisionError) as context:
                CanvasDrag(start=start, source_center=center, target=target, hold_ms=0)
            self.assertEqual(context.exception.code, "invalid_canvas_drag")

    def test_accepts_an_empty_plan_and_the_eight_action_limit(self) -> None:
        empty = CanvasActionPlan(actions=())
        self.assertEqual(empty.actions, ())
        self.assertFalse(hasattr(empty, "__dict__"))
        full = tuple(CanvasClick(point=Point(x=0.05 * (index + 1), y=0.5)) for index in range(8))
        self.assertEqual(len(CanvasActionPlan(actions=full).actions), 8)

    def test_accepts_zero_and_one_thousand_millisecond_holds(self) -> None:
        for hold_ms in (0, 1000):
            drag = CanvasDrag(
                start=Point(x=0.1, y=0.1),
                source_center=Point(x=0.1, y=0.1),
                target=Point(x=0.9, y=0.9),
                hold_ms=hold_ms,
            )
            self.assertEqual(drag.hold_ms, hold_ms)

    def test_rejects_an_invalid_click(self) -> None:
        for value in ((0.25, 0.5), "0.25,0.5", None, {"x": 0.25, "y": 0.5}, 0.5):
            with self.assertRaises(VisionError) as context:
                CanvasClick(point=value)
            self.assertEqual(context.exception.code, "invalid_canvas_click")

    def test_rejects_an_invalid_drag(self) -> None:
        valid = Point(x=0.5, y=0.5)
        builders = (
            lambda: CanvasDrag(start=(0.1, 0.1), source_center=(0.1, 0.1), target=valid, hold_ms=0),
            lambda: CanvasDrag(start=valid, source_center=(0.5, 0.5), target=valid, hold_ms=0),
            lambda: CanvasDrag(start=valid, source_center=valid, target="0.9,0.9", hold_ms=0),
            lambda: CanvasDrag(start=valid, source_center=valid, target=valid, hold_ms=True),
            lambda: CanvasDrag(start=valid, source_center=valid, target=valid, hold_ms=500.0),
            lambda: CanvasDrag(start=valid, source_center=valid, target=valid, hold_ms="500"),
            lambda: CanvasDrag(start=valid, source_center=valid, target=valid, hold_ms=-1),
            lambda: CanvasDrag(start=valid, source_center=valid, target=valid, hold_ms=1001),
        )
        for build in builders:
            with self.assertRaises(VisionError) as context:
                build()
            self.assertEqual(context.exception.code, "invalid_canvas_drag")

    def test_rejects_an_invalid_plan(self) -> None:
        click = CanvasClick(point=Point(x=0.25, y=0.5))
        builders = (
            lambda: CanvasActionPlan(actions=[click]),
            lambda: CanvasActionPlan(actions=(None,)),
            lambda: CanvasActionPlan(actions=(click, (0.25, 0.5))),
            lambda: CanvasActionPlan(actions=(click,) * 9),
        )
        for build in builders:
            with self.assertRaises(VisionError) as context:
                build()
            self.assertEqual(context.exception.code, "invalid_canvas_action_plan")

    def test_repr_holds_only_the_planned_values(self) -> None:
        plan = CanvasActionPlan(actions=(CanvasClick(point=Point(x=0.25, y=0.5)),))
        self.assertEqual(repr(plan), "CanvasActionPlan(actions=(CanvasClick(point=Point(x=0.25, y=0.5)),))")


class CanvasPlanRequestTests(_ClientTestCase):
    """The canvas action-plan request the client posts and the responses it accepts."""

    async def test_plans_a_click(self) -> None:
        self.recorder.responder = _responder(_CLICK_PLAN)
        result = await self._plan_canvas(self._client())
        self.assertEqual(result, CanvasActionPlan(actions=(CanvasClick(point=Point(x=0.25, y=0.5)),)))

    async def test_plans_a_drag(self) -> None:
        self.recorder.responder = _responder(_DRAG_PLAN)
        result = await self._plan_canvas(self._client())
        expected = CanvasDrag(
            start=Point(x=0.25, y=0.5), source_center=Point(x=0.25, y=0.5), target=Point(x=0.75, y=0.5), hold_ms=500
        )
        self.assertEqual(result.actions, (expected,))

    async def test_plans_a_drag_with_an_off_center_grab(self) -> None:
        content = (
            '{"actions": [{"op": "drag", "from": {"x": 0.19, "y": 0.39}, '
            '"source_center": {"x": 0.15, "y": 0.38}, "to": {"x": 0.84, "y": 0.5}, '
            '"hold_ms": 500}]}'
        )
        self.recorder.responder = _responder(content)
        result = await self._plan_canvas(self._client())
        action = result.actions[0]
        self.assertIsInstance(action, CanvasDrag)
        if isinstance(action, CanvasDrag):
            self.assertAlmostEqual(action.release.x, 0.88)

    async def test_rejects_a_drag_whose_release_leaves_the_canvas(self) -> None:
        content = (
            '{"actions": [{"op": "drag", "from": {"x": 0.8, "y": 0.5}, '
            '"source_center": {"x": 0.1, "y": 0.5}, "to": {"x": 0.9, "y": 0.5}, '
            '"hold_ms": 500}]}'
        )
        self.recorder.responder = _responder(content)
        await self._expect_canvas_code("malformed_response")

    async def test_plans_mixed_actions_in_the_given_order(self) -> None:
        content = (
            '{"actions": ['
            '{"op": "drag", "from": {"x": 0.3, "y": 0.4}, "source_center": {"x": 0.3, "y": 0.4}, '
            '"to": {"x": 0.6, "y": 0.7}, "hold_ms": 0}, '
            '{"op": "click", "x": 0.1, "y": 0.2}]}'
        )
        self.recorder.responder = _responder(content)
        result = await self._plan_canvas(self._client())
        self.assertEqual(
            result.actions,
            (
                CanvasDrag(
                    start=Point(x=0.3, y=0.4), source_center=Point(x=0.3, y=0.4), target=Point(x=0.6, y=0.7), hold_ms=0
                ),
                CanvasClick(point=Point(x=0.1, y=0.2)),
            ),
        )

    async def test_accepts_an_empty_plan(self) -> None:
        self.recorder.responder = _responder('{"actions": []}')
        result = await self._plan_canvas(self._client())
        self.assertEqual(result.actions, ())

    async def test_accepts_eight_actions(self) -> None:
        self.recorder.responder = _responder(_actions(8))
        result = await self._plan_canvas(self._client())
        self.assertEqual(len(result.actions), 8)

    async def test_rejects_nine_actions(self) -> None:
        self.recorder.responder = _responder(_actions(9))
        await self._expect_canvas_code("malformed_response")

    async def test_posts_the_canvas_shape(self) -> None:
        self.recorder.responder = _responder(_CLICK_PLAN)
        await self._plan_canvas(self._client())
        self.assertEqual(len(self.recorder.calls), 1)
        call = self.recorder.calls[0]
        self.assertEqual(call["path"], "/v1/chat/completions")
        payload = json.loads(call["body"])
        self.assertEqual(set(payload), {"model", "temperature", "messages"})
        self.assertEqual(payload["model"], "arbitrary-model-id")
        system = payload["messages"][0]
        self.assertEqual(system["role"], "system")
        for expected in ("actions", "click", "drag", "hold_ms", "normalized", "empty list", "source_center"):
            self.assertIn(expected, system["content"])
        parts = payload["messages"][1]["content"]
        text_part = next(part for part in parts if part["type"] == "text")
        self.assertEqual(text_part["text"], "Drag the animal to the side of the scale that needs it")
        image_part = next(part for part in parts if part["type"] == "image_url")
        expected_image = "data:image/png;base64," + base64.b64encode(_IMAGE).decode("ascii")
        self.assertEqual(image_part["image_url"]["url"], expected_image)

    async def test_rejects_duplicate_json_keys(self) -> None:
        for content in (
            '{"actions": [{"op": "click", "x": 0.25, "x": 0.5, "y": 0.5}]}',
            '{"actions": [], "actions": [{"op": "click", "x": 0.25, "y": 0.5}]}',
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, '
                '"hold_ms": 0, "hold_ms": 500}]}'
            ),
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "x": 0.3, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 0}]}'
            ),
        ):
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_extra_keys_at_every_level(self) -> None:
        for content in (
            '{"actions": [{"op": "click", "x": 0.25, "y": 0.5}], "note": "x"}',
            '{"actions": [{"op": "click", "x": 0.25, "y": 0.5, "selector": "#target"}]}',
            '{"actions": [{"op": "click", "x": 0.25, "y": 0.5}], "url": "https://evil.example/click"}',
            '{"actions": [{"op": "click", "x": 0.25, "y": 0.5}], "script": "document.cookie"}',
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5, "z": 0.1}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 0}]}'
            ),
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, '
                '"hold_ms": 0, "key": "Enter"}]}'
            ),
        ):
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_unknown_or_missing_operations(self) -> None:
        for content in (
            '{"actions": [{"op": "keypress", "x": 0.25, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": 0.25, "y": 0.5}, {"op": "scroll", "x": 0.5, "y": 0.5}]}',
            '{"actions": [{"x": 0.25, "y": 0.5}]}',
            '{"actions": [{"op": 1, "x": 0.25, "y": 0.5}]}',
            '{"actions": [{"op": ["click"], "x": 0.25, "y": 0.5}]}',
        ):
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_invalid_coordinates(self) -> None:
        for content in (
            '{"actions": [{"op": "click", "x": true, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": "0.25", "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": null, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": NaN, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": Infinity, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": -Infinity, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": 0.0, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": 1.0, "y": 0.5}]}',
            '{"actions": [{"op": "click", "x": 0.25, "y": -0.5}]}',
            '{"actions": [{"op": "click", "x": 0.25, "y": 1.5}]}',
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": true, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 0}]}'
            ),
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 1.0, "y": 0.5}, "hold_ms": 0}]}'
            ),
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": [0.75, 0.5], "hold_ms": 0}]}'
            ),
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 0}]}'
            ),
            '{"actions": [{"op": "click", "x": ' + "9" * 1000 + ', "y": 0.5}]}',
        ):
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_an_invalid_hold_duration(self) -> None:
        for hold_ms in ('"500"', "500.0", "-1", "1001", "true"):
            content = (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, '
                '"hold_ms": ' + hold_ms + "}]}"
            )
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_missing_fields_and_wrong_shapes(self) -> None:
        for content in (
            '{"actions": [{"op": "click"}]}',
            '{"actions": [{"op": "click", "x": 0.25}]}',
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}}]}'
            ),
            '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, "hold_ms": 500}]}',
            (
                '{"actions": [{"op": "drag", "from": {"x": 0.25, "y": 0.5}, '
                '"source_center": {"x": 0.25, "y": 0.5}, "to": {"x": 0.75, "y": 0.5}, '
                '"hold_ms": 500, "steps": 3}]}'
            ),
            '{"actions": [0.5]}',
            '{"actions": {}}',
            '{"actions": 0.5}',
            '{"points": []}',
            '{"cells": [0]}',
            "[]",
        ):
            self.recorder.responder = _responder(content)
            await self._expect_canvas_code("malformed_response")

    async def test_rejects_a_provider_refusal_without_echoing_it(self) -> None:
        self.recorder.responder = _responder("I cannot help with that request.")
        client = self._client()
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(client)
        self.assertEqual(context.exception.code, "malformed_response")
        self.assertNotIn("cannot help", str(context.exception))

    async def test_rejects_a_non_200_status(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(status=503, text="unavailable")

        self.recorder.responder = respond
        await self._expect_canvas_code("unexpected_status")

    async def test_timeout_maps_to_a_fixed_code(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.3)
            return _envelope(_DRAG_PLAN)

        self.recorder.responder = respond
        await self._expect_canvas_code("request_timeout", seconds_remaining=0.05)

    async def test_cancellation_propagates(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            await asyncio.sleep(0.5)
            return _envelope(_DRAG_PLAN)

        self.recorder.responder = respond
        client = self._client()
        task = asyncio.create_task(self._plan_canvas(client))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_transmits_no_cookie_or_proxy_headers(self) -> None:
        self.recorder.responder = _responder(_CLICK_PLAN)
        await self._plan_canvas(self._client())
        headers = self.recorder.calls[0]["headers"]
        self.assertNotIn("Cookie", headers)
        self.assertFalse(any(name.lower().startswith("proxy") for name in headers))
        self.assertNotIn("Authorization", headers)

    async def test_sends_a_fresh_header_uuid_and_leaks_no_secret(self) -> None:
        self.recorder.responder = _responder(_CLICK_PLAN)
        client = self._client(api_key="sk-plan-secret", headers={"X-Request-Nonce": "$uuid"})
        await self._plan_canvas(client)
        await self._plan_canvas(client)
        values = self._header_values("X-Request-Nonce")
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertEqual(uuid.UUID(value).version, 4)
        self.assertNotEqual(values[0], values[1])
        self.assertNotIn(values[0], repr(client))
        self.assertNotIn("sk-plan-secret", repr(client))
        self.recorder.responder = _responder('{"actions": [{"op": "click", "x": 0.25, "y": 0.5}], "url": "x"}')
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(client)
        self.assertEqual(context.exception.code, "malformed_response")
        self.assertNotIn("sk-plan-secret", str(context.exception))
        self.assertNotIn(values[0], str(context.exception))
        self.assertNotIn(self.endpoint, str(context.exception))

    async def test_rejects_invalid_arguments_before_posting(self) -> None:
        await self._expect_canvas_validation("invalid_image", image=b"GIF89a" + b"\x00" * 40)
        await self._expect_canvas_validation("invalid_instruction", instruction="a" * 2049)
        await self._expect_canvas_validation("invalid_instruction", instruction="")
        await self._expect_canvas_validation("invalid_budget", seconds_remaining=0.0)
        await self._expect_canvas_validation("invalid_budget", seconds_remaining=10**1000)

    async def test_native_timeout_uses_caller_budget(self) -> None:
        self.assertEqual(_validate_budget(120.0), 120.0)
        self.assertEqual(_validate_budget(10.0), 10.0)
        self.recorder.responder = _responder(_CLICK_PLAN)
        result = await self._plan_canvas(self._client(), seconds_remaining=180.0)
        self.assertEqual(len(result.actions), 1)

    async def test_aclose_blocks_plan_canvas(self) -> None:
        client = self._client()
        await client.aclose()
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(client)
        self.assertEqual(context.exception.code, "client_closed")


class ResponsesRequestTests(_ClientTestCase):
    """The endpoint path selects native Responses input/output without a bridge."""

    def _responses_client(
        self,
        *,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        path: str = "/v1/responses",
    ) -> CloudVlmClient:
        endpoint = str(self.server.make_url(path))
        client = CloudVlmClient(
            endpoint,
            "muse-spark-1.3-contributor",
            api_key=api_key,
            headers=headers,
            reasoning_effort=reasoning_effort,
        )
        self.addAsyncCleanup(client.aclose)
        return client

    async def _respond_with(self, content: object) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return _responses_envelope(content)

        self.recorder.responder = respond

    async def test_posts_native_multimodal_responses_shape(self) -> None:
        await self._respond_with(_VALID_INTERPRETATION)
        result = await self._interpret(self._responses_client())
        self.assertEqual(result.labels, ("alpha",))
        call = self.recorder.calls[0]
        self.assertEqual(call["path"], "/v1/responses")
        payload = json.loads(call["body"])
        self.assertEqual(payload["model"], "muse-spark-1.3-contributor")
        self.assertNotIn("messages", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("reasoning", payload)
        system, user = payload["input"]
        self.assertEqual(system["role"], "system")
        self.assertIn("kind", system["content"][0]["text"])
        self.assertEqual(system["content"][0]["type"], "input_text")
        self.assertEqual(user["role"], "user")
        self.assertIn("alpha", user["content"][0]["text"])
        self.assertEqual(user["content"][0]["type"], "input_text")
        self.assertEqual(user["content"][1]["type"], "input_image")
        self.assertEqual(user["content"][1]["image_url"], "data:image/png;base64," + base64.b64encode(_IMAGE).decode())

    async def test_all_operations_use_native_responses(self) -> None:
        client = self._responses_client()
        await self._respond_with('{"cells": [4]}')
        self.assertEqual((await self._select_grid(client)).cells, (4,))
        await self._respond_with('{"points": [{"x": 0.2, "y": 0.4}]}')
        self.assertEqual((await self._select_points(client)).points, (Point(0.2, 0.4),))
        await self._respond_with(_DRAG_PLAN)
        plan = await self._plan_canvas(client)
        drag = plan.actions[0]
        self.assertIsInstance(drag, CanvasDrag)
        if isinstance(drag, CanvasDrag):
            self.assertEqual(drag.release, Point(0.75, 0.5))
        self.assertEqual(len(self.recorder.calls), 3)
        for call in self.recorder.calls:
            payload = json.loads(call["body"])
            self.assertIn("input", payload)
            self.assertNotIn("messages", payload)
            self.assertEqual(call["path"], "/v1/responses")

    async def test_trailing_slash_is_responses_but_other_paths_remain_chat(self) -> None:
        await self._respond_with(_VALID_INTERPRETATION)
        await self._interpret(self._responses_client(path="/v1/responses/"))
        self.assertIn("input", json.loads(self.recorder.calls[0]["body"]))
        self.recorder.responder = _responder(_VALID_INTERPRETATION)
        client = self._responses_client(path="/v1/not-responses")
        await self._interpret(client)
        self.assertIn("messages", json.loads(self.recorder.calls[1]["body"]))

    async def test_ignores_reasoning_and_accepts_split_text_parts(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.json_response(
                {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "private"}]},
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": '{"cells": '},
                                {"type": "output_text", "text": "[1]}"},
                            ],
                        },
                    ],
                }
            )

        self.recorder.responder = respond
        self.assertEqual((await self._select_grid(self._responses_client())).cells, (1,))

    async def test_rejects_incomplete_refusal_tool_call_and_ambiguous_messages(self) -> None:
        invalid_outputs = (
            {
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _VALID_INTERPRETATION}],
                    }
                ],
            },
            {
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}],
            },
            {"status": "completed", "output": [{"type": "function_call", "name": "browser_click"}]},
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _VALID_INTERPRETATION}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _VALID_INTERPRETATION}],
                    },
                ],
            },
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "output_text", "text": _VALID_INTERPRETATION}],
                    }
                ],
            },
            {"status": "completed", "output": []},
        )
        for document in invalid_outputs:

            async def respond(_request: web.Request, body: object = document) -> web.StreamResponse:
                return web.json_response(body)

            self.recorder.responder = respond
            with self.subTest(document=document), self.assertRaises(VisionError) as context:
                await self._interpret(self._responses_client())
            self.assertEqual(context.exception.code, "malformed_response")

    async def test_rejects_duplicate_envelope_keys_and_invalid_json_plan(self) -> None:
        async def respond(_request: web.Request) -> web.StreamResponse:
            return web.Response(
                text='{"status":"completed","status":"completed","output":[]}', content_type="application/json"
            )

        self.recorder.responder = respond
        with self.assertRaises(VisionError) as context:
            await self._interpret(self._responses_client())
        self.assertEqual(context.exception.code, "malformed_response")
        await self._respond_with('{"actions": [{"op": "drag", "from": {"x": 0.2, "y": 0.3}}]}')
        with self.assertRaises(VisionError) as context:
            await self._plan_canvas(self._responses_client())
        self.assertEqual(context.exception.code, "malformed_response")

    async def test_reads_full_bounded_split_responses_body(self) -> None:
        body = json.dumps(
            {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": [], "padding": "x" * 5000},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": _VALID_INTERPRETATION}],
                    },
                ],
            }
        ).encode()

        async def respond(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(status=200, headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(body[:2048])
            await asyncio.sleep(0.05)
            await response.write(body[2048:])
            await response.write_eof()
            return response

        self.recorder.responder = respond
        self.assertEqual((await self._interpret(self._responses_client())).labels, ("alpha",))

    async def test_minimal_reasoning_is_sent_on_native_responses_only(self) -> None:
        await self._respond_with(_DRAG_PLAN)
        client = self._responses_client(reasoning_effort="minimal")
        await self._plan_canvas(client)
        payload = json.loads(self.recorder.calls[0]["body"])
        self.assertEqual(payload["reasoning"], {"effort": "minimal"})
        self.assertNotIn("messages", payload)

    async def test_invalid_reasoning_or_chat_endpoint_is_rejected_before_network(self) -> None:
        for effort, path in (("verbose", "/v1/responses"), ("minimal", "/v1/chat/completions")):
            with self.subTest(effort=effort, path=path), self.assertRaises(VisionError) as context:
                self._responses_client(reasoning_effort=effort, path=path)
            self.assertEqual(context.exception.code, "invalid_reasoning_effort")
        self.assertEqual(self.recorder.calls, [])

    async def test_transport_keeps_key_header_and_cookie_guards(self) -> None:
        await self._respond_with('{"cells": [1]}')
        client = self._responses_client(api_key="sk-secret", headers={"X-Request-Nonce": "$uuid"})
        await self._select_grid(client)
        await self._select_grid(client)
        self.assertEqual(self._header_values("Authorization"), ["Bearer sk-secret"] * 2)
        ids = self._header_values("X-Request-Nonce")
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])
        for value in ids:
            self.assertEqual(uuid.UUID(value).version, 4)
        for call in self.recorder.calls:
            self.assertNotIn("Cookie", call["headers"])
            self.assertFalse(any(name.lower().startswith("proxy") for name in call["headers"]))
        self.assertNotIn("sk-secret", repr(client))
