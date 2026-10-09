"""Internal HTTP response byte/header seams: decoded entity bytes and repeated header pairs.

The HTTP transport carries curl's own decoded entity bytes and the final response's repeated header
pairs through ``HttpResult`` and ``FetchResult`` without re-encoding the text. Every test here talks
to the loopback HTTP/1.1 server and stand-in context from ``test_service_http_transport`` and the
backend routing fixture from ``test_service_routing_backend``; no browser is launched, nothing
external is contacted, and the new fields never enter the JSON solution.
"""

from __future__ import annotations

import gzip
import json
from unittest import TestCase

from prowl.service.backend import FetchResult
from prowl.service.classification import SUCCESS
from prowl.service.http_transport import HttpResult
from prowl.service.protocol import BROWSER_MODE, HTTP_MODE, solution_payload
from test_service_http_transport import _Reply, _routes, _TransportTestCase
from test_service_routing_backend import _request, _RoutingFixture, _StubHttpClient


class ResponseByteSeamTests(_TransportTestCase):
    """A final HTTP response carries curl's decoded bytes and its own repeated header pairs."""

    async def test_final_response_exposes_decoded_bytes_and_repeated_headers(self) -> None:
        payload = b"\x00\xff\xfeok"
        reply: _Reply = (
            200,
            [
                ("Content-Type", "text/html"),
                ("X-Repeat", "one"),
                ("X-Repeat", "two"),
                ("Set-Cookie", "a=1; Path=/"),
                ("Set-Cookie", "b=2; Path=/"),
            ],
            payload,
        )
        server = await self.start(_routes({"/page": reply}))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/page", deadline_seconds=10)

        self.assertIsInstance(result.body_bytes, bytes)
        self.assertEqual(result.body_bytes, payload)
        # The text is curl's replacement decode, so re-encoding it cannot reproduce the bytes.
        self.assertNotEqual(result.body.encode("utf-8"), payload)
        self.assertIn("\ufffd", result.body)
        pairs = [(name.lower(), value) for name, value in result.header_items or ()]
        self.assertEqual([value for name, value in pairs if name == "x-repeat"], ["one", "two"])
        # The existing text mapping and cookie surfaces behave exactly as before.
        self.assertEqual(result.headers["x-repeat"], "one, two")
        self.assertEqual(result.set_cookie_headers, ("a=1; Path=/", "b=2; Path=/"))
        self.assertEqual(self.context.names(), {"a", "b"})
        self.assertEqual({cookie["name"] for cookie in result.cookies}, {"a", "b"})
        self.assertEqual(result.classify().category, SUCCESS)

    async def test_compressed_content_exposes_decoded_entity_bytes(self) -> None:
        payload = b"\x00\xffdecoded"
        compressed = gzip.compress(payload)
        server = await self.start(
            _routes({"/gzip": (200, [("Content-Encoding", "gzip")], compressed)}),
        )
        result = await self.make_client().fetch(server.base_url + "/gzip", deadline_seconds=10)
        self.assertEqual(result.body_bytes, payload)
        self.assertEqual(result.headers["content-encoding"], "gzip")
        self.assertIn(("content-encoding", "gzip"), tuple((k.lower(), v) for k, v in result.header_items or ()))

    async def test_empty_http_body_is_empty_bytes_not_none(self) -> None:
        server = await self.start(_routes({"/empty": (200, [], b"")}))
        client = self.make_client()

        result = await client.fetch(server.base_url + "/empty", deadline_seconds=10)

        self.assertIsNotNone(result.body_bytes)
        self.assertEqual(result.body_bytes, b"")
        self.assertEqual(result.body, "")

    async def test_header_items_hold_only_the_final_response_pairs(self) -> None:
        reply: _Reply = (200, [("X-Marker", "one"), ("X-Marker", "two")], b"final")
        server = await self.start(
            _routes(
                {
                    "/start": (
                        302,
                        [
                            ("Location", "/end"),
                            ("X-Hop", "intermediate"),
                            ("Set-Cookie", "hop=1; Path=/"),
                        ],
                        b"",
                    ),
                    "/end": reply,
                },
            ),
        )
        client = self.make_client()

        result = await client.fetch(server.base_url + "/start", deadline_seconds=10)

        pairs = tuple((name.lower(), value) for name, value in result.header_items or ())
        self.assertEqual([value for name, value in pairs if name == "x-marker"], ["one", "two"])
        self.assertNotIn("x-hop", {name for name, _ in pairs})
        # The redirect's accumulated set-cookie is still reported and its text mapping unchanged.
        self.assertEqual(result.set_cookie_headers, ("hop=1; Path=/",))
        self.assertEqual(result.url, server.base_url + "/end")


class HttpStageBytePropagationTests(_RoutingFixture):
    """The HTTP stage copies the byte/header fields into the fetch result without touching text."""

    async def test_http_stage_propagates_bytes_and_header_pairs(self) -> None:
        payload = b"\x00\xffok"
        header_items = (("Content-Type", "text/html"), ("X-Marker", "one"), ("X-Marker", "two"))
        http_result = HttpResult(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            body="<html><body>ok</body></html>",
            cookies=[],
            set_cookie_headers=(),
            body_bytes=payload,
            header_items=header_items,
        )
        backend, _recorder, _clients, _site = self._backend(http_client=_StubHttpClient(result=http_result))

        result = await backend.fetch(None, _request(mode=HTTP_MODE))

        self.assertIs(result.body_bytes, payload)
        self.assertEqual(result.header_items, header_items)
        self.assertEqual(result.response, http_result.body)
        self.assertEqual(result.status_code, 200)
        self.assertIsNotNone(result.classification)
        assert result.classification is not None
        self.assertEqual(result.classification.category, SUCCESS)
        await backend.aclose()

    async def test_browser_stage_leaves_the_byte_fields_none(self) -> None:
        backend, _recorder, _clients, _site = self._backend()

        result = await backend.fetch(None, _request(mode=BROWSER_MODE))

        self.assertIsNone(result.body_bytes)
        self.assertIsNone(result.header_items)
        await backend.aclose()


class DtoDefaultsTests(TestCase):
    """The new fields default to ``None`` and never enter the JSON solution."""

    def test_positional_http_result_keeps_none_defaults(self) -> None:
        result = HttpResult("https://example.com/", 200, {}, "x", [], ())

        self.assertIsNone(result.body_bytes)
        self.assertIsNone(result.header_items)

    def test_positional_fetch_result_keeps_none_defaults(self) -> None:
        result = FetchResult("https://example.com/", 200, {}, "x", [], "ua")

        self.assertIsNone(result.body_bytes)
        self.assertIsNone(result.header_items)
        self.assertIsNone(result.mode)
        self.assertIsNone(result.classification)

    def test_solution_payload_stays_text_only(self) -> None:
        payload = solution_payload(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            response="<html>ok</html>",
            cookies=[],
            user_agent="ua",
        )

        self.assertEqual(
            set(payload),
            {"url", "status", "headers", "response", "cookies", "userAgent"},
        )
        rendered = json.dumps(payload)
        self.assertIn("<html>ok</html>", rendered)
        self.assertNotIn("body_bytes", rendered)
        self.assertNotIn("header_items", rendered)
