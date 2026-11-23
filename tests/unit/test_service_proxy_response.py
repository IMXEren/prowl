"""Pure projection of a fetched result into a safe, bytes-exact proxy response.

Every test builds a real :class:`FetchResult` DTO; nothing here launches a browser, contacts the
network, or fabricates a native result. The projection only chooses a representation, filters
headers, and recomputes framing.
"""

from __future__ import annotations

from unittest import TestCase

from prowl.browser.proxy.response import ProxyResponse, ProxyResponseError, project_response
from prowl.service.backend import FetchResult
from prowl.service.protocol import BROWSER_MODE, HTTP_MODE, ExecutionMode


def _http_result(
    *,
    status_code: int | None = 200,
    body_bytes: bytes | None = b"body",
    header_items: tuple[tuple[str, str], ...] | None = (("Content-Type", "text/plain"),),
) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=status_code,
        headers={},
        response="",
        cookies=[],
        user_agent="ua",
        mode=HTTP_MODE,
        body_bytes=body_bytes,
        header_items=header_items,
    )


def _rendered_result(
    *,
    status_code: int = 200,
    response: str = "<html>ok</html>",
    headers: dict[str, str] | None = None,
    mode: ExecutionMode | None = BROWSER_MODE,
) -> FetchResult:
    return FetchResult(
        url="https://example.com/",
        status_code=status_code,
        headers={} if headers is None else headers,
        response=response,
        cookies=[],
        user_agent="ua",
        mode=mode,
    )


def _values(response: ProxyResponse, name: str) -> list[str]:
    lowered = name.lower()
    return [value for key, value in response.headers if key.lower() == lowered]


def _names(response: ProxyResponse) -> set[str]:
    return {key.lower() for key, _ in response.headers}


class HttpOriginProjectionTests(TestCase):
    """An HTTP result is projected from curl's decoded bytes and its original final pairs."""

    def test_binary_body_and_repeated_set_cookie_are_preserved(self) -> None:
        payload = b"\x00\xff\xfeok"
        result = _http_result(
            body_bytes=payload,
            header_items=(
                ("Content-Type", "text/html"),
                ("Set-Cookie", "a=1; Path=/"),
                ("Set-Cookie", "b=2; Path=/"),
            ),
        )

        response = project_response(result)

        self.assertIs(response.body, payload)
        self.assertEqual(response.status, 200)
        self.assertEqual(_values(response, "set-cookie"), ["a=1; Path=/", "b=2; Path=/"])
        self.assertEqual(_values(response, "content-length"), [str(len(payload))])
        self.assertEqual(_values(response, "x-prowl-representation"), ["origin"])

    def test_gzip_decoded_body_drops_encoding_and_stale_validators(self) -> None:
        result = _http_result(
            body_bytes=b"decoded",
            header_items=(
                ("Content-Encoding", "gzip"),
                ("Content-Type", "text/html"),
                ("ETag", 'W/"abc"'),
                ("Last-Modified", "Wed, 21 Oct 2015 07:28:00 GMT"),
                ("Content-Range", "bytes 0-6/7"),
            ),
        )

        response = project_response(result)

        self.assertEqual(response.body, b"decoded")
        self.assertEqual(_names(response) & {"content-encoding", "etag", "last-modified", "content-range"}, set())
        self.assertEqual(_values(response, "content-type"), ["text/html"])
        self.assertEqual(_values(response, "content-length"), ["7"])

    def test_plain_body_retains_validators_and_range_data(self) -> None:
        result = _http_result(
            body_bytes=b"plain",
            header_items=(
                ("Content-Type", "application/json"),
                ("ETag", '"v1"'),
                ("Accept-Ranges", "bytes"),
            ),
        )

        response = project_response(result)

        self.assertEqual(_values(response, "etag"), ['"v1"'])
        self.assertEqual(_values(response, "accept-ranges"), ["bytes"])
        self.assertEqual(_values(response, "content-type"), ["application/json"])

    def test_empty_body_is_genuine_bytes_not_none(self) -> None:
        result = _http_result(body_bytes=b"", header_items=(("Content-Type", "text/plain"),))

        response = project_response(result)

        self.assertEqual(response.body, b"")
        self.assertEqual(_values(response, "content-length"), ["0"])


class HopByHopFilterTests(TestCase):
    """Hop-by-hop fields and every name a ``Connection`` field nominates are dropped."""

    def test_connection_nominated_headers_filtered_case_insensitively(self) -> None:
        result = _http_result(
            header_items=(
                ("Connection", "Keep-Alive, X-Custom"),
                ("Keep-Alive", "timeout=5"),
                ("X-Custom", "drop-me"),
                ("X-Kept", "keep-me"),
                ("Transfer-Encoding", "chunked"),
            ),
        )

        response = project_response(result)

        self.assertEqual(_names(response) & {"connection", "keep-alive", "x-custom", "transfer-encoding"}, set())
        self.assertEqual(_values(response, "x-kept"), ["keep-me"])


class RenderedProjectionTests(TestCase):
    """Rendered text is UTF-8, forced HTML, and marked so no stale origin metadata survives."""

    def test_rendered_html_gets_marker_and_drops_stale_headers(self) -> None:
        result = _rendered_result(
            response="<html>caf\u00e9</html>",
            headers={
                "Content-Type": "application/pdf",
                "ETag": '"origin"',
                "Set-Cookie": "session=1; Path=/",
                "X-Kept": "keep-me",
            },
        )

        response = project_response(result)

        self.assertEqual(response.body, "<html>caf\u00e9</html>".encode("utf-8"))
        self.assertEqual(_values(response, "x-prowl-representation"), ["rendered"])
        self.assertEqual(_values(response, "content-type"), ["text/html; charset=utf-8"])
        self.assertEqual(_values(response, "content-length"), [str(len(response.body))])
        self.assertEqual(_names(response) & {"etag", "set-cookie"}, set())
        self.assertEqual(_values(response, "x-kept"), ["keep-me"])

    def test_missing_mode_is_treated_as_rendered(self) -> None:
        response = project_response(_rendered_result(response="hi", mode=None))

        self.assertEqual(_values(response, "x-prowl-representation"), ["rendered"])
        self.assertEqual(response.body, b"hi")

    def test_forged_marker_is_replaced_not_duplicated(self) -> None:
        result = _rendered_result(headers={"X-Prowl-Representation": "forged"})

        response = project_response(result)

        self.assertEqual(_values(response, "x-prowl-representation"), ["rendered"])


class StatusRuleTests(TestCase):
    """Statuses that have no projectable reply, and 204/205/304 framing, are handled explicitly."""

    def test_unprojectable_statuses_are_rejected(self) -> None:
        for status in (None, 101, 199, 600):
            with self.subTest(status=status), self.assertRaises(ProxyResponseError):
                project_response(_http_result(status_code=status))

    def test_http_result_without_origin_representation_is_rejected(self) -> None:
        with self.assertRaises(ProxyResponseError):
            project_response(_http_result(body_bytes=None))
        with self.assertRaises(ProxyResponseError):
            project_response(_http_result(header_items=None))

    def test_204_205_304_body_and_length_rules(self) -> None:
        no_content = project_response(_http_result(status_code=204, body_bytes=b"ignored"))
        self.assertEqual(no_content.body, b"")
        self.assertNotIn("content-length", _names(no_content))

        reset = project_response(_http_result(status_code=205, body_bytes=b"ignored"))
        self.assertEqual(reset.body, b"")
        self.assertEqual(_values(reset, "content-length"), ["0"])

        not_modified = project_response(_http_result(status_code=304, body_bytes=b""))
        self.assertEqual(not_modified.body, b"")
        self.assertNotIn("content-length", _names(not_modified))


class UnsafeHeaderTests(TestCase):
    """Retained fields must be serializable HTTP/1 without ever echoing the offending value."""

    def test_unsafe_retained_values_are_rejected_generically(self) -> None:
        for header_items in (
            (("X-Bad", "secret\r\nInjected: 1"),),
            (("X-Bad", "snowman \u2603"),),
            (("X Bad", "value"),),
        ):
            with self.subTest(header_items=header_items):
                with self.assertRaises(ProxyResponseError) as caught:
                    project_response(_http_result(header_items=header_items))
                self.assertNotIn("secret", str(caught.exception))
                self.assertNotIn("snowman", str(caught.exception))
