"""Pure tests for request preparation: target binding, header policy, raw cookie, exact body.

Every test builds a real :class:`ProxyRequest` DTO and calls :func:`prepare_request`. Nothing here
opens a socket, terminates TLS, launches a browser, or touches native cookie state.
"""

from __future__ import annotations

from unittest import TestCase

from prowl.browser.proxy.http1 import ProxyProtocolError, ProxyRequest
from prowl.browser.proxy.request import PreparedRequest, connect_authority, prepare_request


def _request(
    method: str = "GET",
    target: str = "https://example.com/",
    headers: tuple[tuple[str, str], ...] = (("Host", "example.com"),),
    body: bytes = b"",
) -> ProxyRequest:
    return ProxyRequest(method=method, target=target, headers=headers, body=body)


class TargetPreservationTests(TestCase):
    def test_query_and_percent_escapes_preserved_and_body_is_same_object(self) -> None:
        body = b"\x00\xffbin\x80POST"
        prepared = prepare_request(
            _request(
                method="POST",
                target="https://example.com/a%2Fb?q=1&x=%20",
                headers=(("Host", "example.com"), ("Content-Type", "application/octet-stream")),
                body=body,
            )
        )
        self.assertIsInstance(prepared, PreparedRequest)
        self.assertEqual(prepared.method, "POST")
        self.assertEqual(prepared.url, "https://example.com/a%2Fb?q=1&x=%20")
        self.assertEqual(prepared.headers, {"content-type": "application/octet-stream"})
        self.assertIs(prepared.body, body)

    def test_origin_form_outside_tunnel_rejected(self) -> None:
        with self.assertRaises(ProxyProtocolError) as ctx:
            prepare_request(_request(target="/path?x=1"))
        self.assertEqual(ctx.exception.status, 400)

    def test_absolute_url_rejects_scheme_fragment_credentials_and_unsafe_chars(self) -> None:
        targets = (
            "ftp://example.com/",
            "https://example.com/#frag",
            "https://example.com/?x=1#frag",
            "https://user:pw@example.com/",
            "https://@example.com/",
            "https://example.com/#",
            "https://example.com:/",
            "https://example.com/a\\b",
            "https://example.com/a b",
            "http://example.com\u0000/",
        )
        for target in targets:
            with self.assertRaises(ProxyProtocolError) as ctx:
                prepare_request(_request(target=target))
            self.assertEqual(ctx.exception.status, 400)

    def test_connect_and_unsupported_methods(self) -> None:
        with self.assertRaises(ProxyProtocolError) as connect_ctx:
            prepare_request(_request(method="CONNECT", target="example.com:443", headers=()))
        self.assertEqual(connect_ctx.exception.status, 501)

        with self.assertRaises(ProxyProtocolError) as put_ctx:
            prepare_request(_request(method="PUT"))
        self.assertEqual(put_ctx.exception.status, 405)


class HeaderPolicyTests(TestCase):
    def test_encoded_body_is_rejected_instead_of_losing_its_encoding(self) -> None:
        with self.assertRaises(ProxyProtocolError) as ctx:
            prepare_request(_request(method="POST", headers=(("Content-Encoding", "gzip"),), body=b"compressed"))
        self.assertEqual(ctx.exception.status, 415)
        body = b"plain"
        prepared = prepare_request(_request(method="POST", headers=(("Content-Encoding", "identity"),), body=body))
        self.assertIs(prepared.body, body)
        self.assertNotIn("content-encoding", prepared.headers)

    def test_browser_owned_headers_dropped_and_custom_retained(self) -> None:
        prepared = prepare_request(
            _request(
                headers=(
                    ("Host", "example.com"),
                    ("User-Agent", "native"),
                    ("Accept-Language", "fr"),
                    ("Sec-Fetch-Site", "cross-site"),
                    ("Proxy-Authorization", "Basic xyz"),
                    ("Authorization", "Bearer sekret"),
                    ("X-Trace", "abc"),
                )
            )
        )
        self.assertEqual(prepared.headers, {"authorization": "Bearer sekret", "x-trace": "abc"})

    def test_duplicate_retained_header_rejected_without_echoing_value(self) -> None:
        with self.assertRaises(ProxyProtocolError) as ctx:
            prepare_request(
                _request(
                    headers=(
                        ("Host", "example.com"),
                        ("Authorization", "Bearer sekret"),
                        ("authorization", "Bearer other"),
                    )
                )
            )
        self.assertEqual(ctx.exception.status, 400)
        self.assertNotIn("sekret", str(ctx.exception))

    def test_host_mismatch_and_duplicate_host_rejected(self) -> None:
        for headers in (
            (("Host", "other.example"),),
            (("Host", "example.com"), ("host", "example.com")),
            (("Host", "example.com:8443"),),
        ):
            with self.assertRaises(ProxyProtocolError) as ctx:
                prepare_request(_request(headers=headers))
            self.assertEqual(ctx.exception.status, 400)

    def test_host_default_and_explicit_port_accepted(self) -> None:
        self.assertEqual(
            prepare_request(_request(headers=(("Host", "example.com:443"),))).url,
            "https://example.com/",
        )


class CookieTests(TestCase):
    def test_connection_nomination_suppresses_raw_cookie(self) -> None:
        prepared = prepare_request(
            _request(headers=(("Host", "example.com"), ("Connection", "Cookie"), ("Cookie", "a=1")))
        )
        self.assertIsNone(prepared.cookie_header)

    def test_multiple_cookie_fields_joined_in_wire_order(self) -> None:
        prepared = prepare_request(
            _request(
                headers=(
                    ("Host", "example.com"),
                    ("Cookie", "a=1"),
                    ("Cookie", "a=2; b=3"),
                )
            )
        )
        self.assertEqual(prepared.cookie_header, "a=1; a=2; b=3")

    def test_absent_cookie_is_none(self) -> None:
        self.assertIsNone(prepare_request(_request()).cookie_header)


class ConnectAuthorityTests(TestCase):
    def test_valid_domain_and_ipv6_authorities(self) -> None:
        self.assertEqual(connect_authority("Example.COM:8443"), ("example.com", 8443))
        self.assertEqual(connect_authority("[2001:DB8::1]:443"), ("2001:db8::1", 443))

    def test_bad_authorities_raise_generic_400_without_echoing(self) -> None:
        bad = (
            "example.com",
            "example.com:0",
            "example.com:70000",
            "user@example.com:443",
            "@example.com:443",
            "example.com:443?",
            "example.com:443#",
            "example.com:443/path",
            "example.com:443?q=1",
            "example.com:443#f",
            "http://example.com:443",
            "exa mple.com:443",
        )
        for value in bad:
            with self.assertRaises(ProxyProtocolError) as ctx:
                connect_authority(value)
            self.assertEqual(ctx.exception.status, 400)
            self.assertNotIn("example.com", str(ctx.exception))


class TunnelBindingTests(TestCase):
    def test_origin_form_bound_literally_and_double_slash_stays_put(self) -> None:
        prepared = prepare_request(
            _request(target="//other.example/x", headers=(("Host", "tunnel.example:443"),)),
            tunnel_authority="tunnel.example:443",
        )
        self.assertEqual(prepared.url, "https://tunnel.example:443//other.example/x")

    def test_origin_form_rejects_unsafe_characters_and_fragments(self) -> None:
        for target in ("/path#fragment", "/path#", "/bad\\path", "/bad path", "/bad\x00path"):
            with self.subTest(target=target), self.assertRaises(ProxyProtocolError):
                prepare_request(
                    _request(target=target, headers=(("Host", "tunnel.example"),)),
                    tunnel_authority="tunnel.example:443",
                )

    def test_absolute_https_url_matching_authority_accepted(self) -> None:
        prepared = prepare_request(
            _request(target="https://tunnel.example/", headers=(("Host", "tunnel.example"),)),
            tunnel_authority="tunnel.example:443",
        )
        self.assertEqual(prepared.url, "https://tunnel.example/")

    def test_cross_host_port_and_scheme_rejected(self) -> None:
        for target in (
            "https://other.example/",
            "https://tunnel.example:8443/",
            "http://tunnel.example/",
        ):
            with self.assertRaises(ProxyProtocolError) as ctx:
                prepare_request(
                    _request(target=target, headers=(("Host", "tunnel.example"),)),
                    tunnel_authority="tunnel.example:443",
                )
            self.assertEqual(ctx.exception.status, 400)
