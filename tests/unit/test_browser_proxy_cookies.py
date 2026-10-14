"""Contract tests for resolving a caller Cookie header against native context cookies.

Pure and offline: no browser, no network, no native IO. Native cookies are real typed
Playwright ``Cookie`` records and every update is a real ``SetCookieParam``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase

from prowl.browser.proxy.cookies import CookieHeaderError, request_cookie_updates

if TYPE_CHECKING:
    from playwright._impl._api_structures import Cookie

_URL = "https://example.test/app"
_MESSAGE = "request cookie header cannot be applied safely"


def _cookie(
    name: str,
    value: str,
    *,
    domain: str = "example.test",
    path: str = "/",
    partition_key: str | None = None,
) -> Cookie:
    cookie: Cookie = {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "expires": 1700000000.0,
        "httpOnly": False,
        "secure": False,
        "sameSite": "Lax",
    }
    if partition_key is not None:
        cookie["partitionKey"] = partition_key
    return cookie


class NewCookieTests(TestCase):
    """A name absent from the native store becomes one new URL-scoped cookie."""

    def test_new_cookie_carries_the_url_and_only_the_url(self) -> None:
        updates = request_cookie_updates("sid=abc", [], _URL)

        self.assertEqual(updates, [{"name": "sid", "value": "abc", "url": _URL}])

    def test_raw_values_are_preserved_without_unquoting_or_reencoding(self) -> None:
        updates = request_cookie_updates('empty=; quoted="xy"', [], _URL)

        self.assertEqual(
            updates,
            [
                {"name": "empty", "value": "", "url": _URL},
                {"name": "quoted", "value": '"xy"', "url": _URL},
            ],
        )


class ExistingCookieTests(TestCase):
    """A known name patches its existing cookie in place, or no-ops when unchanged."""

    def test_changed_value_retains_the_full_accepted_identity(self) -> None:
        existing: Cookie = {
            "name": "sid",
            "value": "old",
            "domain": ".example.test",
            "path": "/app",
            "expires": 1700000000.5,
            "httpOnly": True,
            "secure": True,
            "sameSite": "Strict",
        }

        updates = request_cookie_updates("sid=new", [existing], _URL)

        self.assertEqual(
            updates,
            [
                {
                    "name": "sid",
                    "value": "new",
                    "domain": ".example.test",
                    "path": "/app",
                    "expires": 1700000000.5,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Strict",
                }
            ],
        )
        self.assertNotIn("url", updates[0])

    def test_same_value_is_a_noop(self) -> None:
        self.assertEqual(request_cookie_updates("sid=old", [_cookie("sid", "old")], _URL), [])

    def test_identical_duplicate_values_across_scopes_are_untouched(self) -> None:
        existing = [
            _cookie("sid", "dup", domain="example.test", path="/"),
            _cookie("sid", "dup", domain="other.test", path="/deep"),
        ]

        self.assertEqual(request_cookie_updates("sid=dup; sid=dup", existing, _URL), [])


class AmbiguousCookieTests(TestCase):
    """Every duplicate situation that would need a scope guess fails closed."""

    def test_single_value_for_two_scopes_is_rejected(self) -> None:
        existing = [_cookie("sid", "one", path="/"), _cookie("sid", "two", path="/deep")]

        with self.assertRaises(CookieHeaderError) as caught:
            request_cookie_updates("sid=two", existing, _URL)

        self.assertEqual(str(caught.exception), _MESSAGE)

    def test_partial_duplicate_match_is_rejected(self) -> None:
        existing = [_cookie("sid", "one", path="/"), _cookie("sid", "two", path="/deep")]

        with self.assertRaises(CookieHeaderError):
            request_cookie_updates("sid=one; sid=one", existing, _URL)

    def test_duplicate_unknown_name_is_rejected(self) -> None:
        with self.assertRaises(CookieHeaderError):
            request_cookie_updates("fresh=1; fresh=2", [], _URL)


class ScopeAndOmissionTests(TestCase):
    """The header patches only the names it carries and never rewrites the rest."""

    def test_omitted_names_are_left_alone_and_an_empty_header_is_a_noop(self) -> None:
        existing = [_cookie("keep", "value")]

        self.assertEqual(request_cookie_updates("", existing, _URL), [])
        self.assertEqual(request_cookie_updates("   ", existing, _URL), [])

        updates = request_cookie_updates("other=1", existing, _URL)

        self.assertEqual(updates, [{"name": "other", "value": "1", "url": _URL}])
        self.assertEqual([update.get("name") for update in updates], ["other"])


class MalformedHeaderTests(TestCase):
    """A header that cannot be parsed safely returns one generic, credential-free error."""

    def test_malformed_headers_raise_the_generic_error_without_echoing_input(self) -> None:
        cases = ["token", "bad name=SECRET", 'value="unterminated', "a=sp ace", ";"]

        for header in cases:
            with self.subTest(header=header), self.assertRaises(CookieHeaderError) as caught:
                request_cookie_updates(header, [_cookie("token", "SECRET")], _URL)

            self.assertEqual(str(caught.exception), _MESSAGE)
            self.assertNotIn("SECRET", str(caught.exception))
            self.assertNotIn(_URL, str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)


class PartitionedCookieTests(TestCase):
    """A partitioned native cookie is left untouched unchanged and fails closed on change."""

    def test_partitioned_cookie_is_untouched_when_unchanged_and_rejected_when_changed(self) -> None:
        # The installed Playwright Cookie type exposes partitionKey as Optional[str].
        partitioned = _cookie("sid", "one", partition_key="https://example.test")

        self.assertEqual(request_cookie_updates("sid=one", [partitioned], _URL), [])
        with self.assertRaises(CookieHeaderError):
            request_cookie_updates("sid=two", [partitioned], _URL)
