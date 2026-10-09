"""Media suppression is a strict fetch-only flag with no extra response fields."""

from unittest import IsolatedAsyncioTestCase, TestCase

from prowl.service.app import Service
from prowl.service.protocol import FetchCommand, ProtocolError, parse_request
from test_service_snapshot_options import _backend, _config, _result

_GET = {"cmd": "request.get", "url": "https://example.com/"}


class MediaOptionParsingTests(TestCase):
    def test_get_and_post_accept_boolean_flag_with_default_false(self) -> None:
        for command_name in ("request.get", "request.post"):
            payload = {**_GET, "cmd": command_name}
            if command_name == "request.post":
                payload["postData"] = "a=1"
            command = parse_request(payload)
            self.assertIsInstance(command, FetchCommand)
            assert isinstance(command, FetchCommand)
            self.assertFalse(command.disable_media)
            enabled = parse_request({**payload, "disableMedia": True})
            assert isinstance(enabled, FetchCommand)
            self.assertTrue(enabled.disable_media)

    def test_invalid_values_and_non_fetch_commands_are_rejected(self) -> None:
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(ProtocolError) as caught:
                parse_request({**_GET, "disableMedia": value})
            self.assertEqual(caught.exception.message, "disableMedia must be a boolean")
        for payload in (
            {"cmd": "sessions.create", "disableMedia": True},
            {"cmd": "cookies.get", "disableMedia": True},
            {"cmd": "browser.open", "url": _GET["url"], "disableMedia": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                parse_request(payload)


class MediaOptionDispatchTests(IsolatedAsyncioTestCase):
    async def test_media_flag_is_forwarded_even_for_cookie_only_requests(self) -> None:
        backend = _backend(_result())
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)
        status, body = await service.handle(
            {**_GET, "disableMedia": True, "returnOnlyCookies": True, "waitInSeconds": 1}
        )
        self.assertEqual(status, 200)
        backend.fetch.assert_awaited_once()
        request = backend.fetch.await_args.args[1]
        self.assertTrue(request.disable_media)
        self.assertEqual(request.wait_in_seconds, 0)
        self.assertEqual(body["solution"]["response"], "")
        self.assertNotIn("disableMedia", body["solution"])

    async def test_defaults_and_invalid_flags_preserve_dispatch_contract(self) -> None:
        backend = _backend(_result())
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)
        await service.handle(_GET)
        self.assertFalse(backend.fetch.await_args.args[1].disable_media)
        backend.fetch.reset_mock()
        status, body = await service.handle({**_GET, "disableMedia": "secret"})
        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "error")
        self.assertNotIn("secret", body["message"])
        backend.fetch.assert_not_awaited()
