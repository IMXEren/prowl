"""The public ``Service.fetch`` seam: a typed ``FetchResult`` before JSON projection.

A forward proxy needs a fetch's bytes and headers before they are flattened into the
FlareSolverr JSON body, so ``Service.fetch`` runs the existing lease, concurrency bound, and
timeout path and hands back the native :class:`FetchResult`. ``_handle_fetch`` only projects
that result, and the JSON reply keeps its original shape. No browser, process, or network is
touched; every backend is a complete ``Mock(spec=Backend)``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import Backend, FetchRequest, FetchResult
from prowl.service.errors import CallerSafeError
from prowl.service.protocol import AUTO_MODE, BROWSER_MODE, FetchCommand, parse_request
from prowl.service.sessions import ISOLATED_MODE, _Entry

_EGRESS_URL = "socks5://127.0.0.1:10001"
_OTHER_URL = "socks5://127.0.0.1:10002"
_GET = {"cmd": "request.get", "url": "https://example.com/"}


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-fetch-result-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-fetch-result-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


def _result(**overrides: Any) -> FetchResult:
    values: dict[str, Any] = {
        "url": "https://example.com/",
        "status_code": 200,
        "headers": {"content-type": "text/html"},
        "response": "hello",
        "cookies": [],
        "user_agent": "Mozilla/5.0 (Test)",
    }
    values.update(overrides)
    return FetchResult(**values)


def _backend(result: FetchResult) -> Mock:
    """A complete protocol backend double whose ``fetch`` yields *result*."""
    backend = Mock(spec=Backend)
    backend.fetch = AsyncMock(return_value=result)
    backend.close_session = AsyncMock()
    backend.aclose = AsyncMock()
    return backend


def _command(**fields: Any) -> FetchCommand:
    command = parse_request({**_GET, **fields})
    assert isinstance(command, FetchCommand)
    return command


class TypedFetchResultTests(IsolatedAsyncioTestCase):
    """``Service.fetch`` returns the backend's own result, not a JSON projection."""

    async def test_the_identical_result_carries_bytes_and_repeated_headers(self) -> None:
        body = b"\xff\xfe\x00payload"
        items = (("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"))
        result = _result(body_bytes=body, header_items=items)
        backend = _backend(result)
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        returned = await service.fetch(_command())

        self.assertIs(returned, result)
        self.assertEqual(returned.body_bytes, body)
        self.assertEqual(returned.header_items, items)


class JsonProjectionTests(IsolatedAsyncioTestCase):
    """The legacy JSON reply keeps its shape and never exposes the byte fields."""

    async def test_handle_projects_the_existing_solution_without_byte_fields(self) -> None:
        result = _result(body_bytes=b"\x00\x01")
        backend = _backend(result)
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        status, body = await service.handle(dict(_GET))

        self.assertEqual(status, 200)
        solution = body["solution"]
        self.assertEqual(
            set(solution),
            {"url", "status", "headers", "response", "cookies", "userAgent"},
        )
        self.assertEqual(solution["response"], "hello")
        self.assertNotIn("body_bytes", solution)
        self.assertNotIn("header_items", solution)

    async def test_return_only_cookies_clears_the_json_response_not_the_typed_result(self) -> None:
        result = _result(response="body", body_bytes=b"body")
        backend = _backend(result)
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        _status, body = await service.handle({**_GET, "returnOnlyCookies": True})
        self.assertIn("response", body["solution"])
        self.assertEqual(body["solution"]["response"], "")

        returned = await service.fetch(_command())
        self.assertEqual(returned.response, "body")
        self.assertEqual(returned.body_bytes, b"body")


class LeaseBindingTests(IsolatedAsyncioTestCase):
    """The typed fetch reuses the lease's bound mode and egress, not a fresh selection."""

    def _service(self, backend: Mock) -> Service:
        service = Service(_config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}), backend)
        self.addAsyncCleanup(service.aclose)
        return service

    async def test_isolated_metadata_and_request_mode_reach_the_backend(self) -> None:
        backend = _backend(_result())
        service = self._service(backend)
        await service.sessions.create("s", mode=ISOLATED_MODE, egress="decodo")

        await service.fetch(_command(session="s", mode="auto"))
        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.session_mode, ISOLATED_MODE)
        self.assertEqual(request.egress, "decodo")
        self.assertEqual(request.mode, AUTO_MODE)

        backend.fetch.reset_mock()
        await service.fetch(_command(session="s"))
        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.session_mode, ISOLATED_MODE)
        self.assertEqual(request.mode, BROWSER_MODE)

    async def test_a_named_selector_is_honored_without_rotation(self) -> None:
        backend = _backend(_result())
        service = self._service(backend)

        await service.fetch(_command(proxy={"name": "warp"}))

        _session_id, request = backend.fetch.await_args.args
        self.assertEqual(request.egress, "warp")


class LeaseHeldThroughFetchTests(IsolatedAsyncioTestCase):
    """The session lease is held across the whole backend await, so destroy drains it."""

    async def test_destroy_waits_until_the_fetch_releases_the_lease(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        result = _result()
        backend = _backend(result)

        async def blocking_fetch(_session_id: str | None, _request: FetchRequest) -> FetchResult:
            entered.set()
            await release.wait()
            return result

        backend.fetch = AsyncMock(side_effect=blocking_fetch)
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)
        await service.sessions.create("s", mode=ISOLATED_MODE)

        fetch_task = asyncio.create_task(service.fetch(_command(session="s")))
        await asyncio.wait_for(entered.wait(), timeout=1)

        closing = asyncio.Event()
        original = service.sessions._start_close

        def start_close(
            session_id: str,
            entry: _Entry,
            *,
            by_destroy: bool,
            evicted: bool = False,
            shutdown: bool = False,
        ) -> None:
            original(session_id, entry, by_destroy=by_destroy, evicted=evicted, shutdown=shutdown)
            closing.set()

        with patch.object(service.sessions, "_start_close", side_effect=start_close):
            destroy_task = asyncio.create_task(service.sessions.destroy("s"))
            await asyncio.wait_for(closing.wait(), 1)
            self.assertFalse(destroy_task.done())
            backend.close_session.assert_not_awaited()
            release.set()
            self.assertIs(await asyncio.wait_for(fetch_task, timeout=1), result)
            await asyncio.wait_for(destroy_task, timeout=1)
            backend.close_session.assert_awaited_once_with("s")


class CancellationAndErrorPropagationTests(IsolatedAsyncioTestCase):
    """Cancellation frees the concurrency bound and errors propagate from the typed seam."""

    async def test_cancellation_releases_the_global_semaphore(self) -> None:
        entered = asyncio.Event()
        never = asyncio.Event()
        result = _result()
        backend = _backend(result)

        async def blocking_fetch(_session_id: str | None, _request: FetchRequest) -> FetchResult:
            entered.set()
            await never.wait()
            return result

        backend.fetch = AsyncMock(side_effect=blocking_fetch)
        service = Service(_config(max_concurrency=1), backend)
        self.addAsyncCleanup(service.aclose)

        task = asyncio.create_task(service.fetch(_command()))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        backend.fetch = AsyncMock(return_value=result)
        self.assertIs(await asyncio.wait_for(service.fetch(_command()), timeout=1), result)

    async def test_caller_safe_and_timeout_errors_propagate_while_handle_maps_them(self) -> None:
        backend = _backend(_result())
        service = Service(_config(), backend)
        self.addAsyncCleanup(service.aclose)

        backend.fetch = AsyncMock(side_effect=CallerSafeError("boom"))
        with self.assertRaises(CallerSafeError):
            await service.fetch(_command())
        _status, body = await service.handle(dict(_GET))
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "boom")

        backend.fetch = AsyncMock(side_effect=TimeoutError())
        with self.assertRaises(TimeoutError):
            await service.fetch(_command())
        _status, body = await service.handle(dict(_GET))
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["message"], "request timed out")
