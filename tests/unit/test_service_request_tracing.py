"""Request-correlation tracing for the service.

Every command carries one opaque id for the whole of its execution: the ``Service`` binds it for
the duration and logs a single finish line under it, and the HTTP handler returns it as
``X-Request-ID``. An inbound ``X-Request-ID`` is ignored outright, so a caller can never spoof or
steer a correlation id, and the bound id is reset even when the command is cancelled.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from loguru import logger

from prowl.service.app import Service, ServiceConfig, create_app
from prowl.service.protocol import ok_response

if TYPE_CHECKING:
    from prowl.service.backend import (
        CookieQuery,
        FetchRequest,
        FetchResult,
        InteractiveOpenResult,
        InteractiveRequest,
        InteractiveTab,
    )

_HEADER = "X-Request-ID"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")


class StubBackend:
    """A complete, inert ``Backend`` double: no browser, no network, no default facade."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        message = "fetch is not expected"
        raise AssertionError(message)

    async def open_interactive(self, request: InteractiveRequest) -> InteractiveOpenResult:
        message = "open_interactive is not expected"
        raise AssertionError(message)

    async def close_interactive(self, tab_id: str | None) -> list[str]:
        return []

    async def list_interactive(self, tab_id: str | None = None) -> list[InteractiveTab]:
        return []

    async def list_cookies(self, query: CookieQuery) -> list[dict[str, Any]]:
        return []

    async def close_session(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True


def _config() -> ServiceConfig:
    return ServiceConfig(max_concurrency=2)


def _capture_records(case: IsolatedAsyncioTestCase) -> list[tuple[str, dict[str, Any]]]:
    """Collect loguru records (formatted message plus ``extra``) for *case*, dropped on cleanup."""
    records: list[tuple[str, dict[str, Any]]] = []

    def sink(message: Any) -> None:
        records.append((message.record["message"], dict(message.record["extra"])))

    handler_id = logger.add(sink, level="DEBUG")
    case.addCleanup(logger.remove, handler_id)
    return records


class HttpRequestIdTests(IsolatedAsyncioTestCase):
    """Every HTTP command carries a fresh opaque id in its response header."""

    async def asyncSetUp(self) -> None:
        self.backend = StubBackend()
        self.client = TestClient(TestServer(create_app(_config(), self.backend)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_command_gets_a_lowercase_hex_id_and_an_unchanged_body(self) -> None:
        response = await self.client.post("/v1", json={"cmd": "sessions.list"})
        self.assertEqual(response.status, 200)
        self.assertRegex(response.headers[_HEADER], _HEX32)
        body = await response.json()
        self.assertEqual(
            set(body),
            {"status", "message", "startTimestamp", "endTimestamp", "version", "solution", "sessions"},
        )
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["message"], "ok")
        self.assertEqual(body["sessions"], [])
        self.assertEqual(body["solution"], {})

    async def test_each_request_mints_a_fresh_id(self) -> None:
        first = await self.client.post("/v1", json={"cmd": "sessions.list"})
        second = await self.client.post("/v1", json={"cmd": "sessions.list"})
        first_id = first.headers[_HEADER]
        second_id = second.headers[_HEADER]
        self.assertRegex(first_id, _HEX32)
        self.assertRegex(second_id, _HEX32)
        self.assertNotEqual(first_id, second_id)

    async def test_malformed_json_gets_a_400_with_an_id(self) -> None:
        response = await self.client.post("/v1", data="{not json")
        self.assertEqual(response.status, 400)
        self.assertRegex(response.headers[_HEADER], _HEX32)
        body = await response.json()
        self.assertEqual(body["status"], "error")

    async def test_inbound_request_id_is_ignored_and_never_logged(self) -> None:
        records = _capture_records(self)
        response = await self.client.post(
            "/v1",
            json={"cmd": "sessions.list"},
            headers={_HEADER: "caller-supplied-value"},
        )
        self.assertEqual(response.status, 200)
        returned = response.headers[_HEADER]
        self.assertRegex(returned, _HEX32)
        self.assertNotEqual(returned, "caller-supplied-value")
        self.assertNotIn("caller-supplied-value", repr(records))
        finished = [extra["request_id"] for msg, extra in records if "finished in" in msg]
        self.assertEqual(finished, [returned])


class ServiceCorrelationTests(IsolatedAsyncioTestCase):
    """The bound id correlates a command's own logs and survives cancellation cleanly."""

    async def test_concurrent_commands_keep_distinct_correlated_records(self) -> None:
        records = _capture_records(self)
        service = Service(_config(), StubBackend())
        self.addAsyncCleanup(service.aclose)
        both_in = asyncio.Event()
        arrived = 0

        async def dispatch(start: int, _command: Any) -> dict[str, Any]:
            nonlocal arrived
            logger.debug("dispatch enter")
            arrived += 1
            if arrived == 2:
                both_in.set()
            await both_in.wait()
            return ok_response(start)

        async def run(name: str) -> None:
            with logger.contextualize(handle=name):
                await service.handle({"cmd": "sessions.list"})

        with patch.object(service, "_dispatch", dispatch):
            await asyncio.wait_for(asyncio.gather(run("one"), run("two")), timeout=1)

        dispatched = {extra["handle"]: extra["request_id"] for msg, extra in records if msg == "dispatch enter"}
        finished = {extra["handle"]: extra["request_id"] for msg, extra in records if "finished in" in msg}
        self.assertEqual(set(dispatched), {"one", "two"})
        self.assertEqual(set(finished), {"one", "two"})
        for name in ("one", "two"):
            self.assertEqual(dispatched[name], finished[name])
            self.assertRegex(dispatched[name], _HEX32)
        self.assertNotEqual(dispatched["one"], dispatched["two"])

    async def test_cancellation_logs_a_correlated_finish_without_bleeding(self) -> None:
        records = _capture_records(self)
        service = Service(_config(), StubBackend())
        self.addAsyncCleanup(service.aclose)
        started = asyncio.Event()
        never = asyncio.Event()

        async def dispatch(start: int, _command: Any) -> dict[str, Any]:
            started.set()
            await never.wait()
            return ok_response(start)

        async def run_tagged() -> None:
            with logger.contextualize(phase="cancelled"):
                try:
                    await service.handle({"cmd": "sessions.list"})
                except asyncio.CancelledError:
                    logger.debug("caught cancellation")
                    raise

        with patch.object(service, "_dispatch", dispatch):
            task = asyncio.create_task(run_tagged())
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        finishes = [
            extra["request_id"] for msg, extra in records if "finished in" in msg and extra.get("phase") == "cancelled"
        ]
        self.assertEqual(len(finishes), 1)
        cancelled_id = finishes[0]
        self.assertRegex(cancelled_id, _HEX32)

        caught = [extra for msg, extra in records if msg == "caught cancellation"]
        self.assertEqual(caught, [{"phase": "cancelled"}])

        _, body = await service.handle({"cmd": "sessions.list"})
        self.assertEqual(body["status"], "ok")
        later = [extra for msg, extra in records if "finished in" in msg][-1]
        self.assertRegex(later["request_id"], _HEX32)
        self.assertNotEqual(later["request_id"], cancelled_id)
        self.assertNotIn("phase", later)
