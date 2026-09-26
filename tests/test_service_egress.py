"""Tests for named egresses: configuration, selection, isolation, and idling.

The pool is exercised with a stub browser class so lazy creation, idle teardown,
and shutdown are asserted without launching a browser. Selection is exercised at
the service layer with the in-memory backend double, so no process is involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from prowl.browser.config import BrowserConfig
from prowl.browser.egress import (
    DEFAULT_EGRESS_NAME,
    EgressError,
    EgressPool,
    create_egress_browser,
    derive_egress_paths,
    parse_egress_spec,
)
from prowl.service.app import Service, ServiceConfig
from prowl.service.backend import FetchRequest, FetchResult

_EGRESS_URL = "socks5://127.0.0.1:10001"
_OTHER_URL = "socks5://127.0.0.1:10002"


class FakeBackend:
    """In-memory backend double recording every request it receives."""

    def __init__(self, *, delay: float = 0.0) -> None:
        self.requests: list[tuple[str | None, FetchRequest]] = []
        self.active = 0
        self.max_active = 0
        self._delay = delay

    def result(self) -> FetchResult:
        return FetchResult(
            url="https://example.com/",
            status_code=200,
            headers={"content-type": "text/html"},
            response="<html><body>ok</body></html>",
            cookies=[],
            user_agent="Mozilla/5.0 (Test)",
        )

    async def start(self) -> None:
        """No-op for the double."""

    async def close_session(self, session_id: str) -> None:
        """No-op for the double."""

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        self.requests.append((session_id, request))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            return self.result()
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        """No-op for the double."""


def _config(**overrides: Any) -> ServiceConfig:
    values: dict[str, Any] = {
        "proxy_url": None,
        "egresses": {},
        "profile_dir": "/tmp/prowl-test-profile",  # noqa: S108
        "profile_archive": "/tmp/prowl-test-profile.zip",  # noqa: S108
    }
    values.update(overrides)
    return ServiceConfig(**values)


async def _fetch(service: Service, proxy: dict[str, str], session: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"cmd": "request.get", "url": "https://example.com/", "proxy": proxy}
    if session is not None:
        payload["session"] = session
    _status, body = await service.handle(payload)
    return body


class EgressSpecTests(TestCase):
    """The ``name=url`` list is parsed strictly and fails loudly."""

    def test_pairs_are_parsed(self) -> None:
        parsed = parse_egress_spec(f"decodo={_EGRESS_URL},warp={_OTHER_URL}")
        self.assertEqual(parsed, {"decodo": _EGRESS_URL, "warp": _OTHER_URL})

    def test_blank_entries_are_skipped(self) -> None:
        self.assertEqual(parse_egress_spec(" , decodo=x:1 , "), {"decodo": "x:1"})

    def test_empty_spec_is_empty(self) -> None:
        self.assertEqual(parse_egress_spec(""), {})

    def test_entry_without_a_separator_is_rejected(self) -> None:
        with self.assertRaises(EgressError) as ctx:
            parse_egress_spec("decodo")
        self.assertIn("name=url", str(ctx.exception))

    def test_entry_with_an_empty_name_or_url_is_rejected(self) -> None:
        for spec in ("=socks5://127.0.0.1:1", "decodo="):
            with self.assertRaises(EgressError, msg=spec):
                parse_egress_spec(spec)

    def test_duplicate_names_are_rejected(self) -> None:
        with self.assertRaises(EgressError) as ctx:
            parse_egress_spec(f"decodo={_EGRESS_URL},decodo={_OTHER_URL}")
        self.assertIn("duplicate", str(ctx.exception))

    def test_reserved_default_name_is_rejected(self) -> None:
        with self.assertRaises(EgressError) as ctx:
            parse_egress_spec(f"{DEFAULT_EGRESS_NAME}={_EGRESS_URL}")
        self.assertIn("reserved", str(ctx.exception))

    def test_path_unsafe_names_are_rejected(self) -> None:
        for name in ("../evil", "with space", "slash/name", ".hidden"):
            with self.assertRaises(EgressError, msg=name):
                parse_egress_spec(f"{name}={_EGRESS_URL}")


class EgressPathTests(TestCase):
    """Each egress gets its own profile directory and archive."""

    def test_default_keeps_configured_paths(self) -> None:
        paths = derive_egress_paths("/state/profile", "/state/browser-profile.zip", DEFAULT_EGRESS_NAME)
        self.assertEqual(paths, ("/state/profile", "/state/browser-profile.zip"))

    def test_named_egress_derives_both_paths(self) -> None:
        directory, archive = derive_egress_paths("/state/profile", "/state/browser-profile.zip", "decodo")
        self.assertEqual(Path(directory), Path("/state/profile/decodo"))
        self.assertEqual(Path(archive), Path("/state/browser-profile-decodo.zip"))

    def test_named_egress_paths_are_distinct_per_name(self) -> None:
        first = derive_egress_paths("/state/profile", "/state/p.zip", "a")
        second = derive_egress_paths("/state/profile", "/state/p.zip", "b")
        self.assertNotEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])


class EgressBrowserIsolationTests(TestCase):
    """An egress browser owns its own runtime, lifecycle, profile, and proxy."""

    def test_egress_browsers_do_not_share_state(self) -> None:
        first = create_egress_browser(
            name="a",
            proxy_url=_EGRESS_URL,
            profile_dir="/state/profile/a",
            profile_archive="/state/browser-profile-a.zip",
            preferred_cdp_port=9300,
        )
        second = create_egress_browser(
            name="b",
            proxy_url=_OTHER_URL,
            profile_dir="/state/profile/b",
            profile_archive="/state/browser-profile-b.zip",
            preferred_cdp_port=9410,
        )
        self.assertIsNot(first._runtime, second._runtime)
        self.assertIsNot(first._lifecycle, second._lifecycle)
        self.assertEqual(first._lifecycle.proxy_url, _EGRESS_URL)
        self.assertEqual(second._lifecycle.proxy_url, _OTHER_URL)
        self.assertEqual(first._lifecycle.profile_dir, "/state/profile/a")
        self.assertEqual(second._lifecycle.profile_dir, "/state/profile/b")

    def test_egress_browser_does_not_touch_the_default_singleton(self) -> None:
        from prowl.browser.browser import Browser  # noqa: PLC0415

        default_runtime = Browser._runtime
        egress = create_egress_browser(
            name="a",
            proxy_url=_EGRESS_URL,
            profile_dir="/state/profile/a",
            profile_archive="/state/browser-profile-a.zip",
            preferred_cdp_port=9300,
        )
        self.assertIsNot(egress._runtime, default_runtime)
        self.assertIs(Browser._runtime, default_runtime)
        self.assertFalse(egress.is_running())


class StubEgressBrowser:
    """Stub browser class standing in for one egress browser."""

    def __init__(self, *, shutdown_gate: asyncio.Event | None = None) -> None:
        self.shutdown_calls = 0
        self.shutdown_started = asyncio.Event()
        self._shutdown_gate = shutdown_gate

    async def shutdown(self) -> None:
        """Record the call, staying open until the gate is set when one is given."""
        self.shutdown_started.set()
        if self._shutdown_gate is not None:
            await self._shutdown_gate.wait()
        self.shutdown_calls += 1


async def _settle(*, ticks: int = 10) -> None:
    """Yield to the event loop enough times for pending tasks to make progress."""
    for _ in range(ticks):
        await asyncio.sleep(0.005)


async def _wait_for(event: asyncio.Event, *, attempts: int = 400) -> None:
    """Wait until *event* is set, failing rather than hanging when it never is."""
    for _ in range(attempts):
        if event.is_set():
            return
        await asyncio.sleep(0.005)
    msg = "the expected state was not reached in time"
    raise AssertionError(msg)


async def _wait_for_shutdown(stub: StubEgressBrowser, *, attempts: int = 400) -> None:
    """Wait until *stub* has recorded one completed shutdown."""
    for _ in range(attempts):
        if stub.shutdown_calls >= 1:
            return
        await asyncio.sleep(0.005)
    msg = "the stub browser was not shut down in time"
    raise AssertionError(msg)


class EgressPoolTests(IsolatedAsyncioTestCase):
    """Egress browsers are created lazily and shut down once idle."""

    def _pool(
        self,
        *,
        idle_seconds: float = 0.01,
        shutdown_gate: asyncio.Event | None = None,
        created: list[str] | None = None,
    ) -> tuple[EgressPool, dict[str, StubEgressBrowser]]:
        stubs: dict[str, StubEgressBrowser] = {}

        def _create(**kwargs: Any) -> type[StubEgressBrowser]:
            name = str(kwargs["name"])
            stub = StubEgressBrowser(shutdown_gate=shutdown_gate)
            stubs[name] = stub
            if created is not None:
                created.append(name)
            return type(f"Stub_{name}", (StubEgressBrowser,), {"shutdown": stub.shutdown})

        pool = EgressPool(
            BrowserConfig(profile_dir="/tmp/p", profile_archive="/tmp/p.zip"),  # noqa: S108
            {"a": _EGRESS_URL, "b": _OTHER_URL},
            idle_seconds=idle_seconds,
        )
        patcher = patch("prowl.browser.egress.create_egress_browser", side_effect=_create)
        patcher.start()
        self.addCleanup(patcher.stop)
        return pool, stubs

    async def test_no_browser_is_created_before_first_use(self) -> None:
        pool, stubs = self._pool()
        self.assertEqual(pool.live_names(), ())
        self.assertEqual(stubs, {})

    async def test_acquire_creates_the_browser_on_first_use(self) -> None:
        pool, stubs = self._pool()
        await pool.acquire("a")
        self.assertEqual(pool.live_names(), ("a",))
        self.assertEqual(sorted(stubs), ["a"])
        await pool.release("a")

    async def test_repeated_use_reuses_one_browser(self) -> None:
        pool, stubs = self._pool()
        first = await pool.acquire("a")
        await pool.release("a")
        second = await pool.acquire("a")
        await pool.release("a")
        self.assertIs(first, second)
        self.assertEqual(len(stubs), 1)

    async def test_unknown_egress_is_rejected(self) -> None:
        pool, _stubs = self._pool()
        with self.assertRaises(EgressError):
            await pool.acquire("missing")

    async def test_idle_egress_is_shut_down(self) -> None:
        pool, stubs = self._pool()
        await pool.acquire("a")
        await pool.release("a")
        await asyncio.sleep(0.1)
        self.assertEqual(pool.live_names(), ())
        self.assertEqual(stubs["a"].shutdown_calls, 1)

    async def test_new_use_cancels_the_pending_teardown(self) -> None:
        pool, stubs = self._pool(idle_seconds=0.05)
        await pool.acquire("a")
        await pool.release("a")
        await pool.acquire("a")
        await asyncio.sleep(0.15)
        self.assertEqual(stubs["a"].shutdown_calls, 0)
        self.assertEqual(pool.live_names(), ("a",))
        await pool.release("a")

    async def test_busy_egress_is_not_shut_down(self) -> None:
        pool, stubs = self._pool(idle_seconds=0.02)
        await pool.acquire("a")
        await pool.acquire("a")
        await pool.release("a")
        await asyncio.sleep(0.1)
        self.assertEqual(stubs["a"].shutdown_calls, 0)
        await pool.release("a")

    async def test_aclose_shuts_down_every_live_egress(self) -> None:
        pool, stubs = self._pool()
        await pool.acquire("a")
        await pool.acquire("b")
        await pool.aclose()
        self.assertEqual(stubs["a"].shutdown_calls, 1)
        self.assertEqual(stubs["b"].shutdown_calls, 1)
        self.assertEqual(pool.live_names(), ())

    async def test_acquire_waits_for_an_in_flight_shutdown(self) -> None:
        """A replacement browser starts only after the previous one has closed."""
        gate = asyncio.Event()
        created: list[str] = []
        pool, stubs = self._pool(idle_seconds=0.01, shutdown_gate=gate, created=created)
        first = await pool.acquire("a")
        await pool.release("a")
        await _wait_for(stubs["a"].shutdown_started)
        self.assertEqual(created, ["a"])

        pending = asyncio.create_task(pool.acquire("a"))
        await _settle()
        # The closing browser still owns the profile directory and the profile archive.
        self.assertFalse(pending.done())
        self.assertEqual(created, ["a"])

        gate.set()
        replacement = await pending
        self.assertIsNot(replacement, first)
        self.assertEqual(created, ["a", "a"])
        await pool.release("a")
        await pool.aclose()

    async def test_two_acquires_during_a_shutdown_share_one_browser(self) -> None:
        """Both callers waiting on one shutdown get the same replacement browser."""
        gate = asyncio.Event()
        created: list[str] = []
        pool, stubs = self._pool(idle_seconds=0.01, shutdown_gate=gate, created=created)
        await pool.acquire("a")
        await pool.release("a")
        await _wait_for(stubs["a"].shutdown_started)

        pending = [asyncio.create_task(pool.acquire("a")) for _ in range(2)]
        await _settle()
        self.assertEqual(created, ["a"])

        gate.set()
        browsers = await asyncio.gather(*pending)
        self.assertIs(browsers[0], browsers[1])
        self.assertEqual(created, ["a", "a"])
        await pool.release("a")
        await pool.release("a")
        await pool.aclose()

    async def test_cancelling_an_acquire_that_waits_for_a_shutdown_keeps_the_pool_usable(self) -> None:
        """A cancelled waiter leaves nothing claimed and the next acquire works."""
        gate = asyncio.Event()
        pool, stubs = self._pool(idle_seconds=0.01, shutdown_gate=gate)
        first = await pool.acquire("a")
        await pool.release("a")
        await _wait_for(stubs["a"].shutdown_started)

        pending = asyncio.create_task(pool.acquire("a"))
        await _settle()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        gate.set()
        replacement = await pool.acquire("a")
        self.assertIsNot(replacement, first)
        self.assertEqual(pool.live_names(), ("a",))
        await pool.release("a")
        await pool.aclose()

    async def test_aclose_completes_while_a_shutdown_is_in_flight(self) -> None:
        """Closing the pool does not wait for a shutdown that is already running."""
        gate = asyncio.Event()
        pool, stubs = self._pool(idle_seconds=0.01, shutdown_gate=gate)
        await pool.acquire("a")
        await pool.release("a")
        await _wait_for(stubs["a"].shutdown_started)
        # The egress is still tracked while its browser closes, so nothing can take the
        # profile it is still using.
        self.assertEqual(pool.live_names(), ("a",))

        await asyncio.wait_for(pool.aclose(), timeout=2.0)
        self.assertEqual(pool.live_names(), ())

        gate.set()
        await _wait_for_shutdown(stubs["a"])
        self.assertEqual(stubs["a"].shutdown_calls, 1)


class EgressConfigTests(IsolatedAsyncioTestCase):
    """Egress configuration is read and validated at startup."""

    async def test_egresses_are_read_from_the_environment(self) -> None:
        env = {"PROWL_EGRESSES": f"decodo={_EGRESS_URL},warp={_OTHER_URL}"}
        with patch.dict("os.environ", env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.egresses, {"decodo": _EGRESS_URL, "warp": _OTHER_URL})

    async def test_egress_idle_seconds_is_read(self) -> None:
        env = {"PROWL_EGRESS_IDLE_SECONDS": "12.5"}
        with patch.dict("os.environ", env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.egress_idle_seconds, 12.5)

    async def test_a_non_numeric_egress_idle_delay_is_rejected(self) -> None:
        env = {"PROWL_EGRESS_IDLE_SECONDS": "soon"}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(EgressError) as ctx:
            ServiceConfig.from_env()
        self.assertIn("PROWL_EGRESS_IDLE_SECONDS", str(ctx.exception))

    async def test_a_negative_egress_idle_delay_is_clamped(self) -> None:
        env = {"PROWL_EGRESS_IDLE_SECONDS": "-5"}
        with patch.dict("os.environ", env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.egress_idle_seconds, 0.0)

    async def test_absent_egress_list_leaves_only_the_default(self) -> None:
        env = {"PROWL_PROXY_URL": _EGRESS_URL}
        with patch.dict("os.environ", env, clear=True):
            config = ServiceConfig.from_env()
        self.assertEqual(config.egresses, {})
        self.assertEqual(config.proxy_url, _EGRESS_URL)

    async def test_egress_with_embedded_credentials_is_rejected(self) -> None:
        env = {"PROWL_EGRESSES": "decodo=socks5://user:secret@127.0.0.1:1080"}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(EgressError) as ctx:
            ServiceConfig.from_env()
        self.assertIn("decodo", str(ctx.exception))
        self.assertNotIn("secret", str(ctx.exception))

    async def test_egress_with_unsupported_scheme_is_rejected(self) -> None:
        env = {"PROWL_EGRESSES": "decodo=ftp://127.0.0.1:1080"}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(EgressError) as ctx:
            ServiceConfig.from_env()
        self.assertIn("scheme", str(ctx.exception))

    async def test_malformed_egress_list_is_rejected(self) -> None:
        env = {"PROWL_EGRESSES": "decodo"}
        with patch.dict("os.environ", env, clear=True), self.assertRaises(EgressError):
            ServiceConfig.from_env()

    async def test_service_rejects_a_reserved_name(self) -> None:
        with self.assertRaises(ValueError):
            Service(_config(egresses={DEFAULT_EGRESS_NAME: _EGRESS_URL}), FakeBackend())

    async def test_service_rejects_an_unusable_egress_url(self) -> None:
        with self.assertRaises(ValueError):
            Service(_config(egresses={"decodo": "socks5://user:secret@127.0.0.1:1080"}), FakeBackend())


class EgressSelectionTests(IsolatedAsyncioTestCase):
    """A request selects a configured egress by name or by url."""

    def _service(self, **overrides: Any) -> tuple[Service, FakeBackend]:
        backend = FakeBackend()
        config = _config(
            proxy_url=overrides.pop("proxy_url", None),
            egresses=overrides.pop("egresses", {"decodo": _EGRESS_URL}),
            **overrides,
        )
        return Service(config, backend), backend

    async def test_no_proxy_selects_the_default_egress(self) -> None:
        service, backend = self._service()
        await service.handle({"cmd": "request.get", "url": "https://example.com/"})
        self.assertEqual(backend.requests[0][1].egress, DEFAULT_EGRESS_NAME)

    async def test_name_selects_the_named_egress(self) -> None:
        service, backend = self._service()
        body = await _fetch(service, {"name": "decodo"})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.requests[0][1].egress, "decodo")

    async def test_url_maps_to_the_matching_named_egress(self) -> None:
        service, backend = self._service()
        body = await _fetch(service, {"url": _EGRESS_URL})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.requests[0][1].egress, "decodo")

    async def test_url_maps_to_the_default_egress_when_it_matches(self) -> None:
        service, backend = self._service(proxy_url=_OTHER_URL)
        body = await _fetch(service, {"url": _OTHER_URL})
        self.assertEqual(body["status"], "ok")
        self.assertEqual(backend.requests[0][1].egress, DEFAULT_EGRESS_NAME)

    async def test_unknown_name_is_rejected_without_echoing_a_url(self) -> None:
        service, backend = self._service()
        body = await _fetch(service, {"name": "nope"})
        self.assertEqual(body["status"], "error")
        self.assertIn("egress", body["message"])
        self.assertNotIn(_EGRESS_URL, body["message"])
        self.assertEqual(backend.requests, [])

    async def test_unlisted_url_is_rejected(self) -> None:
        service, _backend = self._service()
        body = await _fetch(service, {"url": "socks5://10.0.0.1:9999"})
        self.assertEqual(body["status"], "error")
        self.assertIn("not permitted", body["message"])
        self.assertNotIn("10.0.0.1", body["message"])

    async def test_naming_both_a_url_and_a_name_is_rejected(self) -> None:
        service, _backend = self._service()
        status, _body = await service.handle(
            {
                "cmd": "request.get",
                "url": "https://example.com/",
                "proxy": {"url": _EGRESS_URL, "name": "decodo"},
            },
        )
        self.assertEqual(status, 400)

    async def test_unknown_proxy_field_is_rejected(self) -> None:
        service, _backend = self._service()
        status, _body = await service.handle(
            {"cmd": "request.get", "url": "https://example.com/", "proxy": {"egress": "decodo"}},
        )
        self.assertEqual(status, 400)

    async def test_empty_proxy_name_is_rejected(self) -> None:
        service, _backend = self._service()
        status, _body = await service.handle(
            {"cmd": "request.get", "url": "https://example.com/", "proxy": {"name": ""}},
        )
        self.assertEqual(status, 400)


class EgressSerializationTests(IsolatedAsyncioTestCase):
    """Anonymous requests serialize per egress, not process-wide."""

    async def test_two_anonymous_requests_on_one_egress_serialize(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(egresses={"decodo": _EGRESS_URL}, max_concurrency=2), backend)
        await asyncio.gather(
            _fetch(service, {"name": "decodo"}),
            _fetch(service, {"name": "decodo"}),
        )
        self.assertEqual(backend.max_active, 1)

    async def test_anonymous_requests_on_different_egresses_overlap(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(
            _config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}, max_concurrency=2),
            backend,
        )
        await asyncio.gather(
            _fetch(service, {"name": "decodo"}),
            _fetch(service, {"name": "warp"}),
        )
        self.assertEqual(backend.max_active, 2)

    async def test_the_default_egress_keeps_its_own_lock(self) -> None:
        backend = FakeBackend(delay=0.05)
        service = Service(_config(egresses={"decodo": _EGRESS_URL}, max_concurrency=2), backend)
        await asyncio.gather(
            service.handle({"cmd": "request.get", "url": "https://example.com/"}),
            _fetch(service, {"name": "decodo"}),
        )
        self.assertEqual(backend.max_active, 2)


class EgressSessionBindingTests(IsolatedAsyncioTestCase):
    """A session is bound to the egress of its first use."""

    def _service(self) -> tuple[Service, FakeBackend]:
        backend = FakeBackend()
        service = Service(_config(egresses={"decodo": _EGRESS_URL, "warp": _OTHER_URL}), backend)
        return service, backend

    async def test_repeated_use_of_one_egress_is_fine(self) -> None:
        service, _backend = self._service()
        first = await _fetch(service, {"name": "decodo"}, session="s")
        second = await _fetch(service, {"name": "decodo"}, session="s")
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")

    async def test_a_disagreeing_egress_is_rejected(self) -> None:
        service, _backend = self._service()
        await _fetch(service, {"name": "decodo"}, session="s")
        body = await _fetch(service, {"name": "warp"}, session="s")
        self.assertEqual(body["status"], "error")
        self.assertIn("egress", body["message"])

    async def test_the_default_egress_binding_is_enforced_too(self) -> None:
        service, _backend = self._service()
        await service.handle({"cmd": "request.get", "url": "https://example.com/", "session": "s"})
        body = await _fetch(service, {"name": "decodo"}, session="s")
        self.assertEqual(body["status"], "error")

    async def test_a_destroyed_session_can_be_rebound(self) -> None:
        service, _backend = self._service()
        await _fetch(service, {"name": "decodo"}, session="s")
        await service.handle({"cmd": "sessions.destroy", "session": "s"})
        body = await _fetch(service, {"name": "warp"}, session="s")
        self.assertEqual(body["status"], "ok")


class EgressBackendRoutingTests(IsolatedAsyncioTestCase):
    """The backend routes each fetch to its egress browser."""

    async def test_default_egress_uses_the_process_browser(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        calls: list[str] = []

        class FakeBrowser:
            @classmethod
            async def start(cls) -> None:
                calls.append("start")

            @classmethod
            async def create(cls) -> Any:
                calls.append("create")
                return _fake_group()

            @classmethod
            async def shutdown(cls) -> None:
                calls.append("shutdown")

        backend = backend_module.BrowserBackend(egresses={"decodo": _EGRESS_URL})
        with (
            patch.object(backend_module, "Browser", FakeBrowser),
            patch.object(backend_module, "resolve_site", lambda _group, _url: _FakeSite()),
        ):
            await backend.fetch(None, FetchRequest(url="https://example.com/"))
        self.assertEqual(calls, ["start", "create"])

    async def test_named_egress_uses_its_own_browser(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        calls: list[str] = []

        def _create(**kwargs: Any) -> type[Any]:
            name = str(kwargs["name"])

            class FakeEgressBrowser:
                @classmethod
                async def start(cls) -> None:
                    calls.append(f"start:{name}")

                @classmethod
                async def create(cls) -> Any:
                    calls.append(f"create:{name}")
                    return _fake_group()

                @classmethod
                async def shutdown(cls) -> None:
                    calls.append(f"shutdown:{name}")

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(egresses={"decodo": _EGRESS_URL}, egress_idle_seconds=60.0)
        with (
            patch("prowl.browser.egress.create_egress_browser", side_effect=_create),
            patch.object(backend_module, "resolve_site", lambda _group, _url: _FakeSite()),
        ):
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))
            await backend.aclose()
        self.assertEqual(calls, ["start:decodo", "create:decodo", "shutdown:decodo"])

    async def test_an_egress_the_backend_does_not_know_is_rejected(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        backend = backend_module.BrowserBackend()
        with self.assertRaises(EgressError):
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))

    async def test_a_named_egress_start_failure_releases_the_claim(self) -> None:
        """A start that fails before any group exists still owes the egress claim back."""
        from prowl.service import backend as backend_module  # noqa: PLC0415

        shutdown_log: list[str] = []

        def _create(**kwargs: Any) -> type[Any]:
            name = str(kwargs["name"])

            class FakeEgressBrowser:
                @classmethod
                async def start(cls) -> None:
                    msg = "start failed"
                    raise RuntimeError(msg)

                @classmethod
                async def create(cls) -> Any:
                    msg = "create must not run after a failed start"
                    raise AssertionError(msg)

                @classmethod
                async def shutdown(cls) -> None:
                    shutdown_log.append(name)

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(egresses={"decodo": _EGRESS_URL}, egress_idle_seconds=0.01)
        with (
            patch("prowl.browser.egress.create_egress_browser", side_effect=_create),
            self.assertRaises(RuntimeError),
        ):
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))
        # Releasing the claim lets the pool idle the egress browser out again.
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_a_named_egress_create_failure_releases_the_claim(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        shutdown_log: list[str] = []

        def _create(**kwargs: Any) -> type[Any]:
            name = str(kwargs["name"])

            class FakeEgressBrowser:
                @classmethod
                async def start(cls) -> None:
                    """No-op."""

                @classmethod
                async def create(cls) -> Any:
                    msg = "create failed"
                    raise RuntimeError(msg)

                @classmethod
                async def shutdown(cls) -> None:
                    shutdown_log.append(name)

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(egresses={"decodo": _EGRESS_URL}, egress_idle_seconds=0.01)
        with (
            patch("prowl.browser.egress.create_egress_browser", side_effect=_create),
            self.assertRaises(RuntimeError),
        ):
            await backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo"))
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()

    async def test_a_named_egress_create_cancellation_releases_the_claim(self) -> None:
        from prowl.service import backend as backend_module  # noqa: PLC0415

        shutdown_log: list[str] = []
        entered = asyncio.Event()
        gate = asyncio.Event()

        def _create(**kwargs: Any) -> type[Any]:
            name = str(kwargs["name"])

            class FakeEgressBrowser:
                @classmethod
                async def start(cls) -> None:
                    """No-op."""

                @classmethod
                async def create(cls) -> Any:
                    entered.set()
                    await gate.wait()
                    return _fake_group()

                @classmethod
                async def shutdown(cls) -> None:
                    shutdown_log.append(name)

            return FakeEgressBrowser

        backend = backend_module.BrowserBackend(egresses={"decodo": _EGRESS_URL}, egress_idle_seconds=0.01)
        with patch("prowl.browser.egress.create_egress_browser", side_effect=_create):
            task = asyncio.create_task(
                backend.fetch(None, FetchRequest(url="https://example.com/", egress="decodo")),
            )
            await asyncio.wait_for(entered.wait(), timeout=2.0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await asyncio.sleep(0.1)
        self.assertEqual(shutdown_log, ["decodo"])
        await backend.aclose()


async def _noop_async(*_args: Any, **_kwargs: Any) -> None:
    """Awaitable no-op used to stand in for a tab group quit."""


def _fake_group() -> Any:
    """Return a minimal tab group double with an awaitable parent tab."""

    class FakeTab:
        async def set_cookies(self, cookies: Any) -> None:
            """Accept cookies, as the real parent tab does."""

    class FakePd:
        async def get_cookies(self) -> list[Any]:
            """Report no cookies."""
            return []

    tab = FakeTab()
    pd = FakePd()

    class FakeGroup:
        @property
        async def ptab(self) -> FakeTab:
            """Return the parent tab, matching the real async property."""
            return tab

        def pd(self) -> FakePd:
            """Return the cookie reader."""
            return pd

        async def quit(self) -> None:
            """Close the group."""

    return FakeGroup()


class _FakeSite:
    """Minimal site double returning a fixed source."""

    async def get(self, url: str, timeout: int, **_kwargs: Any) -> Any:
        from prowl.browser.site import Source  # noqa: PLC0415

        return Source(
            source="<html></html>",
            status_code=200,
            headers={},
            user_agent="Mozilla/5.0 (Test)",
            url=url,
        )
