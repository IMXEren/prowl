"""Tests for the PageHandler challenge-solver seam.

The native tab is replaced by a spec-constrained double so the exact native
calls, ownership capture, and stop/retry behavior are verified without a live
browser. Solvers are doubles implementing the ChallengeSolver protocol.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Frame, Page
from pydoll.browser.tab import Tab
from pydoll.protocol.dom.types import Node as PDNode
from pydoll.protocol.dom.types import ShadowRootType

from prowl.browser.exceptions import PageLoadError
from prowl.browser.page_handler import PageHandler, resolve_page_handler
from prowl.browser.solvers import ChallengeSolver, CloudflareSolver, click_embedded_turnstile
from prowl.browser.solvers.cloudflare import _closed_body_shadows

if TYPE_CHECKING:
    from prowl.browser.browser import TabGroup

_CHALLENGE_BODY: dict[str, str | bool] = {
    "body": "<html><head><title>Just a moment...</title></head></html>",
    "base64Encoded": False,
}
_PLAIN_BODY: dict[str, str | bool] = {
    "body": "<html><head><title>Welcome</title></head></html>",
    "base64Encoded": False,
}
_TARGET = "https://protected.example/login"


def _tab_group(tab: Mock) -> Mock:
    group = Mock()
    group.ptab = tab
    return group


def _site(solver: ChallengeSolver | None = None) -> tuple[PageHandler, Mock]:
    tab = Mock(spec=Tab)
    site = PageHandler(_tab_group(tab))
    if solver is not None:
        site.challenge_solver = solver
    site.tab = tab
    site.start = time.perf_counter()
    site.timeout = 30
    return site, tab


class CloudflareSolverTests(IsolatedAsyncioTestCase):
    """The built-in solver issues exactly the current native auto-solve calls."""

    async def test_builtin_solver_uses_exact_native_calls(self) -> None:
        tab = Mock(spec=Tab)
        solver = CloudflareSolver()

        await solver.start(tab)
        tab.enable_auto_solve_cloudflare_captcha.assert_awaited_once_with(time_before_click=1, time_to_wait_captcha=30)

        await solver.stop(tab)
        tab.disable_auto_solve_cloudflare_captcha.assert_awaited_once_with()


class PageHandlerChallengeSolverTests(IsolatedAsyncioTestCase):
    """PageHandler activates and retires exactly one owned solver/tab pair."""

    async def test_registered_site_override_runs_on_challenge_title(self) -> None:
        tab = Mock(spec=Tab)

        override = Mock(spec=ChallengeSolver)

        class ProtectedSite(PageHandler):
            def __init__(self, tg: TabGroup) -> None:
                super().__init__(tg)
                self.challenge_solver = override

        with patch("prowl.browser.page_handler._page_handlers", [("protected.example", ProtectedSite)]):
            site = resolve_page_handler(_tab_group(tab), _TARGET)
        self.assertIsInstance(site, ProtectedSite)
        site.tab = tab
        site.start = time.perf_counter()
        site.timeout = 30

        with patch.object(CloudflareSolver, "start", AsyncMock()) as default_start:
            await site._check_cf_encounter(_TARGET, _CHALLENGE_BODY)

        override.start.assert_awaited_once_with(tab)
        default_start.assert_not_awaited()
        self.assertTrue(site.cf_auto_solve_enabled)
        self.assertTrue(site.cf_encountered)

    async def test_plain_title_does_not_start_solver(self) -> None:
        solver = Mock(spec=ChallengeSolver)
        site, _tab = _site(solver)

        await site._check_cf_encounter(_TARGET, _PLAIN_BODY)

        solver.start.assert_not_awaited()
        self.assertFalse(site.cf_encountered)
        self.assertFalse(site.cf_auto_solve_enabled)

    async def test_duplicate_activation_starts_once(self) -> None:
        solver = Mock(spec=ChallengeSolver)
        site, tab = _site(solver)

        await site.start_challenge_solver()
        await site.start_challenge_solver()

        solver.start.assert_awaited_once_with(tab)
        self.assertTrue(site.cf_auto_solve_enabled)

    async def test_cancelled_start_still_stops_captured_tab(self) -> None:
        solver = Mock(spec=ChallengeSolver)
        solver.start = AsyncMock(side_effect=asyncio.CancelledError())
        site, tab = _site(solver)

        with self.assertRaises(asyncio.CancelledError):
            await site.start_challenge_solver()
        self.assertEqual(site._active_solver, (solver, tab))
        self.assertFalse(site.cf_auto_solve_enabled)

        await site._cleanup()

        solver.stop.assert_awaited_once_with(tab)
        self.assertIsNone(site._active_solver)
        self.assertTrue(site.cleanup_done)

    async def test_failed_stop_retains_ownership_and_blocks_reset(self) -> None:
        solver = Mock(spec=ChallengeSolver)
        solver.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
        site, tab = _site(solver)

        await site.start_challenge_solver()

        with self.assertRaises(RuntimeError):
            await site._cleanup()
        self.assertFalse(site.cleanup_done)
        self.assertEqual(site._active_solver, (solver, tab))
        self.assertTrue(site.cf_auto_solve_enabled)
        tab.disable_page_events.assert_awaited_once()

        with self.assertRaises(PageLoadError):
            site._reset()

        # A changed class selection still retries the captured solver/tab.
        replacement = Mock(spec=ChallengeSolver)
        site.challenge_solver = replacement
        solver.stop = AsyncMock()
        await site._cleanup()

        solver.stop.assert_awaited_once_with(tab)
        replacement.stop.assert_not_awaited()
        self.assertIsNone(site._active_solver)
        self.assertFalse(site.cf_auto_solve_enabled)
        self.assertTrue(site.cleanup_done)

    async def test_cleanup_without_challenge_keeps_default_behavior(self) -> None:
        solver = Mock(spec=ChallengeSolver)
        site, tab = _site(solver)

        await site._cleanup()

        solver.stop.assert_not_awaited()
        tab.disable_page_events.assert_awaited_once()
        tab.disable_fetch_events.assert_awaited_once()
        self.assertTrue(site.cleanup_done)
        self.assertFalse(site.cf_auto_solve_enabled)


def _iframe_tab(node_ids: list[int]) -> Mock:
    tab = Mock(spec=Tab)
    tab._connection_handler = Mock()
    tab._browser_context_id = "owned-context"
    responses = [
        {
            "result": {
                "targetInfos": [
                    {
                        "type": "iframe",
                        "targetId": "target",
                        "url": "https://challenges.cloudflare.com/widget",
                        "browserContextId": "owned-context",
                    }
                ]
            }
        },
        {"result": {"sessionId": "owned-session"}},
        {
            "result": {
                "root": {
                    "children": [
                        {
                            "children": [
                                {
                                    "nodeName": "BODY",
                                    "shadowRoots": [{"nodeId": 5, "shadowRootType": ShadowRootType.CLOSED}],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        {"result": {"nodeIds": node_ids}},
    ]
    if len(node_ids) == 1:
        responses.append({"result": {"object": {"objectId": "native-checkbox"}}})
    responses.append({"result": {}})
    tab._execute_command = AsyncMock(side_effect=responses)
    return tab


def _owned_page() -> Mock:
    page = Mock(spec=Page)
    frame = Mock(spec=Frame)
    frame.url = "https://challenges.cloudflare.com/widget"
    page.frames = [frame]
    return page


class EmbeddedWidgetTests(IsolatedAsyncioTestCase):
    """Only one real checkbox in a Cloudflare iframe may receive a native click."""

    async def test_closed_shadow_body_is_the_only_candidate(self) -> None:
        root: PDNode = {
            "children": [
                {
                    "children": [
                        {"nodeName": "BODY", "shadowRoots": [{"nodeId": 5, "shadowRootType": ShadowRootType.CLOSED}]}
                    ]
                }
            ]
        }
        self.assertEqual(_closed_body_shadows(root), [5])
        self.assertEqual(_closed_body_shadows({"children": [{"children": [{"nodeName": "BODY"}]}]}), [])

    async def test_native_click_routes_to_unique_closed_shadow_checkbox_and_detaches(self) -> None:
        tab = _iframe_tab([12])
        with patch("prowl.browser.solvers.cloudflare.WebElement") as native_element:
            native_element.return_value.click = AsyncMock()
            self.assertTrue(await click_embedded_turnstile(tab, _owned_page()))
        native_element.return_value.click.assert_awaited_once()
        self.assertEqual(native_element.return_value._routing_session_id, "owned-session")
        self.assertEqual(tab._execute_command.await_count, 6)
        self.assertEqual(tab._execute_command.await_args.args[0]["method"], "Target.detachFromTarget")

    async def test_other_context_or_page_frame_is_never_attached(self) -> None:
        tab = _iframe_tab([12])
        page = _owned_page()
        tab._browser_context_id = "another-context"
        self.assertFalse(await click_embedded_turnstile(tab, page))
        self.assertEqual(tab._execute_command.await_count, 1)

        tab = _iframe_tab([12])
        page.frames[0].url = "https://example.com/"
        self.assertFalse(await click_embedded_turnstile(tab, page))
        tab._execute_command.assert_not_awaited()

    async def test_ambiguous_checkboxes_are_not_clicked_and_session_is_detached(self) -> None:
        tab = _iframe_tab([12, 13])
        with patch("prowl.browser.solvers.cloudflare.WebElement") as native_element:
            self.assertFalse(await click_embedded_turnstile(tab, _owned_page()))
        native_element.assert_not_called()
        self.assertEqual(tab._execute_command.await_args.args[0]["method"], "Target.detachFromTarget")
