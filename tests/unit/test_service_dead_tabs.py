"""Dead interactive tabs: a tab whose native page is gone is retired, never reused or listed.

A live isolated session's interactive tab can outlive the page it was opened in once native recovery
retires a dead browser generation. These tests pin the backend's response with spec mocks of the
real native types: a dead tab is closed through the existing ownership path, releasing a shared
tab's claim exactly once while an isolated tab never releases its session's lifetime claim, every
dead tab is attempted even when one close fails, and a live tab is left alone. No process is
launched and no network is touched.
"""

from __future__ import annotations

from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

from playwright.async_api import Browser as PWBrowser
from playwright.async_api import BrowserContext as PWBrowserContext
from playwright.async_api import Page as PWPage

from prowl.browser.browser import TabGroup
from prowl.browser.driver.contexts import BrowserContextHandle
from prowl.browser.exceptions import BrowserTabError
from prowl.service import backend as backend_module
from prowl.service.backend import BrowserBackend, InteractiveRequest, InteractiveTab

_LiveTab = backend_module._LiveTab

_URL = "https://example.com/"


def _context_handle(*, context_closed: bool = False, browser_connected: bool = True) -> BrowserContextHandle:
    """A real handle around spec mocks of a native context and its browser."""
    browser = Mock(spec=PWBrowser)
    browser.is_connected.return_value = browser_connected
    context = Mock(spec=PWBrowserContext)
    context.is_closed.return_value = context_closed
    context.browser = browser
    return BrowserContextHandle(session_id=None, context=context, owner=object())


def _group(
    handle: BrowserContextHandle | None,
    *,
    page_closed: bool = False,
    page_missing: bool = False,
) -> Any:
    """A spec mock of a real ``TabGroup`` whose native page is live, closed or unmapped."""
    group = Mock(spec=TabGroup)
    group.context = handle
    group.quit = AsyncMock()
    if page_missing:
        type(group).ppage = PropertyMock(side_effect=BrowserTabError("the page is gone"))
    else:
        page = Mock(spec=PWPage)
        page.is_closed.return_value = page_closed
        group.ppage = page
    return group


def _tab(tab_id: str, *, egress: str = "default") -> InteractiveTab:
    return InteractiveTab(tab_id=tab_id, url=_URL, title="", status_code=200, requested_url=_URL, egress=egress)


def _entry(
    group: Any,
    *,
    tab_id: str = "tab-1",
    session_id: str | None = None,
    owns_claim: bool = True,
    egress: str = "default",
) -> Any:
    return _LiveTab(tab=_tab(tab_id, egress=egress), group=group, session_id=session_id, owns_claim=owns_claim)


class DeadInteractiveTabTests(IsolatedAsyncioTestCase):
    """Retirement through the existing close path, preserving isolated lifetime claims."""

    async def test_a_dead_shared_tab_is_not_reused_or_listed_and_releases_once(self) -> None:
        backend = BrowserBackend()
        group = _group(_context_handle(), page_closed=True)
        backend._tabs["tab-1"] = _entry(group)
        with patch.object(backend, "_release", new_callable=AsyncMock) as release:
            self.assertEqual(await backend.list_interactive(), [])
            self.assertIsNone(await backend._find_reusable_tab(InteractiveRequest(url=_URL)))
            group.quit.assert_awaited_once()
            release.assert_awaited_once()
        self.assertEqual(backend._tabs, {})

    async def test_a_dead_isolated_tab_never_releases_the_lifetime_claim(self) -> None:
        backend = BrowserBackend()
        group = _group(_context_handle(), page_closed=True)
        backend._tabs["tab-1"] = _entry(group, session_id="s1", owns_claim=False, egress="decodo")
        with patch.object(backend, "_release", new_callable=AsyncMock) as release:
            self.assertEqual(await backend._retire_dead_tabs(), ["tab-1"])
            group.quit.assert_awaited_once()
            release.assert_not_awaited()
        self.assertEqual(backend._tabs, {})

    async def test_a_closed_or_unmapped_page_with_a_live_context_is_dead(self) -> None:
        backend = BrowserBackend()
        closed_group = _group(_context_handle(), page_closed=True)
        missing_group = _group(_context_handle(), page_missing=True)
        backend._tabs["tab-closed"] = _entry(closed_group, tab_id="tab-closed")
        backend._tabs["tab-missing"] = _entry(missing_group, tab_id="tab-missing")
        with patch.object(backend, "_release", new_callable=AsyncMock) as release:
            self.assertEqual(await backend.list_interactive(), [])
            closed_group.quit.assert_awaited_once()
            missing_group.quit.assert_awaited_once()
            self.assertEqual(release.await_count, 2)
        self.assertEqual(backend._tabs, {})

    async def test_a_live_tab_is_still_reused_and_listed(self) -> None:
        backend = BrowserBackend()
        group = _group(_context_handle(), page_closed=False)
        entry = _entry(group)
        backend._tabs["tab-1"] = entry
        with patch.object(backend, "_release", new_callable=AsyncMock) as release:
            self.assertIs(await backend._find_reusable_tab(InteractiveRequest(url=_URL)), entry)
            listed = await backend.list_interactive()
            self.assertEqual([tab.tab_id for tab in listed], ["tab-1"])
            group.quit.assert_not_awaited()
            release.assert_not_awaited()

    async def test_every_dead_tab_is_retired_even_when_one_close_fails(self) -> None:
        backend = BrowserBackend()
        group_a = _group(_context_handle(), page_closed=True)
        group_b = _group(_context_handle(), page_closed=True)
        group_b.quit.side_effect = RuntimeError("quit failed")
        backend._tabs["tab-a"] = _entry(group_a, tab_id="tab-a")
        backend._tabs["tab-b"] = _entry(group_b, tab_id="tab-b")
        with patch.object(backend, "_release", new_callable=AsyncMock) as release:
            with self.assertRaises(RuntimeError):
                await backend.list_interactive()
            group_a.quit.assert_awaited_once()
            group_b.quit.assert_awaited_once()
            self.assertEqual(release.await_count, 2)
        self.assertEqual(backend._tabs, {})
