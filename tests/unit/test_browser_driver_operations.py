"""Tests for concrete browser driver operations."""

from __future__ import annotations

from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from playwright.async_api import BrowserContext as PWBrowserContext
from pydoll.browser.chromium import Chrome

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.driver import BrowserRuntimeState
from prowl.browser.exceptions import BrowserTabError
from prowl.browser.lifecycle import BrowserLifecycle


class _FakeCdpSession:
    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.detached = False

    async def send(self, command: str) -> dict[str, dict[str, str]]:
        if command != "Target.getTargetInfo":
            msg = f"unexpected command {command}"
            raise AssertionError(msg)
        return {"targetInfo": {"targetId": self.target_id}}

    async def detach(self) -> None:
        self.detached = True


def _fake_context(target_ids: list[str]) -> MagicMock:
    context = MagicMock(spec=PWBrowserContext)
    pages: list[MagicMock] = []
    context.pages_created = pages
    context.is_closed.return_value = False

    async def new_page() -> MagicMock:
        page = MagicMock(name=f"page-{len(pages)}")
        page.context = context
        page.close = AsyncMock()
        pages.append(page)
        return page

    async def new_cdp_session(page: MagicMock) -> _FakeCdpSession:
        return _FakeCdpSession(target_ids[pages.index(page)])

    context.new_page = AsyncMock(side_effect=new_page)
    context.new_cdp_session = AsyncMock(side_effect=new_cdp_session)
    return context


def _fake_pydoll() -> MagicMock:
    pd = MagicMock(spec=Chrome)
    opened: dict[str, MagicMock] = {}
    pd._tabs_opened = opened
    pd.get_opened_tabs = AsyncMock(side_effect=lambda: list(opened.values()))
    pd.set_window_minimized = AsyncMock()
    pd._get_tab_kwargs.side_effect = lambda target_id, browser_context_id: {
        "target_id": target_id,
        "browser_context_id": browser_context_id,
    }
    return pd


def _fake_pd_tab(_pd: object, **kwargs: str) -> MagicMock:
    """Create a mock PyDoll tab for a target id."""
    return MagicMock(_target_id=kwargs["target_id"])


class BrowserDriverOperationsTests(IsolatedAsyncioTestCase):
    """Concrete driver operation behavior without launching browser resources."""

    def setUp(self) -> None:
        """Install isolated browser runtime state."""
        self.original_runtime = Browser._runtime
        self.original_lifecycle = Browser._lifecycle
        Browser._runtime = BrowserRuntimeState(max_groups=3)
        Browser._lifecycle = BrowserLifecycle(Browser._runtime)
        self.pydoll = _fake_pydoll()
        Browser._runtime.shared_pd = self.pydoll

    def tearDown(self) -> None:
        """Restore browser runtime state."""
        Browser._runtime = self.original_runtime
        Browser._lifecycle = self.original_lifecycle

    async def test_create_group_creates_page_tracks_maps_and_active_group(self) -> None:
        """Happy path: group creation owns page maps, pydoll bookkeeping, and active set."""
        context = _fake_context(["parent"])
        Browser._runtime.main_ctx = context
        with patch("prowl.browser.driver.runtime.PDTab", side_effect=_fake_pd_tab):
            group = await Browser._create_from_running()

        page = context.pages_created[0]
        self.assertEqual(group.target_id, "parent")
        self.assertEqual(group.gid, 1)
        self.assertIs(Browser._runtime.target_to_page_map["parent"], page)
        self.assertIs(Browser._runtime.page_to_group[page], group)
        self.assertIn(group, Browser._runtime.active_groups)
        self.assertIn("parent", self.pydoll._tabs_opened)

    async def test_new_tab_and_popup_attach_to_existing_group_for_multiple_targets(self) -> None:
        """Input variation: child tabs and popups attach across distinct target ids."""
        context = _fake_context(["parent", "child", "popup"])
        Browser._runtime.main_ctx = context
        with patch("prowl.browser.driver.runtime.PDTab", side_effect=_fake_pd_tab):
            group = await Browser._create_from_running()
            child_tab = await group.new_tab()
            popup_page = await context.new_page()
            popup_page.opener = AsyncMock(return_value=group.ppage)
            await Browser._handle_popup_page(popup_page)

        self.assertEqual(child_tab._target_id, "child")
        self.assertEqual(group.child_target_ids, ["child", "popup"])
        self.assertIs(Browser._runtime.target_to_page_map["popup"], popup_page)
        self.assertIs(Browser._runtime.page_to_group[popup_page], group)

    async def test_close_child_removes_maps_and_rejects_foreign_target(self) -> None:
        """Error path: foreign targets raise while valid child cleanup is precise."""
        Browser._runtime.main_ctx = _fake_context(["parent", "child"])
        with patch("prowl.browser.driver.runtime.PDTab", side_effect=_fake_pd_tab):
            group = await Browser._create_from_running()
            await group.new_tab()

        with self.assertRaisesRegex(BrowserTabError, "does not belong"):
            await group.close("foreign")
        await group.close("child")

        self.assertNotIn("child", Browser._runtime.target_to_page_map)
        self.assertNotIn("child", self.pydoll._tabs_opened)
        self.assertEqual(group.child_target_ids, [])

    async def test_quit_is_idempotent_and_releases_one_slot(self) -> None:
        """State transition: legal quit releases once; repeated quit is idempotent."""
        await Browser._runtime.group_semaphore.acquire()
        Browser._runtime.main_ctx = _fake_context(["parent", "child"])
        with patch("prowl.browser.driver.runtime.PDTab", side_effect=_fake_pd_tab):
            group = await Browser._create_from_running()
            await group.new_tab()

        before = Browser._runtime.group_semaphore._value
        await group.quit()
        after_first = Browser._runtime.group_semaphore._value
        await group.quit()

        self.assertEqual(after_first, before + 1)
        self.assertEqual(Browser._runtime.group_semaphore._value, after_first)
        self.assertNotIn(group, Browser._runtime.active_groups)
        self.assertEqual(Browser._runtime.target_to_page_map, {})

    async def test_boundaries_missing_pages_and_pd_allow_cleanup(self) -> None:
        """Boundary: missing pages and absent pydoll client do not block cleanup."""
        group = TabGroup("parent", 1)
        group.child_target_ids.append("child")
        Browser._runtime.active_groups.add(group)
        Browser._runtime.shared_pd = None

        await group.close("child")
        await group.quit()

        self.assertEqual(group.child_target_ids, [])
        self.assertEqual(Browser._runtime.active_groups, set())

    async def test_minimize_uses_shared_pydoll_when_present_only(self) -> None:
        """Invariant: minimizing delegates only when a concrete pydoll client exists."""
        await Browser.minimize_main_window()
        Browser._runtime.shared_pd = None
        await Browser.minimize_main_window()

        self.pydoll.set_window_minimized.assert_awaited_once()


class TypeIgnoreGateTests(TestCase):
    """Deterministic type-ignore gate for touched browser files."""

    def test_touched_browser_files_have_no_broad_type_ignores(self) -> None:
        """Error path: touched files reject broad or nonconforming type ignores."""
        bad: list[str] = []
        for path in [
            "src/prowl/browser/browser.py",
            "src/prowl/browser/driver/runtime.py",
            "src/prowl/browser/page_handler.py",
        ]:
            for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
                if "type: ignore" in line and "type: ignore[" not in line:
                    bad.append(f"{path}:{line_no}:{line.strip()}")
        self.assertEqual(bad, [])
