"""Tab groups created through a named egress must belong to that egress's browser.

A group resolves its own tabs through its owner, so a group bound to the default browser while
its tabs live in an egress browser raises BrowserTabError against a browser that is working
perfectly. That is what a real deployment showed, and what these tests pin: the stubs the other
tests use never reach the point where the wrong owner is consulted.

``BrowserRuntimeState`` is a slots dataclass, so its methods are patched on the class rather than
on an instance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Self, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from prowl.browser.browser import Browser, TabGroup
from prowl.browser.driver.runtime import BrowserRuntimeState
from prowl.browser.egress import create_egress_browser
from prowl.browser.lifecycle.startup import BrowserLifecycle

_TARGET_ID = "target-one"


def _egress(name: str, tmp: Path, port: int) -> type[Browser]:
    """Return an egress browser subclass with its own runtime, as the pool builds one."""
    return create_egress_browser(
        name=name,
        proxy_url=f"socks5://127.0.0.1:{port}",
        profile_dir=str(tmp / name / "profile"),
        profile_archive=str(tmp / f"profile-{name}.zip"),
        preferred_cdp_port=port,
    )


class EgressGroupOwnershipTests(IsolatedAsyncioTestCase):
    """A group binds to the browser that created it, and resolves its tabs there."""

    def setUp(self: Self) -> None:
        """Give each test its own profile paths, and remember the default browser runtime."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile_root = Path(tmp.name)
        self.original_runtime = Browser._runtime

    def tearDown(self: Self) -> None:
        """Restore the default browser runtime."""
        Browser._runtime = self.original_runtime

    async def test_happy_path_group_created_through_an_egress_owns_that_egress(self: Self) -> None:
        """Happy path: a group made through an egress subclass binds to that subclass."""
        egress = _egress("one", self.profile_root, 9300)
        with (
            patch.object(BrowserRuntimeState, "create_page", AsyncMock(return_value=(_TARGET_ID, object()))),
            patch.object(BrowserRuntimeState, "allocate_group_id", MagicMock(return_value=1)),
        ):
            group = await egress._create_from_running()

        self.assertIsInstance(group, TabGroup)
        self.assertIs(group._owner, egress)
        self.assertIsNot(group._owner, Browser)

    async def test_state_transition_egress_create_binds_the_owner_into_the_factory(self: Self) -> None:
        """State transition: the create path hands the runtime a factory bound to itself."""
        egress = _egress("one", self.profile_root, 9300)
        runtime = MagicMock()
        runtime.create_tab_group = AsyncMock(return_value=TabGroup("target", 1))
        egress._runtime = cast("BrowserRuntimeState", runtime)
        egress._lifecycle = BrowserLifecycle(cast("BrowserRuntimeState", runtime))

        await egress.create()

        factory = runtime.create_tab_group.await_args.args[0]
        bound = factory("target", 1)
        self.assertIsInstance(bound, TabGroup)
        self.assertIs(bound._owner, egress)

    async def test_invariant_the_default_browser_group_stays_bound_to_the_default_browser(self: Self) -> None:
        """Invariant: the default path keeps building plain groups owned by Browser."""
        Browser._runtime = BrowserRuntimeState(max_groups=3)
        with (
            patch.object(BrowserRuntimeState, "create_page", AsyncMock(return_value=("target-default", object()))),
            patch.object(BrowserRuntimeState, "allocate_group_id", MagicMock(return_value=2)),
        ):
            group = await Browser._create_from_running()

        self.assertIs(type(group), TabGroup)
        self.assertIs(group._owner, Browser)

    async def test_invariant_each_egress_resolves_tabs_through_its_own_browser(self: Self) -> None:
        """Invariant: a tab lookup goes through the group's own browser, never the default one."""
        one = _egress("one", self.profile_root, 9300)
        two = _egress("two", self.profile_root, 9410)
        default_runtime = MagicMock()
        default_runtime.get_pd_tab = AsyncMock(return_value="default-tab")
        Browser._runtime = cast("BrowserRuntimeState", default_runtime)

        async def _tab_for(runtime: BrowserRuntimeState, _target_id: str) -> str:
            """Return the tab that belongs to the runtime being asked."""
            return "two-tab" if runtime is two._runtime else "one-tab"

        groups: dict[str, TabGroup] = {}
        with (
            patch.object(BrowserRuntimeState, "get_pd_tab", _tab_for),
            patch.object(BrowserRuntimeState, "create_page", AsyncMock(return_value=(_TARGET_ID, object()))),
            patch.object(BrowserRuntimeState, "allocate_group_id", MagicMock(return_value=1)),
        ):
            for name, egress in (("one", one), ("two", two)):
                groups[name] = await egress._create_from_running()

            for name, egress in (("one", one), ("two", two)):
                group = groups[name]
                self.assertIs(group._owner, egress)
                self.assertEqual(await group.ptab, f"{name}-tab")

        default_runtime.get_pd_tab.assert_not_awaited()
