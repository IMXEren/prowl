"""Native Cloudflare challenge handling for the selected page and tab.

A solver activates and deactivates native challenge handling on one owned
native tab. Solvers never decide that a challenge is solved: the native
response and the loaded DOM remain the authority on success.
"""

from typing import Protocol
from urllib.parse import urlsplit

from playwright.async_api import Page
from pydoll.browser.tab import Tab
from pydoll.commands.dom_commands import DomCommands
from pydoll.commands.target_commands import TargetCommands
from pydoll.elements.web_element import WebElement
from pydoll.protocol.dom.types import Node as PDNode


class ChallengeSolver(Protocol):
    """Activates and deactivates native challenge handling on one tab."""

    async def start(self, tab: Tab) -> None:
        """Begin native challenge handling on *tab*."""
        ...

    async def stop(self, tab: Tab) -> None:
        """End native challenge handling on *tab*."""
        ...


class CloudflareSolver:
    """Cloudflare checkbox solver backed by the native auto-solve calls."""

    async def start(self, tab: Tab) -> None:
        """Enable the native Cloudflare auto-solve on *tab*."""
        await tab.enable_auto_solve_cloudflare_captcha(time_before_click=1, time_to_wait_captcha=30)

    async def stop(self, tab: Tab) -> None:
        """Disable the native Cloudflare auto-solve on *tab*."""
        await tab.disable_auto_solve_cloudflare_captcha()


def _closed_body_shadows(root: PDNode) -> list[int]:
    ids: list[int] = []
    for html in root.get("children", []):
        for body in html.get("children", []):
            if body.get("nodeName") == "BODY":
                ids.extend(
                    node_id
                    for shadow in body.get("shadowRoots", [])
                    if shadow.get("shadowRootType") == "closed"
                    if (node_id := shadow.get("nodeId")) is not None
                )
    return ids


async def click_embedded_turnstile(tab: Tab, page: Page) -> bool:
    """Click one checkbox owned by the selected page and browser context."""
    context_id = tab._browser_context_id  # noqa: SLF001
    if not context_id:
        return False
    frame_urls = {frame.url for frame in page.frames if urlsplit(frame.url).hostname == "challenges.cloudflare.com"}
    if not frame_urls:
        return False
    targets = await tab._execute_command(TargetCommands.get_targets())  # noqa: SLF001
    sessions: list[str] = []
    matches: list[tuple[str, int]] = []
    try:
        for target in targets.get("result", {}).get("targetInfos", []):
            if target.get("type") != "iframe":
                continue
            if target.get("browserContextId") != context_id or target.get("url") not in frame_urls:
                continue
            attached = await tab._execute_command(  # noqa: SLF001
                TargetCommands.attach_to_target(target["targetId"], flatten=True)
            )
            session = attached["result"]["sessionId"]
            sessions.append(session)
            document = DomCommands.get_document(depth=-1, pierce=True)
            document["sessionId"] = session
            root = (await tab._execute_command(document)).get("result", {}).get("root", {})  # noqa: SLF001
            for shadow_id in _closed_body_shadows(root):
                query = DomCommands.query_selector_all(shadow_id, 'input[type="checkbox"]')
                query["sessionId"] = session
                ids = (await tab._execute_command(query)).get("result", {}).get("nodeIds", [])  # noqa: SLF001
                matches.extend((session, node_id) for node_id in ids)
        if len(matches) != 1:
            return False
        session, node_id = matches[0]
        command = DomCommands.resolve_node(node_id=node_id)
        command["sessionId"] = session
        object_id = (await tab._execute_command(command)).get("result", {}).get("object", {}).get("objectId")  # noqa: SLF001
        if not object_id:
            return False
        handler = tab._connection_handler  # noqa: SLF001
        checkbox = WebElement(object_id, handler)
        checkbox._routing_session_handler = handler  # noqa: SLF001
        checkbox._routing_session_id = session  # noqa: SLF001
        await checkbox.click()
        return True
    finally:
        for session in reversed(sessions):
            await tab._execute_command(TargetCommands.detach_from_target(session))  # noqa: SLF001
