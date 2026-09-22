"""Browser backend that executes fetch commands against the reusable core.

Every fetch runs in its own tab group inside the single shared browser
process and closes that group on every path, including errors and
cancellation. The backend owns no per-session state; logical-session
serialization is owned by the session registry.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from prowl.browser import Browser, BrowserConfig
from prowl.browser.site import resolve_site

#: Cookie fields FlareSolverr clients expect in ``solution.cookies``.
_COOKIE_FIELDS = ("name", "value", "domain", "path", "expires", "secure", "httpOnly", "sameSite")


@dataclass(slots=True)
class FetchRequest:
    """A validated fetch to execute in the browser."""

    url: str
    method: str = "GET"
    timeout_seconds: int = 60
    headers: dict[str, str] = field(default_factory=dict)
    cookies: list[dict[str, Any]] = field(default_factory=list)
    post_data: str = ""


@dataclass(slots=True)
class FetchResult:
    """The outcome of a browser fetch."""

    url: str
    status_code: int | None
    headers: dict[str, str]
    response: str
    cookies: list[dict[str, Any]]
    user_agent: str | None


@runtime_checkable
class Backend(Protocol):
    """The browser operations the service depends on."""

    async def start(self) -> None:
        """Start the shared browser process."""
        ...

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        """Execute *request*, serialized within *session_id* when supplied."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Release any backend state for *session_id*; idempotent no-op here."""
        ...

    async def aclose(self) -> None:
        """Release all browser resources."""
        ...


class BrowserBackend:
    """Concrete backend owning the shared browser process.

    The backend starts the browser once, performs each fetch in a fresh tab
    group, and shuts the browser down. It holds no per-session locks; the
    service's session registry serializes per-session work.
    """

    def __init__(self, browser_config: BrowserConfig | None = None) -> None:
        self._browser_config = browser_config

    async def start(self) -> None:
        """Apply launch configuration and start the shared browser process."""
        if self._browser_config is not None:
            Browser.configure(self._browser_config)
        await Browser.start()

    async def close_session(self, session_id: str) -> None:
        """No-op: the backend retains no per-session state."""

    async def fetch(self, session_id: str | None, request: FetchRequest) -> FetchResult:
        """Execute *request* in a fresh tab group and close it on every path."""
        await Browser.start()
        group = await Browser.create()
        try:
            return await self._fetch_in_group(group, request)
        finally:
            with contextlib.suppress(Exception):
                await group.quit()

    @staticmethod
    async def _fetch_in_group(group: Any, request: FetchRequest) -> FetchResult:
        tab = await group.ptab
        if request.cookies:
            await tab.set_cookies(request.cookies)

        site = resolve_site(group, request.url)
        if request.method == "POST":
            source = await site.post(
                request.url,
                request.timeout_seconds,
                post_data=request.post_data,
                headers=request.headers,
            )
        else:
            source = await site.get(request.url, request.timeout_seconds, headers=request.headers)

        cookies = _normalize_cookies(await _collect_cookies(group))
        return FetchResult(
            url=source.url or request.url,
            status_code=source.status_code if isinstance(source.status_code, int) else None,
            headers=dict(source.headers),
            response=source.text,
            cookies=cookies,
            user_agent=source.user_agent,
        )

    async def aclose(self) -> None:
        """Shut the shared browser down."""
        try:
            await Browser.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to shut the browser down: {type(exc).__name__}")


async def _collect_cookies(group: Any) -> list[dict[str, Any]]:
    """Return the shared browser context's cookies in FlareSolverr shape."""
    try:
        raw = await group.pd().get_cookies()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read browser cookies: {type(exc).__name__}")
        return []
    return [cookie for cookie in raw if isinstance(cookie, dict)]


def _normalize_cookies(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce pydoll cookie dicts to the FlareSolverr cookie shape."""
    normalized: list[dict[str, Any]] = []
    for cookie in raw:
        entry = {key: cookie[key] for key in _COOKIE_FIELDS if key in cookie and cookie[key] is not None}
        if "name" in entry and "value" in entry:
            normalized.append(entry)
    return normalized


__all__ = [
    "Backend",
    "BrowserBackend",
    "FetchRequest",
    "FetchResult",
]
