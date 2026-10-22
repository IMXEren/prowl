"""Dispatch prepared proxy requests through service admission and response projection."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Final

from prowl.browser.headers import HEADER_SCOPE_ORIGIN
from prowl.browser.proxy.http1 import ProxyProtocolError, ProxyRequest
from prowl.browser.proxy.request import prepare_request
from prowl.browser.proxy.response import ProxyResponse, project_response
from prowl.service.protocol import (
    AUTO_MODE,
    CMD_REQUEST_GET,
    CMD_REQUEST_POST,
    DEFAULT_TIMEOUT_MS,
    HTTP_MODE,
    FetchCommand,
)

if TYPE_CHECKING:
    from prowl.service.app import Service
    from prowl.service.protocol import ProxySelection
    from prowl.service.sessions import SessionMode

__all__ = ["dispatch_request"]

#: Inbound correlation field that is ignored outright and never forwarded to the origin.
_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"


async def dispatch_request(  # noqa: PLR0913 - the listener passes its trusted bindings explicitly
    service: Service,
    request: ProxyRequest,
    *,
    tunnel_authority: str | None = None,
    session: str | None = None,
    session_mode: SessionMode | None = None,
    proxy: ProxySelection | None = None,
) -> ProxyResponse:
    """Use trusted listener bindings; GET may escalate, while raw POST stays HTTP-only."""
    prepared = prepare_request(_without_request_id(request), tunnel_authority=tunnel_authority)
    is_post = prepared.method == "POST"
    if not is_post and prepared.body:
        raise ProxyProtocolError(400)

    command = FetchCommand(
        cmd=CMD_REQUEST_POST if is_post else CMD_REQUEST_GET,
        url=prepared.url,
        timeout_ms=DEFAULT_TIMEOUT_MS,
        session=session,
        session_mode=session_mode,
        headers=prepared.headers,
        header_scope=HEADER_SCOPE_ORIGIN,
        proxy=proxy,
        mode=HTTP_MODE if is_post else AUTO_MODE,
    )
    result = await service.fetch(
        command,
        body_bytes=prepared.body if is_post else None,
        cookie_header=prepared.cookie_header,
    )
    return project_response(result)


def _without_request_id(request: ProxyRequest) -> ProxyRequest:
    """Drop the ignored inbound ``X-Request-ID`` field so it never reaches the origin.

    Only that one field is removed; the caller's frozen DTO is never mutated.
    """
    headers = tuple((name, value) for name, value in request.headers if name.lower() != _REQUEST_ID_HEADER.lower())
    return replace(request, headers=headers)
