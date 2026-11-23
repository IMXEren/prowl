"""Browser response interception follows relative redirects to the final document."""

import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from pydoll.browser.tab import Tab

from prowl.browser.browser import TabGroup
from prowl.browser.page_handler import PageHandler


class BrowserRedirectTests(IsolatedAsyncioTestCase):
    async def test_relative_location_completes_on_final_document(self) -> None:
        site = PageHandler(Mock(spec=TabGroup))
        tab = Mock(spec=Tab)
        tab.continue_request = AsyncMock()
        tab._execute_command = AsyncMock(return_value={"result": {"body": "<html>ok</html>", "base64Encoded": False}})
        site.tab = tab
        site.url = "https://example.test/demo/cloudflare"

        await site._on_request(
            {
                "params": {
                    "requestId": "redirect",
                    "request": {"url": site.url, "headers": {"User-Agent": "browser"}},
                    "responseStatusCode": 302,
                    "responseHeaders": [{"name": "Location", "value": "/demo"}],
                }
            }
        )
        self.assertEqual(site.redirected_url, "https://example.test/demo")
        self.assertFalse(site.response_found)

        with (
            patch.object(site, "_check_cf_encounter", AsyncMock()),
            patch.object(site, "_remove_network_listeners", AsyncMock()),
        ):
            await site._on_request(
                {
                    "params": {
                        "requestId": "document",
                        "request": {"url": "https://example.test/demo", "headers": {"User-Agent": "browser"}},
                        "responseStatusCode": 200,
                        "responseHeaders": [{"name": "Content-Type", "value": "text/html"}],
                    }
                }
            )
        self.assertTrue(site.response_found)
        self.assertEqual(site.status_code, 200)
        headers = site.response_headers
        assert headers is not None
        self.assertEqual(headers["Content-Type"], "text/html")
        await asyncio.wait_for(site._loaded.wait(), timeout=0.1)

    async def test_relative_query_location_resolves_against_current_hop(self) -> None:
        site = PageHandler(Mock(spec=TabGroup))
        tab = Mock(spec=Tab)
        tab.continue_request = AsyncMock()
        site.tab = tab
        site.url = "https://example.test/directory/start"
        await site._on_request(
            {
                "params": {
                    "requestId": "redirect",
                    "request": {"url": site.url, "headers": {"User-Agent": "browser"}},
                    "responseStatusCode": 302,
                    "responseHeaders": [{"name": "Location", "value": "next?step=1"}],
                }
            }
        )
        self.assertEqual(site.redirected_url, "https://example.test/directory/next?step=1")
