# CDP gateway

Set `PROWL_CDP_TOKEN_FILE` to a file containing a secret of at least 32 characters. The gateway is disabled without it. Every request, including each WebSocket upgrade, requires `Authorization: Bearer <token>`.

1. `POST /cdp/reservations` with an optional JSON `egress` name reserves a tab. The response contains its `tab` reservation ID and CDP `target` ID.
2. `GET /cdp/reservations/{tab}/json/version` returns the browser WebSocket URL. `GET /cdp/reservations/{tab}/json/list` returns page WebSocket URLs. Both are authenticated.
3. Connect to `/cdp/reservations/{tab}/socket` for browser-level CDP. Page clients open `/devtools/page/{target}?reservation={tab}` with the same bearer header. Page sockets are available while the browser socket is connected. Browser-socket closure closes its page sockets and releases the reservation.
4. `DELETE /cdp/reservations/{tab}` releases a reservation that was never connected or explicitly closes one that is still active.

The browser debugging port remains loopback-only. A gateway credential grants browser-level CDP access to the selected browser process, including its other tabs. Only give the credential to trusted plugins. The gateway does not change Prowl's primary browser controller.
