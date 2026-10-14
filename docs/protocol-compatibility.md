# FlareSolverr compatibility

Prowl implements the FlareSolverr-style `/v1` JSON API, but it is not an exact replacement for every FlareSolverr behavior. Unknown commands and fields are rejected rather than ignored.

## Commands and options

The common commands are `request.get`, `request.post`, `sessions.create`, `sessions.list`, and `sessions.destroy`. Prowl also provides `browser.open`, `browser.list`, `browser.close`, and `cookies.list`.

Fetches accept `url`, `session`, `session_ttl_minutes`, `maxTimeout`, `cookies`, and `returnOnlyCookies`. `request.post` accepts `postData`.

| Option | Behavior |
| --- | --- |
| `waitInSeconds` | Wait before the final DOM and cookie snapshot, within `maxTimeout`. Skipped when `returnOnlyCookies` is true. |
| `returnScreenshot` | Include a Base64 PNG in `solution.screenshot`, including for cookie-only requests. |
| `disableMedia` | Block images, stylesheets, and fonts on the selected page. Requests handled by service workers may bypass this filter. |
| `solveCaptcha` | Attempt a supported widget solver, including embedded Turnstile, on browser/auto GET. See [CAPTCHA support](captcha-solvers.md). |

Screenshot, delayed-snapshot, media-filter, and explicit-verification requests require browser work. An eligible auto request goes directly to the browser; explicit HTTP requests cannot use these browser-only options.

## Differences to account for

### Execution mode

Omitting `mode` now selects `auto`: eligible GETs try HTTP first and escalate when browser work is required. Use explicit `browser` to retain browser-only behavior, including JavaScript rendering. `http` uses the identity-matched HTTP transport without browser fallback. Auto POSTs go directly to the browser to avoid replay.

The JSON POST API remains separate from the forward proxy's raw-body POST path. Raw proxy POSTs use HTTP only and are never replayed through the browser.

### Sessions

Sessions are shared by default: sessions on one egress use the same persistent browser profile. `sessionMode: "isolated"` gives a named session its own browser context, not its own browser process or device identity.

A session's mode and bound egress cannot change during its lifetime. TTL measures idle time; active and queued requests do not expire. See [browser profiles](identity-profiles.md) for storage and persistence.

### Proxies

Clients select an egress configured by the service through `proxy.name`. The compatibility selector `proxy.url` must match a configured egress; it does not allow an arbitrary per-request proxy. Caller-supplied proxy URLs cannot contain credentials.

Authenticated HTTP(S) upstreams are configured by the operator and selected by name. Authenticated SOCKS upstreams are not supported.

### Responses

Prowl reports observed response status and headers when available. Do not assume that a browser result always has status 200 or empty headers.

`returnOnlyCookies` keeps Prowl's existing response projection; clients should not rely on it removing fields exactly as FlareSolverr does. HTTP/auto results can also include execution diagnostics.

CAPTCHA response tokens do not guarantee that the target site's server will accept them.

## References

- [FlareSolverr API documentation](https://github.com/FlareSolverr/FlareSolverr#api)
- [Prowl service and configuration](../README.md)
