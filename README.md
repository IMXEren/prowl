# prowl

A reusable browser automation core, a FlareSolverr-compatible HTTP service, and a
small CLI for loading pages through a real CloakBrowser/pydoll browser. Prowl renders
JavaScript, persists browser state, performs browser-origin GET and POST requests, and
returns the resulting document, headers, and cookies to callers that need more than a
plain HTTP client.

## Architecture

One repository owns the whole stack:

```
src/prowl/
  browser/            reusable core (extracted code, behaviour preserved)
    browser.py        Browser process manager + TabGroup
    site.py           Site/Source, challenge handling, DOM tree building, POST fetch
    cookies.py        cookie persistence helpers
    driver/           concrete Playwright + pydoll driver ownership
    lifecycle/        startup, profile warm/pack, shutdown state machine
    profile/          Chromium search-engine injection
  shutdown.py         generic process signal/cancellation coordinator
  service/            FlareSolverr-compatible HTTP service (aiohttp)
  cli.py              `prowl fetch` / `prowl serve`
```

The core is production browser code. It keeps its tested dual Playwright +
pydoll/CloakBrowser behaviour, persistent profile warming and packing, cookie handling,
supported challenge handling, and cancellation/cleanup semantics. The service layer adds
environment-driven configuration, explicitly scoped custom request headers, and a real
browser `POST` path, and the generic signal coordinator moved from `src/signals.py` into
`prowl.shutdown`, keeping the package self-contained.

## Profiles are persistent trust assets

The browser runs from one persistent Chrome profile directory (`PROWL_PROFILE_DIR`,
`/state/profile` in Docker), packed on shutdown into `PROWL_PROFILE_ARCHIVE`
(`/state/browser-profile.zip` in Docker) on the same writable `prowl-state` volume.
Cookies, storage, login state, and site reputation can be bound to that profile and to
the browser fingerprint, so the profile is a long-lived asset rather than scratch state:

* On startup the profile is restored from `PROWL_PROFILE_ARCHIVE` when present, otherwise it
  is primed headlessly once and the search engine is injected.
* On shutdown the profile is packed back to the archive, skipping caches and journals.
* Warm the profile once against the target site; later requests reuse the established trust.

## One shared profile; logical sessions are names, not profiles

The service keeps a single browser process and a single persistent profile. A logical
session (`sessions.create`, or a `session` name on a request) is only a name with an
optional TTL plus a serialization lock over that same shared profile. Sessions share
cookies and trust, and an idle session retains **no** tab group and consumes no browser
capacity. Every fetch creates a fresh tab group, runs inside it, and closes it on
success, error, and cancellation. Do not present sessions as isolation.

`PROWL_MAX_SESSIONS` bounds how many logical sessions may exist at once;
exceeding it returns a deterministic caller-safe error. `session_ttl_minutes` on
`sessions.create` or a request sets an expiry that is enforced lazily and refreshed on
each use; `sessions.list` omits expired sessions.

## Concurrency

* `PROWL_MAX_CONCURRENCY` bounds how many fetches run at once (default `1`: this
  stack intentionally owns one persistent trust profile and one free browser session).
* Each logical session has its own lock, so per-session work is serialized.
* Anonymous (session-less) requests are serialized against each other.
* Each request runs under a deadline derived from `maxTimeout` plus a small cleanup slack;
  every fetch closes its tab group on every path, including cancellation.

## Egress / proxy

One browser and one profile have exactly one egress. `PROWL_PROXY_URL` configures
that egress process-wide and is passed to CloakBrowser as `--proxy-server` (the URL is
never logged). A request may include `proxy.url` only when it exactly matches the configured
process proxy; any absent or mismatched configuration is rejected deterministically. Proxy
credentials are never included in responses or logs. For authenticated upstreams, point
`PROWL_PROXY_URL` at a local credential-injecting proxy. A proxy URL that embeds
credentials (`user:pass@`) is rejected, so a configured or request proxy is always an
authenticated-free hop; put the credential injection at that local hop instead.

## HTTP API (FlareSolverr v1 subset)

`POST /v1` with a JSON body:

| cmd | fields |
| --- | --- |
| `request.get` | `url`, `maxTimeout`, `session`, `session_ttl_minutes`, `headers`, `headerScope`, `cookies`, `returnOnlyCookies`, `proxy` |
| `request.post` | as `request.get` except `headerScope`, plus `postData` (string, or object sent as JSON) |
| `sessions.create` | optional `session` name, optional `session_ttl_minutes` |
| `sessions.list` | — |
| `sessions.destroy` | `session` |

Unknown fields are rejected rather than ignored. A GET without custom headers leaves all
headers under browser control. Custom GET headers require an explicit `headerScope`:
`document` applies them only to the initial main-frame navigation, while `origin` applies
them only to requests with the target URL's exact scheme, host, and effective port. Neither
scope sends headers to redirects on another origin, subdomains, or third-party resources.
Browser-controlled and fingerprint headers (`Host`, `Cookie`, `User-Agent`, `Accept`,
`Origin`, `Referer`, `Sec-*`, and similar) are always rejected. `request.post` accepts only
`Content-Type`, scoped naturally to its single page-context fetch.

For an authenticated initial navigation without exposing the credential to subresources:

```json
{
  "cmd": "request.get",
  "url": "https://example.com/private",
  "headers": { "Authorization": "Bearer token" },
  "headerScope": "document"
}
```

Use `origin` instead only when every request to that exact origin needs the custom header.

Response envelope:

```json
{
  "status": "ok",
  "message": "",
  "startTimestamp": 1710000000000,
  "endTimestamp": 1710000001234,
  "version": "prowl/0.1.0",
  "solution": {
    "url": "https://example.com/",
    "status": 200,
    "headers": { "content-type": "text/html" },
    "response": "<!DOCTYPE html>...",
    "cookies": [{ "name": "session_id", "value": "...", "domain": ".example.com", "path": "/" }],
    "userAgent": "Mozilla/5.0 ..."
  }
}
```

`request.post` first loads the request URL's origin root in the same tab and profile, then
issues a real `POST` from the page's own `fetch` implementation, so it keeps the browser's
origin, cookies, TLS fingerprint, and headers and is never downgraded to GET. Loading the
origin root also warms the page for same-origin APIs and supported site protections. A
challenge that appears only on the POST response itself is not automatically solved.

Errors return `"status": "error"` with a caller-safe `message`. Validated command errors
use HTTP 200 (FlareSolverr clients read `status` from the body); malformed transport
payloads use HTTP 400. Messages and logs never contain tracebacks, credentials, proxy
passwords, profile paths, or browser internals.

Other endpoints: `GET /healthz`, `GET /readyz`.

### Example

```bash
curl -s http://127.0.0.1:8191/v1 -H 'Content-Type: application/json' -d '{
  "cmd": "request.get",
  "url": "https://example.com/",
  "maxTimeout": 60000,
  "session": "example"
}' | jq '.solution.cookies'
```

## CLI

```bash
# one-shot URL fetch
prowl fetch https://example.com/ --timeout 60 -o page.html

# run the service
prowl serve --host 0.0.0.0 --port 8191
```

Exit status: `0` on success, `1` when the fetch fails, `2` for CLI usage errors.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

The compose file runs the service next to an Xvfb display (headed browser needs a
display), mounts the `prowl-state` volume at `/state` for the persistent profile, sets
`shm_size: 2gb`, and
healthchecks `/healthz`. The service listens on `8191` on the Compose network only; it is
not published to the host by default. For an opt-in localhost-only debugging bind, use the
commented `ports` block in `docker-compose.yml`. The image installs the CloakBrowser binary
and the GeoIP database, runs as a non-root user, and keeps Chromium's `--no-sandbox`
because containers do not grant the user namespaces its sandbox needs. The image is built
locally from this public source (Compose tag `prowl:local`); no prebuilt Prowl image is
published, so there is nothing to pull.

Optionally improve fingerprint fidelity with a trusted Windows font archive. Put
`fonts.zip` in a directory outside the source tree and provide that directory as a
read-only named build context:

```bash
mkdir -p ../prowl-private-fonts
cp /private/path/fonts.zip ../prowl-private-fonts/fonts.zip
docker build \
  --build-context windows_fonts=../prowl-private-fonts \
  -t prowl:local .
```

The archive is not copied into Prowl's source context. BuildKit secrets are not used
because their payload is limited to 500 KiB, which is too small for the font archive.

## Browser-backed access and live smoke tests

The service keeps one persistent, warmed CloakBrowser profile and performs requests inside
that real browser session. This preserves site-issued cookies and browser state across
requests instead of fabricating or synthesizing them. Supported challenge pages can be
handled as part of normal navigation, while ordinary JavaScript-heavy pages use the same
browser path without any challenge-specific behavior.

Live websites are **smoke tests**, not deterministic CI. The default test suite never
touches the network: the browser is replaced by a fake backend and a local HTTP fixture, so
runs are reproducible offline. Any live target check is opt-in and must be run explicitly.

## Releases

- `main` produces stable GitHub Releases.
- `dev` produces prerelease GitHub Releases on the `dev` channel.
- Release assets are the audited Python wheel and source archive.

Prowl is not published to package or container registries. Install from source, a
GitHub Release asset, or a pinned Git revision, and build the container locally.

## License

Prowl source is distributed under **GPL-3.0-only**. See `LICENSE`.

Powered by [CloakBrowser](https://github.com/CloakHQ/CloakBrowser). The image build
downloads the CloakBrowser Binary from the official CloakHQ channel. That Binary is
proprietary and is **not** covered by Prowl's GPL license; it is governed by its own
[CloakBrowser Binary License](https://github.com/CloakHQ/CloakBrowser/blob/main/BINARY-LICENSE.md).
An image built here that contains the Binary is for personal/internal use and must not be
redistributed or published without a separate license from CloakHQ.

## Development

```bash
uv venv
uv pip install -e '.[dev]'
uv run pytest
uv run ruff format --check . && uv run ruff check .
uv run python -m build
```

For an opt-in manual browser check against a URL:

```bash
uv run python -m prowl.browser.test https://example.com/ --timeout 60
```
