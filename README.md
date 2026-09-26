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

The default egress's browser runs from the configured persistent Chrome profile directory
(`PROWL_PROFILE_DIR`, `/state/profile` in Docker), packed on shutdown into
`PROWL_PROFILE_ARCHIVE` (`/state/browser-profile.zip` in Docker) on the same writable
`prowl-state` volume. Each egress named in `PROWL_EGRESSES` gets its own browser with its own
profile directory inside that one (`/state/profile/<name>`) and its own archive beside that one
(`/state/browser-profile-<name>.zip`), so profiles are never shared between egresses.
Cookies, storage, login state, and site reputation can be bound to a profile and to
the browser fingerprint, so a profile is a long-lived asset rather than scratch state:

* On startup the profile is restored from `PROWL_PROFILE_ARCHIVE` when present, otherwise it
  is primed headlessly once and the search engine is injected.
* On shutdown the profile is packed back to the archive, skipping caches and journals.
* Warm the profile once against the target site; later requests reuse the established trust.

## One shared profile; logical sessions are names, not profiles

The service keeps one browser process and one persistent profile per egress. A logical
session (`sessions.create`, or a `session` name on a request) is only a name with an
optional TTL plus a serialization lock over that egress's profile. Sessions on the same
egress share cookies and trust, and an idle session retains **no** tab group and consumes no
browser capacity. Every fetch creates a fresh tab group, runs inside it, and closes it on
success, error, and cancellation. Do not present sessions as isolation.

`PROWL_MAX_SESSIONS` bounds how many logical sessions may exist at once;
exceeding it returns a deterministic caller-safe error. `session_ttl_minutes` on
`sessions.create` or a request sets an expiry that is enforced lazily and refreshed on
each use; `sessions.list` omits expired sessions.

## Concurrency

* `PROWL_MAX_CONCURRENCY` bounds how many browser operations run at once: a fetch, a
  `browser.open`, and a `cookies.list` each take a slot. It is also the capacity at which an open
  interactive tab may be taken over, so open tabs and in-flight operations count together against
  it (default `1`: this stack intentionally owns one persistent trust profile and one free browser
  session).
* Each logical session has its own lock, so per-session work is serialized.
* Anonymous (session-less) requests are serialized against each other within their egress.
* Each request runs under a deadline derived from `maxTimeout` plus a small cleanup slack;
  every fetch closes its tab group on every path, including cancellation.

## Egress / proxy

One browser and one profile have exactly one egress. `PROWL_PROXY_URL` configures the
process-wide egress, which is named `default` and is passed to CloakBrowser as
`--proxy-server` (the URL is never logged).

`PROWL_EGRESSES` adds more egresses as a comma separated `name=url` list, for example
`PROWL_EGRESSES=decodo=socks5://127.0.0.1:10001,warp=socks5://127.0.0.1:10002`. Each named
egress owns its own browser process, profile directory, and profile archive, so it keeps
its own warm trust, cookies, and clearance. Names are limited to letters, digits, dot,
dash, and underscore, because they become path components, and `default` is reserved for
`PROWL_PROXY_URL`.

A request selects an egress with `proxy`:

* `proxy.url` keeps its original meaning: it is accepted only when it equals a configured
egress URL, which is what an existing FlareSolverr client sends.
* `proxy.name` selects one configured egress by name.

Anything else is rejected deterministically, and an unlisted URL or an unknown name is
reported without echoing a URL. A named egress browser starts on first use and is shut down
after `PROWL_EGRESS_IDLE_SECONDS` without work (default `300`), so an egress nothing is using
costs no memory. A logical session is bound to the egress of its first use, so a request
naming a different egress for the same session fails instead of silently mixing the two.

Proxy credentials are never included in responses or logs. For authenticated upstreams, point
the egress at a local credential-injecting proxy. A proxy URL that embeds credentials
(`user:pass@`) is rejected, so a configured or request proxy is always an authenticated-free
hop; put the credential injection at that local hop instead.

## HTTP API (FlareSolverr v1 subset)

`POST /v1` with a JSON body:

| cmd | fields |
| --- | --- |
| `request.get` | `url`, `maxTimeout`, `session`, `session_ttl_minutes`, `headers`, `headerScope`, `cookies`, `returnOnlyCookies`, `proxy` |
| `request.post` | as `request.get` except `headerScope`, plus `postData` (string, or object sent as JSON) |
| `sessions.create` | optional `session` name, optional `session_ttl_minutes` |
| `sessions.list` | none |
| `sessions.destroy` | `session` |
| `browser.open` | `url`, optional `maxTimeout`, `session`, `cookies` (installed before the navigation), optional `newTab` (a second tab for a url already open), `proxy` |
| `browser.close` | optional `tab`; without it every interactive tab is closed |
| `browser.list` | optional `tab`; refreshing only the named tab keeps it alive |
| `cookies.list` | optional `url` (only the cookies the browser would send there), `proxy` |

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

## Interactive browser tabs

A fetch closes its tab group when it returns, so nothing stays on screen. An interactive tab is
the other case: it is opened, navigated, and left open, so a person can watch and drive it on
the headed browser's X display, for example over VNC.

`browser.open` returns the tab's `id`, `url`, `title` and `status`. The egress is never part of
a response, so a reply cannot disclose a proxy url.

Opening a url that is already open on that egress does not open a second tab: the existing tab is
handed back unchanged with `reused: true` and the message `Tab reused`, and only its idle
countdown is refreshed. That is what stops a caller whose own request timed out from leaving a
tab nobody can close by asking again. `newTab: true` overrides the reuse and forces a second tab
for the same url.

An interactive tab runs in the egress's own browser, and therefore shares that egress's profile
directory and profile archive with its fetches. That is deliberate: a Cloudflare clearance
earned by a person clicking through a challenge is bound to the address that solved it, so it
has to be the clearance the fetches then use. While a tab is open the egress is held, so the
idle pool cannot shut that browser down underneath it.

Tabs share the profile, not the work. A fetch still runs in its own transient tab group, so an
open tab does not block fetches and a fetch never closes an open tab. Anonymous `browser.open`
calls serialize per egress, like anonymous fetches, because they touch the one profile.

A tab is kept until it is closed. Set `PROWL_INTERACTIVE_IDLE_SECONDS` to a positive number of
seconds to close one that has gone untouched for that long; `browser.list` refreshes the
countdown only for the `tab` it names, so a client displaying one tab keeps that tab open
without holding every other forgotten tab alive. Unset or `0` disables the timeout.

A caller that loses track of its tabs cannot crowd out the fetch path. When open tabs and
in-flight operations together reach `PROWL_MAX_CONCURRENCY` and a fetch or a new tab is about to
run, the least recently used tab is taken over first, but never the only one: a lone tab is always
left alone because it is the one most likely being watched. `browser.list` refreshes the position
of the named `tab`, so a client displaying a tab protects it from takeover. This is on by default
(`PROWL_STEAL_LEAST_RECENT=true`); set `PROWL_STEAL_LEAST_RECENT=0` to keep every tab until its
idle timeout or an explicit `browser.close`.

```json
{ "cmd": "browser.open", "url": "https://example.com/", "proxy": { "name": "decodo" } }
```

```json
{
  "status": "ok",
  "message": "Tab opened",
  "tab": { "id": "tab-1", "url": "https://example.com/", "title": "Example Domain", "status": 200 },
  "reused": false
}
```

`browser.close` reports the ids it closed, and closing a tab that is not open is a no-op, so the
command is safe to retry.

## Window size and display

The browser renders into a headed window on the container's X display. By default it launches at a
default screen size of `1920x980`, which the page is laid out for and which may be larger than the
display the window is shown on. `PROWL_WINDOW_SIZE` sets the launch size explicitly as
`WIDTHxHEIGHT`, for example `1600x900`.

The configured size is not only the window. The page viewport and the screen size the page reports
are set to the same value, so the window, the viewport, and the reported screen agree, and a page
lays out at the size that is actually shown instead of wider than the window it is rendered in.
That alignment is the whole effect: it makes the sizes agree rather than disguising any of them,
and it is not an anti-fingerprinting measure. Leaving it unset keeps the default size.

The window is always placed at the origin, so the window manager fits it to the screen rather than
Chromium restoring a placement saved for a larger display, which would leave the window's right and
bottom edges out of reach.

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
because containers do not grant the user namespaces its sandbox needs. Compose builds
locally from this public source with the tag `prowl:local`. Release builds also publish an
authenticated, private image at `ghcr.io/imxeren/prowl`; no public Prowl image is published.

Prowl's image helper optionally improves fingerprint fidelity with a trusted Windows font
archive while keeping it outside the source context and Git history:

```bash
# Public-source build without private fonts
bash .github/scripts/build-image.sh --load --tag prowl:local

# Local archive (resources/fonts.zip is also accepted)
bash .github/scripts/build-image.sh \
  --font-archive /private/path/fonts.zip \
  --load --tag prowl:local

# Or fetch fonts.zip from a private Git LFS repository
PROWL_FONT_GITHUB_TOKEN=github_pat_... \
PROWL_FONT_REPOSITORY=owner/font-assets \
bash .github/scripts/build-image.sh --load --tag prowl:local
```

`PROWL_FONT_REPOSITORY` defaults to Prowl's configured font repository;
`PROWL_FONT_REF` and `PROWL_FONT_ARCHIVE_PATH` select another revision or path. Temporary
checkout and font material are removed after every build. If neither a local archive nor a
font token is supplied, the same helper builds successfully with the Dockerfile's empty
font context. Advanced callers can still provide a directory containing `fonts.zip` as
Docker's read-only `windows_fonts` named build context. BuildKit secrets are not used
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

## Extensions and managed policy

Both are deployment configuration rather than per-request input, because the browser and its
persistent profile are shared by every request. Both are read at launch, so a change needs a
restart.

`PROWL_EXTENSIONS_DIR` points at a directory of unpacked extensions. Each immediate
subdirectory holding a `manifest.json` is loaded with `--load-extension` and
`--disable-extensions-except`, so the browser runs exactly that set. A subdirectory without a
manifest, or with one that will not parse, is skipped with a warning rather than failing the
launch. An absent or empty directory launches exactly as it did before the option existed.

Two things are worth stating plainly. A modern Chromium no longer runs Manifest V2 extensions,
so an ad blocker has to be a Manifest V3 build: uBlock Origin Lite rather than the original
uBlock Origin. And an extension runs inside the pages it applies to, so it is visible to them
and is one more thing that distinguishes this browser from a stock one, which matters for a
browser whose value is passing bot checks.

`PROWL_POLICY_DIR` points at a directory of managed policy JSON files. Chromium reads managed
policy from a system directory whose path depends on how the build is branded: a Chromium build
reads `/etc/chromium/policies/managed` and a Chrome-branded build reads
`/etc/opt/chrome/policies/managed`. A build cannot be asked which it is, so Prowl writes every
policy file into each of those directories that it can, and a file under a path the build does
not read is inert. When the container runs unprivileged it cannot create them, and the operator
mounts the policy directory at one of those paths instead; Prowl says which directories it could
not write.

Policies set what policies can set. For an unpacked extension that is who may run it, which
hosts it may reach, and whether it is pinned to the toolbar, through `ExtensionSettings`:

```json
{
  "ExtensionSettings": {
    "*": {
      "toolbar_pin": "force_pinned",
      "runtime_allowed_hosts": ["https://*"],
      "runtime_blocked_hosts": []
    }
  }
}
```

What policy does **not** cover is a permission a user normally grants by clicking, such as
uBlock Origin Lite's "Allow User Scripts". That value lives in the profile rather than in policy,
and Prowl does not seed it: the mechanism is a profile key the extension system owns
(`extensions.settings.<id>.granted_permissions`), the id of an unpacked extension is derived from
its path, and no browser launch was available to prove that writing it has the intended effect.
Rather than ship a guess, the option is left out and the limit is documented here. What does work
is that a click is a one-time cost: Prowl owns its profile and packs it on shutdown, so a toggle
granted once inside the browser survives every restart, so a deployment that needs a preset
toggle should treat the click as the supported path for now.

## Releases

- `main` produces stable GitHub Releases and immutable `ghcr.io/imxeren/prowl:<version>`
  image tags, with `latest` moving to the newest stable image.
- `dev` produces prerelease GitHub Releases on the `dev` channel and immutable versioned
  image tags, with `dev` moving to the newest prerelease image.
- Release assets are the audited Python wheel and source archive.
- Each published image is a multi-platform index covering `linux/amd64` and `linux/arm64`.
  Both platforms are built natively on their own runner, `amd64` by the release job and
  `arm64` on an arm64 runner that merges its manifest into the tags the release created,
  so no emulation is involved. A failure there leaves the `amd64` image published and
  unchanged rather than replacing it with something incomplete.
- The GHCR package is private and requires authorization; it is not a public distribution
  channel. Retention keeps the newest two stable and three prerelease image versions.
  A multi-platform image is one tagged index plus one untagged manifest per platform, so
  retention resolves the platform manifests of every image it keeps and never removes one
  a retained image still references; only manifests whose parent it removed are pruned, and
  a resolution failure leaves untagged manifests in place.
- The image is published with a personal access token, never `GITHUB_TOKEN`. A
  `GITHUB_TOKEN` push links the package to the workflow repository, and a package linked to a
  public repository is created public. A token push creates an unlinked package, and an
  unlinked package is private, so the published image is private by construction.
  `PROWL_GHCR_TOKEN` needs `write:packages`, and `delete:packages` as well if retention is to
  prune old versions.
- The guards around that are checks, not the mechanism. Before anything is uploaded, an
  existing package must report private, and the same is verified after the push. A package
  that does not exist yet is allowed through, because the token push creates it unlinked and
  private, so the first release needs no manual bootstrap.

Prowl is not published to a public package or container registry. Install from source, a
GitHub Release asset, or a pinned Git revision, build locally, or authenticate to the
private image when access has been granted.

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
