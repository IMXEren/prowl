# Visual CAPTCHA adapters

Image solving is opt-in per service and request. Set `PROWL_VISION_ENABLED=true` and use
`solveCaptcha=true` on a browser/auto GET. The default remains off.

hCaptcha and reCAPTCHA v2 images also require `PROWL_VISION_VLM_ENABLED=true`,
`PROWL_VISION_VLM_ENDPOINT`, and `PROWL_VISION_VLM_MODEL`. Put credentials in
`PROWL_VISION_VLM_API_KEY`, not tracked files. `PROWL_VISION_VLM_HEADERS` accepts a JSON object
of operator-supplied header names and values. A literal value is forwarded unchanged; the exact
value `$uuid` generates a new UUID per request. No provider header name is built in.
`PROWL_VISION_VLM_REASONING_EFFORT` applies only to Responses endpoints. Remote endpoints require
verified HTTPS; HTTP is accepted only for an exact loopback address.

GeeTest uses a local pixel matcher, not a neural model or cloud upload. v3 reads canvas layers;
v4 decodes provider images with Pillow from the optional `vision` extra. Prowl ships no ONNX
inference backend or NumPy dependency.

The adapters operate in the selected page's context under the original deadline. Playwright
reads challenge frames and images; PyDoll drives clicks and drags. The dispatcher rejects mixed
providers, stale puzzles and ambiguous targets. A fresh provider response, not visual appearance,
determines success. Only fixed outcome statuses and reasons are exposed through the service.

## Supported challenges

- **hCaptcha:** 3x3 DOM tiles and bounded canvas click/drag challenges. Unique page breadcrumbs
  track progression, up to eight pages. An observed rejection allows at most two extra provider
  pages. The adapter requires a fresh `h-captcha-response`; unreadable transitions stop.
- **reCAPTCHA v2:** static 3x3/4x4 grids and refreshing 3x3 elimination puzzles. Selected tiles
  must refresh before the next assessment; Verify is pressed only when no matches remain. An
  observed rejection allows at most two extra provider pages. Only a fresh
  `g-recaptcha-response` counts; navigation is not success.
- **GeeTest Slide:** ambiguous gap matches or stale layouts stop. v3 requires fresh validation
  fields; v4 requires a successful same-frame `/verify` response with a `pass_token`. That token
  alone is not the complete off-browser site-validation payload.

These shapes have returned real provider responses on selected public demos, but results vary.
Audio, other layouts and reliable success on arbitrary sites are not established. Tests under
`tests/e2e/` contact live providers; `tests/integration/` uses controlled fixtures.
