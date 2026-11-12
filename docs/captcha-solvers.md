# CAPTCHA support

Prowl attempts supported challenges in the selected browser page. A fresh provider response is required for a reported solve; a checked box, changed screen, or navigation is not sufficient. The site's server may still reject the response.

| Provider | Support | Limits |
| --- | --- | --- |
| Cloudflare interstitial | Automatic browser handling | A loaded page is not necessarily a cleared challenge. |
| Turnstile | Native checkbox interaction, with a visible-checkbox fallback | Provider risk checks may withhold a token. |
| ALTCHA | Widget proof of work and response extraction | Expensive puzzles may exceed the request deadline. |
| Friendly Captcha | Widget interaction and response extraction | The live playground case uses One-click mode; default Smart mode has not passed that test. |
| reCAPTCHA v2 | Checkbox and opt-in cloud VLM image adapter | Static 3x3/4x4 and refreshing 3x3 elimination grids have passed live; other layouts and audio are unsupported. Google interstitial outcomes vary. |
| reCAPTCHA v3 | Site JavaScript executes normally | No Prowl solver or `solveCaptcha` token-return path; the site evaluates its score. |
| hCaptcha | Checkbox and opt-in cloud VLM tile/canvas adapter | DOM grids and canvas click/drag challenges have passed live; audio and broad reliability are unsupported. |
| GeeTest Slide v3/v4 | Opt-in local image matching and native slider drag | Both slide demos have passed live; other GeeTest modes and broad reliability are unsupported. |

## Requesting a solve

Use `solveCaptcha` on a browser or auto `request.get`:

```json
{
  "cmd": "request.get",
  "url": "https://example.com/form",
  "mode": "auto",
  "solveCaptcha": true
}
```

Auto selects the browser directly; POST and explicit HTTP requests cannot solve widgets. On success, `solution.captcha_provider` and `solution.captcha_token` contain the selected provider and its fresh response. These fields are absent when no response was obtained. Cloudflare interstitial handling also runs without `solveCaptcha` on the browser path.

Authenticated CDP reservations arm the same Cloudflare solver before caller-driven navigation and run the existing visible-widget dispatcher on the reserved page. Solvers act on that tab only and stop when its reservation closes. Local GeeTest and cloud VLM adapters still require their existing opt-ins; page verification and site acceptance are not guaranteed.

Visual adapters require `PROWL_VISION_ENABLED=true`. hCaptcha and reCAPTCHA v2 image solving also require an enabled and configured cloud VLM. GeeTest needs no VLM or neural model; its v4 image decoder uses Pillow from the optional `vision` extra. See [visual adapters](vision-engine.md) for configuration and supported challenge shapes.

Request solvers stay within the original request deadline; reserved-tab attempts are bounded separately. Solvers do not submit the application's form. Do not log or persist returned tokens. Live tests are opt-in under `tests/e2e/`; controlled browser fixtures under `tests/integration/` do not establish provider acceptance.
