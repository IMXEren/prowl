# CAPTCHA support

Prowl uses browser interactions and the widgets' own proof-of-work implementations. It does not use paid solving services, audio transcription, or image classifiers.

| Provider | Status | Limitations |
| --- | --- | --- |
| Cloudflare interstitials and Turnstile | Native solver with a targeted checkbox fallback | Live success is intermittent. Provider risk checks can reject the interaction or withhold a token. |
| ALTCHA | Widget verification and response extraction | Tested with a signed local challenge. Expensive puzzles can exceed the request timeout. |
| Friendly Captcha | Widget start/button interaction and response extraction | Returned a fresh response on the official v2 playground. The site's server must still accept it. |
| reCAPTCHA v2 | TODO; a limited checkbox attempt exists | No fresh token obtained in public-demo testing. Image grids and audio challenges are unsupported. |
| hCaptcha | TODO; a limited checkbox attempt exists | No fresh token obtained in public-demo testing. Image grids and audio challenges are unsupported. |
| GeeTest | TODO | No solver implemented. |

## Requesting a solve

Cloudflare interstitial handling runs automatically in the browser path. To attempt an embedded Turnstile, reCAPTCHA v2, hCaptcha, ALTCHA, or Friendly Captcha widget, set `solveCaptcha` on `request.get`:

```json
{
  "cmd": "request.get",
  "url": "https://example.com/form",
  "mode": "auto",
  "solveCaptcha": true
}
```

This option requires browser or auto mode. Auto goes directly to the browser rather than trying HTTP first. POST and explicit HTTP mode are rejected.

On success, `solution.captcha_provider` identifies the provider and `solution.captcha_token` contains its response. These fields are omitted when no fresh response is obtained. A checkbox attempt is not a promise that an interactive challenge can be solved.

Turnstile uses the same `solveCaptcha` flag and reports `captcha_provider: "turnstile"`. Prowl first tries the native checkbox click in the selected page's Cloudflare frame. If that target is unavailable, it finds a single visible checkbox in a Cloudflare frame, temporarily widens the frame and clicks the control. It does not guess a click position or use keyboard navigation. A confirmed widget response or a changed nonempty token input is required.

## What counts as success

Solvers work on the selected page within the original request timeout. They refuse ambiguous widget selections and do not return prefilled tokens as newly solved responses. A checked box or a visible "Success" message alone is not enough.

A fresh token confirms that the widget produced a response. It does not confirm login, form submission, or server-side acceptance. Prowl does not submit the form for you.

Treat returned tokens as secrets and avoid logging them.
