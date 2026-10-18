# Live CAPTCHA tests

These cases contact public provider demos through Prowl's browser path with temporary profiles, normal TLS verification and bounded deadlines. The default offline suite skips the entire module. Token values are not logged.

```powershell
$env:PROWL_E2E='1'; uv run pytest tests/e2e -q -s
```

To record the selected tab, set `PROWL_E2E_RECORD=1`. Videos are saved to the ignored `tests/e2e-records/` directory. Without this flag, the browser does not record video or create that directory. Recordings may contain sensitive page content; do not commit or share them.

Select a case with `-k`, such as `-k altcha`. hCaptcha and reCAPTCHA v2 skip unless `PROWL_VISION_VLM_ENDPOINT` and `PROWL_VISION_VLM_MODEL` are set. Use `PROWL_VISION_VLM_API_KEY` only if the endpoint requires it. GeeTest needs no VLM; v4 uses Pillow from the optional `vision` extra.

For an endpoint that requires a custom header, set `PROWL_VISION_VLM_HEADERS` to a JSON name/value object. The exact value `$uuid` generates a fresh UUID per request; other values are forwarded unchanged. Supply the header name yourself, for example:

```powershell
$headers = @{}
$headers[(Read-Host 'Required VLM header name')] = '$uuid'
$env:PROWL_VISION_VLM_HEADERS = ConvertTo-Json -Compress $headers
```

Use `[string](Get-Random)` instead of `'$uuid'` for one value reused across the run. The Friendly playground case selects One-click mode; it does not test the default Smart mode. The Cloudflare interstitial case also requires an operator-authorized `PROWL_E2E_INTERSTITIAL_URL` that serves an actual challenge.

Provider failures are test failures, not automatic retries. A fresh widget response does not prove that the application's server accepted it. Controlled fixtures under `tests/integration/` test mechanics without contacting providers.
