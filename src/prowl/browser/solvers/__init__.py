"""Local challenge solvers for selected browser pages."""

from prowl.browser.solvers.checkbox import solve_hcaptcha_checkbox, solve_recaptcha_checkbox, solve_visible_checkbox
from prowl.browser.solvers.cloudflare import ChallengeSolver, CloudflareSolver, click_embedded_turnstile
from prowl.browser.solvers.dispatch import solve_visible_captcha
from prowl.browser.solvers.pow import solve_altcha, solve_friendly_captcha

__all__ = [
    "ChallengeSolver",
    "CloudflareSolver",
    "click_embedded_turnstile",
    "solve_altcha",
    "solve_friendly_captcha",
    "solve_hcaptcha_checkbox",
    "solve_recaptcha_checkbox",
    "solve_visible_captcha",
    "solve_visible_checkbox",
]
