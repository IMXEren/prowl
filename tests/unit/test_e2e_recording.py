"""Live CAPTCHA recording remains optional and local to the test tree."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch


def test_recording_requires_explicit_opt_in() -> None:
    with patch.dict(os.environ, {"PROWL_E2E": "1"}):
        suite = importlib.import_module("tests.e2e.test_captcha_providers")

    with patch.dict(os.environ, {"PROWL_E2E_RECORD": ""}):
        assert suite._recording_path(suite.CASES[0]) is None
    with patch.dict(os.environ, {"PROWL_E2E_RECORD": "true"}):
        assert suite._recording_path(suite.CASES[0]) is None
    source = suite.__file__
    assert source is not None
    records = Path(source).resolve().parents[1] / "e2e-records"
    existed = records.exists()
    with patch.dict(os.environ, {"PROWL_E2E_RECORD": "1"}):
        target = suite._recording_path(suite.CASES[0])
    assert target is not None
    assert target.parent == records
    assert target.name.startswith("turnstile-")
    assert target.suffix == ".webm"
    assert records.exists() is existed
