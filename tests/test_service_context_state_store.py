import json
import os
import re
from pathlib import Path

import pytest
from playwright.async_api import StorageState

from prowl.service import context_state as context_state_module
from prowl.service.context_state import ContextStateError, ContextStateStore
from prowl.service.errors import CallerSafeError

_EGRESS = "direct"
_SESSION = "sess-1"


def _native_state() -> StorageState:
    return {
        "cookies": [
            {
                "name": "sid",
                "value": "abc123",
                "domain": "example.test",
                "path": "/",
                "expires": 1893456000.0,
                "httpOnly": True,
                "secure": True,
                "sameSite": "Lax",
            }
        ],
        "origins": [
            {
                "origin": "https://example.test",
                "localStorage": [{"name": "token", "value": "t"}],
            }
        ],
    }


def _envelope(*, version: object, binding: object, state: object) -> bytes:
    return json.dumps({"version": version, "binding": binding, "state": state}).encode("utf-8")


def test_construction_missing_load_and_delete_touch_no_files(tmp_path: Path) -> None:
    root = tmp_path / "runtime" / "context-state"
    store = ContextStateStore(root)

    assert store.load(_EGRESS, _SESSION) is None
    assert store.delete(_EGRESS, _SESSION) is None
    assert not root.exists()
    assert not (tmp_path / "runtime").exists()


def test_native_state_roundtrip_preserves_nested_containers(tmp_path: Path) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    state = _native_state()

    store.save(_EGRESS, _SESSION, state)
    path = next(root.iterdir())
    envelope = json.loads(path.read_bytes())
    database = [{"name": "app-db", "version": 1, "stores": []}]
    envelope["state"]["origins"][0]["indexedDB"] = database
    path.write_bytes(json.dumps(envelope).encode("utf-8"))
    loaded = store.load(_EGRESS, _SESSION)
    assert loaded is not None
    assert loaded is not state
    store.save(_EGRESS, _SESSION, loaded)
    assert json.loads(path.read_bytes())["state"] == envelope["state"]

    replacement = StorageState(cookies=[], origins=[])
    store.save(_EGRESS, _SESSION, replacement)

    assert store.load(_EGRESS, _SESSION) == replacement
    assert len(list(root.iterdir())) == 1


def test_filenames_are_safe_stable_and_distinct(tmp_path: Path) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    tricky = "tenant/../ünïcode'\u2028sess"
    state = _native_state()

    store.save(_EGRESS, tricky, state)
    store.save(_EGRESS, tricky, state)
    store.save("other-egress", tricky, state)
    store.save(_EGRESS, "sess-2", state)

    entries = list(root.iterdir())
    names = sorted(entry.name for entry in entries)
    assert len(names) == 3
    assert all(entry.is_file() for entry in entries)
    assert all(re.fullmatch(r"[0-9a-f]{64}\.json", name) for name in names)
    assert all("/" not in name and "\\" not in name for name in names)
    assert store.load(_EGRESS, tricky) == state
    assert store.load("other-egress", tricky) == state
    assert store.load(_EGRESS, "sess-2") == state


def test_invalid_envelopes_raise_generic_errors(tmp_path: Path) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    store.save(_EGRESS, _SESSION, _native_state())
    path = next(root.iterdir())
    marker = "super-secret-value"

    bad_bodies = [
        b"{not json",
        b"{}",
        b"[]",
        b"\xff",
        b"[" * 1100 + b"]" * 1100,
        _envelope(version=True, binding=path.stem, state={}),
        _envelope(version=1.0, binding=path.stem, state={}),
        _envelope(version=1, binding=path.stem, state={"cookies": None}),
        _envelope(version=1, binding=path.stem, state={"origins": None}),
        _envelope(version=2, binding=path.stem, state={"cookies": [{"value": marker}]}),
        _envelope(version=1, binding="0" * 64, state={"cookies": [{"value": marker}]}),
        _envelope(version=1, binding=path.stem, state=[]),
        _envelope(version=1, binding=path.stem, state={"cookies": marker}),
        _envelope(version=1, binding=path.stem, state={"origins": [marker]}),
    ]
    for body in bad_bodies:
        path.write_bytes(body)
        with pytest.raises(ContextStateError) as error:
            store.load(_EGRESS, _SESSION)
        message = str(error.value)
        assert marker not in message
        assert str(root) not in message
        assert path.name not in message


def test_failed_promotion_keeps_committed_snapshot_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    store.save(_EGRESS, _SESSION, _native_state())
    committed_name = next(root.iterdir()).name
    committed = store.load(_EGRESS, _SESSION)

    def _fail_replace(_source: object, _target: object) -> None:
        msg = "replace failed"
        raise OSError(msg)

    # Path.replace delegates to os.replace, so the same atomic seam is patched.
    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(ContextStateError):
        store.save(_EGRESS, _SESSION, StorageState(cookies=[], origins=[]))
    monkeypatch.undo()

    assert [entry.name for entry in root.iterdir()] == [committed_name]
    assert store.load(_EGRESS, _SESSION) == committed


def test_size_bounds_reject_oversized_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    monkeypatch.setattr(context_state_module, "MAX_CONTEXT_STATE_BYTES", 4096)
    store.save(_EGRESS, _SESSION, _native_state())
    committed = store.load(_EGRESS, _SESSION)
    assert committed is not None

    monkeypatch.setattr(context_state_module, "MAX_CONTEXT_STATE_BYTES", 8)
    with pytest.raises(ContextStateError):
        store.load(_EGRESS, _SESSION)

    oversized = StorageState(
        cookies=[{"name": "n", "value": "v" * 64, "domain": "example.test", "path": "/"}],
        origins=[],
    )
    monkeypatch.setattr(context_state_module, "MAX_CONTEXT_STATE_BYTES", 16)
    with pytest.raises(ContextStateError):
        store.save(_EGRESS, _SESSION, oversized)

    monkeypatch.setattr(context_state_module, "MAX_CONTEXT_STATE_BYTES", 4096)
    assert store.load(_EGRESS, _SESSION) == committed


def test_delete_is_idempotent_and_leaves_other_bindings(tmp_path: Path) -> None:
    root = tmp_path / "state"
    store = ContextStateStore(root)
    state = _native_state()
    store.save(_EGRESS, _SESSION, state)
    store.save(_EGRESS, "sess-2", state)
    store.save("other-egress", _SESSION, state)

    store.delete(_EGRESS, _SESSION)
    store.delete(_EGRESS, _SESSION)

    assert store.load(_EGRESS, _SESSION) is None
    assert store.load(_EGRESS, "sess-2") == state
    assert store.load("other-egress", _SESSION) == state
    assert len(list(root.iterdir())) == 2


def test_context_state_error_is_caller_safe() -> None:
    assert issubclass(ContextStateError, CallerSafeError)


def test_unpaired_unicode_surrogates_roundtrip(tmp_path: Path) -> None:
    store = ContextStateStore(tmp_path / "state")
    state = _native_state()
    state.get("origins", [])[0].get("localStorage", [])[0]["value"] = "value-\ud800"
    store.save(_EGRESS, "session-\ud800", state)
    assert store.load(_EGRESS, "session-\ud800") == state


def test_io_error_does_not_expose_private_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ContextStateStore(tmp_path / "private-state")

    def fail_open(*_args: object, **_kwargs: object) -> None:
        message = "private path or state"
        raise PermissionError(message)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(ContextStateError) as error:
        store.load(_EGRESS, _SESSION)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
