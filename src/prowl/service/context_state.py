"""On-disk storage for the storage state of optional isolated contexts.

Each ``(egress, session)`` binding owns exactly one file under a configured
private root and holds a native Playwright storage state. The root is an
operator-owned private directory: POSIX permissions are requested on a
best-effort basis and inherited Windows ACLs are left untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from prowl.service.errors import CallerSafeError

if TYPE_CHECKING:
    from playwright.async_api import StorageState

MAX_CONTEXT_STATE_BYTES = 64 * 1024 * 1024

_ENVELOPE_VERSION = 1
_READ_FAILED = "persisted context state could not be read"
_UNUSABLE = "persisted context state is not usable"
_SERIALIZE_FAILED = "context state could not be serialized"
_TOO_LARGE = "context state exceeds the persistence size limit"
_WRITE_FAILED = "context state could not be persisted"
_DELETE_FAILED = "persisted context state could not be removed"
_NATIVE_CONTAINERS = ("cookies", "origins")


class ContextStateError(CallerSafeError):
    """Raised when persisted context state cannot be read, written, or removed."""


class ContextStateStore:
    """Store one native storage state per (egress, session) binding."""

    def __init__(self, root: Path) -> None:
        """Record the private runtime directory holding the state files."""
        self._root = Path(root)

    def load(self, egress: str, session_id: str) -> StorageState | None:
        """Return the stored state for the binding, or ``None`` when it has none."""
        binding = _binding_key(egress, session_id)
        path = self._path(binding)
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_CONTEXT_STATE_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError:
            raise ContextStateError(_READ_FAILED) from None
        if len(data) > MAX_CONTEXT_STATE_BYTES:
            raise ContextStateError(_UNUSABLE)
        return _decode_state(data, binding)

    def save(self, egress: str, session_id: str, state: StorageState) -> None:
        """Atomically replace the binding's file with ``state``."""
        binding = _binding_key(egress, session_id)
        path = self._path(binding)
        body = _encode_state(binding, state)
        if len(body) > MAX_CONTEXT_STATE_BYTES:
            raise ContextStateError(_TOO_LARGE)
        temp_path: Path | None = None
        try:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self._root, suffix=".tmp", delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(body)
                handle.flush()
            temp_path.replace(path)
            temp_path = None
        except OSError:
            raise ContextStateError(_WRITE_FAILED) from None
        finally:
            if temp_path is not None:
                with contextlib.suppress(OSError):
                    temp_path.unlink()

    def delete(self, egress: str, session_id: str) -> None:
        """Remove the binding's file; deleting an absent binding is a no-op."""
        try:
            self._path(_binding_key(egress, session_id)).unlink()
        except FileNotFoundError:
            return
        except OSError:
            raise ContextStateError(_DELETE_FAILED) from None

    def _path(self, binding: str) -> Path:
        return self._root / f"{binding}.json"


def _binding_key(egress: str, session_id: str) -> str:
    canonical = json.dumps([egress, session_id], separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encode_state(binding: str, state: StorageState) -> bytes:
    envelope = {"version": _ENVELOPE_VERSION, "binding": binding, "state": state}
    try:
        return json.dumps(envelope, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ContextStateError(_SERIALIZE_FAILED) from None


def _decode_state(data: bytes, binding: str) -> StorageState:
    try:
        envelope = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ContextStateError(_UNUSABLE) from None
    if not isinstance(envelope, dict):
        raise ContextStateError(_UNUSABLE)
    version = envelope.get("version")
    if type(version) is not int or version != _ENVELOPE_VERSION:
        raise ContextStateError(_UNUSABLE)
    if envelope.get("binding") != binding:
        raise ContextStateError(_UNUSABLE)
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ContextStateError(_UNUSABLE)
    for name in _NATIVE_CONTAINERS:
        if name not in state:
            continue
        entries = state[name]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ContextStateError(_UNUSABLE)
    # The checks above bound the JSON boundary; validating the native schema
    # further is the browser's job when the state is restored.
    return cast("StorageState", state)
