"""Shared bounded vision client inputs and categorized errors."""

from __future__ import annotations

from typing import Final, Literal

type TaskKind = Literal["classification", "detection"]
TASK_KINDS: Final = ("classification", "detection")
MAX_INPUT_PIXELS: Final = 1_048_576


class VisionError(Exception):
    """An error identified only by a fixed, safe category."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
