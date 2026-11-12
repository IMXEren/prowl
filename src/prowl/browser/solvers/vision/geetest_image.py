"""Bounded local matching of GeeTest v3 and v4 slider cut-outs."""

from __future__ import annotations

from dataclasses import dataclass

_MIN_CONTRAST = 45.0
_MIN_SEPARATION = 20.0
_ALPHA_THRESHOLD = 128
_MIN_SIDE = 8
_MIN_ROW_VARIATION = 2
_OUTSIDE_RANGE = 12


@dataclass(frozen=True, slots=True)
class RgbaImage:
    """Bounded, row-major, 8-bit RGBA pixels from one challenge image."""

    width: int
    height: int
    pixels: bytes


def _luminance(image: RgbaImage, x: int, y: int) -> int:
    offset = (y * image.width + x) * 4
    pixels = image.pixels
    return (299 * pixels[offset] + 587 * pixels[offset + 1] + 114 * pixels[offset + 2]) // 1000


def _inside(piece: RgbaImage) -> list[tuple[int, int]]:
    """Sample the opaque shape, rejecting a flat opaque rectangle."""
    row_widths = [
        sum(piece.pixels[(y * piece.width + x) * 4 + 3] >= _ALPHA_THRESHOLD for x in range(piece.width))
        for y in range(1, piece.height - 1)
    ]
    if not row_widths or max(row_widths) - min(row_widths) < _MIN_ROW_VARIATION:
        return []
    return [
        (x, y)
        for y in range(5, piece.height - 5, 2)
        for x in range(5, piece.width - 5, 2)
        if piece.pixels[(y * piece.width + x) * 4 + 3] >= _ALPHA_THRESHOLD
    ]


def _darkness(background: RgbaImage, left: int, top: int, piece: RgbaImage, inside: list[tuple[int, int]]) -> float:
    dark = sum(_luminance(background, left + x, top + y) for x, y in inside) / len(inside)
    neighbors = [
        _luminance(background, x, top + y)
        for y in range(2, piece.height - 2, 2)
        for x in (
            *range(max(0, left - _OUTSIDE_RANGE), max(0, left - 3), 2),
            *range(
                min(background.width, left + piece.width + 3),
                min(background.width, left + piece.width + _OUTSIDE_RANGE),
                2,
            ),
        )
    ]
    return sum(neighbors) / len(neighbors) - dark if neighbors else 0.0


def match_geetest_piece(background: RgbaImage, piece: RgbaImage, top: int) -> int | None:
    """Locate a unique dark cut-out, or return ``None`` if it is ambiguous."""
    if (
        background.width < 2 * piece.width
        or piece.width < _MIN_SIDE
        or piece.height < _MIN_SIDE
        or top < 0
        or top + piece.height >= background.height
        or len(background.pixels) != background.width * background.height * 4
        or len(piece.pixels) != piece.width * piece.height * 4
    ):
        return None
    inside = _inside(piece)
    if not inside:
        return None
    scores = [
        (_darkness(background, left, top, piece, inside), left)
        for left in range(piece.width // 2, background.width - piece.width)
    ]
    if not scores:
        return None
    best_score, best_x = max(scores)
    next_score = max((score for score, x in scores if abs(x - best_x) > piece.width // 2), default=float("-inf"))
    if best_score < _MIN_CONTRAST or best_score - next_score < _MIN_SEPARATION:
        return None
    return best_x
