"""GeeTest slide matcher and token/geometry guards, without provider traffic."""

from __future__ import annotations

import hashlib
import unittest
from typing import cast

from prowl.browser.solvers.vision.geetest import _fresh_validation, _positions, _Slide, _slide
from prowl.browser.solvers.vision.geetest_image import RgbaImage, match_geetest_piece


class GeeTestImageTests(unittest.TestCase):
    def setUp(self) -> None:
        source = bytearray(hashlib.shake_256(b"synthetic-geetest-pattern").digest(120 * 65 * 4))
        for index in range(3, len(source), 4):
            source[index] = 255
        self.top, self.target = 18, 74
        width, height = 19, 21
        piece = bytearray()
        for y in range(height):
            for x in range(width):
                index = ((self.top + y) * 120 + self.target + x) * 4
                piece.extend(source[index : index + 4])
                if y in (0, height - 1) or (x in (0, width - 1) and not 7 <= y <= 13):
                    piece[-1] = 0
        self.piece = RgbaImage(width, height, bytes(piece))
        for y in range(height):
            for x in range(width):
                if piece[(y * width + x) * 4 + 3] >= 128:
                    index = ((self.top + y) * 120 + self.target + x) * 4
                    source[index : index + 3] = bytes(channel // 3 for channel in source[index : index + 3])
        self.background = RgbaImage(120, 65, bytes(source))

    def test_finds_unique_gap(self) -> None:
        match = match_geetest_piece(self.background, self.piece, self.top)
        self.assertIsNotNone(match)
        self.assertLessEqual(abs(cast("int", match) - self.target), 3)

    def test_rejects_repeated_texture(self) -> None:
        source = bytearray(self.background.pixels)
        for y in range(self.piece.height):
            for x in range(self.piece.width):
                at = ((self.top + y) * 120 + 37 + x) * 4
                original = ((self.top + y) * 120 + self.target + x) * 4
                source[at : at + 4] = source[original : original + 4]
        duplicated = RgbaImage(120, 65, bytes(source))
        self.assertIsNone(match_geetest_piece(duplicated, self.piece, self.top))

    def test_invalid_shape_and_top_refused(self) -> None:
        self.assertIsNone(match_geetest_piece(self.background, self.piece, -1))
        self.assertIsNone(match_geetest_piece(self.background, self.piece, 55))
        self.assertIsNone(match_geetest_piece(self.background, RgbaImage(19, 21, b"x"), self.top))

    def test_flat_piece_refused(self) -> None:
        piece = RgbaImage(19, 21, bytes([128, 128, 128, 255]) * (19 * 21))
        self.assertIsNone(match_geetest_piece(self.background, piece, self.top))

    def test_no_dark_cutout_refused(self) -> None:
        plain = RgbaImage(120, 65, bytes([190, 190, 190, 255]) * (120 * 65))
        self.assertIsNone(match_geetest_piece(plain, self.piece, self.top))

    def test_moderate_contrast_cutout_is_usable(self) -> None:
        pixels = bytearray(bytes([200, 200, 200, 255]) * (120 * 65))
        for y in range(self.piece.height):
            for x in range(self.piece.width):
                if self.piece.pixels[(y * self.piece.width + x) * 4 + 3] >= 128:
                    index = ((self.top + y) * 120 + self.target + x) * 4
                    pixels[index : index + 3] = b"\x89\x89\x89"
        result = match_geetest_piece(RgbaImage(120, 65, bytes(pixels)), self.piece, self.top)
        self.assertIsNotNone(result)
        self.assertLessEqual(abs(cast("int", result) - self.target), 3)


class GeeTestAdapterGuardsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.background = RgbaImage(260, 160, b"\x00" * (260 * 160 * 4))
        self.piece = RgbaImage(47, 40, b"\x00" * (47 * 40 * 4))
        self.slide = _Slide(
            self.background,
            self.piece,
            0,
            99,
            (724.6, 248.1, 258.76, 159.24),
            (708.65, 410.3, 65.65, 65.65),
            (724.6, 422.26, 258.73, 37.81),
        )

    def test_target_maps_canvas_pixels_to_css_slider_distance(self) -> None:
        result = _positions(self.slide, 181)
        self.assertIsNotNone(result)
        start, end = cast("tuple[tuple[float, float], tuple[float, float]]", result)
        self.assertAlmostEqual(end[0] - start[0], 181 * 258.76 / 260)
        self.assertEqual(start[1], end[1])

    def test_out_of_rail_target_refused(self) -> None:
        self.assertIsNone(_positions(self.slide, 250))

    def test_token_requires_three_provider_fields_and_fresh_pair(self) -> None:
        before = ("challenge", "", "")
        self.assertEqual(_fresh_validation(("challenge", "actual-token", "actual-code"), before), "actual-token")
        self.assertIsNone(_fresh_validation(("challenge", "actual-token", ""), before))
        self.assertIsNone(_fresh_validation(("challenge", "old", "code"), ("challenge", "old", "code")))
        self.assertIsNone(_fresh_validation(("", "actual-token", "actual-code"), before))

    def test_canvas_payload_size_and_values_bounded(self) -> None:
        self.assertIsNone(
            _slide({"width": 99999999, "height": 160, "piece_width": 47, "piece_height": 40, "left": 0, "top": 99})
        )
        self.assertIsNone(
            _slide(
                {
                    "width": 260,
                    "height": 160,
                    "piece_width": 47,
                    "piece_height": 40,
                    "left": 0,
                    "top": 99,
                    "background": [],
                    "piece": [],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
