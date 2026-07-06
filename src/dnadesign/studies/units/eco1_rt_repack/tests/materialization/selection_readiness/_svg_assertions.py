"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_svg_assertions.py

SVG geometry assertions for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re


def assert_heatmap_cells_are_square(svg_text: str, *, row_count: int, column_count: int) -> None:
    expected_ratio = column_count / row_count
    assert any(
        width > 80.0 and height > 80.0 and abs((width / height) - expected_ratio) <= 0.03
        for _x, _y, width, height in _svg_clip_rects(svg_text)
    )


def assert_svg_has_square_panel(svg_text: str) -> None:
    assert any(
        width > 100.0 and height > 100.0 and abs(width - height) <= 1.0
        for _x, _y, width, height in _svg_clip_rects(svg_text)
    )


def _svg_clip_rects(svg_text: str) -> list[tuple[float, float, float, float]]:
    return [
        tuple(float(value) for value in match)
        for match in re.findall(
            r'<clipPath id="[^"]+">\s*<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"',
            svg_text,
        )
    ]
