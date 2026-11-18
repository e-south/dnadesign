"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/public/typography.py

Measure literal text through the font geometry used by native SVG captions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core.vector import TextMeasurement, finite_tuple, validate_literal_text


def measure_text(text: str, *, family: str, size_pt: float) -> TextMeasurement:
    """Return advance and ink bounds in points, without drawing or allocating a figure.

    Bounds are left/bottom/right/top relative to the baseline, with positive y
    upward. Empty/space-only text has no ink but retains its measured advance.
    Text is literal, single-line and must be covered by the selected font.
    """
    validate_literal_text(text, allow_blank=True)
    finite_tuple((size_pt,), 1, "size_pt", positive=True)
    from ..render.fonts import measure_literal_text, resolve_font

    font = resolve_font(family)
    metrics = measure_literal_text(text, font, size_pt)
    return TextMeasurement(
        text,
        font,
        float(size_pt),
        metrics.advance_pt,
        (metrics.left_pt, metrics.bottom_pt, metrics.right_pt, metrics.top_pt),
    )
