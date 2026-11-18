"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/svg_caption.py

Measure editable SVG caption origins and ink bounds before drawing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import LayoutError
from ..core.vector import IllustrationCaption, ResolvedFont
from .fonts import measure_literal_text


@dataclass(frozen=True)
class CaptionGeometry:
    origin_pt: tuple[float, float]
    bounds_pt: tuple[float, float, float, float]


def measure_caption(
    caption: IllustrationCaption | None,
    font: ResolvedFont | None,
    illustration_bounds_pt: tuple[float, float, float, float],
    *,
    asset_id: str,
    px_to_pt: float,
    canvas_size_pt: tuple[float, float],
) -> CaptionGeometry | None:
    if caption is None or not caption.visible:
        return None
    if font is None:
        raise LayoutError(f"asset {asset_id!r} caption has no admitted font")
    x0, y0, x1, y1 = illustration_bounds_pt
    cx = x0 + caption.anchor_fraction[0] * (x1 - x0) + caption.offset_px[0] * px_to_pt
    cy = y0 + caption.anchor_fraction[1] * (y1 - y0) + caption.offset_px[1] * px_to_pt
    metrics = measure_literal_text(caption.text, font, caption.size_pt)
    anchor_fraction = {"start": 0.0, "middle": 0.5, "end": 1.0}[caption.text_anchor]
    left = cx - metrics.advance_pt * anchor_fraction
    bounds = (left + metrics.left_pt, cy - metrics.top_pt, left + metrics.right_pt, cy - metrics.bottom_pt)
    width, height = canvas_size_pt
    if (
        not all(math.isfinite(value) for value in (*bounds, cx, cy))
        or min(bounds[:2]) < 0
        or bounds[2] > width
        or bounds[3] > height
    ):
        raise LayoutError(
            f"asset {asset_id!r} caption bounds {bounds} pt exceed the {width:g} x {height:g} pt canvas; "
            "move the caption or increase the canvas; text will not be shrunk or clipped"
        )
    return CaptionGeometry((cx, cy), bounds)
