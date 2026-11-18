"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/layout_frame.py

Admit and place measured sequence geometry on an explicit physical canvas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

from ..config import SequenceLayout, Style
from ..core import LayoutError, Record, RenderingError
from .layout import LayoutContext, compute_layout, translate_layout

_LEGACY_KEYS = (
    "cell_width_scale",
    "fixed_content_top_extent_px",
    "fixed_content_bottom_extent_px",
    "fixed_content_radius_px",
    "video_extra_bottom_padding_px",
)


def _optional_meta_float(record: Record, key: str) -> float | None:
    raw = record.meta.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderingError(f"record.meta.{key} must be numeric when set") from exc


def prepare_sequence_layout(record: Record, style: Style, request: SequenceLayout | None) -> LayoutContext:
    if request is None:
        scale = _optional_meta_float(record, "cell_width_scale")
        scale = 1.0 if scale is None else scale
        if not math.isfinite(scale) or scale <= 0:
            raise RenderingError("record.meta.cell_width_scale must be finite and > 0")
        extra = _optional_meta_float(record, "video_extra_bottom_padding_px")
        extra = 0.0 if extra is None else extra
        if not math.isfinite(extra) or extra < 0:
            raise RenderingError("record.meta.video_extra_bottom_padding_px must be finite and >= 0")
        return compute_layout(
            record,
            style,
            cell_width_scale=scale,
            fixed_content_top_extent_px=_optional_meta_float(record, "fixed_content_top_extent_px"),
            fixed_content_bottom_extent_px=_optional_meta_float(record, "fixed_content_bottom_extent_px"),
            fixed_content_radius_px=_optional_meta_float(record, "fixed_content_radius_px"),
            extra_bottom_padding_px=extra,
        )
    if not isinstance(request, SequenceLayout):
        raise LayoutError("layout must be a SequenceLayout; use SequenceLayout.from_mapping for mappings")
    conflicts = [key for key in _LEGACY_KEYS if record.meta.get(key) is not None]
    if conflicts:
        raise LayoutError(
            f"record.meta fields {conflicts} compete with explicit layout; migrate them into SequenceLayout"
        )
    if style.figure_scale != 1:
        raise LayoutError("explicit layout requires style.figure_scale=1 to preserve point units")
    unit = style.dpi / 72.0
    for name, value in request.to_mapping().items():
        values = value if isinstance(value, list) else (value,)
        if any(not math.isfinite(item * unit) for item in values):
            raise LayoutError(f"layout.{name} cannot be represented at {style.dpi:g} dpi")
    natural = compute_layout(
        record,
        style,
        base_pitch_pt=request.base_pitch_pt,
        fixed_content_top_extent_px=None if request.top_extent_pt is None else request.top_extent_pt * unit,
        fixed_content_bottom_extent_px=None if request.bottom_extent_pt is None else request.bottom_extent_pt * unit,
        extra_bottom_padding_px=(request.extra_bottom_padding_pt or 0.0) * unit,
    )
    return _place_canvas(natural, request, unit=unit)


def _place_canvas(natural: LayoutContext, request: SequenceLayout, *, unit: float) -> LayoutContext:
    width, height = natural.width, natural.height
    if request.canvas_size_pt is not None:
        width, height = (value * unit for value in request.canvas_size_pt)
    if not all(math.isfinite(value) for value in (natural.width, natural.height, width, height)):
        raise LayoutError("layout produced non-finite canvas bounds; choose smaller physical dimensions")
    dx, dy = 0.0, height - natural.height
    if request.sequence_origin_pt is not None:
        x, y = request.sequence_origin_pt
        dx = x * unit - natural.x_left
        dy = height - y * unit - natural.y_forward
    bounds = (dx, height - dy - natural.height, dx + natural.width, height - dy)
    tolerance = 1e-7
    if min(bounds[0], bounds[1]) < -tolerance or bounds[2] > width + tolerance or bounds[3] > height + tolerance:
        raise LayoutError(
            f"layout canvas {width / unit:g} x {height / unit:g} pt cannot contain measured bounds "
            f"{tuple(round(value / unit, 4) for value in bounds)} pt at the requested sequence origin; "
            "increase the canvas or move the origin; type and pitch will not be shrunk"
        )
    return translate_layout(natural, dx=dx, dy=dy, width=width, height=height)
