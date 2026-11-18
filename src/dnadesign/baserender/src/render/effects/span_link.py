"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/effects/span_link.py

Generic span-link effect drawer connecting feature ids or explicit spans.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import matplotlib.colors as mcolors

from ...core import Effect, LayoutError, Record, RenderingError, Span
from ..layout import LayoutContext, measure_text_width_px
from ..palette import Palette


def _span_mid_x(span: Span, layout: LayoutContext) -> float:
    return layout.x_left + ((span.start + span.end) / 2.0) * layout.cw


def _resolve_endpoints(
    effect: Effect, layout: LayoutContext, feature_boxes: dict[str, tuple[float, float, float, float]]
) -> tuple[float, float]:
    target = effect.target

    if "from_feature_id" in target and "to_feature_id" in target:
        left = feature_boxes.get(str(target["from_feature_id"]))
        right = feature_boxes.get(str(target["to_feature_id"]))
        if left is None or right is None:
            raise RenderingError("span_link references unknown feature id(s)")
        from_x = left[2]
        to_x = right[0]
        if from_x > to_x:
            from_x, to_x = to_x, from_x
        return from_x, to_x

    if "from_span" in target and "to_span" in target:
        from_raw = target["from_span"]
        to_raw = target["to_span"]
        if not isinstance(from_raw, Mapping) or not isinstance(to_raw, Mapping):
            raise RenderingError("span_link span targets must be mappings")
        from_span = Span(start=int(from_raw["start"]), end=int(from_raw["end"]), strand=from_raw.get("strand"))
        to_span = Span(start=int(to_raw["start"]), end=int(to_raw["end"]), strand=to_raw.get("strand"))
        x1 = _span_mid_x(from_span, layout)
        x2 = _span_mid_x(to_span, layout)
        return (x1, x2) if x1 <= x2 else (x2, x1)

    raise RenderingError("span_link target must provide from/to feature ids or from/to spans")


@dataclass(frozen=True)
class SpanLinkGeometry:
    x1: float
    x2: float
    y: float
    label: str
    font_size_pt: float | None
    gap: float
    label_bounds: tuple[float, float, float, float] | None
    occupied_boxes: tuple[tuple[float, float, float, float], ...]


def compute_span_link_geometry(
    effect: Effect,
    layout: LayoutContext,
    style,
    feature_boxes: Mapping[str, tuple[float, float, float, float]],
    *,
    preserve_type: bool = False,
    enforce_bounds: bool = True,
) -> SpanLinkGeometry:
    """Resolve and validate span-link geometry without allocating artists."""

    x1, x2 = _resolve_endpoints(effect, layout, feature_boxes)

    lane = str(effect.params.get("lane", "top")).lower()
    if lane not in {"top", "bottom"}:
        raise RenderingError(f"span_link lane must be top|bottom, got {lane!r}")

    track_raw = effect.render.get("track", 0)
    try:
        track = int(track_raw)
    except Exception as exc:
        raise RenderingError("span_link render.track must be int") from exc

    if lane == "top":
        y = layout.y_forward + layout.feature_track_base_offset_up + track * layout.feature_track_step
    else:
        y = layout.y_reverse - layout.feature_track_base_offset_down - track * layout.feature_track_step

    inner_margin_bp = effect.params.get("inner_margin_bp", style.span_link_inner_margin_bp)
    try:
        inner_margin_bp = float(inner_margin_bp)
    except Exception as exc:
        raise RenderingError("span_link params.inner_margin_bp must be numeric") from exc
    if not math.isfinite(inner_margin_bp) or inner_margin_bp < 0:
        raise RenderingError("span_link params.inner_margin_bp must be finite and >= 0")

    inner_margin_px = inner_margin_bp * layout.cw
    x1 += inner_margin_px
    x2 -= inner_margin_px
    if x2 <= x1:
        raise RenderingError("span_link collapsed geometry after applying inner_margin_bp")

    label = str(effect.params.get("label", "")).strip()
    shrink = effect.params.get("shrink_label_to_fit", True)
    if not isinstance(shrink, bool):
        raise RenderingError("span_link params.shrink_label_to_fit must be bool")
    if preserve_type and effect.params.get("shrink_label_to_fit") is True:
        raise RenderingError(
            "native or explicit-layout output rejects span_link.params.shrink_label_to_fit=True; "
            "remove it or set False and choose the font size explicitly"
        )
    font_size = None
    gap = 0.0
    label_bounds = None
    mid = (x1 + x2) / 2.0
    available = x2 - x1
    text_height = 0.0
    if label:
        try:
            base_size = float(style.font_size_span_link_label or max(6, style.font_size_seq))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RenderingError("span_link font size must be finite and >= 6 pt") from exc
        if not math.isfinite(base_size) or base_size < 6:
            raise RenderingError("span_link font size must be finite and >= 6 pt")
        font_size = base_size
        label_width = measure_text_width_px(label, style.font_label, font_size, style.dpi)
        if label_width + 12.0 > 0.85 * available:
            if preserve_type:
                raise LayoutError(
                    f"span_link label {label!r} at {font_size:g} pt requires "
                    f"{(label_width + 12.0) / 0.85:g} px but the span allows {available:g} px; "
                    "increase pitch, reduce the inner margin or explicitly choose another label/font size"
                )
            if shrink:
                scale = (0.85 * max(4.0, available)) / max(1.0, label_width)
                font_size = float(max(6, int(base_size * min(1.0, scale))))
                label_width = measure_text_width_px(label, style.font_label, font_size, style.dpi)
        text_height = max(8.0, font_size / 72.0 * style.dpi)
        gap = min(available * 0.9, label_width + 12.0)
        label_bounds = (
            mid - (label_width + 12.0) / 2,
            y - text_height / 2,
            mid + (label_width + 12.0) / 2,
            y + text_height / 2,
        )
        if preserve_type and (
            min(label_bounds[:2]) < 0 or label_bounds[2] > layout.width or label_bounds[3] > layout.height
        ):
            raise LayoutError(
                f"span_link label {label!r} bounds {label_bounds} px exceed the canvas; increase layout clearances"
            )
    line_half_height = max(2.0, text_height * 0.33)
    tick_half_height = 4.0
    tick_half_width = 1.2
    occupied = [
        (x1 - tick_half_width, y - tick_half_height, x1 + tick_half_width, y + tick_half_height),
        (x2 - tick_half_width, y - tick_half_height, x2 + tick_half_width, y + tick_half_height),
    ]
    if label:
        left_end, right_start = mid - gap / 2, mid + gap / 2
        if left_end > x1:
            occupied.append((x1, y - line_half_height, left_end, y + line_half_height))
        if x2 > right_start:
            occupied.append((right_start, y - line_half_height, x2, y + line_half_height))
        occupied.append(label_bounds)
    else:
        occupied.append((x1, y - line_half_height, x2, y + line_half_height))
    for bounds in occupied if enforce_bounds else ():
        if (
            not all(math.isfinite(value) for value in bounds)
            or min(bounds[:2]) < 0
            or bounds[2] > layout.width
            or bounds[3] > layout.height
        ):
            raise LayoutError(
                f"span_link bounds {tuple(float(value) for value in bounds)} px exceed the "
                f"{layout.width:g} x {layout.height:g} px canvas; "
                "increase layout clearances or revise the declared track"
            )
    return SpanLinkGeometry(x1, x2, y, label, font_size, gap, label_bounds, tuple(occupied))


def validate_span_link(
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    _ = (record, palette)
    compute_span_link_geometry(effect, layout, style, feature_boxes)
    for key in ("color", "label_color"):
        if key in effect.params and not mcolors.is_color_like(effect.params[key]):
            raise RenderingError(f"span_link params.{key} must be a valid color")


def draw_span_link(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
    *,
    geometry: SpanLinkGeometry | None = None,
) -> None:
    if geometry is None:
        geometry = compute_span_link_geometry(effect, layout, style, feature_boxes)
    x1, x2, y, label = geometry.x1, geometry.x2, geometry.y, geometry.label
    color = effect.params.get("color", style.span_link_color)
    label_color = effect.params.get("label_color", style.span_link_label_color or color)
    line_width = max(0.8, float(getattr(style, "span_link_line_width", 1.1)))
    tick_line_width = max(0.8, float(getattr(style, "span_link_tick_line_width", line_width)))
    if label:
        mid = (x1 + x2) / 2.0
        left_end = mid - geometry.gap / 2.0
        right_start = mid + geometry.gap / 2.0

        ax.plot([x1, left_end], [y, y], color=color, lw=line_width, zorder=5)
        ax.plot([right_start, x2], [y, y], color=color, lw=line_width, zorder=5)
        ax.text(
            mid,
            y,
            label,
            ha="center",
            va="center",
            fontsize=geometry.font_size_pt * float(style.figure_scale),
            family=style.font_label,
            color=label_color,
            zorder=6,
        )
    else:
        ax.plot([x1, x2], [y, y], color=color, lw=line_width, zorder=5)

    tick = 6.0
    ax.plot([x1, x1], [y - tick / 2, y + tick / 2], color=color, lw=tick_line_width, zorder=5)
    ax.plot([x2, x2], [y - tick / 2, y + tick / 2], color=color, lw=tick_line_width, zorder=5)
