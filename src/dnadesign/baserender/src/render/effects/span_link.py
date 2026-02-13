"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/span_link.py

Generic span-link effect drawer connecting feature ids or explicit spans.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ...core import Effect, Record, RenderingError, Span
from ..layout import LayoutContext
from ..palette import Palette


def _text_px_width(text: str, family: str, size_pt: int, dpi: int) -> float:
    prop = FontProperties(family=family, size=size_pt)
    bbox = TextPath((0, 0), text, prop=prop).get_extents()
    return bbox.width / 72.0 * dpi


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
        if not isinstance(from_raw, dict) or not isinstance(to_raw, dict):
            raise RenderingError("span_link span targets must be mappings")
        from_span = Span(start=int(from_raw["start"]), end=int(from_raw["end"]), strand=from_raw.get("strand"))
        to_span = Span(start=int(to_raw["start"]), end=int(to_raw["end"]), strand=to_raw.get("strand"))
        x1 = _span_mid_x(from_span, layout)
        x2 = _span_mid_x(to_span, layout)
        return (x1, x2) if x1 <= x2 else (x2, x1)

    raise RenderingError("span_link target must provide from/to feature ids or from/to spans")


def draw_span_link(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
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
        center = layout.y_forward + layout.feature_track_base_offset + track * layout.feature_track_step
        y = center + layout.kmer_box_height * 0.5
    else:
        center = layout.y_reverse - layout.feature_track_base_offset - track * layout.feature_track_step
        y = center - layout.kmer_box_height * 0.5

    inner_margin_bp = effect.params.get("inner_margin_bp", style.span_link_inner_margin_bp)
    try:
        inner_margin_bp = float(inner_margin_bp)
    except Exception as exc:
        raise RenderingError("span_link params.inner_margin_bp must be numeric") from exc
    if inner_margin_bp < 0:
        raise RenderingError("span_link params.inner_margin_bp must be >= 0")
    inner_margin_px = inner_margin_bp * layout.cw

    x1 = x1 + inner_margin_px
    x2 = x2 - inner_margin_px
    if x2 <= x1:
        raise RenderingError("span_link collapsed geometry after applying inner_margin_bp")

    label = str(effect.params.get("label", "")).strip()
    color = "#9CA3AF"
    base_fs = max(6, style.font_size_label - 2)

    if label:
        label_w = _text_px_width(label, style.font_label, base_fs, style.dpi)
        avail = max(4.0, x2 - x1)
        if label_w + 12.0 > 0.85 * avail:
            scale = (0.85 * avail) / max(1.0, label_w)
            fs = max(6, int(base_fs * min(1.0, scale)))
            label_w = _text_px_width(label, style.font_label, fs, style.dpi)
        else:
            fs = base_fs
        gap = min(avail * 0.9, label_w + 12.0)
        mid = (x1 + x2) / 2.0
        left_end = mid - gap / 2.0
        right_start = mid + gap / 2.0

        ax.plot([x1, left_end], [y, y], color=color, lw=1.1, zorder=5)
        ax.plot([right_start, x2], [y, y], color=color, lw=1.1, zorder=5)
        ax.text(mid, y, label, ha="center", va="center", fontsize=fs, family=style.font_label, color=color, zorder=6)
    else:
        ax.plot([x1, x2], [y, y], color=color, lw=1.1, zorder=5)

    tick = 6.0
    ax.plot([x1, x1], [y - tick / 2, y + tick / 2], color=color, lw=1.1, zorder=5)
    ax.plot([x2, x2], [y - tick / 2, y + tick / 2], color=color, lw=1.1, zorder=5)
