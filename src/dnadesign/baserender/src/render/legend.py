"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/legend.py

Plan and paint bottom legends independently of sequence-row orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Sequence

import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

from ..config import Style
from ..core import LayoutError, Record, RenderingError
from .fonts import measure_literal_text, resolve_font
from .layout import LayoutContext
from .palette import Palette, darken_rgb

_ROW_HEIGHT_EM = 1.4  # Full font box plus clearance, in physical point units.


def legend_entries_for_record(record: Record) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    labels = dict(record.display.tag_labels)
    excluded = {
        str(tag).strip()
        for tag in record.meta.get("legend_exclude_tags", ())
        if isinstance(tag, str) and str(tag).strip()
    }

    for feature in record.features:
        for tag in feature.tags:
            if tag in seen or tag in excluded:
                continue
            seen.add(tag)
            entries.append((tag, labels.get(tag, tag)))

    return entries


@dataclass(frozen=True)
class PlannedLegendEntry:
    tag: str
    label: str
    font_size_pt: float
    font_path: str
    font_family: str
    color: tuple[float, float, float]
    patch_bounds: tuple[float, float, float, float]
    text_origin: tuple[float, float]
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class _LegendRows:
    font_size: float
    patch_w: float
    patch_h: float
    gap_patch_text: float
    gap_x: float
    row_height: float
    row_gap: float
    widths: tuple[float, ...]
    left_insets: tuple[float, ...]
    rows: tuple[tuple[int, ...], ...]

    @property
    def height(self) -> float:
        return len(self.rows) * self.row_height + (len(self.rows) - 1) * self.row_gap


def _wrap_rows(widths: Sequence[float], gap: float, available: float) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    current: list[int] = []
    used = 0.0
    for index, width in enumerate(widths):
        if current and used + gap + width > available:
            rows.append(tuple(current))
            current, used = [], 0.0
        used += (gap if current else 0.0) + width
        current.append(index)
    if current:
        rows.append(tuple(current))
    return tuple(rows)


def _content_bottom(layout: LayoutContext, reserved_boxes: Sequence[tuple[float, float, float, float]]) -> float:
    return min(
        layout.y_forward - layout.sequence_extent_down,
        layout.y_reverse - layout.sequence_extent_down,
        *(item.y - item.h / 2 for item in layout.placements),
        *layout.motif_logo_y0_by_effect.values(),
        *(box[1] for box in reserved_boxes),
    )


def plan_bottom_legend(
    entries: Sequence[tuple[str, str]],
    style: Style,
    layout: LayoutContext,
    palette: Palette,
    *,
    preserve_type: bool = False,
    reserved_boxes: Sequence[tuple[float, float, float, float]] = (),
) -> tuple[PlannedLegendEntry, ...]:
    """Measure a complete footer before drawing; no hidden overflow fallback."""
    if not entries:
        return ()
    if not mcolors.is_color_like(style.color_sequence):
        raise RenderingError("bottom legend color_sequence must be a valid color")
    try:
        colors = tuple(palette.color_for(tag) for tag, _label in entries)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderingError("bottom legend received invalid palette evidence") from exc
    for _tag, label in entries:
        if any(unicodedata.category(char) in {"Cc", "Zl", "Zp"} and char != "\n" for char in label):
            raise RenderingError("bottom legend labels must be literal text without control characters")
    scale = float(style.figure_scale)
    uniform = bool(style.uniform_display_font_size)
    base_size = float(style.display_font_size() if uniform else style.legend_font_size)
    font = resolve_font(style.font_label)
    unit = style.dpi / 72.0 / scale
    side_pad = max(6.0, min(12.0, float(style.padding_x) * 0.35)) if uniform else float(style.padding_x)
    available = layout.width - 2 * side_pad
    minimum_y = float(style.legend_pad_px)
    gap_to_content = 0.0 if style.legend_content_gap_px is None else float(style.legend_content_gap_px) / scale
    ceiling = min(layout.height, _content_bottom(layout, reserved_boxes) - gap_to_content)
    safe_height = ceiling - minimum_y
    declared_height = float(style.legend_height_px)
    # Candidates taller than the available surface cannot fit. Exclude them
    # before font measurement rather than scanning an unbounded requested size.
    max_fitted_size = min(base_size, max(0.0, safe_height / unit / _ROW_HEIGHT_EM))
    sizes = (base_size,) if preserve_type or uniform else range(math.floor(max_fitted_size), 5, -1)

    def candidate(size: float, geometry_scale: float, *, wrap: bool, height_limit: float) -> _LegendRows | None:
        if not math.isfinite(size * unit * _ROW_HEIGHT_EM) or size * unit * _ROW_HEIGHT_EM > height_limit:
            return None
        font_scale = size / base_size
        if preserve_type:
            patch_w, patch_h = float(style.legend_patch_w) / scale, float(style.legend_patch_h) / scale
            text_gap = float(style.legend_gap_patch_text) / scale
        else:
            patch_w = max(8.0, float(style.legend_patch_w) * (1.0 if uniform else 0.55)) / scale
            patch_w = max(patch_w, float(style.legend_patch_w) * font_scale * geometry_scale / scale)
            patch_h = max(6.0, float(style.legend_patch_h) * (1.0 if uniform else 0.55)) / scale
            patch_h = max(patch_h, float(style.legend_patch_h) * font_scale * geometry_scale / scale)
            min_text_gap = max(12.0, float(style.legend_gap_patch_text) * 0.6) if uniform else 2.0
            text_gap = max(min_text_gap, float(style.legend_gap_patch_text) * min(font_scale, geometry_scale)) / scale
        text_widths, left_insets = [], []
        max_lines = 1
        for _tag, label in entries:
            lines = label.split("\n")
            max_lines = max(max_lines, len(lines))
            metrics = [measure_literal_text(line, font, size) for line in lines if line.strip()]
            left = min(0.0, *(item.left_pt for item in metrics)) if metrics else 0.0
            right = max(0.0, *(max(item.advance_pt, item.right_pt) for item in metrics)) if metrics else 0.0
            left_insets.append(-left * unit)
            text_widths.append((right - left) * unit)
        widths = tuple(patch_w + text_gap + width for width in text_widths)
        if available <= 0 or any(width > available for width in widths):
            return None
        gap = float(style.legend_gap_x) * min(font_scale, geometry_scale) / scale
        minimum_gap = max(24.0, float(style.legend_gap_x) * 0.4) / scale if uniform and not preserve_type else 0.0
        gap = max(minimum_gap, gap)
        if wrap:
            rows = _wrap_rows(widths, gap, available)
        else:
            if len(widths) > 1:
                max_gap = (available - sum(widths)) / (len(widths) - 1)
                if (preserve_type and gap > max_gap) or max_gap < minimum_gap:
                    return None
                gap = min(gap, max_gap)
            if sum(widths) + gap * (len(widths) - 1) > available:
                return None
            rows = (tuple(range(len(entries))),)
        row_height = max(patch_h, size * unit * (_ROW_HEIGHT_EM + 1.2 * (max_lines - 1)))
        row_gap = max(2.0 / scale, min(8.0 / scale, row_height * 0.22)) if len(rows) > 1 else 0.0
        result = _LegendRows(
            size, patch_w, patch_h, text_gap, gap, row_height, row_gap, widths, tuple(left_insets), rows
        )
        return result if math.isfinite(result.height) and result.height <= height_limit else None

    selected = None
    limits = (min(declared_height, safe_height),) if preserve_type else (min(declared_height, safe_height), safe_height)
    for limit in limits:
        for wrap in (False, True):
            for size in sizes:
                geometry_scales = (1.0,) if preserve_type or uniform or wrap else (1.0, 0.92, 0.84, 0.76)
                for geometry_scale in geometry_scales:
                    selected = candidate(size, geometry_scale, wrap=wrap, height_limit=limit)
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    if selected is None:
        policy = "preserving type" if preserve_type or uniform else "after supported font fitting"
        raise LayoutError(
            f"bottom legend at {base_size:g} pt cannot fit its {available:g} px-wide, "
            f"{min(declared_height, safe_height):g} px footer {policy}; "
            "increase canvas width or legend_height_px, move content or explicitly choose another font size"
        )
    y = (
        style.legend_origin_y(total_rows_height=selected.height)
        if style.legend_content_gap_px is None
        else max(minimum_y, ceiling - selected.height)
    )
    if y < 0 or y + selected.height > ceiling + 1e-7:
        raise LayoutError("bottom legend footer overlaps content or exceeds canvas; increase footer clearance")
    result = []
    for row in selected.rows:
        row_width = sum(selected.widths[index] for index in row) + selected.gap_x * (len(row) - 1)
        x = max(side_pad, (layout.width - row_width) / 2) if style.legend_center else side_pad
        for index in row:
            tag, label = entries[index]
            patch_y = y + (selected.row_height - selected.patch_h) / 2
            result.append(
                PlannedLegendEntry(
                    tag,
                    label,
                    selected.font_size,
                    font.path,
                    font.resolved_family,
                    colors[index],
                    (x, patch_y, x + selected.patch_w, patch_y + selected.patch_h),
                    (
                        x + selected.patch_w + selected.gap_patch_text + selected.left_insets[index],
                        y + selected.row_height / 2,
                    ),
                    (x, y, x + selected.widths[index], y + selected.row_height),
                )
            )
            x += selected.widths[index] + selected.gap_x
        y += selected.row_height + selected.row_gap
    return tuple(result)


def draw_bottom_legend(ax, entries: Sequence[PlannedLegendEntry], style: Style) -> None:
    for entry in entries:
        color = entry.color
        x0, y0, x1, y1 = entry.patch_bounds
        ax.add_artist(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle=f"round,pad=0.0,rounding_size={min(2.5, (x1 - x0) / 2, (y1 - y0) / 2)}",
                linewidth=0.7,
                facecolor=color,
                edgecolor=darken_rgb(color, factor=0.76),
                alpha=1.0,
                zorder=10,
                clip_on=False,
            )
        )
        ax.text(
            *entry.text_origin,
            entry.label,
            va="center",
            ha="left",
            fontproperties=FontProperties(fname=entry.font_path, family=entry.font_family, size=entry.font_size_pt),
            parse_math=False,
            usetex=False,
            color=style.color_sequence,
            zorder=10,
            clip_on=False,
        )
