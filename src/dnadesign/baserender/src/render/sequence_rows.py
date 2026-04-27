"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/sequence_rows.py

Sequence-row renderer for Record v1 with kmer features, effects, overlays, and legend.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Affine2D

from ..config import Style
from ..core import Record, RenderingError
from .effects.motif_logo import MotifLogoGeometry, compute_motif_logo_geometry
from .effects.registry import draw_effect
from .layout import LayoutContext, comp, compute_layout, measure_text_width_px
from .palette import Palette

_NEAR_FEATURE_ANNOTATION_SOURCES = frozenset(
    {
        "densegen_promoter",
        "usr_genbank",
        "usr_genbank_promoter",
    }
)


def _feature_source(feature) -> str:
    return str(feature.attrs.get("source", "")).strip().lower()


def _uses_near_feature_annotation_label(feature) -> bool:
    if _feature_source(feature) not in _NEAR_FEATURE_ANNOTATION_SOURCES:
        return False
    return str(feature.attrs.get("display_label", "")).strip() != ""


@dataclass(frozen=True)
class SequenceRowsRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        record = record.validate()
        show_two = bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"})
        fixed_content_top_extent_px: float | None = None
        fixed_content_bottom_extent_px: float | None = None
        fixed_content_radius_px: float | None = None
        extra_bottom_padding_px: float = 0.0
        if isinstance(record.meta, Mapping):
            raw_top_extent = record.meta.get("fixed_content_top_extent_px")
            if raw_top_extent is not None:
                try:
                    fixed_content_top_extent_px = float(raw_top_extent)
                except Exception as exc:
                    raise RenderingError("record.meta.fixed_content_top_extent_px must be numeric when set") from exc
            raw_bottom_extent = record.meta.get("fixed_content_bottom_extent_px")
            if raw_bottom_extent is not None:
                try:
                    fixed_content_bottom_extent_px = float(raw_bottom_extent)
                except Exception as exc:
                    raise RenderingError("record.meta.fixed_content_bottom_extent_px must be numeric when set") from exc
            raw_radius = record.meta.get("fixed_content_radius_px")
            if raw_radius is not None:
                try:
                    fixed_content_radius_px = float(raw_radius)
                except Exception as exc:
                    raise RenderingError("record.meta.fixed_content_radius_px must be numeric when set") from exc
            raw_extra_bottom_padding = record.meta.get("video_extra_bottom_padding_px")
            if raw_extra_bottom_padding is not None:
                try:
                    extra_bottom_padding_px = float(raw_extra_bottom_padding)
                except Exception as exc:
                    raise RenderingError("record.meta.video_extra_bottom_padding_px must be numeric when set") from exc
                if not math.isfinite(extra_bottom_padding_px) or extra_bottom_padding_px < 0.0:
                    raise RenderingError("record.meta.video_extra_bottom_padding_px must be finite and >= 0")
        layout = compute_layout(
            record,
            style,
            fixed_content_top_extent_px=fixed_content_top_extent_px,
            fixed_content_bottom_extent_px=fixed_content_bottom_extent_px,
            fixed_content_radius_px=fixed_content_radius_px,
            extra_bottom_padding_px=extra_bottom_padding_px,
        )

        motif_geometries: list[MotifLogoGeometry] = []
        for effect_index, effect in enumerate(record.effects):
            if effect.kind != "motif_logo":
                continue
            motif_geometries.append(
                compute_motif_logo_geometry(
                    record=record,
                    effect_index=effect_index,
                    layout=layout,
                    style=style,
                    feature_boxes={},
                )
            )

        tone_fwd: Sequence[float] | None = None
        tone_rev: Sequence[float] | None = None
        explicit_complement_sequence: str | None = None
        base_highlights: Mapping[str, Sequence[int]] = {}
        base_highlight_color: Mapping[str, str] = {}
        dim_base_indices: Mapping[str, Sequence[int]] = {}
        span_backdrops: Sequence[Mapping[str, object]] = ()
        if isinstance(record.meta, Mapping):
            raw_complement = record.meta.get("complement_sequence")
            if isinstance(raw_complement, str) and len(raw_complement) == len(record.sequence):
                explicit_complement_sequence = raw_complement
            raw_highlights = record.meta.get("base_highlights")
            if isinstance(raw_highlights, Mapping):
                base_highlights = raw_highlights
            raw_highlight_color = record.meta.get("base_highlight_color")
            if isinstance(raw_highlight_color, Mapping):
                base_highlight_color = raw_highlight_color
            raw_dim_indices = record.meta.get("dim_base_indices")
            if isinstance(raw_dim_indices, Mapping):
                dim_base_indices = raw_dim_indices
            raw_span_backdrops = record.meta.get("span_backdrops")
            if isinstance(raw_span_backdrops, Sequence) and not isinstance(raw_span_backdrops, (str, bytes)):
                span_backdrops = raw_span_backdrops
        if bool(style.sequence.bold_consensus_bases) and motif_geometries:
            tone_fwd, tone_rev = _sequence_tone_strengths(
                record,
                motif_geometries,
                q_low=float(style.sequence.tone_quantile_low),
                q_high=float(style.sequence.tone_quantile_high),
                complement_sequence=explicit_complement_sequence,
            )

        fig_scale = float(style.figure_scale)
        has_trajectory_panel = record.display.trajectory_panel is not None
        sequence_width_px = float(layout.width) * fig_scale
        sequence_height_px = float(layout.height) * fig_scale
        if has_trajectory_panel:
            panel_side_px = min(
                max(sequence_height_px * 0.46, 96.0),
                max(88.0, sequence_height_px * 0.64),
            )
            panel_left_pad_px = max(80.0, panel_side_px * 0.55)
            panel_gap_px = max(20.0, sequence_height_px * 0.08)
            sequence_right_pad_px = max(6.0, sequence_height_px * 0.03)
            total_width_px = (
                panel_left_pad_px + panel_side_px + panel_gap_px + sequence_width_px + sequence_right_pad_px
            )
            fig = plt.figure(
                figsize=(total_width_px / style.dpi, sequence_height_px / style.dpi),
                dpi=style.dpi,
            )
            panel_y0_px = (sequence_height_px - panel_side_px) / 2.0
            panel_y0_px = min(panel_y0_px + max(6.0, sequence_height_px * 0.03), sequence_height_px - panel_side_px)
            panel_ax = fig.add_axes(
                [
                    panel_left_pad_px / total_width_px,
                    panel_y0_px / sequence_height_px,
                    panel_side_px / total_width_px,
                    panel_side_px / sequence_height_px,
                ],
                zorder=5.0,
            )
            panel_ax.set_box_aspect(1.0)
            sequence_x0_px = panel_left_pad_px + panel_side_px + panel_gap_px
            ax = fig.add_axes([sequence_x0_px / total_width_px, 0.0, sequence_width_px / total_width_px, 1.0])
        else:
            fig = plt.figure(
                figsize=(sequence_width_px / style.dpi, sequence_height_px / style.dpi),
                dpi=style.dpi,
            )
            panel_ax = None
            ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax._dnadesign_record_meta = record.meta

        x0 = layout.x_left
        _draw_span_backdrops(ax, layout, span_backdrops, show_two=show_two)
        _draw_sequence(
            ax,
            record.sequence,
            x0,
            layout.y_forward,
            layout.cw,
            style,
            "5'",
            "3'",
            tone_strengths=tone_fwd,
            row_id="fwd",
            highlight_indices=base_highlights.get("primary"),
            highlight_color=base_highlight_color.get("primary"),
            dim_indices=dim_base_indices.get("primary"),
        )
        if show_two:
            _draw_sequence(
                ax,
                explicit_complement_sequence or comp(record.sequence),
                x0,
                layout.y_reverse,
                layout.cw,
                style,
                "3'",
                "5'",
                tone_strengths=tone_rev,
                row_id="rev",
                highlight_indices=base_highlights.get("complement"),
                highlight_color=base_highlight_color.get("complement"),
                dim_indices=dim_base_indices.get("complement"),
            )
            _draw_connectors(ax, len(record.sequence), x0, layout.cw, layout, style)
        _draw_row_labels(ax, record, layout, style)
        _draw_segment_labels(ax, record, layout, style)
        if bool(style.show_coordinate_ticks):
            _draw_coordinate_ticks(ax, record, layout, style)

        feature_boxes = dict(layout.feature_boxes)
        feature_box_pad = float(style.kmer.pad_x_px)
        near_annotation_feature_boxes: list[tuple[float, float, float, float]] = []
        for placement in layout.placements:
            feature = record.features[placement.feature_index]
            if not _uses_near_feature_annotation_label(feature):
                continue
            x0 = placement.x - feature_box_pad
            x1 = placement.x + placement.w + feature_box_pad
            near_annotation_feature_boxes.append(
                (
                    x0,
                    placement.y - placement.h / 2.0,
                    x1,
                    placement.y + placement.h / 2.0,
                )
            )

        # Draw feature boxes first.
        for placement in layout.placements:
            feature = record.features[placement.feature_index]
            tag = str(feature.attrs.get("style_token", "")) or (feature.tags[0] if feature.tags else feature.kind)
            color = palette.color_for(tag)
            label = feature.label or ""
            if not placement.above and feature.kind != "interval_annotation":
                label = label[::-1]
            placement_box = (
                placement.x - feature_box_pad,
                placement.y - placement.h / 2.0,
                placement.x + placement.w + feature_box_pad,
                placement.y + placement.h / 2.0,
            )
            draw_label = True
            if not _uses_near_feature_annotation_label(feature) and any(
                _boxes_overlap(placement_box, annotation_box) for annotation_box in near_annotation_feature_boxes
            ):
                draw_label = False
            if (
                feature.kind == "interval_annotation"
                and str(feature.attrs.get("shape", "")).strip().lower() == "underline"
            ):
                _draw_interval_underline(
                    ax,
                    placement.x,
                    placement.y,
                    placement.w,
                    placement.h,
                    label,
                    color,
                    style,
                    draw_label=True,
                )
            else:
                _draw_feature_box(
                    ax,
                    placement.x,
                    placement.y,
                    placement.w,
                    placement.h,
                    label,
                    color,
                    style,
                    cw=layout.cw,
                    ch=layout.ch,
                    draw_label=draw_label,
                )

            feature_boxes[placement.feature_id] = (
                placement.x,
                placement.y - placement.h / 2.0,
                placement.x + placement.w,
                placement.y + placement.h / 2.0,
            )

        # Draw effects with strict unknown-kind failure from registry.
        for effect in record.effects:
            draw_effect(ax, effect, record, layout, style, palette, feature_boxes)

        _draw_fixed_element_annotations(ax, record, layout, palette, style)

        legend_mode = str(style.legend_mode).lower()
        if style.legend and legend_mode == "inline":
            _draw_inline_feature_labels(ax, record, layout, palette, style)

        _draw_motif_scale_bar(ax, motif_geometries, layout, style)

        if style.legend and legend_mode == "bottom":
            from .legend import legend_entries_for_record

            _draw_legend(ax, legend_entries_for_record(record), palette, style, layout.width)

        if record.display.overlay_text:
            _draw_overlay(ax, layout, style, record.display.overlay_text)
        if panel_ax is not None and record.display.trajectory_panel is not None:
            _draw_trajectory_panel(panel_ax, record.display.trajectory_panel, style)

        ax.set_xlim(0, layout.width)
        ax.set_ylim(0, layout.height)
        return fig


@lru_cache(maxsize=2048)
def _mono_text_path(char: str, font_family: str, size_pt: int, weight: str = "normal") -> TextPath:
    prop = FontProperties(family=font_family, size=size_pt, weight=weight)
    return TextPath((0, 0), char, prop=prop, usetex=False)


@lru_cache(maxsize=128)
def _mono_ag_mid_px(font_family: str, size_pt: int, dpi: int, weight: str = "normal") -> float:
    prop = FontProperties(family=font_family, size=size_pt, weight=weight)
    px_per_pt = dpi / 72.0
    ag = TextPath((0, 0), "Ag", prop=prop, usetex=False).get_extents()
    return ((ag.y0 + ag.y1) / 2.0) * px_per_pt


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _column_information_bits(row: Sequence[float]) -> float:
    entropy = 0.0
    for prob in row[:4]:
        p = float(prob)
        if p > 0.0:
            entropy += -p * math.log2(p)
    return max(0.0, 2.0 - entropy)


def _row_prob_for_base(row: Sequence[float], base: str) -> float:
    idx_by_base = {"A": 0, "C": 1, "G": 2, "T": 3}
    idx = idx_by_base.get(base)
    if idx is None:
        return 0.0
    return float(row[idx])


def _normalize_tone_scores(
    raw_scores: Sequence[float],
    cover_counts: Sequence[int],
    *,
    q_low: float,
    q_high: float,
) -> tuple[float, ...]:
    if q_low < 0.0 or q_low > 1.0 or q_high < 0.0 or q_high > 1.0 or q_low >= q_high:
        raise RenderingError("sequence tone quantiles must satisfy 0 <= low < high <= 1")

    covered = [float(s) for s, c in zip(raw_scores, cover_counts) if c > 0]
    if not covered:
        return tuple(0.0 for _ in raw_scores)

    lo = _quantile(covered, q_low)
    hi = _quantile(covered, q_high)
    eps = 1e-12

    if hi <= (lo + eps):
        max_score = max(covered)
        if max_score <= eps:
            return tuple(0.0 for _ in raw_scores)
        return tuple(1.0 if c > 0 and s > eps else 0.0 for s, c in zip(raw_scores, cover_counts))

    inv = 1.0 / (hi - lo)
    out: list[float] = []
    for score, covered_count in zip(raw_scores, cover_counts):
        if covered_count <= 0:
            out.append(0.0)
            continue
        norm = (float(score) - lo) * inv
        out.append(max(0.0, min(1.0, norm)))
    return tuple(out)


def _sequence_tone_strengths(
    record: Record,
    motif_geometries: Sequence[MotifLogoGeometry],
    *,
    q_low: float,
    q_high: float,
    complement_sequence: str | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = len(record.sequence)
    seq_fwd = record.sequence.upper()
    seq_rev = (complement_sequence or comp(record.sequence)).upper()
    covered_fwd = [0 for _ in range(n)]
    covered_rev = [0 for _ in range(n)]
    accum_fwd = [0.0 for _ in range(n)]
    accum_rev = [0.0 for _ in range(n)]
    weight_fwd = [0.0 for _ in range(n)]
    weight_rev = [0.0 for _ in range(n)]
    feature_by_id = {feature.id: feature for feature in record.features if feature.id is not None}

    for geometry in motif_geometries:
        feature = feature_by_id.get(geometry.feature_id)
        if feature is None:
            raise RenderingError(
                f"motif_logo target feature not found during sequence tone scoring: {geometry.feature_id!r}"
            )

        for offset, row in enumerate(geometry.matrix):
            pos = geometry.render_start + offset
            if pos < 0 or pos >= n:
                raise RenderingError(
                    f"motif_logo geometry out of sequence bounds while computing sequence tone scores: pos={pos}, n={n}"
                )
            if len(row) < 4:
                raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
            info_weight = max(0.0, min(1.0, _column_information_bits(row) / 2.0))
            if info_weight <= 0.0:
                continue
            if feature.span.strand == "fwd":
                p_fwd = _row_prob_for_base(row, seq_fwd[pos])
                covered_fwd[pos] += 1
                accum_fwd[pos] += info_weight * p_fwd
                weight_fwd[pos] += info_weight
            elif feature.span.strand == "rev":
                p_rev = _row_prob_for_base(row, seq_rev[pos])
                covered_rev[pos] += 1
                accum_rev[pos] += info_weight * p_rev
                weight_rev[pos] += info_weight
            else:
                raise RenderingError(f"Unknown feature strand while scoring sequence tone: {feature.span.strand!r}")

    raw_fwd = [(accum_fwd[i] / weight_fwd[i]) if weight_fwd[i] > 0.0 else 0.0 for i in range(n)]
    raw_rev = [(accum_rev[i] / weight_rev[i]) if weight_rev[i] > 0.0 else 0.0 for i in range(n)]
    tone_fwd = _normalize_tone_scores(
        raw_fwd,
        covered_fwd,
        q_low=q_low,
        q_high=q_high,
    )
    tone_rev = _normalize_tone_scores(
        raw_rev,
        covered_rev,
        q_low=q_low,
        q_high=q_high,
    )
    return tone_fwd, tone_rev


def _mix_colors(light_hex: str, dark_hex: str, strength: float) -> tuple[float, float, float]:
    light = mcolors.to_rgb(light_hex)
    dark = mcolors.to_rgb(dark_hex)
    t = max(0.0, min(1.0, float(strength)))
    return (
        light[0] + (dark[0] - light[0]) * t,
        light[1] + (dark[1] - light[1]) * t,
        light[2] + (dark[2] - light[2]) * t,
    )


def _darken_rgb(color: object, *, factor: float) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    scale = min(1.0, max(0.0, float(factor)))
    return (r * scale, g * scale, b * scale)


def _capitalize_first(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    return t[0].upper() + t[1:]


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def _span_link_label_boxes(
    record: Record, layout: LayoutContext, style: Style
) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    feature_boxes = layout.feature_boxes
    for effect in record.effects:
        if effect.kind != "span_link":
            continue

        target = effect.target
        if "from_feature_id" in target and "to_feature_id" in target:
            left = feature_boxes.get(str(target["from_feature_id"]))
            right = feature_boxes.get(str(target["to_feature_id"]))
            if left is None or right is None:
                continue
            x1 = float(left[2])
            x2 = float(right[0])
            if x1 > x2:
                x1, x2 = x2, x1
        elif "from_span" in target and "to_span" in target:
            from_raw = target["from_span"]
            to_raw = target["to_span"]
            if not isinstance(from_raw, dict) or not isinstance(to_raw, dict):
                continue
            x1 = layout.x_left + ((int(from_raw["start"]) + int(from_raw["end"])) / 2.0) * layout.cw
            x2 = layout.x_left + ((int(to_raw["start"]) + int(to_raw["end"])) / 2.0) * layout.cw
            if x1 > x2:
                x1, x2 = x2, x1
        else:
            continue

        lane = str(effect.params.get("lane", "top")).lower()
        if lane not in {"top", "bottom"}:
            continue
        try:
            track = int(effect.render.get("track", 0))
        except Exception:
            continue
        if lane == "top":
            y = layout.y_forward + layout.feature_track_base_offset_up + track * layout.feature_track_step
        else:
            y = layout.y_reverse - layout.feature_track_base_offset_down - track * layout.feature_track_step

        inner_margin_bp = effect.params.get("inner_margin_bp", style.span_link_inner_margin_bp)
        try:
            inner_margin_bp = float(inner_margin_bp)
        except Exception:
            inner_margin_bp = float(style.span_link_inner_margin_bp)
        if inner_margin_bp < 0:
            inner_margin_bp = 0.0
        inner_margin_px = inner_margin_bp * layout.cw
        x1 = x1 + inner_margin_px
        x2 = x2 - inner_margin_px
        if x2 <= x1:
            continue

        base_fs = (
            max(6, int(round(style.display_font_size())))
            if bool(style.uniform_display_font_size)
            else max(6, style.font_size_label - 2)
        )
        avail = max(4.0, x2 - x1)
        label = str(effect.params.get("label", "")).strip()
        fs = base_fs
        text_h = max(8.0, (float(fs) / 72.0) * float(style.dpi))
        line_half_h = max(2.0, text_h * 0.33)
        tick_half_h = 4.0
        tick_half_w = 1.2
        if label == "":
            boxes.append((x1, y - line_half_h, x2, y + line_half_h))
            boxes.append((x1 - tick_half_w, y - tick_half_h, x1 + tick_half_w, y + tick_half_h))
            boxes.append((x2 - tick_half_w, y - tick_half_h, x2 + tick_half_w, y + tick_half_h))
            continue

        label_w = _text_px_width(label, style.font_label, fs, style.dpi)
        if label_w + 12.0 > 0.85 * avail:
            scale = (0.85 * avail) / max(1.0, label_w)
            fs = max(6, int(base_fs * min(1.0, scale)))
            label_w = _text_px_width(label, style.font_label, fs, style.dpi)
            text_h = max(8.0, (float(fs) / 72.0) * float(style.dpi))
            line_half_h = max(2.0, text_h * 0.33)
        gap = min(avail * 0.9, label_w + 12.0)
        mid = (x1 + x2) / 2.0
        left_end = mid - gap / 2.0
        right_start = mid + gap / 2.0
        if left_end > x1:
            boxes.append((x1, y - line_half_h, left_end, y + line_half_h))
        if x2 > right_start:
            boxes.append((right_start, y - line_half_h, x2, y + line_half_h))
        boxes.append((x1 - tick_half_w, y - tick_half_h, x1 + tick_half_w, y + tick_half_h))
        boxes.append((x2 - tick_half_w, y - tick_half_h, x2 + tick_half_w, y + tick_half_h))
        boxes.append((mid - gap / 2.0, y - text_h / 2.0, mid + gap / 2.0, y + text_h / 2.0))
    return boxes


def _compact_fixed_element_annotation_label(raw_label: str) -> str:
    text = str(raw_label).strip()
    lowered = text.lower()
    for token in ("-35 site", "-10 site"):
        idx = lowered.find(token)
        if idx >= 0:
            return text[idx:].strip()
    return text


def _fixed_element_annotation_font_size(style: Style) -> float:
    if bool(style.uniform_display_font_size):
        return style.display_font_size()
    return float(max(6, style.font_size_label, style.font_size_seq))


def _draw_fixed_element_annotations(ax, record: Record, layout: LayoutContext, palette: Palette, style: Style) -> None:
    if not layout.placements:
        return

    labels = dict(record.display.tag_labels)
    margin = max(float(style.legend_inline_margin_cells) * layout.cw, float(style.kmer.pad_x_px) + 4.0)
    x_min = float(style.padding_x)
    x_max = float(layout.width - style.padding_x)

    feature_box_pad = float(style.kmer.pad_x_px)
    occupied_boxes: list[tuple[float, float, float, float]] = []
    for placement in layout.placements:
        occupied_boxes.append(
            (
                placement.x - feature_box_pad,
                placement.y - placement.h / 2.0,
                placement.x + placement.w + feature_box_pad,
                placement.y + placement.h / 2.0,
            )
        )
    occupied_boxes.extend(_span_link_label_boxes(record, layout, style))

    placed_label_boxes: list[tuple[float, float, float, float]] = []

    def _candidate_box(
        *,
        x_anchor: float,
        y_anchor: float,
        ha: str,
        text_w: float,
        text_h: float,
    ) -> tuple[float, float, float, float]:
        if ha == "left":
            x0 = float(x_anchor)
            x1 = float(x_anchor) + float(text_w)
        elif ha == "right":
            x0 = float(x_anchor) - float(text_w)
            x1 = float(x_anchor)
        else:
            x0 = float(x_anchor) - float(text_w) / 2.0
            x1 = float(x_anchor) + float(text_w) / 2.0
        y0 = float(y_anchor) - float(text_h) / 2.0
        y1 = float(y_anchor) + float(text_h) / 2.0
        return (x0, y0, x1, y1)

    placement_by_id = {placement.feature_id: placement for placement in layout.placements}
    force_baseline_label_ids: set[str] = set()

    def _annotation_label_geometry(
        placement,
    ) -> tuple[str, float, float, float, float, float, float, float] | None:
        feature = record.features[placement.feature_index]
        if not _uses_near_feature_annotation_label(feature):
            return None
        tag = feature.tags[0] if feature.tags else feature.kind
        fallback = tag.split(":")[-1] if ":" in tag else tag
        feature_display_label = feature.attrs.get("display_label")
        raw_label = str(feature_display_label or labels.get(tag, fallback))
        text = _compact_fixed_element_annotation_label(raw_label)
        if not text:
            return None
        text_size = _fixed_element_annotation_font_size(style)
        text_w = _text_px_width(text, style.font_label, text_size, style.dpi)
        text_h = max(8.0, (float(text_size) / 72.0) * float(style.dpi))
        center_x = placement.x + placement.w / 2.0
        top_gap = max(4.0, feature_box_pad + text_h * 0.25)
        top_y = placement.y + placement.h / 2.0 + top_gap + text_h / 2.0
        right_x = placement.x + placement.w + margin
        left_x = placement.x - margin
        return text, text_size, text_w, text_h, center_x, top_y, right_x, left_x

    def _candidate_fits(
        bbox: tuple[float, float, float, float],
        *,
        occupied: Sequence[tuple[float, float, float, float]],
    ) -> bool:
        if bbox[0] < x_min or bbox[2] > x_max:
            return False
        if bbox[1] < 0.0 or bbox[3] > float(layout.height):
            return False
        return not any(_boxes_overlap(bbox, occupied_box) for occupied_box in occupied)

    for effect in record.effects:
        if effect.kind != "span_link":
            continue
        from_id = effect.target.get("from_feature_id")
        to_id = effect.target.get("to_feature_id")
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            continue
        from_placement = placement_by_id.get(from_id)
        to_placement = placement_by_id.get(to_id)
        if from_placement is None or to_placement is None:
            continue
        endpoint_bboxes: list[tuple[float, float, float, float]] = []
        can_place_both_above = True
        for endpoint in (from_placement, to_placement):
            geometry = _annotation_label_geometry(endpoint)
            if geometry is None:
                can_place_both_above = False
                break
            _text, _text_size, text_w, text_h, center_x, top_y, _right_x, _left_x = geometry
            bbox = _candidate_box(
                x_anchor=center_x,
                y_anchor=top_y,
                ha="center",
                text_w=text_w,
                text_h=text_h,
            )
            if not _candidate_fits(bbox, occupied=occupied_boxes):
                can_place_both_above = False
                break
            if any(_boxes_overlap(bbox, existing) for existing in endpoint_bboxes):
                can_place_both_above = False
                break
            endpoint_bboxes.append(bbox)
        if not can_place_both_above:
            force_baseline_label_ids.update({from_id, to_id})

    for placement in layout.placements:
        feature = record.features[placement.feature_index]
        if not _uses_near_feature_annotation_label(feature):
            continue

        tag = feature.tags[0] if feature.tags else feature.kind
        geometry = _annotation_label_geometry(placement)
        if geometry is None:
            continue
        text, text_size, text_w, text_h, center_x, top_y, right_x, left_x = geometry

        if placement.feature_id in force_baseline_label_ids:
            candidates = (
                (right_x, placement.y, "left"),
                (left_x, placement.y, "right"),
                (center_x, top_y, "center"),
            )
        else:
            candidates = (
                (center_x, top_y, "center"),
                (right_x, placement.y, "left"),
                (left_x, placement.y, "right"),
            )

        selected: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for x_anchor, y_anchor, ha in candidates:
            bbox = _candidate_box(x_anchor=x_anchor, y_anchor=y_anchor, ha=ha, text_w=text_w, text_h=text_h)
            if bbox[0] < x_min or bbox[2] > x_max:
                continue
            if bbox[1] < 0.0 or bbox[3] > float(layout.height):
                continue
            if any(_boxes_overlap(bbox, occupied) for occupied in occupied_boxes):
                continue
            if any(_boxes_overlap(bbox, occupied) for occupied in placed_label_boxes):
                continue
            selected = (x_anchor, y_anchor, ha, bbox)
            break

        if selected is None:
            continue

        x_anchor, y_anchor, ha, bbox = selected
        annotation_color = _darken_rgb(palette.color_for(tag), factor=0.6)
        ax.text(
            x_anchor,
            y_anchor,
            text,
            ha=ha,
            va="center",
            fontsize=text_size,
            family=style.font_label,
            color=annotation_color,
            zorder=6.2,
            clip_on=False,
        )
        placed_label_boxes.append(bbox)


def _draw_inline_feature_labels(ax, record: Record, layout: LayoutContext, palette: Palette, style: Style) -> None:
    if not layout.placements:
        return

    labels = dict(record.display.tag_labels)
    side_pref = str(style.legend_inline_side).lower()
    margin = float(style.legend_inline_margin_cells) * layout.cw
    box_pad = float(style.kmer.pad_x_px)
    x_min = style.padding_x
    x_max = layout.width - style.padding_x

    text_h = max(8.0, (float(style.legend_font_size) / 72.0) * float(style.dpi))
    lateral_step = max(8.0, box_pad + text_h * 0.75)
    max_lateral_steps = max(4, int(math.ceil((x_max - x_min) / max(1.0, lateral_step))))

    occupied_boxes: list[tuple[float, float, float, float]] = []
    for placement in layout.placements:
        occupied_boxes.append(
            (
                placement.x - box_pad,
                placement.y - placement.h / 2.0,
                placement.x + placement.w + box_pad,
                placement.y + placement.h / 2.0,
            )
        )
    occupied_boxes.extend(_span_link_label_boxes(record, layout, style))
    sequence_x0 = float(layout.x_left)
    sequence_x1 = float(layout.x_left + len(record.sequence) * layout.cw)
    occupied_boxes.append(
        (
            sequence_x0,
            float(layout.y_forward - layout.sequence_extent_down),
            sequence_x1,
            float(layout.y_forward + layout.sequence_extent_up),
        )
    )
    if bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"}):
        occupied_boxes.append(
            (
                sequence_x0,
                float(layout.y_reverse - layout.sequence_extent_down),
                sequence_x1,
                float(layout.y_reverse + layout.sequence_extent_up),
            )
        )

    for effect_index, effect in enumerate(record.effects):
        if effect.kind != "motif_logo":
            continue
        feature_id_raw = effect.target.get("feature_id")
        if not isinstance(feature_id_raw, str) or not feature_id_raw:
            continue
        feature_box = layout.feature_boxes.get(feature_id_raw)
        if feature_box is None:
            continue
        motif_y0 = layout.motif_logo_y0_by_effect.get(effect_index)
        if motif_y0 is None:
            continue
        occupied_boxes.append(
            (
                float(feature_box[0]),
                float(motif_y0),
                float(feature_box[2]),
                float(motif_y0 + layout.motif_logo_height),
            )
        )

    placed_label_boxes: list[tuple[float, float, float, float]] = []

    def _candidate_box(
        *,
        x_anchor: float,
        y_anchor: float,
        ha: str,
        text_w: float,
    ) -> tuple[float, float, float, float]:
        if ha == "left":
            x0 = float(x_anchor)
            x1 = float(x_anchor) + float(text_w)
        elif ha == "right":
            x0 = float(x_anchor) - float(text_w)
            x1 = float(x_anchor)
        else:
            x0 = float(x_anchor) - float(text_w) / 2.0
            x1 = float(x_anchor) + float(text_w) / 2.0
        y0 = float(y_anchor) - text_h / 2.0
        y1 = float(y_anchor) + text_h / 2.0
        return (x0, y0, x1, y1)

    for placement in layout.placements:
        feature = record.features[placement.feature_index]
        source = str(feature.attrs.get("source", "")).strip().lower()
        if source == "densegen_promoter":
            continue
        tag = feature.tags[0] if feature.tags else feature.kind
        fallback = tag.split(":")[-1] if ":" in tag else tag
        raw_label = str(labels.get(tag, fallback))
        text = _capitalize_first(raw_label)
        if not text:
            continue

        text_w = _text_px_width(text, style.font_label, style.legend_font_size, style.dpi)
        x_left = placement.x - margin
        x_right = placement.x + placement.w + margin
        left_room = (x_left - text_w) - x_min
        right_room = x_max - (x_right + text_w)

        if side_pref == "left":
            side_order = ("left", "right")
        elif side_pref == "right":
            side_order = ("right", "left")
        else:
            side_order = ("right", "left") if right_room >= left_room else ("left", "right")

        candidates: list[tuple[float, float, str]] = []
        for step in range(0, max_lateral_steps + 1):
            delta = lateral_step * float(step)
            for side in side_order:
                if side == "right":
                    candidates.append((x_right + delta, placement.y, "left"))
                else:
                    candidates.append((x_left - delta, placement.y, "right"))

        selected: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for x_anchor, y_anchor, ha in candidates:
            bbox = _candidate_box(x_anchor=x_anchor, y_anchor=y_anchor, ha=ha, text_w=text_w)
            if bbox[0] < x_min or bbox[2] > x_max:
                continue
            if bbox[1] < 0.0 or bbox[3] > float(layout.height):
                continue
            if any(_boxes_overlap(bbox, occupied) for occupied in occupied_boxes):
                continue
            if any(_boxes_overlap(bbox, occupied) for occupied in placed_label_boxes):
                continue
            selected = (x_anchor, y_anchor, ha, bbox)
            break
        if selected is None:
            continue
        x_text, y_text, ha, bbox = selected

        ax.text(
            x_text,
            y_text,
            text,
            ha=ha,
            va="center",
            fontsize=style.legend_font_size,
            family=style.font_label,
            color=palette.color_for(tag),
            zorder=6.2,
            clip_on=False,
        )
        placed_label_boxes.append(bbox)


def _actual_content_top(layout: LayoutContext) -> float:
    top = max(
        float(layout.y_forward + layout.sequence_extent_up),
        float(layout.y_reverse + layout.sequence_extent_up),
    )
    for placement in layout.placements:
        top = max(top, float(placement.y + (placement.h / 2.0)))
    for y0 in layout.motif_logo_y0_by_effect.values():
        top = max(top, float(y0 + layout.motif_logo_height))
    return float(top)


def _draw_overlay(ax, layout: LayoutContext, style: Style, text: str) -> None:
    align = str(style.overlay_align).lower()
    if align == "center":
        x = layout.width / 2.0
        ha = "center"
    elif align == "right":
        x = layout.width - style.padding_x
        ha = "right"
    else:
        x = style.padding_x
        ha = "left"
    synthetic_top_pad = max(0.0, float(layout.content_top) - _actual_content_top(layout))
    title_size = (
        style.display_font_size()
        if bool(style.uniform_display_font_size)
        else max(float(style.font_size_label), float(style.font_size_seq))
    )
    title_lines = max(1, len([line for line in str(text).splitlines() if line.strip()]))
    title_line_height = max((title_size / 72.0 * style.dpi) * 1.05, layout.ch * 0.5)
    title_block_height = title_line_height * title_lines
    min_overlay_y = _actual_content_top(layout) + title_block_height + max(4.0, style.font_size_label * 0.25)
    overlay_y = (
        layout.height
        - max(4.0, style.padding_y * 0.5)
        - synthetic_top_pad
        - max(0.0, float(style.overlay_title_gap_reduction_px))
    )
    overlay_y = max(min_overlay_y, overlay_y)
    ax.text(
        x,
        overlay_y,
        text,
        ha=ha,
        va="top",
        fontsize=title_size,
        family=style.font_label,
        color=style.overlay_title_color,
        alpha=0.95,
        zorder=15,
        clip_on=False,
    )


def _window_bboxes_overlap(a, b) -> bool:
    return float(max(a.x0, b.x0)) < float(min(a.x1, b.x1)) and float(max(a.y0, b.y0)) < float(min(a.y1, b.y1))


def _axis_label_overlaps_ticks(label_artist, tick_artists, *, renderer) -> bool:
    label_box = label_artist.get_window_extent(renderer=renderer)
    for tick_artist in tick_artists:
        if not str(tick_artist.get_text()).strip():
            continue
        if _window_bboxes_overlap(label_box, tick_artist.get_window_extent(renderer=renderer)):
            return True
    return False


def _tick_labels_overlap_each_other(tick_artists, *, renderer) -> bool:
    boxes = [
        tick_artist.get_window_extent(renderer=renderer)
        for tick_artist in tick_artists
        if str(tick_artist.get_text()).strip()
    ]
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            if _window_bboxes_overlap(box, other):
                return True
    return False


def _format_compact_axis_value(value: float, _position: int) -> str:
    numeric = float(value)
    magnitude = abs(numeric)
    if magnitude >= 1_000_000.0:
        scaled = numeric / 1_000_000.0
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if magnitude >= 1_000.0:
        scaled = numeric / 1_000.0
        precision = 0 if abs(scaled) >= 100.0 else 1
        if precision == 0:
            text = f"{scaled:.0f}"
        else:
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    if magnitude >= 1.0:
        rounded = round(numeric)
        if abs(numeric - rounded) < 1.0e-6:
            return f"{int(rounded)}"
        return f"{numeric:.2f}".rstrip("0").rstrip(".")
    return f"{numeric:.3g}"


def _draw_trajectory_panel(panel_ax, panel, style: Style) -> None:
    x = tuple(float(v) for v in panel.x)
    y = tuple(float(v) for v in panel.y)
    point_index = int(panel.point_index)
    panel_ax.set_axisbelow(True)
    panel_ax.set_facecolor("#ffffff")
    for spine_name, spine in panel_ax.spines.items():
        if spine_name in {"top", "right"}:
            spine.set_visible(False)
            continue
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.9)
    panel_ax.plot(x, y, color="#475569", lw=1.8, zorder=2)
    panel_ax.scatter([x[point_index]], [y[point_index]], color="#dc2626", s=26, zorder=3)
    panel_ax.grid(True, alpha=0.20, lw=0.6, color="#9ca3af", zorder=0)

    label_size = float(max(9, min(13, int(round(float(style.font_size_label) * 0.7)))))
    tick_size = float(max(8, min(12, label_size - 1.0)))
    x_label = str(panel.x_label).strip() if panel.x_label is not None else ""
    y_label = str(panel.y_label).strip() if panel.y_label is not None else ""
    if not x_label:
        x_label = "Sweep"
    if not y_label:
        y_label = "Best objective"
    panel_ax.set_xlabel(x_label, fontsize=label_size, color="#334155", labelpad=4.0)
    panel_ax.set_ylabel(y_label, fontsize=label_size, color="#334155", labelpad=1.5)
    panel_ax.xaxis.set_label_position("bottom")
    panel_ax.yaxis.set_label_position("left")
    x_tick_pad = 3.0
    y_tick_pad = 2.5
    panel_ax.tick_params(
        axis="x",
        labelsize=tick_size,
        colors="#475569",
        length=2.0,
        pad=x_tick_pad,
        direction="out",
        bottom=True,
        labelbottom=True,
        top=False,
        labeltop=False,
    )
    panel_ax.tick_params(
        axis="y",
        labelsize=tick_size,
        colors="#475569",
        length=2.0,
        pad=y_tick_pad,
        direction="out",
        left=True,
        labelleft=True,
        right=False,
        labelright=False,
    )
    panel_ax.xaxis.set_major_formatter(FuncFormatter(_format_compact_axis_value))
    panel_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))
    panel_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))

    x_min = float(min(x))
    x_max = float(max(x))
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_span = max(x_max - x_min, 1.0)
    x_pad = max(1.0, 0.02 * x_span)
    y_min = float(min(y))
    y_max = float(max(y))
    y_span = max(y_max - y_min, 1.0e-6)
    panel_ax.set_xlim(x_min - x_pad, x_max + x_pad)
    panel_ax.set_ylim(y_min - (0.05 * y_span), y_max + (0.12 * y_span))

    figure = panel_ax.figure
    for _ in range(12):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel_bbox = panel_ax.get_window_extent(renderer=renderer)
        x_tick_labels = [tick for tick in panel_ax.get_xticklabels() if str(tick.get_text()).strip()]
        y_tick_labels = [tick for tick in panel_ax.get_yticklabels() if str(tick.get_text()).strip()]
        x_tick_bboxes = [tick.get_window_extent(renderer=renderer) for tick in x_tick_labels]
        y_tick_bboxes = [tick.get_window_extent(renderer=renderer) for tick in y_tick_labels]
        x_label_bbox = panel_ax.xaxis.label.get_window_extent(renderer=renderer)
        y_label_bbox = panel_ax.yaxis.label.get_window_extent(renderer=renderer)
        x_ticks_below_plot = not x_tick_bboxes or (
            max(float(box.y1) for box in x_tick_bboxes) <= float(panel_bbox.y0) + 1.0
        )
        x_label_below_ticks = not x_tick_bboxes or (
            float(x_label_bbox.y1) <= min(float(box.y0) for box in x_tick_bboxes) + 1.0
        )
        y_ticks_left_plot = not y_tick_bboxes or (
            max(float(box.x1) for box in y_tick_bboxes) <= float(panel_bbox.x0) + 1.0
        )
        y_label_left_ticks = not y_tick_bboxes or (
            float(y_label_bbox.x1) <= min(float(box.x0) for box in y_tick_bboxes) + 1.0
        )
        x_ticks_overlap = _tick_labels_overlap_each_other(x_tick_labels, renderer=renderer)
        y_ticks_overlap = _tick_labels_overlap_each_other(y_tick_labels, renderer=renderer)
        x_label_overlap = _axis_label_overlaps_ticks(panel_ax.xaxis.label, x_tick_labels, renderer=renderer)
        y_label_overlap = _axis_label_overlaps_ticks(panel_ax.yaxis.label, y_tick_labels, renderer=renderer)
        x_label_inside_figure = float(x_label_bbox.y0) >= 0.0
        y_label_inside_figure = float(y_label_bbox.x0) >= 0.0

        changed = False
        if not x_label_inside_figure:
            new_x_labelpad = max(1.0, float(panel_ax.xaxis.labelpad) - 0.6)
            if new_x_labelpad < float(panel_ax.xaxis.labelpad):
                panel_ax.xaxis.labelpad = new_x_labelpad
                changed = True
            new_x_tick_pad = max(1.0, x_tick_pad - 0.4)
            if new_x_tick_pad < x_tick_pad:
                x_tick_pad = new_x_tick_pad
                panel_ax.tick_params(axis="x", pad=x_tick_pad)
                changed = True
        if not y_label_inside_figure:
            new_y_labelpad = max(1.0, float(panel_ax.yaxis.labelpad) - 0.6)
            if new_y_labelpad < float(panel_ax.yaxis.labelpad):
                panel_ax.yaxis.labelpad = new_y_labelpad
                changed = True
            new_y_tick_pad = max(1.0, y_tick_pad - 0.4)
            if new_y_tick_pad < y_tick_pad:
                y_tick_pad = new_y_tick_pad
                panel_ax.tick_params(axis="y", pad=y_tick_pad)
                changed = True
        if not x_ticks_below_plot:
            x_tick_pad = min(12.0, x_tick_pad + 0.8)
            panel_ax.tick_params(axis="x", pad=x_tick_pad)
            changed = True
        if not y_ticks_left_plot:
            y_tick_pad = min(12.0, y_tick_pad + 0.8)
            panel_ax.tick_params(axis="y", pad=y_tick_pad)
            changed = True
        if not x_label_below_ticks:
            panel_ax.xaxis.labelpad = min(12.0, float(panel_ax.xaxis.labelpad) + 0.8)
            changed = True
        if not y_label_left_ticks:
            panel_ax.yaxis.labelpad = min(14.0, float(panel_ax.yaxis.labelpad) + 0.8)
            changed = True
        if x_ticks_overlap:
            panel_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
            changed = True
        if y_ticks_overlap:
            panel_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            changed = True
        if x_label_overlap or y_label_overlap:
            new_tick_size = max(5.0, tick_size - 0.5)
            if new_tick_size < tick_size:
                tick_size = new_tick_size
                panel_ax.tick_params(axis="x", labelsize=tick_size)
                panel_ax.tick_params(axis="y", labelsize=tick_size)
                changed = True
        if not changed:
            break

    figure.canvas.draw()
    panel_ax.xaxis.label.set_clip_on(False)
    panel_ax.yaxis.label.set_clip_on(False)
    for label in [*panel_ax.get_xticklabels(), *panel_ax.get_yticklabels()]:
        label.set_clip_on(False)


def _draw_connectors(ax, n: int, x0: float, cw: float, layout: LayoutContext, style: Style) -> None:
    y_top = float(layout.y_forward)
    y_bottom = float(layout.y_reverse)
    if not style.connectors or y_top <= y_bottom:
        return
    top_row_boundary = y_top - float(layout.sequence_extent_down)
    bottom_row_boundary = y_bottom + float(layout.sequence_extent_up)
    available_gap = top_row_boundary - bottom_row_boundary
    if available_gap <= 0:
        return
    connector_span = max(0.0, available_gap * 0.5)
    center_y = (top_row_boundary + bottom_row_boundary) / 2.0
    y1 = max(bottom_row_boundary, center_y - connector_span / 2.0)
    y2 = min(top_row_boundary, center_y + connector_span / 2.0)
    if y2 <= y1:
        return
    dash_pattern = tuple(float(value) for value in style.connector_dash)
    hidden_indices: set[int] = set()
    cross_indices: set[int] = set()
    overhang_spans: list[tuple[int, int]] = []
    record_meta = getattr(ax, "_dnadesign_record_meta", None)
    if isinstance(record_meta, Mapping):
        hidden_indices = {int(value) for value in record_meta.get("connector_hidden_indices", ())}
        cross_indices = {int(value) for value in record_meta.get("connector_cross_indices", ())}
        raw_spans = record_meta.get("connector_overhang_spans", ())
        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, (str, bytes)):
            for raw in raw_spans:
                if not isinstance(raw, Mapping):
                    continue
                try:
                    start = int(raw.get("start"))
                    end = int(raw.get("end"))
                except Exception:
                    continue
                if end > start:
                    overhang_spans.append((start, end))
    for i in range(n):
        if i in hidden_indices:
            continue
        x = x0 + i * cw + cw / 2.0
        if i in cross_indices:
            dx = max(2.5, cw * 0.24)
            ax.plot(
                [x - dx, x + dx], [y1, y2], color="#6B7280", lw=max(1.1, style.connector_width), alpha=0.95, zorder=1.5
            )
            ax.plot(
                [x - dx, x + dx], [y2, y1], color="#6B7280", lw=max(1.1, style.connector_width), alpha=0.95, zorder=1.5
            )
            continue
        (ln,) = ax.plot(
            [x, x],
            [y1, y2],
            color=style.color_ticks,
            lw=style.connector_width,
            alpha=style.connector_alpha,
            zorder=1,
        )
        if dash_pattern:
            ln.set_dashes(dash_pattern)
    if overhang_spans:
        center_y = (y1 + y2) / 2.0
        for start, end in overhang_spans:
            x_start = x0 + start * cw + cw * 0.12
            x_end = x0 + end * cw - cw * 0.12
            if x_end <= x_start:
                continue
            (ln,) = ax.plot([x_start, x_end], [center_y, center_y], color="#111827", lw=1.35, zorder=1.7)
            ln.set_dashes((3.0, 2.0))


def _draw_span_backdrops(
    ax,
    layout: LayoutContext,
    span_backdrops: Sequence[Mapping[str, object]],
    *,
    show_two: bool,
) -> None:
    for index, raw in enumerate(span_backdrops):
        if not isinstance(raw, Mapping):
            continue
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
            alpha = float(raw.get("alpha"))
            corner_radius = float(raw.get("corner_radius"))
        except Exception:
            continue
        if end <= start:
            continue
        fill = str(raw.get("fill", "")).strip()
        if not fill:
            continue
        cover_rows = str(raw.get("cover_rows", "both")).strip().lower()
        row_bounds: list[tuple[float, float]] = []
        if cover_rows in {"primary", "both"}:
            row_bounds.append(
                (
                    float(layout.y_forward - layout.sequence_extent_down),
                    float(layout.y_forward + layout.sequence_extent_up),
                )
            )
        if show_two and cover_rows in {"complement", "both"}:
            row_bounds.append(
                (
                    float(layout.y_reverse - layout.sequence_extent_down),
                    float(layout.y_reverse + layout.sequence_extent_up),
                )
            )
        elif not row_bounds and cover_rows == "complement":
            row_bounds.append(
                (
                    float(layout.y_forward - layout.sequence_extent_down),
                    float(layout.y_forward + layout.sequence_extent_up),
                )
            )
        if not row_bounds:
            continue
        x = layout.x_left + start * layout.cw
        y0 = min(bound[0] for bound in row_bounds)
        y1 = max(bound[1] for bound in row_bounds)
        ax.add_patch(
            FancyBboxPatch(
                (x, y0),
                (end - start) * layout.cw,
                y1 - y0,
                boxstyle=f"round,pad=0.0,rounding_size={corner_radius}",
                linewidth=0.0,
                facecolor=mcolors.to_rgba(fill, alpha),
                edgecolor="none",
                zorder=0.6,
                clip_on=False,
                gid=f"sequence_backdrop:{index}",
            )
        )


def _draw_row_labels(ax, record: Record, layout: LayoutContext, style: Style) -> None:
    row_labels = record.meta.get("row_labels") if isinstance(record.meta, Mapping) else None
    if not isinstance(row_labels, Mapping):
        return
    terminal_dx = style.font_size_label / 72.0 * style.dpi * 0.8
    terminal_label_width = max(
        measure_text_width_px("5'", style.font_label, style.font_size_label, style.dpi),
        measure_text_width_px("3'", style.font_label, style.font_size_label, style.dpi),
    )
    row_gap = max(8.0, style.font_size_label / 72.0 * style.dpi * 0.35)
    x = layout.x_left - terminal_dx - terminal_label_width - row_gap
    primary = row_labels.get("primary")
    complement = row_labels.get("complement")
    if primary:
        ax.text(
            x,
            layout.y_forward,
            str(primary),
            ha="right",
            va="center",
            fontsize=style.display_font_size() if bool(style.uniform_display_font_size) else style.font_size_label,
            family=style.font_label,
            color="#374151",
            zorder=4.0,
            clip_on=False,
        )
    if bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"}) and complement:
        ax.text(
            x,
            layout.y_reverse,
            str(complement),
            ha="right",
            va="center",
            fontsize=style.display_font_size() if bool(style.uniform_display_font_size) else style.font_size_label,
            family=style.font_label,
            color="#374151",
            zorder=4.0,
            clip_on=False,
        )


def _draw_segment_labels(ax, record: Record, layout: LayoutContext, style: Style) -> None:
    segment_labels = record.meta.get("segment_labels") if isinstance(record.meta, Mapping) else None
    if not isinstance(segment_labels, Sequence) or isinstance(segment_labels, (str, bytes)):
        return
    font_size = style.display_font_size() if bool(style.uniform_display_font_size) else max(9, style.font_size_label)
    line_height = max(10.0, (float(font_size) / 72.0) * float(style.dpi))
    horizontal_gap = max(6.0, layout.cw * 0.25)
    top_tiers: list[list[tuple[float, float]]] = []
    bottom_tiers: list[list[tuple[float, float]]] = []
    placements: list[tuple[str, str, float, int]] = []
    for raw in segment_labels:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        row_id = str(raw.get("row_id", "primary")).strip().lower() or "primary"
        if row_id not in {"primary", "complement"}:
            row_id = "primary"
        x = layout.x_left + ((start + end) / 2.0) * layout.cw
        text_width = _text_px_width(text, style.font_label, int(round(font_size)), style.dpi)
        x0 = x - text_width / 2.0
        x1 = x + text_width / 2.0
        tiers = bottom_tiers if row_id == "complement" else top_tiers
        tier_index = len(tiers)
        for candidate_index, occupied in enumerate(tiers):
            if all((x1 + horizontal_gap) <= left or (x0 - horizontal_gap) >= right for left, right in occupied):
                tier_index = candidate_index
                occupied.append((x0, x1))
                break
        else:
            tiers.append([(x0, x1)])
        placements.append((row_id, text, x, tier_index))

    top_base_y = layout.y_forward + layout.sequence_extent_up + max(18.0, style.font_size_label * 1.4)
    bottom_base_y = layout.y_reverse - layout.sequence_extent_down - max(16.0, style.font_size_label * 1.2)
    show_two = bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"})
    for row_id, text, x, tier_index in placements:
        if row_id == "complement" and show_two:
            y = bottom_base_y - tier_index * line_height * 1.05
            va = "top"
        else:
            y = top_base_y + tier_index * line_height * 1.05
            va = "bottom"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va=va,
            fontsize=font_size,
            family=style.font_label,
            color="#111827",
            zorder=4.3,
            clip_on=False,
        )


def _draw_coordinate_ticks(ax, record: Record, layout: LayoutContext, style: Style) -> None:
    tick_every = 5 if len(record.sequence) > 12 else 2
    show_two = bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"})
    y = (
        layout.y_reverse - max(20.0, style.font_size_label * 1.8)
        if show_two
        else layout.y_forward - max(20.0, style.font_size_label * 1.8)
    )
    for boundary in range(0, len(record.sequence) + 1, tick_every):
        x = layout.x_left + boundary * layout.cw
        ax.plot([x, x], [y + 2.0, y + 8.0], color=style.color_ticks, linewidth=0.8, zorder=1.2)
        ax.text(
            x,
            y,
            str(boundary),
            ha="center",
            va="top",
            fontsize=(
                style.display_font_size()
                if bool(style.uniform_display_font_size)
                else max(7, style.font_size_label - 2)
            ),
            family=style.font_label,
            color=style.color_ticks,
            zorder=1.3,
        )


def _draw_sequence(
    ax,
    seq: str,
    x0: float,
    y_center: float,
    cw: float,
    style: Style,
    left_label: str,
    right_label: str,
    *,
    tone_strengths: Sequence[float] | None = None,
    row_id: str = "fwd",
    highlight_indices: Sequence[int] | None = None,
    highlight_color: str | None = None,
    dim_indices: Sequence[int] | None = None,
) -> None:
    label_dx = style.font_size_label / 72.0 * style.dpi * 0.8
    ax.text(
        x0 - label_dx,
        y_center,
        left_label,
        va="center",
        ha="right",
        fontsize=style.display_font_size() if bool(style.uniform_display_font_size) else style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )
    ax.text(
        x0 + len(seq) * cw + label_dx,
        y_center,
        right_label,
        va="center",
        ha="left",
        fontsize=style.display_font_size() if bool(style.uniform_display_font_size) else style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )

    px_per_pt = style.dpi / 72.0
    highlight_set = {int(index) for index in (highlight_indices or ())}
    dim_set = {int(index) for index in (dim_indices or ())}
    x = x0
    for idx, char in enumerate(seq):
        is_highlighted = idx in highlight_set
        weight = "bold" if is_highlighted else "normal"
        tp = _mono_text_path(char, style.font_mono, style.font_size_seq, weight)
        glyph_color = style.color_sequence
        if tone_strengths is not None:
            strength = tone_strengths[idx] if idx < len(tone_strengths) else 0.0
            glyph_color = _mix_colors(style.sequence.non_consensus_color, style.color_sequence, strength)
        if idx in dim_set and not is_highlighted:
            glyph_color = "#D1D5DB"
        if is_highlighted:
            glyph_color = str(highlight_color).strip() if highlight_color else _darken_rgb(glyph_color, factor=0.72)
        y_mid_px = _mono_ag_mid_px(style.font_mono, style.font_size_seq, style.dpi, weight)
        trans = Affine2D().scale(px_per_pt).translate(x, y_center - y_mid_px) + ax.transData
        patch = PathPatch(
            tp,
            transform=trans,
            facecolor=glyph_color,
            edgecolor="none",
            linewidth=0.0,
            zorder=2,
            clip_on=False,
        )
        gid = f"sequence:{row_id}:{idx}:{char}"
        if is_highlighted:
            gid = f"{gid}:highlight"
        patch.set_gid(gid)
        ax.add_patch(patch)
        x += cw


def _draw_feature_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    facecolor,
    style: Style,
    *,
    cw: float,
    ch: float,
    draw_label: bool = True,
) -> None:
    r = style.kmer.round_px
    pad_x = float(style.kmer.pad_x_px)
    edge_color = _darken_rgb(facecolor, factor=0.78)

    ax.add_patch(
        FancyBboxPatch(
            (x - pad_x, y - h / 2),
            w + 2 * pad_x,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            linewidth=style.kmer.edge_width,
            facecolor=facecolor,
            alpha=style.kmer.fill_alpha,
            edgecolor=edge_color,
            zorder=3,
            clip_on=False,
        )
    )

    if not draw_label or not label:
        return

    px_per_pt = style.dpi / 72.0

    y_text_center = y + float(style.kmer.text_y_nudge_cells) * ch
    for idx, char in enumerate(label):
        if char.isspace():
            continue
        tp = _mono_text_path(char, style.font_mono, style.font_size_seq)
        gb = tp.get_extents()
        gx = ((gb.x0 + gb.x1) / 2.0) * px_per_pt
        gy = ((gb.y0 + gb.y1) / 2.0) * px_per_pt
        x_center = x + (idx + 0.5) * cw
        trans = Affine2D().scale(px_per_pt).translate(x_center - gx, y_text_center - gy) + ax.transData
        ax.add_patch(
            PathPatch(
                tp,
                transform=trans,
                facecolor=style.kmer.text_color,
                edgecolor="none",
                linewidth=0.0,
                zorder=4,
                clip_on=False,
            )
        )


def _draw_interval_underline(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color,
    style: Style,
    *,
    draw_label: bool = True,
) -> None:
    pad_x = float(style.kmer.pad_x_px)
    x0 = float(x) - pad_x
    x1 = float(x) + float(w) + pad_x
    line_y = float(y) - max(2.0, float(h) * 0.24)
    tick_h = max(3.0, float(h) * 0.12)
    ax.plot(
        [x0, x1],
        [line_y, line_y],
        color=color,
        alpha=0.85,
        lw=max(1.4, float(style.kmer.edge_width) * 2.0),
        solid_capstyle="round",
        zorder=3.15,
        clip_on=False,
    )
    ax.plot([x0, x0], [line_y - tick_h, line_y + tick_h], color=color, alpha=0.7, lw=1.0, zorder=3.1, clip_on=False)
    ax.plot([x1, x1], [line_y - tick_h, line_y + tick_h], color=color, alpha=0.7, lw=1.0, zorder=3.1, clip_on=False)

    if not draw_label or not label:
        return
    font_size = max(6, min(int(style.legend_font_size), int(style.font_size_label)))
    text = str(label).strip()
    text_w = _text_px_width(text, style.font_label, font_size, style.dpi)
    available = max(0.0, x1 - x0)
    if text_w <= max(8.0, available - 8.0):
        text_x = x0 + 4.0
        ha = "left"
    else:
        text_x = (x0 + x1) / 2.0
        ha = "center"
    ax.text(
        text_x,
        line_y + max(3.0, float(h) * 0.08),
        text,
        ha=ha,
        va="bottom",
        fontsize=font_size,
        family=style.font_label,
        color=_darken_rgb(color, factor=0.72),
        zorder=6.1,
        clip_on=False,
    )


def _draw_motif_scale_bar(
    ax,
    geometries: Sequence[MotifLogoGeometry],
    layout: LayoutContext,
    style: Style,
) -> None:
    def _draw_bar(*, x: float, y0: float, y1: float, baseline: str) -> None:
        tick_w = 4.0
        ax.plot([x, x], [y0, y1], color=cfg.color, lw=0.9, zorder=7, clip_on=False)
        ax.plot([x - tick_w, x + tick_w], [y0, y0], color=cfg.color, lw=0.9, zorder=7, clip_on=False)
        ax.plot([x - tick_w, x + tick_w], [y1, y1], color=cfg.color, lw=0.9, zorder=7, clip_on=False)

        if str(style.motif_logo.display_mode).lower() == "information":
            max_label = f"{style.motif_logo.height_bits:g} bits"
        else:
            max_label = "1.0"
        if baseline == "top":
            top_label = "0"
            bottom_label = max_label
        else:
            top_label = max_label
            bottom_label = "0"

        ax.text(
            x - 6.0,
            y0,
            bottom_label,
            ha="right",
            va="bottom",
            fontsize=cfg.font_size,
            family=style.font_label,
            color=cfg.color,
            zorder=8,
            clip_on=False,
        )
        ax.text(
            x - 6.0,
            y1,
            top_label,
            ha="right",
            va="top",
            fontsize=cfg.font_size,
            family=style.font_label,
            color=cfg.color,
            zorder=8,
            clip_on=False,
        )

    cfg = style.motif_logo.scale_bar
    if not cfg.enabled:
        return
    if not geometries:
        return

    location = str(cfg.location).lower()
    if location == "left_of_logo":
        pad = float(cfg.pad_cells) * layout.ch
        seen: set[tuple[float, float, float, str]] = set()
        for geometry in geometries:
            x = geometry.x0 - pad
            y0 = geometry.y0
            y1 = geometry.y0 + geometry.height
            # Avoid overdrawing identical bars when multiple effects share exact placement.
            key = (round(x, 4), round(y0, 4), round(y1, 4), geometry.baseline)
            if key in seen:
                continue
            seen.add(key)
            _draw_bar(x=x, y0=y0, y1=y1, baseline=geometry.baseline)
        return

    if location == "top_right":
        candidates = [g for g in geometries if g.above]
        if not candidates:
            return
        ref = max(candidates, key=lambda g: g.y0)
    elif location == "bottom_right":
        candidates = [g for g in geometries if not g.above]
        if not candidates:
            return
        ref = min(candidates, key=lambda g: g.y0)
    else:
        return

    x = layout.width - max(10.0, style.padding_x * 0.55)
    _draw_bar(x=x, y0=ref.y0, y1=ref.y0 + ref.height, baseline=ref.baseline)


def _text_px_width(text: str, family: str, size_pt: int, dpi: int) -> float:
    prop = FontProperties(family=family, size=size_pt)
    bbox = TextPath((0, 0), text, prop=prop).get_extents()
    return bbox.width / 72.0 * dpi


def _draw_legend(ax, legend: Sequence[tuple[str, str]], palette: Palette, style: Style, total_width: float) -> None:
    if not legend:
        return

    coord_scale = max(1.0e-6, float(style.figure_scale))

    def _to_layout_units(value: float) -> float:
        return float(value) / coord_scale

    # Bottom legends act as a footer; they do not need the same horizontal padding as the sequence body.
    side_pad = (
        max(6.0, min(12.0, float(style.padding_x) * 0.35))
        if bool(style.uniform_display_font_size)
        else float(style.padding_x)
    )
    available_width = max(0.0, float(total_width) - (2.0 * side_pad))
    if available_width <= 0.0:
        return

    def _wrap_rows(entry_widths: list[float], *, gap_x: float) -> list[list[int]]:
        rows: list[list[int]] = []
        current: list[int] = []
        current_width = 0.0
        for idx, width in enumerate(entry_widths):
            add_width = width if not current else (gap_x + width)
            if current and (current_width + add_width) > available_width:
                rows.append(current)
                current = [idx]
                current_width = width
                continue
            current.append(idx)
            current_width += add_width
        if current:
            rows.append(current)
        return rows

    base_font_size = max(
        6,
        int(round(style.display_font_size() if bool(style.uniform_display_font_size) else style.legend_font_size)),
    )
    min_font_size = base_font_size if bool(style.uniform_display_font_size) else 6
    is_uniform_footer = bool(style.uniform_display_font_size)
    min_patch_w = max(
        _to_layout_units(8.0),
        _to_layout_units(float(style.legend_patch_w) if is_uniform_footer else float(style.legend_patch_w) * 0.55),
    )
    min_patch_h = max(
        _to_layout_units(6.0),
        _to_layout_units(float(style.legend_patch_h) if is_uniform_footer else float(style.legend_patch_h) * 0.55),
    )
    min_gap_patch_text = max(
        _to_layout_units(2.0),
        _to_layout_units(
            max(12.0, float(style.legend_gap_patch_text) * 0.6)
            if is_uniform_footer
            else float(style.legend_gap_patch_text) * 0.5
        ),
    )
    min_gap_x = max(
        0.0,
        _to_layout_units(max(24.0, float(style.legend_gap_x) * 0.4) if is_uniform_footer else 0.0),
    )
    selected_layout: tuple[int, float, float, float, float, float, float, list[float], list[list[int]]] | None = None

    legend_indices = list(range(len(legend)))
    for font_size in range(base_font_size, min_font_size - 1, -1):
        font_scale = float(font_size) / float(base_font_size)
        geometry_scales = (
            (1.0,)
            if is_uniform_footer
            else (
                1.0,
                0.92,
                0.84,
                0.76,
            )
        )
        for geometry_scale in geometry_scales:
            patch_w = max(
                min_patch_w,
                _to_layout_units(float(style.legend_patch_w) * font_scale * geometry_scale),
            )
            patch_h = max(
                min_patch_h,
                _to_layout_units(float(style.legend_patch_h) * font_scale * geometry_scale),
            )
            gap_patch_text = max(
                min_gap_patch_text,
                _to_layout_units(float(style.legend_gap_patch_text) * min(font_scale, geometry_scale)),
            )
            text_widths = [
                _to_layout_units(_text_px_width(label, style.font_label, font_size, style.dpi))
                for _tag, label in legend
            ]
            entry_widths = [patch_w + gap_patch_text + width for width in text_widths]
            if any(width > available_width for width in entry_widths):
                continue
            requested_gap_x = max(
                min_gap_x,
                _to_layout_units(float(style.legend_gap_x) * min(font_scale, geometry_scale)),
            )
            if len(entry_widths) <= 1:
                gap_x = 0.0
                row_total = sum(entry_widths)
            else:
                fixed_width = sum(entry_widths)
                max_gap = max(0.0, (available_width - fixed_width) / float(len(entry_widths) - 1))
                if max_gap < min_gap_x:
                    continue
                gap_x = min(requested_gap_x, max_gap)
                row_total = fixed_width + float(len(entry_widths) - 1) * gap_x
            if row_total > available_width:
                continue
            row_height = max(patch_h, _to_layout_units(float(font_size) * 1.18))
            if row_height > float(style.legend_height_px):
                continue
            selected_layout = (
                font_size,
                patch_w,
                patch_h,
                gap_patch_text,
                gap_x,
                row_height,
                0.0,
                entry_widths,
                [legend_indices],
            )
            break
        if selected_layout is not None:
            break

    if selected_layout is None:
        for font_size in range(base_font_size, min_font_size - 1, -1):
            scale = float(font_size) / float(base_font_size)
            patch_w = max(min_patch_w, _to_layout_units(float(style.legend_patch_w) * scale))
            patch_h = max(min_patch_h, _to_layout_units(float(style.legend_patch_h) * scale))
            gap_patch_text = max(min_gap_patch_text, _to_layout_units(float(style.legend_gap_patch_text) * scale))
            gap_x_requested = max(0.0, _to_layout_units(float(style.legend_gap_x) * scale))
            text_widths = [
                _to_layout_units(_text_px_width(label, style.font_label, font_size, style.dpi))
                for _tag, label in legend
            ]
            entry_widths = [patch_w + gap_patch_text + width for width in text_widths]
            if any(width > available_width for width in entry_widths):
                continue

            rows = _wrap_rows(entry_widths, gap_x=gap_x_requested)
            gap_x = gap_x_requested
            per_row_max_gap: list[float] = []
            for row in rows:
                if len(row) <= 1:
                    continue
                row_fixed_width = sum(entry_widths[idx] for idx in row)
                per_row_max_gap.append(max(0.0, (available_width - row_fixed_width) / float(len(row) - 1)))
            if per_row_max_gap:
                gap_x = min(gap_x_requested, min(per_row_max_gap))

            row_height = max(patch_h, _to_layout_units(float(font_size) * 1.18))
            row_gap_y = max(_to_layout_units(2.0), min(_to_layout_units(8.0), row_height * 0.22))
            rows_height = (len(rows) * row_height) + (max(0, len(rows) - 1) * row_gap_y)
            if rows_height <= float(style.legend_height_px):
                selected_layout = (
                    font_size,
                    patch_w,
                    patch_h,
                    gap_patch_text,
                    gap_x,
                    row_height,
                    row_gap_y,
                    entry_widths,
                    rows,
                )
                break

    if selected_layout is None:
        font_size = min_font_size
        scale = float(font_size) / float(base_font_size)
        patch_w = max(min_patch_w, _to_layout_units(float(style.legend_patch_w) * scale))
        patch_h = max(min_patch_h, _to_layout_units(float(style.legend_patch_h) * scale))
        gap_patch_text = max(min_gap_patch_text, _to_layout_units(float(style.legend_gap_patch_text) * scale))
        gap_x = max(0.0, _to_layout_units(float(style.legend_gap_x) * scale))
        text_widths = [
            _to_layout_units(_text_px_width(label, style.font_label, font_size, style.dpi)) for _tag, label in legend
        ]
        entry_widths = [patch_w + gap_patch_text + width for width in text_widths]
        rows = _wrap_rows(entry_widths, gap_x=gap_x)
        row_height = max(patch_h, _to_layout_units(float(font_size) * 1.18))
        row_gap_y = max(_to_layout_units(2.0), min(_to_layout_units(8.0), row_height * 0.22))
    else:
        (
            font_size,
            patch_w,
            patch_h,
            gap_patch_text,
            gap_x,
            row_height,
            row_gap_y,
            entry_widths,
            rows,
        ) = selected_layout

    total_rows_height = (len(rows) * row_height) + (max(0, len(rows) - 1) * row_gap_y)
    y = style.legend_origin_y(total_rows_height=total_rows_height)
    for row in rows:
        row_total = sum(entry_widths[idx] for idx in row)
        if len(row) > 1:
            row_total += float(len(row) - 1) * gap_x
        x = (total_width - row_total) / 2.0 if style.legend_center else side_pad
        x = max(x, side_pad)
        for position, idx in enumerate(row):
            tag, label = legend[idx]
            color = palette.color_for(tag)
            edge_color = _darken_rgb(color, factor=0.76)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    patch_w,
                    patch_h,
                    boxstyle="round,pad=0.0,rounding_size=2.5",
                    linewidth=0.7,
                    facecolor=color,
                    alpha=1.0,
                    edgecolor=edge_color,
                    zorder=10,
                    clip_on=False,
                )
            )
            ax.text(
                x + patch_w + gap_patch_text,
                y + patch_h / 2.0,
                label,
                va="center",
                ha="left",
                fontsize=font_size,
                family=style.font_label,
                color=style.color_sequence,
                zorder=10,
                clip_on=False,
            )
            x += entry_widths[idx]
            if position < (len(row) - 1):
                x += gap_x
        y += row_height + row_gap_y
