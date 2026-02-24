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
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ..config import Style
from ..core import Record, RenderingError
from .effects.motif_logo import MotifLogoGeometry, compute_motif_logo_geometry
from .effects.registry import draw_effect
from .layout import LayoutContext, comp, compute_layout
from .palette import Palette


@dataclass(frozen=True)
class SequenceRowsRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        record = record.validate()
        show_two = bool(style.show_reverse_complement and record.alphabet == "DNA")
        layout = compute_layout(record, style)

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
        if bool(style.sequence.bold_consensus_bases) and motif_geometries:
            tone_fwd, tone_rev = _sequence_tone_strengths(
                record,
                motif_geometries,
                q_low=float(style.sequence.tone_quantile_low),
                q_high=float(style.sequence.tone_quantile_high),
            )

        fig_scale = float(style.figure_scale)
        fig = plt.figure(
            figsize=((layout.width / style.dpi) * fig_scale, (layout.height / style.dpi) * fig_scale),
            dpi=style.dpi,
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        x0 = layout.x_left
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
        )
        if show_two:
            _draw_sequence(
                ax,
                comp(record.sequence),
                x0,
                layout.y_reverse,
                layout.cw,
                style,
                "3'",
                "5'",
                tone_strengths=tone_rev,
                row_id="rev",
            )
            _draw_connectors(ax, len(record.sequence), x0, layout.cw, layout.y_forward, layout.y_reverse, style)

        feature_boxes = dict(layout.feature_boxes)

        # Draw feature boxes first.
        for placement in layout.placements:
            feature = record.features[placement.feature_index]
            tag = feature.tags[0] if feature.tags else feature.kind
            color = palette.color_for(tag)
            label = feature.label or ""
            if not placement.above:
                label = label[::-1]
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

        legend_mode = str(style.legend_mode).lower()
        if style.legend and legend_mode == "inline":
            _draw_inline_feature_labels(ax, record, layout, palette, style)

        _draw_motif_scale_bar(ax, motif_geometries, layout, style)

        if style.legend and legend_mode == "bottom":
            from .legend import legend_entries_for_record

            _draw_legend(ax, legend_entries_for_record(record), palette, style, layout.width)

        if record.display.overlay_text:
            _draw_overlay(ax, layout, style, record.display.overlay_text)

        ax.set_xlim(0, layout.width)
        ax.set_ylim(0, layout.height)
        return fig


@lru_cache(maxsize=1024)
def _mono_text_path(char: str, font_family: str, size_pt: int) -> TextPath:
    prop = FontProperties(family=font_family, size=size_pt)
    return TextPath((0, 0), char, prop=prop, usetex=False)


@lru_cache(maxsize=64)
def _mono_ag_mid_px(font_family: str, size_pt: int, dpi: int) -> float:
    prop = FontProperties(family=font_family, size=size_pt)
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
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n = len(record.sequence)
    seq_fwd = record.sequence.upper()
    seq_rev = comp(record.sequence).upper()
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
            pos = feature.span.start + offset
            if pos < 0 or pos >= n:
                raise RenderingError(
                    f"motif_logo geometry out of sequence bounds while computing sequence tone scores: pos={pos}, n={n}"
                )
            if len(row) < 4:
                raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
            info_weight = max(0.0, min(1.0, _column_information_bits(row) / 2.0))
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


def _capitalize_first(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    return t[0].upper() + t[1:]


def _draw_inline_feature_labels(ax, record: Record, layout: LayoutContext, palette: Palette, style: Style) -> None:
    if not layout.placements:
        return

    labels = dict(record.display.tag_labels)
    side_pref = str(style.legend_inline_side).lower()
    margin = float(style.legend_inline_margin_cells) * layout.cw
    box_pad = float(style.kmer.pad_x_px)
    x_min = style.padding_x
    x_max = layout.width - style.padding_x

    def _candidate_position(
        side: str, *, x_left: float, x_right: float, text_w: float
    ) -> tuple[float, str, float, float]:
        if side == "right":
            x_text = min(x_right, x_max - text_w)
            return x_text, "left", x_text, x_text + text_w
        x_text = max(x_left, x_min + text_w)
        return x_text, "right", x_text - text_w, x_text

    def _overlap_score(
        *,
        interval_x0: float,
        interval_x1: float,
        y_anchor: float,
        own_feature_index: int,
    ) -> float:
        score = 0.0
        for other in layout.placements:
            if other.feature_index == own_feature_index:
                continue
            y0 = other.y - other.h / 2.0
            y1 = other.y + other.h / 2.0
            if not (y0 <= y_anchor <= y1):
                continue
            bx0 = other.x - box_pad
            bx1 = other.x + other.w + box_pad
            overlap = min(interval_x1, bx1) - max(interval_x0, bx0)
            if overlap > 0.0:
                score += overlap
        return score

    for placement in layout.placements:
        feature = record.features[placement.feature_index]
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
            side = "left"
        elif side_pref == "right":
            side = "right"
        else:
            side = "right" if right_room >= left_room else "left"

        preferred = _candidate_position(side, x_left=x_left, x_right=x_right, text_w=text_w)
        alternate_side = "left" if side == "right" else "right"
        alternate = _candidate_position(alternate_side, x_left=x_left, x_right=x_right, text_w=text_w)
        preferred_score = _overlap_score(
            interval_x0=preferred[2],
            interval_x1=preferred[3],
            y_anchor=placement.y,
            own_feature_index=placement.feature_index,
        )
        alternate_score = _overlap_score(
            interval_x0=alternate[2],
            interval_x1=alternate[3],
            y_anchor=placement.y,
            own_feature_index=placement.feature_index,
        )
        chosen = preferred if preferred_score <= alternate_score else alternate
        x_text, ha = chosen[0], chosen[1]

        ax.text(
            x_text,
            placement.y,
            text,
            ha=ha,
            va="center",
            fontsize=style.legend_font_size,
            family=style.font_label,
            color=palette.color_for(tag),
            zorder=6.2,
            clip_on=False,
        )


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
    ax.text(
        x,
        layout.height - max(4.0, style.padding_y * 0.5),
        text,
        ha=ha,
        va="top",
        fontsize=style.font_size_label,
        family=style.font_label,
        color="#6B7280",
        alpha=0.95,
        zorder=15,
        clip_on=False,
    )


def _draw_connectors(ax, n: int, x0: float, cw: float, y_top: float, y_bottom: float, style: Style) -> None:
    if not style.connectors or y_top <= y_bottom:
        return
    y1 = y_bottom + 0.35 * (y_top - y_bottom)
    y2 = y_top - 0.35 * (y_top - y_bottom)
    for i in range(n):
        x = x0 + i * cw + cw / 2.0
        (ln,) = ax.plot(
            [x, x],
            [y1, y2],
            color=style.color_ticks,
            lw=style.connector_width,
            alpha=style.connector_alpha,
            zorder=1,
        )
        ln.set_dashes(style.connector_dash)


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
) -> None:
    label_dx = style.font_size_label / 72.0 * style.dpi * 0.8
    ax.text(
        x0 - label_dx,
        y_center,
        left_label,
        va="center",
        ha="right",
        fontsize=style.font_size_label,
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
        fontsize=style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )

    px_per_pt = style.dpi / 72.0
    y_mid_px = _mono_ag_mid_px(style.font_mono, style.font_size_seq, style.dpi)
    x = x0
    for idx, char in enumerate(seq):
        tp = _mono_text_path(char, style.font_mono, style.font_size_seq)
        glyph_color = style.color_sequence
        if tone_strengths is not None:
            strength = tone_strengths[idx] if idx < len(tone_strengths) else 0.0
            glyph_color = _mix_colors(style.sequence.non_consensus_color, style.color_sequence, strength)
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
        patch.set_gid(f"sequence:{row_id}:{idx}:{char}")
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
) -> None:
    r = style.kmer.round_px
    pad_x = float(style.kmer.pad_x_px)

    ax.add_patch(
        FancyBboxPatch(
            (x - pad_x, y - h / 2),
            w + 2 * pad_x,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            linewidth=style.kmer.edge_width,
            facecolor=facecolor,
            alpha=style.kmer.fill_alpha,
            edgecolor=facecolor,
            zorder=3,
            clip_on=False,
        )
    )

    px_per_pt = style.dpi / 72.0

    y_text_center = y + float(style.kmer.text_y_nudge_cells) * ch
    for idx, char in enumerate(label):
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

    entries: list[tuple[str, str, float]] = []
    text_total = 0.0
    for tag, label in legend:
        w = _text_px_width(label, style.font_label, style.legend_font_size, style.dpi)
        entries.append((tag, label, w))
        text_total += w

    n = len(entries)
    total = n * style.legend_patch_w + n * style.legend_gap_patch_text + text_total + (n - 1) * style.legend_gap_x
    x = (total_width - total) / 2.0 if style.legend_center else style.padding_x
    x = max(x, style.padding_x)
    y = style.legend_pad_px

    for i, (tag, label, w) in enumerate(entries):
        color = palette.color_for(tag)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                style.legend_patch_w,
                style.legend_patch_h,
                boxstyle="round,pad=0.0,rounding_size=2.5",
                linewidth=0.0,
                facecolor=color,
                alpha=1.0,
                edgecolor=color,
                zorder=10,
                clip_on=False,
            )
        )
        ax.text(
            x + style.legend_patch_w + style.legend_gap_patch_text,
            y + style.legend_patch_h / 2.0,
            label,
            va="center",
            ha="left",
            fontsize=style.legend_font_size,
            family=style.font_label,
            color=style.color_sequence,
            zorder=10,
            clip_on=False,
        )
        x += style.legend_patch_w + style.legend_gap_patch_text + w
        if i < n - 1:
            x += style.legend_gap_x
