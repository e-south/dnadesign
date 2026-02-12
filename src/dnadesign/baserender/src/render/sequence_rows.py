"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/sequence_rows.py

Sequence-row renderer for Record v1 with kmer features, effects, overlays, and legend.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ..config import Style
from ..core import Record
from .effects.registry import draw_effect
from .layout import LayoutContext, comp, compute_layout
from .palette import Palette


@dataclass(frozen=True)
class SequenceRowsRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        record = record.validate()
        show_two = bool(style.show_reverse_complement and record.alphabet == "DNA")
        layout = compute_layout(record, style)

        fig = plt.figure(figsize=(layout.width / style.dpi, layout.height / style.dpi), dpi=style.dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        x0 = layout.x_left
        _draw_sequence(ax, record.sequence, x0, layout.y_forward, layout.cw, style, "5'", "3'")
        if show_two:
            _draw_sequence(ax, comp(record.sequence), x0, layout.y_reverse, layout.cw, style, "3'", "5'")
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
                above=placement.above,
                cw=layout.cw,
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

        if style.legend:
            from .legend import legend_entries_for_record

            _draw_legend(ax, legend_entries_for_record(record), palette, style, layout.width)

        if record.display.overlay_text:
            _draw_overlay(ax, layout, style, record.display.overlay_text)

        ax.set_xlim(0, layout.width)
        ax.set_ylim(0, layout.height)
        return fig


def _draw_overlay(ax, layout: LayoutContext, style: Style, text: str) -> None:
    ax.text(
        style.padding_x,
        layout.height - 6.0,
        text,
        ha="left",
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
    ax, seq: str, x0: float, y_center: float, cw: float, style: Style, left_label: str, right_label: str
) -> None:
    ax.text(
        x0 - (style.font_size_label / 72.0 * style.dpi * 0.8),
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
        x0 + len(seq) * cw + (style.font_size_label / 72.0 * style.dpi * 0.4),
        y_center,
        right_label,
        va="center",
        ha="left",
        fontsize=style.font_size_label,
        family=style.font_label,
        color=style.color_sequence,
        alpha=0.9,
    )

    prop = FontProperties(family=style.font_mono, size=style.font_size_seq)
    px_per_pt = style.dpi / 72.0
    ag = TextPath((0, 0), "Ag", prop=prop, usetex=False).get_extents()
    y_mid_px = ((ag.y0 + ag.y1) / 2.0) * px_per_pt

    cache: dict[str, TextPath] = {}
    x = x0
    for char in seq:
        tp = cache.get(char)
        if tp is None:
            tp = TextPath((0, 0), char, prop=prop, usetex=False)
            cache[char] = tp
        trans = Affine2D().scale(px_per_pt).translate(x, y_center - y_mid_px) + ax.transData
        ax.add_patch(
            PathPatch(
                tp,
                transform=trans,
                facecolor=style.color_sequence,
                edgecolor="none",
                linewidth=0.0,
                zorder=2,
                clip_on=False,
            )
        )
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
    above: bool,
    cw: float,
) -> None:
    r = style.kmer.round_px
    pad_x = float(style.kmer.pad_x_px)
    yy = y + (h * 0.5 if above else -h * 0.5)

    ax.add_patch(
        FancyBboxPatch(
            (x - pad_x, yy - h / 2),
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

    prop = FontProperties(family=style.font_mono, size=style.font_size_seq)
    px_per_pt = style.dpi / 72.0
    ag = TextPath((0, 0), "Ag", prop=prop, usetex=False).get_extents()
    y_mid_px = ((ag.y0 + ag.y1) / 2.0) * px_per_pt
    cache: dict[str, TextPath] = {}

    xx = x
    for char in label:
        tp = cache.get(char)
        if tp is None:
            tp = TextPath((0, 0), char, prop=prop, usetex=False)
            cache[char] = tp
        align = str(style.kmer.text_v_align).lower()
        y_text_center = yy if align == "center" else y
        y_text_center += float(style.kmer.text_v_offset_px)
        trans = Affine2D().scale(px_per_pt).translate(xx, y_text_center - y_mid_px) + ax.transData
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
        xx += cw


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
