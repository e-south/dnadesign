"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/preligation_junction.py

Exact barcode helix, nick, and termini for one pre-ligation junction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ..sequence_preview import bounded_svg_gid
from .foundation import (
    BACKGROUND,
    BARCODE,
    BARCODE_DARK,
    INK,
    MOLECULAR_ANNOTATION_FONTSIZE,
    MUTED,
    PAIR,
    display_junction_id,
)
from .primitives import draw_compact_base_run, draw_molecular_path


def _draw_internal_termini(
    axis,
    *,
    stem_x: float,
    nick_x: float,
    stem_top: float,
    bottom_y: float,
    gid_prefix: str,
) -> None:
    labels = (
        (stem_x - 0.006, stem_top + 0.025, "3′", "top-left", "right", "bottom"),
        (stem_x + 0.006, stem_top + 0.025, "5′", "top-right", "left", "bottom"),
        (nick_x - 0.004, bottom_y - 0.15, "5′", "bottom-left", "right", "top"),
        (nick_x + 0.004, bottom_y - 0.15, "3′", "bottom-right", "left", "top"),
    )
    for label_x, label_y, text, role, horizontal, vertical in labels:
        artist = axis.text(
            label_x,
            label_y,
            text,
            fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
            color=MUTED,
            ha=horizontal,
            va=vertical,
        )
        artist.set_gid(bounded_svg_gid(f"{gid_prefix}:terminus:{role}"))


def draw_preligation_junction(
    axis,
    junction,
    *,
    left: float,
    base_step: float,
    top_y: float,
    bottom_y: float,
    barcode_base_step_y: float,
) -> None:
    """Draw one exact antiparallel barcode helix and its four strand termini."""

    position = junction.toehold_span.end
    x = left + position * base_step
    offset = base_step * 0.62
    stem_start = top_y + 0.12
    stem_top = stem_start + len(junction.barcode) * barcode_base_step_y
    base_fontsize = max(3.2, min(8.6, barcode_base_step_y * 72.0 * 0.96))
    strand_linewidth = max(3.8, min(8.5, barcode_base_step_y * 72.0))
    prefix = f"junction-three-way-assembly:three-way:{junction.junction_id}"
    for stem_x, sequence, direction, role in (
        (x - offset, junction.barcode, barcode_base_step_y, "barcode"),
        (x + offset, junction.barcode_complement, -barcode_base_step_y, "barcode-complement"),
    ):
        draw_molecular_path(
            axis,
            (stem_x, stem_x),
            (top_y, stem_top),
            color=BARCODE,
            gid=f"{prefix}:{role}:path",
            linewidth=strand_linewidth,
        )
        draw_compact_base_run(
            axis,
            sequence,
            start_x=stem_x,
            start_y=stem_start if direction > 0 else stem_top,
            delta_x=0.0,
            delta_y=direction,
            gid_prefix=f"{prefix}:{role}",
            fontsize=base_fontsize,
        )
    pair_segments = [
        (
            (x - offset, stem_start + (index + 0.5) * barcode_base_step_y),
            (x + offset, stem_start + (index + 0.5) * barcode_base_step_y),
        )
        for index in range(len(junction.barcode))
    ]
    pairs = LineCollection(pair_segments, colors=PAIR, linewidths=0.55, zorder=1)
    pairs.set_gid(bounded_svg_gid(f"{prefix}:barcode-pairs"))
    axis.add_collection(pairs)
    gap = Line2D((x, x), (top_y - 0.08, top_y + 0.08), color=BACKGROUND, linewidth=2.2, zorder=2)
    gap.set_gid(bounded_svg_gid(f"{prefix}:top-gap"))
    axis.add_line(gap)
    nick_x = left + junction.toehold_span.start * base_step
    axis.add_line(
        Line2D(
            (nick_x - 0.003, nick_x + 0.003),
            (bottom_y - 0.07, bottom_y + 0.07),
            color=INK,
            linewidth=1.0,
            zorder=4,
        )
    )
    _draw_internal_termini(
        axis,
        stem_x=x,
        nick_x=nick_x,
        stem_top=stem_top,
        bottom_y=bottom_y,
        gid_prefix=prefix,
    )
    label = axis.text(
        x + offset + max(0.012, base_step),
        (stem_start + stem_top) / 2,
        display_junction_id(junction.junction_id),
        fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
        color=BARCODE_DARK,
        ha="left",
        va="center",
    )
    label.set_gid(bounded_svg_gid(f"{prefix}:label"))


__all__ = ["draw_preligation_junction"]
