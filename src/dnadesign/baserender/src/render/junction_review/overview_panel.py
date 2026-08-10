"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/overview_panel.py

Target-scale composition for the three-way-junction assembly overview.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from .foundation import (
    BARCODE,
    BARCODE_DARK,
    DOMAIN,
    INK,
    MUTED,
    PAIR,
    TOEHOLD,
    TOEHOLD_DARK,
    display_junction_id,
    fragment_order_lengths,
    length_summary,
    safe_identifier,
)
from .primitives import draw_molecular_path, draw_segmented_strand


def draw_overview(axis, review: ThreeWayJunctionReviewV1) -> None:
    """Draw one bounded locator map on target coordinates."""

    axis.set_gid("junction-three-way-assembly:overview")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fragment_count = len(review.geometry.fragments)
    junction_count = len(review.geometry.junctions)
    target_id = safe_identifier(review.target.target_id)
    axis.text(
        0.025,
        0.95,
        (
            f"{junction_count} three-way "
            f"{'junction links' if junction_count == 1 else 'junctions link'} "
            f"{fragment_count} fragments across {target_id}"
        ),
        fontsize=15.0,
        fontweight="semibold",
        color=INK,
        va="top",
    )
    lengths = fragment_order_lengths(review)
    axis.text(
        0.025,
        0.84,
        (
            f"The {len(review.target.sequence_5to3)} bp target uses "
            f"{len(lengths)} fragment oligos spanning {length_summary(lengths)}"
        ),
        fontsize=9.0,
        color=MUTED,
        va="top",
    )
    left, right = 0.055, 0.965
    top_y, bottom_y = 0.47, 0.37
    target_length = len(review.target.sequence_5to3)
    base_step = (right - left) / target_length

    def x_for(coordinate: int) -> float:
        return left + base_step * coordinate

    target_segments = tuple(
        sorted(
            (
                *(
                    (fragment.domain_span.start, fragment.domain_span.end, DOMAIN)
                    for fragment in review.geometry.fragments
                ),
                *(
                    (junction.toehold_span.start, junction.toehold_span.end, TOEHOLD)
                    for junction in review.geometry.junctions
                ),
            ),
            key=lambda item: item[0],
        )
    )
    for y, role in ((top_y, "top"), (bottom_y, "bottom")):
        draw_segmented_strand(
            axis,
            start_x=left,
            center_y=y,
            base_step=base_step,
            length=target_length,
            segments=target_segments,
            height=0.052,
            gid_prefix=f"junction-three-way-assembly:target:{role}",
        )
    for fragment in review.geometry.fragments:
        x0, x1 = x_for(fragment.domain_span.start), x_for(fragment.domain_span.end)
        axis.text(
            (x0 + x1) / 2,
            0.265,
            f"F{fragment.index + 1:02d} · {fragment.role}",
            fontsize=7.2,
            color=INK,
            ha="center",
            va="top",
        )
    for junction in review.geometry.junctions:
        x = x_for(junction.toehold_span.end)
        stem_offset = 0.0038
        stem_top = 0.69
        draw_molecular_path(
            axis,
            [x - stem_offset, x - stem_offset],
            [top_y, stem_top],
            color=BARCODE,
            gid=f"junction-three-way-assembly:{junction.junction_id}:barcode",
            linewidth=7.0,
        )
        draw_molecular_path(
            axis,
            [x + stem_offset, x + stem_offset],
            [stem_top, top_y],
            color=BARCODE,
            gid=f"junction-three-way-assembly:{junction.junction_id}:barcode-complement",
            linewidth=7.0,
        )
        pairs = LineCollection(
            [((x - stem_offset + 0.001, y), (x + stem_offset - 0.001, y)) for y in (0.54, 0.59, 0.64)],
            colors=PAIR,
            linewidths=0.75,
            zorder=1,
        )
        axis.add_collection(pairs)
        axis.text(
            x,
            0.735,
            display_junction_id(junction.junction_id),
            fontsize=7.4,
            color=BARCODE_DARK,
            ha="center",
            va="bottom",
        )
        axis.text(
            x,
            0.705,
            f"bp {junction.toehold_span.start + 1}–{junction.toehold_span.end}",
            fontsize=6.2,
            color=MUTED,
            ha="center",
            va="bottom",
        )
        nick_x = x_for(junction.toehold_span.start)
        axis.add_line(
            Line2D(
                [nick_x - 0.0025, nick_x + 0.0025],
                [bottom_y - 0.032, bottom_y + 0.032],
                linewidth=1.1,
                color=INK,
                zorder=4,
            )
        )
    axis.text(left - 0.008, top_y, "5′", fontsize=7.2, color=MUTED, ha="right", va="center")
    axis.text(right + 0.008, top_y, "3′", fontsize=7.2, color=MUTED, ha="left", va="center")
    axis.text(left - 0.008, bottom_y, "3′", fontsize=7.2, color=MUTED, ha="right", va="center")
    axis.text(right + 0.008, bottom_y, "5′", fontsize=7.2, color=MUTED, ha="left", va="center")
    axis.text(left, 0.57, "barcode stems", fontsize=7.0, color=BARCODE_DARK, ha="left", va="center")
    axis.text(left, 0.43, "toeholds", fontsize=7.0, color=TOEHOLD_DARK, ha="left", va="center")
    axis.text(
        0.025,
        0.08,
        "Each stable junction ID opens an exact nucleotide-level view of the expected local annealing geometry",
        fontsize=8.0,
        color=MUTED,
        va="bottom",
    )
    axis.text(
        0.975,
        0.08,
        "The map is sequence-derived and does not establish assembly success",
        fontsize=8.0,
        color=MUTED,
        ha="right",
        va="bottom",
    )
