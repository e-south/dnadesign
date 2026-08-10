"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/assembly_stages.py

Separate-oligo and expected three-way stages for Junction assembly review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .foundation import BARCODE, BARCODE_DARK, DOMAIN, INK, MUTED, PAIR, TOEHOLD, display_junction_id
from .fragment_geometry import fragment_pair_geometry
from .primitives import draw_molecular_path, draw_segmented_strand


def target_spans(review: ThreeWayJunctionReviewV1, *, offset: int = 0):
    """Return the complete target partition as categorical display spans."""

    return tuple(
        sorted(
            (
                *(
                    (fragment.domain_span.start + offset, fragment.domain_span.end + offset, DOMAIN)
                    for fragment in review.geometry.fragments
                ),
                *(
                    (junction.toehold_span.start + offset, junction.toehold_span.end + offset, TOEHOLD)
                    for junction in review.geometry.junctions
                ),
            ),
            key=lambda item: item[0],
        )
    )


def draw_orders_stage(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> None:
    """Draw each order as its own single strand without inventing annealing."""

    geometries = tuple(fragment_pair_geometry(review, index) for index in range(len(review.strands)))
    left, right, gap = 0.08, 0.96, 0.010
    base_step = (right - left - gap * (len(geometries) - 1)) / sum(geometry.width for geometry in geometries)
    cursor = left
    for index, geometry in enumerate(geometries):
        top = geometry.strand.barcode_bearing_sequence_5to3
        bottom = geometry.bottom_sequence_left_to_right
        top_start = cursor + (geometry.width - len(top)) * base_step / 2
        bottom_start = cursor + (geometry.width - len(bottom)) * base_step / 2
        for start_x, center_y, length, spans, role in (
            (top_start, y, len(top), geometry.top_spans, "top"),
            (bottom_start, y - 0.22, len(bottom), geometry.bottom_spans, "bottom"),
        ):
            draw_segmented_strand(
                axis,
                start_x=start_x,
                center_y=center_y,
                base_step=base_step,
                length=length,
                segments=tuple((a, b, color) for a, b, color, _ in spans),
                height=0.075,
                gid_prefix=f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:{role}",
            )
        label = axis.text(
            cursor + geometry.width * base_step / 2,
            y + 0.13,
            f"F{index + 1:02d}",
            fontsize=9.0,
            color=INK,
            ha="center",
            va="bottom",
        )
        label.set_gid(bounded_svg_gid(f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:label"))
        cursor += geometry.width * base_step + gap
    orientation = axis.text(
        0.5,
        y - 0.38,
        "Upper oligos run 5′→3′; lower oligos are shown antiparallel from 3′→5′",
        fontsize=9.4,
        color=MUTED,
        ha="center",
        va="top",
    )
    orientation.set_gid(bounded_svg_gid("junction-three-way-assembly:orders:orientation"))


def draw_three_way_stage(axis, review: ThreeWayJunctionReviewV1, *, y: float) -> None:
    """Draw both target strands and every planned external barcode helix."""

    left, right = 0.08, 0.96
    target_length = len(review.target.sequence_5to3)
    base_step = (right - left) / target_length
    spans = target_spans(review)
    top_y, bottom_y = y, y - 0.20
    for strand_y, role in ((top_y, "top"), (bottom_y, "bottom")):
        draw_segmented_strand(
            axis,
            start_x=left,
            center_y=strand_y,
            base_step=base_step,
            length=target_length,
            segments=spans,
            height=0.075,
            gid_prefix=f"junction-three-way-assembly:three-way:target:{role}",
        )
    for junction in review.geometry.junctions:
        x = left + junction.toehold_span.end * base_step
        stem_offset = 0.0035
        stem_top = y + 0.34
        for stem_x, direction in ((x - stem_offset, "barcode"), (x + stem_offset, "barcode-complement")):
            draw_molecular_path(
                axis,
                (stem_x, stem_x),
                (top_y, stem_top),
                color=BARCODE,
                gid=f"junction-three-way-assembly:three-way:{junction.junction_id}:{direction}",
                linewidth=6.0,
            )
        pairs = LineCollection(
            [((x - stem_offset, level), (x + stem_offset, level)) for level in (y + 0.10, y + 0.19, y + 0.28)],
            colors=PAIR,
            linewidths=0.65,
            zorder=1,
        )
        pairs.set_gid(bounded_svg_gid(f"junction-three-way-assembly:three-way:{junction.junction_id}:pairs"))
        axis.add_collection(pairs)
        label = axis.text(
            x,
            stem_top + 0.06,
            display_junction_id(junction.junction_id),
            fontsize=8.6,
            color=BARCODE_DARK,
            ha="center",
            va="bottom",
        )
        label.set_gid(bounded_svg_gid(f"junction-three-way-assembly:three-way:{junction.junction_id}:label"))
        nick_x = left + junction.toehold_span.start * base_step
        axis.add_line(
            Line2D(
                (nick_x - 0.0025, nick_x + 0.0025),
                (bottom_y - 0.035, bottom_y + 0.035),
                color=INK,
                linewidth=1.0,
            )
        )
    axis.text(left - 0.008, top_y, "5′", fontsize=9.4, color=MUTED, ha="right", va="center")
    axis.text(right + 0.008, top_y, "3′", fontsize=9.4, color=MUTED, va="center")
    axis.text(left - 0.008, bottom_y, "3′", fontsize=9.4, color=MUTED, ha="right", va="center")
    axis.text(right + 0.008, bottom_y, "5′", fontsize=9.4, color=MUTED, va="center")


__all__ = ["draw_orders_stage", "draw_three_way_stage", "target_spans"]
