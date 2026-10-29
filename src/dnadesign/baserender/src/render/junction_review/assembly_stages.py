"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/assembly_stages.py

Separate-oligo and expected three-way stages for Junction assembly review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_geometry import ORDER_GROUP_GAP_BASES, AssemblyLayout
from .foundation import DOMAIN, INK, MOLECULAR_ANNOTATION_FONTSIZE, MUTED, PRIMER_BINDING_SITE, TOEHOLD
from .fragment_geometry import fragment_pair_geometry
from .primitives import draw_compact_base_run, draw_segmented_strand


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
                *(
                    (
                        primer.target_binding_span.start + offset,
                        primer.target_binding_span.end + offset,
                        PRIMER_BINDING_SITE,
                    )
                    for primer in (review.recovery.forward, review.recovery.reverse)
                ),
            ),
            key=lambda item: item[0],
        )
    )


def draw_orders_stage(
    axis,
    review: ThreeWayJunctionReviewV1,
    *,
    y: float,
    layout: AssemblyLayout,
) -> None:
    """Draw fragment pairs left-to-right at the shared molecular base scale."""

    geometries = tuple(fragment_pair_geometry(review, index) for index in range(len(review.strands)))
    base_step = layout.order_base_step
    cursor = layout.order_left
    for index, geometry in enumerate(geometries):
        top = geometry.strand.barcode_bearing_sequence_5to3
        bottom = geometry.bottom_sequence_left_to_right
        row_start = cursor
        top_start = row_start + geometry.top_offset * base_step
        bottom_start = row_start + geometry.bottom_offset * base_step
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
                height=0.14,
                gid_prefix=f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:{role}",
            )
        for sequence, start_x, strand_y, role in (
            (top, top_start, y, "top"),
            (bottom, bottom_start, y - 0.22, "bottom"),
        ):
            draw_compact_base_run(
                axis,
                sequence,
                start_x=start_x,
                start_y=strand_y,
                delta_x=base_step,
                delta_y=0.0,
                gid_prefix=f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:{role}",
                fontsize=layout.order_fontsize,
            )
        _draw_termini(
            axis,
            start_x=top_start,
            end_x=top_start + len(top) * base_step,
            y=y,
            left_label="5′",
            right_label="3′",
            gid_prefix=f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:top",
        )
        _draw_termini(
            axis,
            start_x=bottom_start,
            end_x=bottom_start + len(bottom) * base_step,
            y=y - 0.22,
            left_label="3′",
            right_label="5′",
            gid_prefix=f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:bottom",
        )
        label = axis.text(
            row_start + geometry.width * base_step / 2,
            y + 0.18,
            f"F{index + 1:02d}",
            fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
            color=INK,
            ha="center",
            va="bottom",
        )
        label.set_gid(bounded_svg_gid(f"junction-three-way-assembly:orders:{geometry.fragment.fragment_id}:label"))
        cursor += (geometry.width + ORDER_GROUP_GAP_BASES) * base_step


def _draw_termini(
    axis,
    *,
    start_x: float,
    end_x: float,
    y: float,
    left_label: str,
    right_label: str,
    gid_prefix: str,
) -> None:
    for x, label, side in ((start_x - 0.006, left_label, "left"), (end_x + 0.006, right_label, "right")):
        artist = axis.text(
            x,
            y,
            label,
            fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
            color=MUTED,
            ha="right" if side == "left" else "left",
            va="center",
        )
        artist.set_gid(bounded_svg_gid(f"{gid_prefix}:terminus:{side}"))


__all__ = ["draw_orders_stage", "target_spans"]
