"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/preligation_panel.py

Exact nucleotide geometry for the Junction pre-ligation assembly state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.collections import LineCollection

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_geometry import AssemblyLayout
from .assembly_stages import target_spans
from .foundation import MOLECULAR_ANNOTATION_FONTSIZE, MUTED, PAIR
from .preligation_junction import draw_preligation_junction
from .primitives import draw_compact_base_run, draw_segmented_strand


def _fragment_intervals(review: ThreeWayJunctionReviewV1):
    for index, fragment in enumerate(review.geometry.fragments):
        previous = None if index == 0 else review.geometry.junctions[index - 1]
        following = None if index == len(review.geometry.fragments) - 1 else review.geometry.junctions[index]
        yield (
            fragment,
            (
                fragment.domain_span.start,
                fragment.domain_span.end if following is None else following.toehold_span.end,
            ),
            (
                fragment.domain_span.start if previous is None else previous.toehold_span.start,
                fragment.domain_span.end,
            ),
        )


def _draw_fragment_bodies(axis, review, *, left: float, base_step: float, top_y: float, bottom_y: float) -> None:
    spans = target_spans(review)
    for fragment, top_interval, bottom_interval in _fragment_intervals(review):
        for (start, end), strand_y, role in (
            (top_interval, top_y, "top"),
            (bottom_interval, bottom_y, "bottom"),
        ):
            draw_segmented_strand(
                axis,
                start_x=left + start * base_step,
                center_y=strand_y,
                base_step=base_step,
                length=end - start,
                segments=tuple(
                    (max(span_start, start) - start, min(span_end, end) - start, color)
                    for span_start, span_end, color in spans
                    if span_start < end and span_end > start
                ),
                height=0.14,
                gid_prefix=(f"junction-three-way-assembly:three-way:fragment:{fragment.fragment_id}:{role}"),
            )


def _draw_target_bases(
    axis,
    review,
    *,
    left: float,
    base_step: float,
    top_y: float,
    bottom_y: float,
    fontsize: float,
) -> None:
    top = review.target.sequence_5to3
    bottom = top.translate(str.maketrans("ACGT", "TGCA"))
    for sequence, strand_y, role in ((top, top_y, "top"), (bottom, bottom_y, "bottom")):
        draw_compact_base_run(
            axis,
            sequence,
            start_x=left,
            start_y=strand_y,
            delta_x=base_step,
            delta_y=0.0,
            gid_prefix=f"junction-three-way-assembly:three-way:{role}",
            fontsize=fontsize,
        )
    pairs = LineCollection(
        [
            (
                (left + (index + 0.5) * base_step, top_y - 0.073),
                (left + (index + 0.5) * base_step, bottom_y + 0.073),
            )
            for index in range(len(top))
        ],
        colors=PAIR,
        linewidths=0.45,
        zorder=1,
    )
    pairs.set_gid(bounded_svg_gid("junction-three-way-assembly:three-way:target-pairs"))
    axis.add_collection(pairs)


def _draw_target_termini(axis, *, left: float, end_x: float, top_y: float, bottom_y: float) -> None:
    for x, strand_y, text, role, horizontal in (
        (left - 0.006, top_y, "5′", "top-left", "right"),
        (left - 0.006, bottom_y, "3′", "bottom-left", "right"),
        (end_x + 0.006, top_y, "3′", "top-right", "left"),
        (end_x + 0.006, bottom_y, "5′", "bottom-right", "left"),
    ):
        artist = axis.text(
            x,
            strand_y,
            text,
            fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
            color=MUTED,
            ha=horizontal,
            va="center",
        )
        artist.set_gid(bounded_svg_gid(f"junction-three-way-assembly:three-way:terminus:{role}"))


def draw_preligation_stage(
    axis,
    review: ThreeWayJunctionReviewV1,
    *,
    first_y: float,
    layout: AssemblyLayout,
) -> None:
    """Draw one continuous pre-ligation target and its exact barcode helices."""

    target_length = len(review.target.sequence_5to3)
    base_step = layout.target_base_step
    left = layout.target_left
    top_y = first_y
    bottom_y = top_y - 0.22
    _draw_fragment_bodies(
        axis,
        review,
        left=left,
        base_step=base_step,
        top_y=top_y,
        bottom_y=bottom_y,
    )
    _draw_target_bases(
        axis,
        review,
        left=left,
        base_step=base_step,
        top_y=top_y,
        bottom_y=bottom_y,
        fontsize=layout.target_fontsize,
    )
    for junction in review.geometry.junctions:
        draw_preligation_junction(
            axis,
            junction,
            left=left,
            base_step=base_step,
            top_y=top_y,
            bottom_y=bottom_y,
            barcode_base_step_y=layout.barcode_base_step_y,
        )
    _draw_target_termini(
        axis,
        left=left,
        end_x=left + target_length * base_step,
        top_y=top_y,
        bottom_y=bottom_y,
    )


__all__ = ["draw_preligation_stage"]
