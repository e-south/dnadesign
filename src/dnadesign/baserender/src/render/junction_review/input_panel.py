"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/input_panel.py

Exact submitted-target row for the Junction assembly-process view.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

from ..sequence_preview import bounded_svg_gid
from .assembly_geometry import AssemblyLayout
from .foundation import (
    INPUT_SPECIFICATION,
    INPUT_SPECIFICATION_EDGE,
    MOLECULAR_ANNOTATION_FONTSIZE,
    PRIMER_BINDING_SITE,
    PRIMER_BINDING_SITE_DARK,
)
from .primitives import draw_compact_base_run, draw_segmented_strand, draw_terminus


def draw_input_target(
    axis,
    review: ThreeWayJunctionReviewV1,
    *,
    y: float,
    layout: AssemblyLayout,
) -> None:
    """Draw the exact submitted 5-prime-to-3-prime target before fragmentation."""

    sequence = review.target.sequence_5to3
    base_step = layout.target_base_step
    left = layout.target_left
    binding_spans = tuple(
        (
            primer.target_binding_span.start,
            primer.target_binding_span.end,
            PRIMER_BINDING_SITE,
        )
        for primer in (review.recovery.forward, review.recovery.reverse)
    )
    draw_segmented_strand(
        axis,
        start_x=left,
        center_y=y,
        base_step=base_step,
        length=len(sequence),
        segments=binding_spans,
        height=0.14,
        gid_prefix="junction-three-way-assembly:input:strand",
        body_color=INPUT_SPECIFICATION,
        outline_color=INPUT_SPECIFICATION_EDGE,
        outline_linestyle=(0, (3, 2)),
    )
    draw_compact_base_run(
        axis,
        sequence,
        start_x=left,
        start_y=y,
        delta_x=base_step,
        delta_y=0.0,
        gid_prefix="junction-three-way-assembly:input",
        fontsize=layout.target_fontsize,
    )
    draw_terminus(
        axis,
        x=left - 0.008,
        y=y,
        text="5′",
        ha="right",
        gid="junction-three-way-assembly:input:terminus:left",
    )
    draw_terminus(
        axis,
        x=left + len(sequence) * base_step + 0.008,
        y=y,
        text="3′",
        ha="left",
        gid="junction-three-way-assembly:input:terminus:right",
    )
    for primer in (review.recovery.forward, review.recovery.reverse):
        start = primer.target_binding_span.start
        end = primer.target_binding_span.end
        label = axis.text(
            left + (start + end) * base_step / 2,
            y + 0.18,
            f"{primer.direction} primer-binding site",
            fontsize=MOLECULAR_ANNOTATION_FONTSIZE,
            color=PRIMER_BINDING_SITE_DARK,
            ha="center",
            va="bottom",
        )
        label.set_gid(bounded_svg_gid(f"junction-three-way-assembly:input:primer-site:{primer.direction}"))


__all__ = ["draw_input_target"]
