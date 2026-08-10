"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/detail_geometry.py

Bounded local geometry for one nucleotide-level three-way-junction detail.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.contracts.visual.three_way_junction_review_v1 import (
    FragmentGeometry,
    JunctionGeometry,
    ThreeWayJunctionReviewV1,
)

CONTEXT_BASES = 6
MAX_RIGHT_CONTEXT_BASES = 16
STRAND_WIDTH = 11.0
COMPONENT_WIDTH = 8.5
TOP_Y = 0.0
BOTTOM_Y = -2.6
STEM_LEFT_X = -0.72
STEM_RIGHT_X = 0.72
STEM_START_Y = 1.0


@dataclass(frozen=True, slots=True)
class LocalJunctionGeometry:
    junction: JunctionGeometry
    left_fragment: FragmentGeometry
    right_fragment: FragmentGeometry
    left_context: str
    right_context: str
    left_is_terminal: bool
    right_is_terminal: bool


def local_junction_geometry(review: ThreeWayJunctionReviewV1, index: int) -> LocalJunctionGeometry:
    """Project one exact interface into a bounded, symmetric review window."""

    junction = review.geometry.junctions[index]
    left_fragment = review.geometry.fragments[index]
    right_fragment = review.geometry.fragments[index + 1]
    target = review.target.sequence_5to3
    left_start = max(left_fragment.domain_span.start, junction.toehold_span.start - CONTEXT_BASES)
    right_context_bases = min(CONTEXT_BASES + len(junction.toehold), MAX_RIGHT_CONTEXT_BASES)
    right_end = min(right_fragment.domain_span.end, junction.toehold_span.end + right_context_bases)
    return LocalJunctionGeometry(
        junction=junction,
        left_fragment=left_fragment,
        right_fragment=right_fragment,
        left_context=target[left_start : junction.toehold_span.start],
        right_context=target[junction.toehold_span.end : right_end],
        left_is_terminal=left_start == 0,
        right_is_terminal=right_end == len(target),
    )


def junction_detail_base_glyph_count(review: ThreeWayJunctionReviewV1, index: int) -> int:
    """Count the per-base text artists required by one local detail view."""

    geometry = local_junction_geometry(review, index)
    junction = geometry.junction
    return 2 * (
        len(geometry.left_context) + len(junction.toehold) + len(geometry.right_context) + len(junction.barcode)
    )
