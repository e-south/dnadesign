"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_review/fragment_geometry.py

Shared strand alignment for Junction fragment review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.contracts.visual.three_way_junction_review_v1 import (
    FragmentGeometry,
    FragmentStrands,
    ThreeWayJunctionReviewV1,
)

from .foundation import BARCODE, DOMAIN, PRIMER_BINDING_SITE, TOEHOLD

ComponentSpan = tuple[int, int, str, str]


@dataclass(frozen=True, slots=True)
class FragmentPairGeometry:
    """One antiparallel fragment pair aligned on its target-derived domain."""

    fragment: FragmentGeometry
    strand: FragmentStrands
    bottom_sequence_left_to_right: str
    top_offset: int
    bottom_offset: int
    paired_start: int
    paired_length: int
    top_spans: tuple[ComponentSpan, ...]
    bottom_spans: tuple[ComponentSpan, ...]

    @property
    def width(self) -> int:
        return max(
            self.top_offset + len(self.strand.barcode_bearing_sequence_5to3),
            self.bottom_offset + len(self.bottom_sequence_left_to_right),
        )


def fragment_pair_geometry(review: ThreeWayJunctionReviewV1, index: int) -> FragmentPairGeometry:
    """Return exact display geometry without changing either stored 5-prime sequence."""

    previous = None if index == 0 else review.geometry.junctions[index - 1]
    following = None if index == len(review.strands) - 1 else review.geometry.junctions[index]
    fragment = review.geometry.fragments[index]
    strand = review.strands[index]
    domain_length = fragment.domain_span.end - fragment.domain_span.start
    top_domain_start = 0 if previous is None else len(previous.barcode)
    bottom_domain_start = 0 if previous is None else len(previous.toehold)
    aligned_domain_start = max(top_domain_start, bottom_domain_start)
    top_spans: list[ComponentSpan] = []
    bottom_spans: list[ComponentSpan] = []
    if previous is not None:
        top_spans.append((0, len(previous.barcode), BARCODE, f"b{index}*"))
        bottom_spans.append((0, len(previous.toehold), TOEHOLD, f"t{index}*"))
    top_spans.append((top_domain_start, top_domain_start + domain_length, DOMAIN, "target"))
    bottom_spans.append((bottom_domain_start, bottom_domain_start + domain_length, DOMAIN, "target"))
    for primer in (review.recovery.forward, review.recovery.reverse):
        overlap_start = max(fragment.domain_span.start, primer.target_binding_span.start)
        overlap_end = min(fragment.domain_span.end, primer.target_binding_span.end)
        if overlap_start < overlap_end:
            relative_start = overlap_start - fragment.domain_span.start
            relative_end = overlap_end - fragment.domain_span.start
            label = f"{primer.direction} primer-binding site"
            top_spans.append(
                (top_domain_start + relative_start, top_domain_start + relative_end, PRIMER_BINDING_SITE, label)
            )
            bottom_spans.append(
                (bottom_domain_start + relative_start, bottom_domain_start + relative_end, PRIMER_BINDING_SITE, label)
            )
    if following is not None:
        start = top_domain_start + domain_length
        top_spans.extend(
            (
                (start, start + len(following.toehold), TOEHOLD, f"t{index + 1}"),
                (
                    start + len(following.toehold),
                    start + len(following.toehold) + len(following.barcode),
                    BARCODE,
                    f"b{index + 1}",
                ),
            )
        )
    return FragmentPairGeometry(
        fragment=fragment,
        strand=strand,
        bottom_sequence_left_to_right=strand.complement_sequence_5to3[::-1],
        top_offset=aligned_domain_start - top_domain_start,
        bottom_offset=aligned_domain_start - bottom_domain_start,
        paired_start=aligned_domain_start,
        paired_length=domain_length,
        top_spans=tuple(top_spans),
        bottom_spans=tuple(bottom_spans),
    )


__all__ = ["ComponentSpan", "FragmentPairGeometry", "fragment_pair_geometry"]
