"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/junction_pairing_layout.py

Bounded layout calculations for nucleotide-level Junction review figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1

BASES_PER_ROW = 120


@dataclass(frozen=True)
class SequenceChunk:
    """One exact, ordered slice of a sequence."""

    start: int
    end: int
    sequence: str


def sequence_chunks(sequence: str, *, width: int = BASES_PER_ROW) -> tuple[SequenceChunk, ...]:
    """Split a non-empty sequence into exact, bounded display rows."""

    if width < 1:
        raise ValueError("sequence chunk width must be positive")
    return tuple(
        SequenceChunk(start=start, end=min(start + width, len(sequence)), sequence=sequence[start : start + width])
        for start in range(0, len(sequence), width)
    )


def complement(sequence: str) -> str:
    """Return the aligned Watson-Crick complement without reversing it."""

    return sequence.translate(str.maketrans("ACGT", "TGCA"))


def _chunk_count(sequence: str) -> int:
    return math.ceil(len(sequence) / BASES_PER_ROW)


def _annealed_display_width(review: ThreeWayJunctionReviewV1, index: int) -> int:
    strand = review.strands[index]
    previous = None if index == 0 else review.geometry.junctions[index - 1]
    top_domain_start = 0 if previous is None else len(previous.barcode)
    bottom_domain_start = 0 if previous is None else len(previous.toehold)
    aligned_start = max(top_domain_start, bottom_domain_start)
    return max(
        aligned_start - top_domain_start + len(strand.barcode_bearing_sequence_5to3),
        aligned_start - bottom_domain_start + len(strand.complement_sequence_5to3),
    )


def expanded_annealed_fragments(review: ThreeWayJunctionReviewV1) -> bool:
    """Return whether every annealed fragment fits a useful expanded view."""

    return len(review.strands) <= 3 and all(
        _annealed_display_width(review, index) <= BASES_PER_ROW for index in range(len(review.strands))
    )


def review_content_height(review: ThreeWayJunctionReviewV1) -> float:
    """Estimate the exact content height before Matplotlib allocates a canvas."""

    recovered_rows = _chunk_count(review.recovery.extended_top_sequence_5to3)
    junction_pair_rows = sum(
        _chunk_count(junction.toehold) + _chunk_count(junction.barcode) for junction in review.geometry.junctions
    )
    oligo_rows = sum(
        _chunk_count(strand.barcode_bearing_sequence_5to3) + _chunk_count(strand.complement_sequence_5to3)
        for strand in review.strands
    )
    primer_rows = _chunk_count(review.recovery.forward.order_sequence_5to3) + _chunk_count(
        review.recovery.reverse.order_sequence_5to3
    )
    annealed_height = 0.34
    if expanded_annealed_fragments(review):
        annealed_height = 0.27 + len(review.strands) * 0.61
    return max(
        6.5,
        2.60
        + recovered_rows * 0.46
        + len(review.geometry.junctions) * 0.20
        + junction_pair_rows * 0.36
        + len(review.strands) * 0.14
        + oligo_rows * 0.19
        + primer_rows * 0.19
        + annealed_height,
    )


__all__ = [
    "BASES_PER_ROW",
    "SequenceChunk",
    "complement",
    "expanded_annealed_fragments",
    "review_content_height",
    "sequence_chunks",
]
