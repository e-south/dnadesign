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


def review_content_height(review: ThreeWayJunctionReviewV1) -> float:
    """Estimate the exact content height before Matplotlib allocates a canvas."""

    target_rows = _chunk_count(review.target.sequence_5to3)
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
    return max(
        6.5,
        2.32
        + target_rows * 0.46
        + len(review.geometry.junctions) * 0.20
        + junction_pair_rows * 0.36
        + len(review.strands) * 0.14
        + oligo_rows * 0.19
        + primer_rows * 0.19,
    )


__all__ = ["BASES_PER_ROW", "SequenceChunk", "complement", "review_content_height", "sequence_chunks"]
