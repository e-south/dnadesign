"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/metrics.py

Bounded pairwise metrics for Junction sequence review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dnadesign.junction.errors import JunctionConfigError
from dnadesign.junction.sequence import (
    levenshtein_distance,
    longest_common_substring_length,
    position_weighted_levenshtein,
)

from .contract import JunctionSequenceDissimilarityV1

MAX_SELECTED_JUNCTIONS = 24
MAX_DYNAMIC_PROGRAMMING_CELLS = 20_000_000


@dataclass(frozen=True, slots=True)
class DissimilaritySelection:
    indices: tuple[int, ...]
    selected_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class DissimilarityMatrices:
    toehold: np.ndarray
    barcode: np.ndarray
    combined: np.ndarray


def resolve_selection(
    review: JunctionSequenceDissimilarityV1,
    junction_ids: Sequence[str] | None,
) -> DissimilaritySelection:
    available = tuple(junction.junction_id for junction in review.junctions)
    if junction_ids is None:
        if len(available) > MAX_SELECTED_JUNCTIONS:
            raise JunctionConfigError(
                f"assembly group has {len(available)} junctions; choose at most {MAX_SELECTED_JUNCTIONS} junction_ids"
            )
        selected = available
    else:
        selected = tuple(str(junction_id).strip() for junction_id in junction_ids)
        if not selected or any(not junction_id for junction_id in selected):
            raise JunctionConfigError("junction_ids must contain at least one non-empty ID")
        if len(selected) != len(set(selected)):
            raise JunctionConfigError("junction_ids must not contain duplicates")
        if len(selected) > MAX_SELECTED_JUNCTIONS:
            raise JunctionConfigError(f"junction_ids must choose at most {MAX_SELECTED_JUNCTIONS} junctions")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise JunctionConfigError(f"junction_ids contains unknown IDs: {unknown}")

    by_id = {junction_id: index for index, junction_id in enumerate(available)}
    pair_count = len(selected) * (len(selected) - 1) // 2
    if pair_count:
        first = review.junctions[by_id[selected[0]]]
        toehold_length = len(first.toehold_sequence_5to3)
        barcode_length = len(first.barcode_sequence_5to3)
        projected_cells = pair_count * (toehold_length**2 + barcode_length**2 + (toehold_length + barcode_length) ** 2)
        if projected_cells > MAX_DYNAMIC_PROGRAMMING_CELLS:
            raise JunctionConfigError(
                f"pairwise work requires {projected_cells} dynamic-programming cells; "
                f"limit {MAX_DYNAMIC_PROGRAMMING_CELLS}. Choose fewer junction_ids"
            )
    return DissimilaritySelection(
        indices=tuple(by_id[junction_id] for junction_id in selected),
        selected_count=len(selected),
        total_count=len(available),
    )


def pairwise_matrices(
    review: JunctionSequenceDissimilarityV1,
    selection: DissimilaritySelection,
) -> DissimilarityMatrices:
    selected = tuple(review.junctions[index] for index in selection.indices)
    count = len(selected)
    toehold = np.zeros((count, count), dtype=np.float64)
    barcode = np.zeros((count, count), dtype=np.float64)
    combined = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        left_junction = selected[left]
        left_combined = left_junction.toehold_sequence_5to3 + left_junction.barcode_sequence_5to3
        for right in range(left + 1, count):
            right_junction = selected[right]
            right_combined = right_junction.toehold_sequence_5to3 + right_junction.barcode_sequence_5to3
            values = (
                position_weighted_levenshtein(
                    left_junction.toehold_sequence_5to3,
                    right_junction.toehold_sequence_5to3,
                ),
                float(
                    levenshtein_distance(
                        left_junction.barcode_sequence_5to3,
                        right_junction.barcode_sequence_5to3,
                    )
                ),
                float(longest_common_substring_length(left_combined, right_combined)),
            )
            for matrix, value in zip((toehold, barcode, combined), values, strict=True):
                matrix[left, right] = value
                matrix[right, left] = value
    return DissimilarityMatrices(toehold=toehold, barcode=barcode, combined=combined)


__all__ = [
    "DissimilarityMatrices",
    "DissimilaritySelection",
    "MAX_DYNAMIC_PROGRAMMING_CELLS",
    "MAX_SELECTED_JUNCTIONS",
    "pairwise_matrices",
    "resolve_selection",
]
