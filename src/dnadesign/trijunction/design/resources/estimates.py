"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/design/resources/estimates.py

Pure integer workload estimates used before search-state allocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dnadesign.trijunction.contracts.request import PlanningProfile

# These ceilings include retained Python integer/tuple objects, hash-table
# slack, dictionary values, and the simultaneous sorted matching index. They
# intentionally exceed shallow CPython measurements used by the regression
# test instead of treating index values as compact C integers.
_SAMPLED_SUBSET_FIXED_BYTES = 1_024
_SAMPLED_SUBSET_INDEX_BYTES = 64
_SAMPLED_MATCHING_FIXED_BYTES = 512
_SAMPLED_MATCHING_INDEX_BYTES = 96


@dataclass(frozen=True, slots=True)
class RequestWorkloadEstimate:
    """Aggregate work estimate for one complete planning request."""

    pool_count: int
    target_count: int
    input_bases: int
    locus_count: int
    toehold_candidate_count: int
    toehold_encoded_bases: int
    toehold_cache_bytes: int
    toehold_distance_lookups: int
    toehold_dp_cells: int
    toehold_search_state_bytes: int
    barcode_candidate_count: int
    barcode_generation_base_visits: int
    barcode_generation_state_bytes: int
    barcode_encoded_bases: int
    barcode_distance_cache_bytes: int
    barcode_subset_lookups: int
    barcode_dp_cells: int
    barcode_subset_state_bytes: int
    matching_substring_visits: int
    matching_state_bytes: int


def toehold_distance_cache_bytes(candidate_count: int) -> int:
    """Return bytes required by the compact uint64 triangular cache."""

    return candidate_count * (candidate_count - 1) // 2 * 8


def barcode_distance_cache_bytes(candidate_count: int) -> int:
    """Return bytes required by the compact uint16 triangular cache."""

    return candidate_count * (candidate_count - 1) // 2 * 2


def sampled_barcode_subset_state_bytes(*, evaluations: int, selected_count: int) -> int:
    """Conservatively model retained tuple, set, score, and rank state."""

    return evaluations * (_SAMPLED_SUBSET_FIXED_BYTES + _SAMPLED_SUBSET_INDEX_BYTES * selected_count)


def sampled_matching_state_bytes(*, evaluations: int, count: int) -> int:
    """Conservatively model retained tuple/set state and its sorted index."""

    return evaluations * (_SAMPLED_MATCHING_FIXED_BYTES + _SAMPLED_MATCHING_INDEX_BYTES * count)


def estimated_toehold_distance_lookups(candidate_counts: tuple[int, ...], iterations: int) -> int:
    """Return the uniform-locus pairwise lookup ceiling used by the search."""

    return iterations * sum(index * count for index, count in enumerate(sorted(candidate_counts)))


def estimate_request_workload(
    *,
    input_bases: int,
    target_count: int,
    pool_locus_counts: tuple[int, ...],
    profile: PlanningProfile,
) -> RequestWorkloadEstimate:
    """Estimate request-wide work using the same bounds as each stage guard."""

    if input_bases < 0 or target_count < 0 or any(count < 0 for count in pool_locus_counts):
        raise ValueError("request workload inputs must be non-negative")

    total_loci = sum(pool_locus_counts)
    total_toehold_candidates = total_loci * profile.search_range
    total_barcode_candidates = total_loci * profile.barcode_pool_factor
    toehold_encoded_bases = total_toehold_candidates * profile.toehold_length

    toehold_cache_bytes = 0
    toehold_distance_lookups = 0
    toehold_dp_cells = 0
    toehold_search_state_bytes = 0
    barcode_generation_state_bytes = 0
    barcode_distance_cache_bytes_total = 0
    barcode_subset_lookups = 0
    barcode_dp_cells = 0
    barcode_subset_state_bytes = 0
    matching_substring_visits = 0
    matching_state_bytes = 0

    barcode_kmers_per_candidate = 2 * max(profile.barcode_length - profile.barcode_pair_k + 1, 0)
    combined_length = profile.toehold_length + profile.barcode_length
    for locus_count in pool_locus_counts:
        toehold_candidate_count = locus_count * profile.search_range
        pool_toehold_pairs = toehold_candidate_count * (toehold_candidate_count - 1) // 2
        pool_toehold_lookups = (
            profile.toehold_search_iterations * profile.search_range * locus_count * (locus_count - 1) // 2
        )
        toehold_cache_bytes += toehold_distance_cache_bytes(toehold_candidate_count)
        toehold_distance_lookups += pool_toehold_lookups
        toehold_dp_cells += (
            min(pool_toehold_pairs, pool_toehold_lookups) * 2 * profile.toehold_length * profile.toehold_length
        )
        toehold_search_state_bytes += profile.toehold_search_iterations * locus_count * 12

        barcode_candidate_count = locus_count * profile.barcode_pool_factor
        barcode_generation_state_bytes += (
            barcode_candidate_count * (profile.barcode_length + 96 * barcode_kmers_per_candidate)
            + locus_count * profile.toehold_length * 96
        )
        barcode_distance_cache_bytes_total += barcode_distance_cache_bytes(barcode_candidate_count)
        barcode_pairs = barcode_candidate_count * (barcode_candidate_count - 1) // 2
        subset_pairs = locus_count * (locus_count - 1) // 2
        pool_subset_lookups = (profile.barcode_subset_iterations + 1) * subset_pairs
        barcode_subset_lookups += pool_subset_lookups
        barcode_dp_cells += min(barcode_pairs, pool_subset_lookups) * profile.barcode_length * profile.barcode_length
        barcode_subset_state_bytes += sampled_barcode_subset_state_bytes(
            evaluations=profile.barcode_subset_iterations + 1,
            selected_count=locus_count,
        )

        factorial = math.factorial(locus_count) if locus_count <= 8 else profile.matching_iterations + 1
        exhaustive = locus_count <= 8 and factorial <= profile.matching_iterations
        matching_evaluations = factorial if exhaustive else profile.matching_iterations + 1
        matching_substring_visits += matching_evaluations * locus_count * combined_length * (combined_length + 1) // 2
        matching_state_bytes += sampled_matching_state_bytes(
            evaluations=matching_evaluations,
            count=locus_count,
        )

    return RequestWorkloadEstimate(
        pool_count=len(pool_locus_counts),
        target_count=target_count,
        input_bases=input_bases,
        locus_count=total_loci,
        toehold_candidate_count=total_toehold_candidates,
        toehold_encoded_bases=toehold_encoded_bases,
        toehold_cache_bytes=toehold_cache_bytes,
        toehold_distance_lookups=toehold_distance_lookups,
        toehold_dp_cells=toehold_dp_cells,
        toehold_search_state_bytes=toehold_search_state_bytes,
        barcode_candidate_count=total_barcode_candidates,
        barcode_generation_base_visits=(
            len(pool_locus_counts) * profile.barcode_generation_attempts * profile.barcode_length
        ),
        barcode_generation_state_bytes=barcode_generation_state_bytes,
        barcode_encoded_bases=total_barcode_candidates * profile.barcode_length,
        barcode_distance_cache_bytes=barcode_distance_cache_bytes_total,
        barcode_subset_lookups=barcode_subset_lookups,
        barcode_dp_cells=barcode_dp_cells,
        barcode_subset_state_bytes=barcode_subset_state_bytes,
        matching_substring_visits=matching_substring_visits,
        matching_state_bytes=matching_state_bytes,
    )
