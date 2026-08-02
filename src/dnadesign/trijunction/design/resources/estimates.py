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

# Toehold search retains one PRNG per trial, three int32 path matrices, and at
# the array peak both the vstack input and unique-path output. During scoring,
# the unique paths coexist with Python path/identity tuples and three mappings
# (paths, scores, and ranks). These allowances include owned Python integers,
# tuple and mapping slack, Fraction/rank values, and ndarray headers.
_TOEHOLD_PRNG_BYTES = 128
_TOEHOLD_NDARRAY_HEADER_BYTES = 128
_TOEHOLD_SCORING_FIXED_BYTES = 2_048
_TOEHOLD_SCORING_LOCUS_BYTES = 128

# A retained k-mer owns both an ASCII string object and one hash-table entry.
# 128 bytes deliberately covers their fixed/object overhead and table slack;
# the k-mer payload is then added explicitly instead of disappearing inside a
# fixed per-entry assumption. The fixed set allowance exceeds an empty CPython
# set and keeps the model conservative for small collections too.
_KMER_SET_FIXED_BYTES = 256
_KMER_ENTRY_OVERHEAD_BYTES = 128


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


def kmer_set_state_bytes(*, sequence_count: int, sequence_length: int, k: int) -> int:
    """Conservatively model one forward/reverse-complement k-mer set."""

    if sequence_count < 0 or sequence_length < 0 or k < 1:
        raise ValueError("k-mer set inputs must be non-negative and k must be positive")
    kmers_per_sequence = 2 * max(sequence_length - k + 1, 0)
    return _KMER_SET_FIXED_BYTES + sequence_count * kmers_per_sequence * (_KMER_ENTRY_OVERHEAD_BYTES + k)


def barcode_generation_state_bytes(
    *,
    toehold_count: int,
    toehold_length: int,
    forbidden_toehold_k: int,
    barcode_count: int,
    barcode_length: int,
    forbidden_barcode_k: int,
) -> int:
    """Model retained strings/indexes plus the peak candidate toehold-k set."""

    return (
        barcode_count * barcode_length
        + kmer_set_state_bytes(
            sequence_count=toehold_count,
            sequence_length=toehold_length,
            k=forbidden_toehold_k,
        )
        + kmer_set_state_bytes(
            sequence_count=barcode_count,
            sequence_length=barcode_length,
            k=forbidden_barcode_k,
        )
        + kmer_set_state_bytes(
            sequence_count=1,
            sequence_length=barcode_length,
            k=forbidden_toehold_k,
        )
    )


def sampled_barcode_subset_state_bytes(*, evaluations: int, selected_count: int) -> int:
    """Conservatively model retained tuple, set, score, and rank state."""

    return evaluations * (_SAMPLED_SUBSET_FIXED_BYTES + _SAMPLED_SUBSET_INDEX_BYTES * selected_count)


def sampled_matching_state_bytes(*, evaluations: int, count: int) -> int:
    """Conservatively model retained tuple/set state and its sorted index."""

    return evaluations * (_SAMPLED_MATCHING_FIXED_BYTES + _SAMPLED_MATCHING_INDEX_BYTES * count)


def capped_toehold_path_count(candidate_counts: tuple[int, ...], *, iterations: int) -> int:
    """Cap the Cartesian path count without constructing an enormous product."""

    if iterations < 0 or any(count < 0 for count in candidate_counts):
        raise ValueError("toehold path-count inputs must be non-negative")
    cap = iterations + 1
    product = 1
    for count in candidate_counts:
        if count == 0:
            return 0
        if product > (cap - 1) // count:
            return cap
        product *= count
    return product


def sampled_toehold_search_state_bytes(*, iterations: int, candidate_counts: tuple[int, ...]) -> int:
    """Conservatively model the peak retained sampled-path search state."""

    evaluations = iterations + 1
    unique_paths = capped_toehold_path_count(candidate_counts, iterations=iterations)
    locus_count = len(candidate_counts)
    prng_state = iterations * _TOEHOLD_PRNG_BYTES
    array_peak = (
        prng_state
        + ((3 * iterations + evaluations + unique_paths) * locus_count * 4)
        + 5 * _TOEHOLD_NDARRAY_HEADER_BYTES
    )
    scoring_peak = (
        prng_state
        + ((3 * iterations + unique_paths) * locus_count * 4)
        + 4 * _TOEHOLD_NDARRAY_HEADER_BYTES
        + unique_paths * (_TOEHOLD_SCORING_FIXED_BYTES + _TOEHOLD_SCORING_LOCUS_BYTES * locus_count)
    )
    return max(array_peak, scoring_peak)


def estimated_toehold_distance_lookups(candidate_counts: tuple[int, ...], iterations: int) -> int:
    """Return search-construction plus final unique-path scoring lookups."""

    locus_count = len(candidate_counts)
    path_pairs = locus_count * (locus_count - 1) // 2
    construction_lookups = iterations * sum(index * count for index, count in enumerate(sorted(candidate_counts)))
    scoring_lookups = capped_toehold_path_count(candidate_counts, iterations=iterations) * path_pairs
    return construction_lookups + scoring_lookups


def estimated_toehold_dp_cells(
    candidate_counts: tuple[int, ...],
    *,
    iterations: int,
    sequence_length: int,
) -> int:
    """Return the combined unique-pair edit-distance ceiling."""

    candidate_count = sum(candidate_counts)
    candidate_pairs = candidate_count * (candidate_count - 1) // 2
    unique_pairs = min(
        candidate_pairs,
        estimated_toehold_distance_lookups(candidate_counts, iterations),
    )
    return unique_pairs * 2 * sequence_length * sequence_length


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
    barcode_generation_state_bytes_total = 0
    barcode_distance_cache_bytes_total = 0
    barcode_subset_lookups = 0
    barcode_dp_cells = 0
    barcode_subset_state_bytes = 0
    matching_substring_visits = 0
    matching_state_bytes = 0

    combined_length = profile.toehold_length + profile.barcode_length
    for locus_count in pool_locus_counts:
        toehold_candidate_count = locus_count * profile.search_range
        candidate_counts = (profile.search_range,) * locus_count
        pool_toehold_lookups = estimated_toehold_distance_lookups(
            candidate_counts,
            profile.toehold_search_iterations,
        )
        toehold_cache_bytes += toehold_distance_cache_bytes(toehold_candidate_count)
        toehold_distance_lookups += pool_toehold_lookups
        toehold_dp_cells += estimated_toehold_dp_cells(
            candidate_counts,
            iterations=profile.toehold_search_iterations,
            sequence_length=profile.toehold_length,
        )
        toehold_search_state_bytes += sampled_toehold_search_state_bytes(
            iterations=profile.toehold_search_iterations,
            candidate_counts=candidate_counts,
        )

        barcode_candidate_count = locus_count * profile.barcode_pool_factor
        barcode_generation_state_bytes_total += barcode_generation_state_bytes(
            toehold_count=locus_count,
            toehold_length=profile.toehold_length,
            forbidden_toehold_k=profile.barcode_toehold_k,
            barcode_count=barcode_candidate_count,
            barcode_length=profile.barcode_length,
            forbidden_barcode_k=profile.barcode_pair_k,
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
        barcode_generation_state_bytes=barcode_generation_state_bytes_total,
        barcode_encoded_bases=total_barcode_candidates * profile.barcode_length,
        barcode_distance_cache_bytes=barcode_distance_cache_bytes_total,
        barcode_subset_lookups=barcode_subset_lookups,
        barcode_dp_cells=barcode_dp_cells,
        barcode_subset_state_bytes=barcode_subset_state_bytes,
        matching_substring_visits=matching_substring_visits,
        matching_state_bytes=matching_state_bytes,
    )
