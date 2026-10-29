"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/test_workload_estimates.py

Request-wide workload and retained-state accounting contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from dnadesign.junction.design.randomness import StablePrng
from dnadesign.junction.design.resources import (
    MAX_TOEHOLD_CACHE_BYTES,
    capped_toehold_path_count,
    estimate_request_workload,
    estimated_toehold_distance_lookups,
    guard_request_workload,
    guard_uniform_toehold_search,
    sampled_barcode_subset_state_bytes,
    sampled_matching_state_bytes,
    sampled_toehold_search_state_bytes,
    toehold_distance_cache_bytes,
)
from dnadesign.junction.design.scoring import RankAggregate
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.tests.performance._factories import planning_profile as _profile


def test_paper_scale_cache_is_bounded_and_full_iteration_budget_is_explicit() -> None:
    loci = 296
    search_range = 15
    candidate_count = loci * search_range

    assert toehold_distance_cache_bytes(candidate_count) == 78_836_640
    assert toehold_distance_cache_bytes(candidate_count) < MAX_TOEHOLD_CACHE_BYTES
    assert estimated_toehold_distance_lookups((search_range,) * loci, 1_000) == 1_353_503_660
    guard_uniform_toehold_search(
        locus_count=loci,
        candidates_per_locus=search_range,
        sequence_length=10,
        iterations=1_000,
    )


def test_toehold_lookup_guard_counts_the_cache_only_weight_pass() -> None:
    lookups = estimated_toehold_distance_lookups((40,) * 100, 3_000)

    assert 1_000_000_000 < lookups <= 2_000_000_000
    guard_uniform_toehold_search(
        locus_count=100,
        candidates_per_locus=40,
        sequence_length=2,
        iterations=3_000,
    )


def test_toehold_workload_includes_two_pass_construction_and_final_scoring() -> None:
    profile = replace(
        _profile(),
        nominal_fragment_oligo_length=64,
        search_range=10,
        toehold_search_iterations=2,
    )

    estimate = estimate_request_workload(
        input_bases=72,
        target_count=1,
        assembly_group_locus_counts=(4,),
        profile=profile,
    )

    one_pass_construction_lookups = 2 * 10 * 4 * 3 // 2
    scoring_lookups = 3 * 4 * 3 // 2
    combined_lookups = 2 * one_pass_construction_lookups + scoring_lookups
    unique_pairs = min((4 * 10) * (4 * 10 - 1) // 2, one_pass_construction_lookups + scoring_lookups)
    assert estimated_toehold_distance_lookups((10,) * 4, 2) == combined_lookups
    assert estimate.toehold_distance_lookups == combined_lookups
    assert estimate.toehold_dp_cells == unique_pairs * 2 * 8 * 8


def test_toehold_unique_path_count_caps_cartesian_product_without_materializing_it() -> None:
    assert capped_toehold_path_count((2, 2, 2, 2), iterations=40) == 16
    assert capped_toehold_path_count((15,) * 296, iterations=1_000) == 1_001
    assert sampled_toehold_search_state_bytes(iterations=100_000, candidate_counts=(1,)) < 64 * 1024 * 1024
    guard_uniform_toehold_search(
        locus_count=1,
        candidates_per_locus=1,
        sequence_length=10,
        iterations=100_000,
    )


def test_normal_multi_assembly_group_workload_is_accepted_and_aggregated() -> None:
    profile = _profile()

    estimate = estimate_request_workload(
        input_bases=144,
        target_count=2,
        assembly_group_locus_counts=(2, 2),
        profile=profile,
    )

    assert estimate.assembly_group_count == 2
    assert estimate.locus_count == 4
    assert estimate.toehold_candidate_count == 8
    assert estimate.barcode_candidate_count == 20
    assert estimate.barcode_generation_base_visits == 3_200_000
    assert estimate.matching_substring_character_visits == 83_200
    guard_request_workload(estimate)


def test_matching_character_work_is_aggregated_across_individually_safe_assembly_groups() -> None:
    profile = replace(_profile(), barcode_generation_attempts=40)

    estimate = estimate_request_workload(
        input_bases=701 * 300,
        target_count=701,
        assembly_group_locus_counts=(8,) * 701,
        profile=profile,
    )

    assert estimate.matching_substring_character_visits == 3_003_644_800
    with pytest.raises(JunctionDesignError, match="Request-wide matching substring character visits"):
        guard_request_workload(estimate)


def test_barcode_generation_state_models_kmer_payloads_and_peak_temporary_set() -> None:
    profile = _profile()

    estimate = estimate_request_workload(
        input_bases=72,
        target_count=1,
        assembly_group_locus_counts=(2,),
        profile=profile,
    )

    # Each k-mer owns a conservatively modeled 128 bytes of Python object and
    # hash-table overhead in addition to its sequence payload.
    toehold_kmers = 2 * 2 * (profile.toehold_length - profile.barcode_toehold_k + 1)
    barcode_kmers = 2 * 10 * (profile.barcode_length - profile.barcode_pair_k + 1)
    temporary_candidate_kmers = 2 * (profile.barcode_length - profile.barcode_toehold_k + 1)
    expected = (
        10 * profile.barcode_length
        + 256
        + toehold_kmers * (128 + profile.barcode_toehold_k)
        + 256
        + barcode_kmers * (128 + profile.barcode_pair_k)
        + 256
        + temporary_candidate_kmers * (128 + profile.barcode_toehold_k)
    )

    assert estimate.barcode_generation_state_bytes == expected


def test_sampled_state_estimates_cover_retained_python_containers() -> None:
    evaluations = 101
    count = 100
    samples = {tuple((*range(count - 1), index)) for index in range(evaluations)}
    scores = {sample: (0, 0) for sample in samples}
    ranks = dict.fromkeys(samples, 0)
    subset_observed = (
        sys.getsizeof(samples)
        + sys.getsizeof(scores)
        + sys.getsizeof(ranks)
        + sum(sys.getsizeof(sample) for sample in samples)
        + sum(sys.getsizeof(score) for score in scores.values())
    )
    sorted_samples = sorted(samples)
    matching_observed = (
        sys.getsizeof(samples) + sys.getsizeof(sorted_samples) + sum(sys.getsizeof(sample) for sample in samples)
    )

    assert sampled_barcode_subset_state_bytes(evaluations=evaluations, selected_count=count) >= subset_observed
    assert sampled_matching_state_bytes(evaluations=evaluations, count=count) >= matching_observed


def test_toehold_state_estimate_covers_simultaneously_retained_search_and_scoring_state() -> None:
    iterations = 101
    locus_count = 100
    evaluations = iterations + 1
    trial_rngs = [StablePrng(index) for index in range(iterations)]
    visit_orders = np.empty((iterations, locus_count), dtype=np.int32)
    selected = np.empty_like(visit_orders)
    canonical_paths = np.empty_like(selected)
    stacked_paths = np.empty((evaluations, locus_count), dtype=np.int32)
    unique_paths = np.empty_like(stacked_paths)
    identities = tuple(
        tuple(("target", index, evaluation if index == 0 else 0) for index in range(locus_count))
        for evaluation in range(evaluations)
    )
    paths = tuple(
        tuple((evaluation + index) % evaluations for index in range(locus_count)) for evaluation in range(evaluations)
    )
    paths_by_identity = dict(zip(identities, paths, strict=True))
    scores = {identity: (0, Fraction(0)) for identity in identities}
    ranks = {
        identity: RankAggregate(
            minimum_rank_fraction=Fraction(1),
            mean_rank_fraction=Fraction(1),
            weighted_score_fraction=Fraction(3, 2),
        )
        for identity in identities
    }

    array_peak = (
        sum(sys.getsizeof(array) for array in (visit_orders, selected, canonical_paths, stacked_paths, unique_paths))
        + sys.getsizeof(trial_rngs)
        + sum(sys.getsizeof(rng) + sys.getsizeof(0) for rng in trial_rngs)
    )
    scoring_peak = (
        sum(sys.getsizeof(array) for array in (visit_orders, selected, canonical_paths, unique_paths))
        + sys.getsizeof(trial_rngs)
        + sum(sys.getsizeof(rng) + sys.getsizeof(0) for rng in trial_rngs)
        + sys.getsizeof(paths_by_identity)
        + sys.getsizeof(scores)
        + sys.getsizeof(ranks)
        + sum(sys.getsizeof(identity) + sum(sys.getsizeof(item) for item in identity) for identity in identities)
        + sum(sys.getsizeof(path) + sum(sys.getsizeof(index) for index in path) for path in paths)
        + sum(sys.getsizeof(score) + sum(sys.getsizeof(value) for value in score) for score in scores.values())
        + sum(
            sys.getsizeof(rank)
            + sys.getsizeof(rank.minimum_rank_fraction)
            + sys.getsizeof(rank.mean_rank_fraction)
            + sys.getsizeof(rank.weighted_score_fraction)
            for rank in ranks.values()
        )
    )

    estimate = sampled_toehold_search_state_bytes(
        iterations=iterations,
        candidate_counts=(2,) * locus_count,
    )
    assert estimate >= max(array_peak, scoring_peak)
