"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_performance_contracts.py

Correctness and resource contracts for TriJunction's hot search paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from dnadesign.trijunction.contracts import (
    OrderPolicy,
    PlanningProfile,
    Primer,
    RecoveryPrimerPair,
    Target,
    TriJunctionRequest,
)
from dnadesign.trijunction.design import barcodes as barcode_module
from dnadesign.trijunction.design import planner as planner_module
from dnadesign.trijunction.design import toeholds as toehold_module
from dnadesign.trijunction.design.barcodes import generate_barcode_candidates
from dnadesign.trijunction.design.loci import ToeholdCandidate, ToeholdLocus, enumerate_loci
from dnadesign.trijunction.design.matching import _matching_score
from dnadesign.trijunction.design.randomness import StablePrng
from dnadesign.trijunction.design.resources import (
    MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS,
    MAX_TOEHOLD_CACHE_BYTES,
    capped_toehold_path_count,
    estimate_request_workload,
    estimated_toehold_distance_lookups,
    guard_barcode_generation,
    guard_barcode_subset_search,
    guard_request_workload,
    guard_uniform_toehold_search,
    sampled_barcode_subset_state_bytes,
    sampled_matching_state_bytes,
    sampled_toehold_search_state_bytes,
    toehold_distance_cache_bytes,
)
from dnadesign.trijunction.design.scoring import RankAggregate
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import (
    longest_common_substring_length,
    position_weighted_levenshtein_units,
    position_weighted_levenshtein_units_many,
    reverse_complement,
)


def _candidate(index: int, sequence: str) -> ToeholdCandidate:
    return ToeholdCandidate(
        target_id="target",
        pool_id="pool",
        locus_index=index,
        candidate_offset=0,
        start=index,
        sequence=sequence,
    )


def _profile() -> PlanningProfile:
    return PlanningProfile(
        oligo_length=46,
        barcode_length=16,
        toehold_length=8,
        search_range=2,
        toehold_search_iterations=40,
        barcode_pool_factor=5,
        barcode_generation_attempts=100_000,
        barcode_toehold_k=4,
        barcode_pair_k=5,
        barcode_subset_iterations=40,
        matching_iterations=100,
        barcode_gc_min=0.25,
        barcode_gc_max=0.75,
        barcode_max_homopolymer=3,
    )


def test_stable_prng_has_a_golden_cross_runtime_stream() -> None:
    stream = StablePrng(0)

    assert [stream.next_u64() for _ in range(5)] == [
        16294208416658607535,
        7960286522194355700,
        487617019471545679,
        17909611376780542444,
        1961750202426094747,
    ]


def test_vectorized_weighted_distances_equal_the_scalar_fixed_point_contract() -> None:
    left = ("ACGATTCGGT", "GATTACAGAT", "ACGTACGTAC", "TTTTTTTTTT")
    right = ("CGCTTAGACT", "TACTAGATTA", "TACGTACGTA", "TTTTTATTTT")

    observed = tuple(int(value) for value in position_weighted_levenshtein_units_many(left, right))
    expected = tuple(position_weighted_levenshtein_units(a, b) for a, b in zip(left, right, strict=True))

    assert observed == expected


def test_duplicate_substring_matching_equals_pairwise_lcs_semantics() -> None:
    candidates = (
        _candidate(0, "ACGATTCGGT"),
        _candidate(1, "TTACGATACC"),
        _candidate(2, "CGGTACTGAA"),
        _candidate(3, "GATCAGGTCA"),
    )
    barcodes = ("ACGTAC", "TGCACT", "GATTAC", "CCTAGG")
    combined = tuple(candidate.sequence + barcode for candidate, barcode in zip(candidates, barcodes, strict=True))
    reference = max(longest_common_substring_length(left, right) for left, right in combinations(combined, 2))

    assert _matching_score(candidates, barcodes) == reference


def test_paper_scale_cache_is_bounded_and_full_iteration_budget_is_explicit() -> None:
    loci = 296
    search_range = 15
    candidate_count = loci * search_range

    assert toehold_distance_cache_bytes(candidate_count) == 78_836_640
    assert toehold_distance_cache_bytes(candidate_count) < MAX_TOEHOLD_CACHE_BYTES
    assert estimated_toehold_distance_lookups((search_range,) * loci, 1_000) == 698_603_660
    guard_uniform_toehold_search(
        locus_count=loci,
        candidates_per_locus=search_range,
        sequence_length=10,
        iterations=1_000,
    )


def test_toehold_workload_includes_final_unique_path_scoring() -> None:
    profile = replace(_profile(), oligo_length=64, search_range=10, toehold_search_iterations=2)

    estimate = estimate_request_workload(
        input_bases=72,
        target_count=1,
        pool_locus_counts=(4,),
        profile=profile,
    )

    construction_lookups = 2 * 10 * 4 * 3 // 2
    scoring_lookups = 3 * 4 * 3 // 2
    combined_lookups = construction_lookups + scoring_lookups
    unique_pairs = min((4 * 10) * (4 * 10 - 1) // 2, combined_lookups)
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


def test_normal_multi_pool_workload_is_accepted_and_aggregated() -> None:
    profile = _profile()

    estimate = estimate_request_workload(
        input_bases=144,
        target_count=2,
        pool_locus_counts=(2, 2),
        profile=profile,
    )

    assert estimate.pool_count == 2
    assert estimate.locus_count == 4
    assert estimate.toehold_candidate_count == 8
    assert estimate.barcode_candidate_count == 20
    assert estimate.barcode_generation_base_visits == 3_200_000
    guard_request_workload(estimate)


def test_barcode_generation_state_models_kmer_payloads_and_peak_temporary_set() -> None:
    profile = _profile()

    estimate = estimate_request_workload(
        input_bases=72,
        target_count=1,
        pool_locus_counts=(2,),
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


def test_confirmed_toehold_resource_undercount_fails_before_numpy_search_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loci = tuple(
        ToeholdLocus(
            target_id="target",
            pool_id="pool",
            index=locus_index,
            candidates=tuple(
                ToeholdCandidate(
                    target_id="target",
                    pool_id="pool",
                    locus_index=locus_index,
                    candidate_offset=offset,
                    start=locus_index + offset,
                    sequence="ACGATTCGGT",
                )
                for offset in range(2)
            ),
        )
        for locus_index in range(14)
    )

    def fail_if_allocated(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("NumPy search allocation must not begin")

    monkeypatch.setattr(toehold_module.np, "asarray", fail_if_allocated)

    assert estimated_toehold_distance_lookups((2,) * 14, 20_000) < 1_000_000_000
    assert 20_000 * 14 * 12 < 64 * 1024 * 1024
    assert (
        sampled_toehold_search_state_bytes(
            iterations=20_000,
            candidate_counts=(2,) * 14,
        )
        > 64 * 1024 * 1024
    )
    with pytest.raises(TriJunctionDesignError, match="sampled-path state"):
        toehold_module.select_toeholds(loci, iterations=20_000, seed=17)


def test_many_safe_pools_fail_request_guard_before_candidate_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    sequence = ("ACGATTCGGTACCTGATGCACTGA" * 4)[:72]
    recovery = RecoveryPrimerPair(
        mode="target_specific",
        forward=Primer(binding_sequence=sequence[:8], five_prime_extension=""),
        reverse=Primer(
            binding_sequence=reverse_complement(sequence[-8:]),
            five_prime_extension="",
        ),
    )
    visits_per_pool = profile.barcode_generation_attempts * profile.barcode_length
    pool_count = MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS // visits_per_pool + 1
    request = TriJunctionRequest(
        schema="dnadesign.trijunction.request.v1",
        seed=17,
        planning=profile,
        targets=tuple(
            Target(
                id=f"target-{index:04d}",
                pool_id=f"pool-{index:04d}",
                sequence=sequence,
                recovery_primers=recovery,
            )
            for index in range(pool_count)
        ),
        order_policy=OrderPolicy(
            synthesis_scale="declared-test-scale",
            barcode_bearing_purification="declared-test-purification",
            complement_purification="declared-test-purification",
            primer_purification="declared-test-purification",
            complement_end_preparation="vendor_5_prime_phosphate",
            max_oligo_length=64,
        ),
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(TriJunctionDesignError, match="Request-wide barcode-generation base visits"):
        planner_module.design_trijunction(request)


def test_oversized_target_fails_before_candidate_materialization() -> None:
    profile = PlanningProfile(
        oligo_length=96,
        barcode_length=22,
        toehold_length=10,
        search_range=15,
        toehold_search_iterations=1_000,
        barcode_pool_factor=5,
        barcode_generation_attempts=100_000,
        barcode_toehold_k=5,
        barcode_pair_k=6,
        barcode_subset_iterations=100,
        matching_iterations=100,
        barcode_gc_min=0.25,
        barcode_gc_max=0.75,
        barcode_max_homopolymer=3,
    )
    target = SimpleNamespace(id="oversized", pool_id="pool", sequence="A" * 100_000)

    with pytest.raises(TriJunctionDesignError, match="memory envelope"):
        enumerate_loci(target, profile)


def test_sequence_and_sampled_state_envelopes_fail_before_allocation() -> None:
    with pytest.raises(TriJunctionDesignError, match="edit-distance envelope"):
        guard_uniform_toehold_search(
            locus_count=100,
            candidates_per_locus=1,
            sequence_length=10_000,
            iterations=1,
        )
    with pytest.raises(TriJunctionDesignError, match="sampled-path state"):
        guard_uniform_toehold_search(
            locus_count=100,
            candidates_per_locus=1,
            sequence_length=10,
            iterations=100_000,
        )


def test_barcode_generation_and_subset_shapes_fail_before_allocation() -> None:
    with pytest.raises(TriJunctionDesignError, match="generation exceeds the explicit state"):
        guard_barcode_generation(
            toehold_count=100,
            toehold_length=10,
            length=22,
            count=100_000,
            forbidden_toehold_k=5,
            forbidden_barcode_k=6,
            max_attempts=100_000,
        )
    with pytest.raises(TriJunctionDesignError, match="distance cache"):
        guard_barcode_subset_search(
            candidate_count=20_000,
            selected_count=100,
            sequence_length=22,
            iterations=10,
        )
    with pytest.raises(TriJunctionDesignError, match="sampled-subset state"):
        guard_barcode_subset_search(
            candidate_count=1,
            selected_count=1,
            sequence_length=22,
            iterations=20_000_000,
        )


def test_large_kmer_payload_fails_before_kmer_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_materialized(*_args: object, **_kwargs: object) -> set[str]:
        raise AssertionError("k-mer materialization must not begin")

    monkeypatch.setattr(barcode_module, "kmer_set_with_reverse_complements", fail_if_materialized)

    with pytest.raises(TriJunctionDesignError, match="generation exceeds the explicit state"):
        generate_barcode_candidates(
            ("A" * 200_000,),
            length=1_000_000,
            count=5,
            forbidden_toehold_k=199_999,
            forbidden_barcode_k=1_000_000,
            gc_min=0.0,
            gc_max=1.0,
            max_homopolymer=1_000_000,
            max_attempts=5,
            seed=17,
        )


def test_large_kmer_payload_fails_request_guard_before_locus_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barcode_length = 1_000_000
    toehold_length = 200_000
    oligo_length = 2_200_001
    target_length = 2_400_001
    sequence = "A" * target_length
    profile = PlanningProfile(
        oligo_length=oligo_length,
        barcode_length=barcode_length,
        toehold_length=toehold_length,
        search_range=1,
        toehold_search_iterations=1,
        barcode_pool_factor=5,
        barcode_generation_attempts=5,
        barcode_toehold_k=199_999,
        barcode_pair_k=1_000_000,
        barcode_subset_iterations=1,
        matching_iterations=1,
        barcode_gc_min=0.0,
        barcode_gc_max=1.0,
        barcode_max_homopolymer=barcode_length,
    )
    request = TriJunctionRequest(
        schema="dnadesign.trijunction.request.v1",
        seed=17,
        planning=profile,
        targets=(
            Target(
                id="large-kmer-payload",
                pool_id="pool",
                sequence=sequence,
                recovery_primers=RecoveryPrimerPair(
                    mode="target_specific",
                    forward=Primer(binding_sequence=sequence[:8], five_prime_extension=""),
                    reverse=Primer(binding_sequence=reverse_complement(sequence[-8:]), five_prime_extension=""),
                ),
            ),
        ),
        order_policy=OrderPolicy(
            synthesis_scale="declared-test-scale",
            barcode_bearing_purification="declared-test-purification",
            complement_purification="declared-test-purification",
            primer_purification="declared-test-purification",
            complement_end_preparation="vendor_5_prime_phosphate",
            max_oligo_length=oligo_length,
        ),
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("locus materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(TriJunctionDesignError, match="Request-wide barcode-generation state bytes"):
        planner_module.design_trijunction(request)


def test_impossible_barcode_count_fails_before_the_generation_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_drawn(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("barcode draws must not begin")

    monkeypatch.setattr(barcode_module, "_random_dna", fail_if_drawn)

    with pytest.raises(TriJunctionDesignError, match="cannot satisfy the declared candidate count"):
        generate_barcode_candidates(
            ("ACGTACGT",),
            length=16,
            count=11,
            forbidden_toehold_k=4,
            forbidden_barcode_k=5,
            gc_min=0.25,
            gc_max=0.75,
            max_homopolymer=3,
            max_attempts=10,
            seed=17,
        )
