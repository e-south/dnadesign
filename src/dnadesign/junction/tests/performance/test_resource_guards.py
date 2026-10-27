"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/test_resource_guards.py

Fail-fast resource guards for junction search and planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dnadesign.junction.contracts import (
    JunctionRequest,
    OrderPolicy,
    PlanningProfile,
    Primer,
    RecoveryPrimerPair,
    Target,
)
from dnadesign.junction.design import barcodes as barcode_module
from dnadesign.junction.design import matching as matching_module
from dnadesign.junction.design import planner as planner_module
from dnadesign.junction.design import toeholds as toehold_module
from dnadesign.junction.design.barcodes import generate_barcode_candidates
from dnadesign.junction.design.loci import ToeholdCandidate, ToeholdLocus, enumerate_loci
from dnadesign.junction.design.resources import (
    MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH,
    MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS,
    estimated_toehold_distance_lookups,
    guard_barcode_generation,
    guard_barcode_subset_search,
    guard_uniform_toehold_search,
    sampled_toehold_search_state_bytes,
)
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.sequence import reverse_complement
from dnadesign.junction.tests.performance._factories import candidate as _candidate
from dnadesign.junction.tests.performance._factories import planning_profile as _profile


def test_matching_guard_rejects_unbounded_substring_character_work_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = (
        _candidate(0, "A" * 4_000),
        _candidate(1, "C" * 4_000),
    )
    barcodes = ("G" * 4_000, "T" * 4_000)

    def fail_if_scored(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("substring scoring must not begin")

    monkeypatch.setattr(matching_module, "_maximum_shared_substring", fail_if_scored)

    with pytest.raises(JunctionDesignError, match="substring character"):
        matching_module.match_barcodes(candidates, barcodes, iterations=2, seed=17)


def test_confirmed_toehold_resource_undercount_fails_before_numpy_search_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loci = tuple(
        ToeholdLocus(
            target_id="target",
            assembly_group_id="assembly",
            index=locus_index,
            candidates=tuple(
                ToeholdCandidate(
                    target_id="target",
                    assembly_group_id="assembly",
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
    with pytest.raises(JunctionDesignError, match="sampled-path state"):
        toehold_module.select_toeholds(loci, iterations=20_000, seed=17)


def test_many_safe_assembly_groups_fail_request_guard_before_candidate_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    visits_per_assembly_group = profile.barcode_generation_attempts * profile.barcode_length
    assembly_group_count = MAX_REQUEST_BARCODE_GENERATION_BASE_VISITS // visits_per_assembly_group + 1
    request = JunctionRequest(
        schema="dnadesign.junction.request.v2",
        seed=17,
        planning=profile,
        targets=tuple(
            Target(
                id=f"target-{index:04d}",
                assembly_group_id=f"assembly-{index:04d}",
                sequence=sequence,
                recovery_primers=recovery,
            )
            for index in range(assembly_group_count)
        ),
        order_policy=OrderPolicy(
            synthesis_scale="declared-test-scale",
            barcode_bearing_purification="declared-test-purification",
            complement_purification="declared-test-purification",
            primer_purification="declared-test-purification",
            complement_end_preparation="vendor_5_prime_phosphate",
            minimum_fragment_oligo_length=1,
            max_oligo_length=64,
        ),
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(JunctionDesignError, match="Request-wide barcode-generation base visits"):
        planner_module.design_junction(request)


def test_oversized_target_fails_before_candidate_materialization() -> None:
    profile = PlanningProfile(
        nominal_fragment_oligo_length=96,
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
    target = SimpleNamespace(id="oversized", assembly_group_id="assembly", sequence="A" * 100_000)

    with pytest.raises(JunctionDesignError, match="memory envelope"):
        enumerate_loci(target, profile)


def test_sequence_and_sampled_state_envelopes_fail_before_allocation() -> None:
    with pytest.raises(JunctionDesignError, match="edit-distance envelope"):
        guard_uniform_toehold_search(
            locus_count=100,
            candidates_per_locus=1,
            sequence_length=10_000,
            iterations=1,
        )
    with pytest.raises(JunctionDesignError, match="sampled-path state"):
        guard_uniform_toehold_search(
            locus_count=100,
            candidates_per_locus=1,
            sequence_length=10,
            iterations=100_000,
        )


def test_barcode_generation_and_subset_shapes_fail_before_allocation() -> None:
    with pytest.raises(JunctionDesignError, match="generation exceeds the explicit state"):
        guard_barcode_generation(
            toehold_count=100,
            toehold_length=10,
            length=22,
            count=100_000,
            forbidden_toehold_k=5,
            forbidden_barcode_k=6,
            max_attempts=100_000,
        )
    with pytest.raises(JunctionDesignError, match="distance cache"):
        guard_barcode_subset_search(
            candidate_count=20_000,
            selected_count=100,
            sequence_length=22,
            iterations=10,
        )
    with pytest.raises(JunctionDesignError, match="sampled-subset state"):
        guard_barcode_subset_search(
            candidate_count=1,
            selected_count=1,
            sequence_length=22,
            iterations=20_000_000,
        )


def test_barcode_distance_representation_limit_is_explicit() -> None:
    guard_barcode_subset_search(
        candidate_count=1,
        selected_count=1,
        sequence_length=MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH,
        iterations=0,
    )

    with pytest.raises(
        JunctionDesignError,
        match=rf"distance-cache representation: requested .*, limit {MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH}",
    ):
        guard_barcode_subset_search(
            candidate_count=1,
            selected_count=1,
            sequence_length=MAX_BARCODE_DISTANCE_SEQUENCE_LENGTH + 1,
            iterations=0,
        )


def test_large_kmer_payload_fails_before_kmer_materialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_materialized(*_args: object, **_kwargs: object) -> set[str]:
        raise AssertionError("k-mer materialization must not begin")

    monkeypatch.setattr(barcode_module, "kmer_set_with_reverse_complements", fail_if_materialized)

    with pytest.raises(JunctionDesignError, match="generation exceeds the explicit state"):
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
    nominal_fragment_oligo_length = 2_200_001
    target_length = 2_400_001
    sequence = "A" * target_length
    profile = PlanningProfile(
        nominal_fragment_oligo_length=nominal_fragment_oligo_length,
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
    request = JunctionRequest(
        schema="dnadesign.junction.request.v2",
        seed=17,
        planning=profile,
        targets=(
            Target(
                id="large-kmer-payload",
                assembly_group_id="assembly",
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
            minimum_fragment_oligo_length=1,
            max_oligo_length=nominal_fragment_oligo_length,
        ),
    )

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("locus materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(JunctionDesignError, match="Request-wide barcode-generation state bytes"):
        planner_module.design_junction(request)


def test_impossible_barcode_count_fails_before_the_generation_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_drawn(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("barcode draws must not begin")

    monkeypatch.setattr(barcode_module, "_random_dna", fail_if_drawn)

    with pytest.raises(JunctionDesignError, match="cannot satisfy the declared candidate count"):
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
