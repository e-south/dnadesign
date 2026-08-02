"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_planner.py

End-to-end contracts for the pure TriJunction planner.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from dnadesign.trijunction.contracts import parse_request
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.sequence import reverse_complement


def _target_sequence(*, offset: int = 0) -> str:
    motif = "ACGATTCGGTACCTGATGCACTGA"
    rotated = motif[offset:] + motif[:offset]
    return (rotated * 4)[:72]


def _primer(binding_sequence: str, *, extension: str = "") -> dict[str, str]:
    return {
        "binding_sequence": binding_sequence,
        "five_prime_extension": extension,
    }


def _request_mapping(*, targets: list[dict] | None = None) -> dict:
    sequence = _target_sequence()
    if targets is None:
        targets = [
            {
                "id": "target-a",
                "pool_id": "pool-a",
                "sequence": sequence,
                "recovery_primers": {
                    "mode": "target_specific",
                    "forward": _primer(sequence[:8]),
                    "reverse": _primer(reverse_complement(sequence[-8:])),
                },
            }
        ]
    return {
        "schema": "dnadesign.trijunction.request.v1",
        "seed": 17,
        "planning": {
            "oligo_length": 46,
            "barcode_length": 16,
            "toehold_length": 8,
            "search_range": 2,
            "toehold_search_iterations": 40,
            "barcode_pool_factor": 5,
            "barcode_generation_attempts": 100_000,
            "barcode_toehold_k": 4,
            "barcode_pair_k": 5,
            "barcode_subset_iterations": 40,
            "matching_iterations": 100,
            "barcode_gc_min": 0.25,
            "barcode_gc_max": 0.75,
            "barcode_max_homopolymer": 3,
        },
        "targets": targets,
        "order_policy": {
            "synthesis_scale": "declared-test-scale",
            "barcode_bearing_purification": "declared-test-purification",
            "complement_purification": "declared-test-purification",
            "primer_purification": "declared-test-purification",
            "complement_end_preparation": "vendor_5_prime_phosphate",
            "max_oligo_length": 64,
        },
    }


def test_design_is_deterministic_and_reconstructs_the_exact_target() -> None:
    request = parse_request(_request_mapping())

    first = design_trijunction(request)
    second = design_trijunction(request)

    assert first == second
    assert first.plan_id.startswith("sha256:")
    target = first.targets[0]
    assert target.reconstructed_target == request.targets[0].sequence
    assert target.assembled_complement == reverse_complement(request.targets[0].sequence)
    assert len(target.fragments) == len(target.junctions) + 1
    assert [target.fragments[0].role, target.fragments[-1].role] == ["first", "last"]
    assert all(junction.complement_nick_geometry_valid for junction in target.junctions)
    assert {junction.complement_end_preparation for junction in target.junctions} == {"vendor_5_prime_phosphate"}
    assert all(order.length == len(order.sequence) for order in first.orders)


def test_fragment_strands_follow_the_paper_defined_equations() -> None:
    request = parse_request(_request_mapping())
    result = design_trijunction(request)
    target = result.targets[0]
    sequence = request.targets[0].sequence
    selected = sorted(
        (junction for junction in result.pools[0].junctions if junction.target_id == target.target_id),
        key=lambda junction: junction.start,
    )
    domains = [sequence[fragment.domain_start : fragment.domain_end] for fragment in target.fragments]

    assert target.fragments[0].barcode_bearing_strand == domains[0] + selected[0].toehold + selected[0].barcode
    assert target.fragments[0].complement_strand == reverse_complement(domains[0])
    for index in range(1, len(target.fragments) - 1):
        previous = selected[index - 1]
        current = selected[index]
        assert target.fragments[index].barcode_bearing_strand == (
            reverse_complement(previous.barcode) + domains[index] + current.toehold + current.barcode
        )
        assert target.fragments[index].complement_strand == (
            reverse_complement(domains[index]) + reverse_complement(previous.toehold)
        )
    assert target.fragments[-1].barcode_bearing_strand == reverse_complement(selected[-1].barcode) + domains[-1]
    assert target.fragments[-1].complement_strand == (
        reverse_complement(domains[-1]) + reverse_complement(selected[-1].toehold)
    )
    assert all(
        evidence.toehold_complement == reverse_complement(evidence.toehold)
        and evidence.barcode_complement == reverse_complement(evidence.barcode)
        for evidence in target.junctions
    )


def test_target_input_order_does_not_change_the_plan() -> None:
    first_sequence = _target_sequence(offset=0)
    second_sequence = _target_sequence(offset=3)
    targets = [
        {
            "id": target_id,
            "pool_id": "pool-a",
            "sequence": sequence,
            "recovery_primers": {
                "mode": "target_specific",
                "forward": _primer(sequence[:8]),
                "reverse": _primer(reverse_complement(sequence[-8:])),
            },
        }
        for target_id, sequence in (("target-a", first_sequence), ("target-b", second_sequence))
    ]
    forward_mapping = _request_mapping(targets=targets)
    reverse_mapping = _request_mapping(targets=list(reversed(targets)))
    forward = design_trijunction(parse_request(forward_mapping))
    reverse = design_trijunction(parse_request(reverse_mapping))

    assert forward == reverse


def test_unrelated_physical_pool_does_not_perturb_existing_pool_search() -> None:
    baseline = design_trijunction(parse_request(_request_mapping()))
    first_sequence = _target_sequence(offset=0)
    second_sequence = _target_sequence(offset=3)
    targets = [
        {
            "id": target_id,
            "pool_id": pool_id,
            "sequence": sequence,
            "recovery_primers": {
                "mode": "target_specific",
                "forward": _primer(sequence[:8]),
                "reverse": _primer(reverse_complement(sequence[-8:])),
            },
        }
        for target_id, pool_id, sequence in (
            ("target-a", "pool-a", first_sequence),
            ("target-b", "pool-b", second_sequence),
        )
    ]

    extended = design_trijunction(parse_request(_request_mapping(targets=targets)))

    baseline_pool = next(pool for pool in baseline.pools if pool.pool_id == "pool-a")
    extended_pool = next(pool for pool in extended.pools if pool.pool_id == "pool-a")
    assert extended_pool == baseline_pool


def test_universal_recovery_requires_one_pair_per_physical_pool() -> None:
    prefix = "ACGTACGT"
    suffix = "GATTACAA"
    sequence_a = prefix + _target_sequence(offset=0)[:56] + suffix
    sequence_b = prefix + _target_sequence(offset=3)[:56] + suffix
    targets = []
    for target_id, sequence, extension in (
        ("target-a", sequence_a, ""),
        ("target-b", sequence_b, "ACGT"),
    ):
        targets.append(
            {
                "id": target_id,
                "pool_id": "pool-a",
                "sequence": sequence,
                "recovery_primers": {
                    "mode": "universal",
                    "forward": _primer(prefix, extension=extension),
                    "reverse": _primer(reverse_complement(suffix)),
                },
            }
        )
    request = parse_request(_request_mapping(targets=targets))

    with pytest.raises(TriJunctionDesignError, match="universal recovery.*primer pairs differ"):
        design_trijunction(request)


def test_universal_recovery_emits_one_shared_order_pair() -> None:
    prefix = "ACGTACGT"
    suffix = "GATTACAA"
    sequence_a = prefix + _target_sequence(offset=0)[:56] + suffix
    sequence_b = prefix + _target_sequence(offset=3)[:56] + suffix
    targets = [
        {
            "id": target_id,
            "pool_id": "pool-a",
            "sequence": sequence,
            "recovery_primers": {
                "mode": "universal",
                "forward": _primer(prefix),
                "reverse": _primer(reverse_complement(suffix)),
            },
        }
        for target_id, sequence in (("target-a", sequence_a), ("target-b", sequence_b))
    ]

    mapping = _request_mapping(targets=targets)
    mapping["planning"]["barcode_pair_k"] = 6
    result = design_trijunction(parse_request(mapping))
    recovery_orders = [order for order in result.orders if order.role.startswith("recovery_")]

    assert len(recovery_orders) == 2
    assert {order.order_id for order in recovery_orders} == {
        "pool-a:universal-recovery-forward",
        "pool-a:universal-recovery-reverse",
    }
    assert {order.target_ids for order in recovery_orders} == {("target-a", "target-b")}


def test_universal_recovery_order_ids_cannot_alias_target_specific_orders() -> None:
    universal_prefix = "ACGTACGT"
    universal_suffix = "GATTACAA"
    universal_sequence = universal_prefix + _target_sequence(offset=3)[:56] + universal_suffix
    target_specific_sequence = _target_sequence(offset=0)
    targets = [
        {
            "id": "pool-a",
            "pool_id": "target-specific-pool",
            "sequence": target_specific_sequence,
            "recovery_primers": {
                "mode": "target_specific",
                "forward": _primer(target_specific_sequence[:8]),
                "reverse": _primer(reverse_complement(target_specific_sequence[-8:])),
            },
        },
        {
            "id": "universal-target",
            "pool_id": "pool-a",
            "sequence": universal_sequence,
            "recovery_primers": {
                "mode": "universal",
                "forward": _primer(universal_prefix),
                "reverse": _primer(reverse_complement(universal_suffix)),
            },
        },
    ]

    result = design_trijunction(parse_request(_request_mapping(targets=targets)))
    order_ids = [order.order_id for order in result.orders]

    assert len(order_ids) == len(set(order_ids))
    assert "pool-a:universal-recovery-forward" in order_ids
    assert "pool-a:universal-recovery-reverse" in order_ids


def test_target_specific_ambiguity_uses_binding_sequences_not_extensions() -> None:
    prefix = "ACGTACGT"
    suffix = "GATTACAA"
    sequence_a = prefix + _target_sequence(offset=0)[:56] + suffix
    sequence_b = prefix + _target_sequence(offset=3)[:56] + suffix
    targets = [
        {
            "id": target_id,
            "pool_id": "pool-a",
            "sequence": sequence,
            "recovery_primers": {
                "mode": "target_specific",
                "forward": _primer(prefix, extension=extension),
                "reverse": _primer(reverse_complement(suffix)),
            },
        }
        for target_id, sequence, extension in (
            ("target-a", sequence_a, ""),
            ("target-b", sequence_b, "ACGT"),
        )
    ]

    with pytest.raises(TriJunctionDesignError, match="Target-specific recovery.*also resolves"):
        design_trijunction(parse_request(_request_mapping(targets=targets)))


def test_recovery_extensions_are_preserved_without_interpreting_downstream_use() -> None:
    mapping = _request_mapping()
    sequence = mapping["targets"][0]["sequence"]
    forward_binding = sequence[:8]
    reverse_binding = reverse_complement(sequence[-8:])
    forward_extension = "GGTCTCA"
    reverse_extension = "CGTCTCA"
    mapping["targets"][0]["recovery_primers"] = {
        "mode": "target_specific",
        "forward": _primer(forward_binding, extension=forward_extension),
        "reverse": _primer(reverse_binding, extension=reverse_extension),
    }

    result = design_trijunction(parse_request(mapping))
    target = result.targets[0]
    recovery = target.recovery

    assert recovery.forward_binding_sequence == forward_binding
    assert recovery.forward_five_prime_extension == forward_extension
    assert recovery.forward_order_sequence == forward_extension + forward_binding
    assert (recovery.forward_start, recovery.forward_end) == (0, len(forward_binding))
    assert recovery.reverse_binding_sequence == reverse_binding
    assert recovery.reverse_five_prime_extension == reverse_extension
    assert recovery.reverse_order_sequence == reverse_extension + reverse_binding
    assert (recovery.reverse_start, recovery.reverse_end) == (
        len(sequence) - len(reverse_binding),
        len(sequence),
    )
    assert recovery.expected_core_product == sequence
    assert recovery.extended_top_strand == forward_extension + sequence + reverse_complement(reverse_extension)
    assert recovery.extended_bottom_strand == (
        reverse_extension + reverse_complement(sequence) + reverse_complement(forward_extension)
    )
    assert recovery.extended_bottom_strand == reverse_complement(recovery.extended_top_strand)
    recovery_orders = {order.role: order.sequence for order in result.orders if order.role.startswith("recovery_")}
    assert recovery_orders == {
        "recovery_forward_primer": recovery.forward_order_sequence,
        "recovery_reverse_primer": recovery.reverse_order_sequence,
    }


def test_downstream_phosphorylation_is_an_explicit_ligation_precondition() -> None:
    mapping = _request_mapping()
    mapping["order_policy"]["complement_end_preparation"] = "downstream_phosphorylation"

    result = design_trijunction(parse_request(mapping))

    assert {junction.complement_end_preparation for junction in result.targets[0].junctions} == {
        "downstream_phosphorylation"
    }
    complement_orders = [order for order in result.orders if order.role == "complement_strand"]
    assert {order.five_prime_state for order in complement_orders} == {"phosphate_required_before_assembly"}


def test_plan_identity_changes_when_seed_changes() -> None:
    first_mapping = _request_mapping()
    second_mapping = deepcopy(first_mapping)
    second_mapping["seed"] = 18

    first = design_trijunction(parse_request(first_mapping))
    second = design_trijunction(parse_request(second_mapping))

    assert first.request_sha256 != second.request_sha256
    assert first.plan_id != second.plan_id
