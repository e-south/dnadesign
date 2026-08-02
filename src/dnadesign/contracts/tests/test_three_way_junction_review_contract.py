"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_three_way_junction_review_contract.py

Validation tests for the neutral three-way-junction review contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1
from dnadesign.contracts.visual.three_way_junction_review_v1 import AssemblyGeometry, AssemblyGroupSearchReview


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def three_way_junction_review_payload() -> dict[str, object]:
    target = "ACGTACGTTGCATGCAGATTACAGGCTAACCGTACGATCGTTAACCGGTTACGATC"
    reverse_binding = _reverse_complement(target[-8:])
    return {
        "contract_kind": "three_way_junction_review_v1",
        "source": {
            "plan_schema": "dnadesign.junction.plan.v1",
            "plan_id": f"sha256:{'a' * 64}",
            "request_sha256": f"sha256:{'b' * 64}",
            "algorithm": "junction.v1",
        },
        "target": {
            "target_id": "target-01",
            "assembly_group_id": "assembly-01",
            "sequence_5to3": target,
            "sequence_sha256": f"sha256:{hashlib.sha256(target.encode()).hexdigest()}",
        },
        "geometry": {
            "fragments": [
                {
                    "fragment_id": "target-01.fragment-01",
                    "index": 0,
                    "role": "first",
                    "domain_span": {"start": 0, "end": 26},
                },
                {
                    "fragment_id": "target-01.fragment-02",
                    "index": 1,
                    "role": "last",
                    "domain_span": {"start": 30, "end": len(target)},
                },
            ],
            "junctions": [
                {
                    "junction_id": "target-01.junction-01",
                    "toehold_span": {"start": 26, "end": 30},
                    "left_fragment_id": "target-01.fragment-01",
                    "right_fragment_id": "target-01.fragment-02",
                    "toehold": target[26:30],
                    "toehold_complement": _reverse_complement(target[26:30]),
                    "barcode": "GACTTGCA",
                    "barcode_complement": _reverse_complement("GACTTGCA"),
                    "complement_nick_sequence_layout_valid": True,
                    "complement_end_preparation": "downstream_phosphorylation",
                }
            ],
        },
        "strands": [
            {
                "fragment_id": "target-01.fragment-01",
                "role": "first",
                "incoming_junction_id": None,
                "outgoing_junction_id": "target-01.junction-01",
                "barcode_bearing_sequence_5to3": target[:30] + "GACTTGCA",
                "complement_sequence_5to3": _reverse_complement(target[:26]),
            },
            {
                "fragment_id": "target-01.fragment-02",
                "role": "last",
                "incoming_junction_id": "target-01.junction-01",
                "outgoing_junction_id": None,
                "barcode_bearing_sequence_5to3": _reverse_complement("GACTTGCA") + target[30:],
                "complement_sequence_5to3": _reverse_complement(target[30:]) + _reverse_complement(target[26:30]),
            },
        ],
        "recovery": {
            "mode": "target_specific",
            "forward": {
                "direction": "forward",
                "binding_sequence_5to3": target[:8],
                "five_prime_extension_5to3": "GG",
                "order_sequence_5to3": "GG" + target[:8],
                "target_binding_span": {"start": 0, "end": 8},
            },
            "reverse": {
                "direction": "reverse",
                "binding_sequence_5to3": reverse_binding,
                "five_prime_extension_5to3": "TT",
                "order_sequence_5to3": "TT" + reverse_binding,
                "target_binding_span": {"start": len(target) - 8, "end": len(target)},
            },
            "first_fragment_id": "target-01.fragment-01",
            "last_fragment_id": "target-01.fragment-02",
            "expected_target_sequence_5to3": target,
            "extended_top_sequence_5to3": "GG" + target + _reverse_complement("TT"),
            "extended_bottom_sequence_5to3": "TT" + _reverse_complement(target) + _reverse_complement("GG"),
        },
        "search": {
            "assembly_group_id": "assembly-01",
            "toehold_seed": 11,
            "barcode_generation_seed": 12,
            "barcode_subset_seed": 13,
            "matching_seed": 14,
            "locus_count": 1,
            "toehold_paths_evaluated": 20,
            "toehold_min_distance": 0.0,
            "toehold_mean_distance": 0.0,
            "toehold_rank_score": 1.5,
            "barcode_candidates_generated": 25,
            "barcode_forbidden_toehold_k": 3,
            "barcode_forbidden_barcode_k": 4,
            "barcode_subsets_evaluated": 20,
            "barcode_min_distance": 0.0,
            "barcode_mean_distance": 0.0,
            "barcode_rank_score": 1.5,
            "matchings_evaluated": 1,
            "matching_max_pairwise_lcs": 0,
            "thermodynamic_screening": "not_run",
        },
        "checks": [
            {
                "subject": {"kind": "target", "id": "target-01"},
                "check": "target_reconstruction",
                "status": "passed",
                "detail": "exact",
            },
            {
                "subject": {"kind": "assembly_group", "id": "assembly-01"},
                "check": "thermodynamic_screening",
                "status": "not_run",
                "detail": "not part of this contract",
            },
        ],
    }


def _two_junction_review_payload() -> dict[str, object]:
    payload = three_way_junction_review_payload()
    target = payload["target"]["sequence_5to3"]
    first_toehold = target[18:22]
    second_toehold = target[35:39]
    first_barcode = "AACCGGTT"
    second_barcode = "GACTTGCA"
    payload["geometry"] = {
        "fragments": [
            {
                "fragment_id": "target-01.fragment-01",
                "index": 0,
                "role": "first",
                "domain_span": {"start": 0, "end": 18},
            },
            {
                "fragment_id": "target-01.fragment-02",
                "index": 1,
                "role": "internal",
                "domain_span": {"start": 22, "end": 35},
            },
            {
                "fragment_id": "target-01.fragment-03",
                "index": 2,
                "role": "last",
                "domain_span": {"start": 39, "end": len(target)},
            },
        ],
        "junctions": [
            {
                "junction_id": "target-01.junction-01",
                "toehold_span": {"start": 18, "end": 22},
                "left_fragment_id": "target-01.fragment-01",
                "right_fragment_id": "target-01.fragment-02",
                "toehold": first_toehold,
                "toehold_complement": _reverse_complement(first_toehold),
                "barcode": first_barcode,
                "barcode_complement": _reverse_complement(first_barcode),
                "complement_nick_sequence_layout_valid": True,
                "complement_end_preparation": "vendor_5_prime_phosphate",
            },
            {
                "junction_id": "target-01.junction-02",
                "toehold_span": {"start": 35, "end": 39},
                "left_fragment_id": "target-01.fragment-02",
                "right_fragment_id": "target-01.fragment-03",
                "toehold": second_toehold,
                "toehold_complement": _reverse_complement(second_toehold),
                "barcode": second_barcode,
                "barcode_complement": _reverse_complement(second_barcode),
                "complement_nick_sequence_layout_valid": True,
                "complement_end_preparation": "downstream_phosphorylation",
            },
        ],
    }
    payload["strands"] = [
        {
            "fragment_id": "target-01.fragment-01",
            "role": "first",
            "incoming_junction_id": None,
            "outgoing_junction_id": "target-01.junction-01",
            "barcode_bearing_sequence_5to3": target[:22] + first_barcode,
            "complement_sequence_5to3": _reverse_complement(target[:18]),
        },
        {
            "fragment_id": "target-01.fragment-02",
            "role": "internal",
            "incoming_junction_id": "target-01.junction-01",
            "outgoing_junction_id": "target-01.junction-02",
            "barcode_bearing_sequence_5to3": (_reverse_complement(first_barcode) + target[22:39] + second_barcode),
            "complement_sequence_5to3": (_reverse_complement(target[22:35]) + _reverse_complement(first_toehold)),
        },
        {
            "fragment_id": "target-01.fragment-03",
            "role": "last",
            "incoming_junction_id": "target-01.junction-02",
            "outgoing_junction_id": None,
            "barcode_bearing_sequence_5to3": _reverse_complement(second_barcode) + target[39:],
            "complement_sequence_5to3": (_reverse_complement(target[39:]) + _reverse_complement(second_toehold)),
        },
    ]
    payload["recovery"]["last_fragment_id"] = "target-01.fragment-03"
    payload["search"].update({"locus_count": 2, "barcode_candidates_generated": 50})
    return payload


def test_review_contract_accepts_complete_neutral_evidence() -> None:
    payload = three_way_junction_review_payload()

    review = ThreeWayJunctionReviewV1.model_validate(payload)

    assert review.target.target_id == "target-01"
    assert review.recovery.forward.order_sequence_5to3 == (
        review.recovery.forward.five_prime_extension_5to3 + review.recovery.forward.binding_sequence_5to3
    )
    assert review.search.thermodynamic_screening == "not_run"
    assert {(check.check, check.status) for check in review.checks} == {
        ("target_reconstruction", "passed"),
        ("thermodynamic_screening", "not_run"),
    }


@pytest.mark.parametrize(
    "field",
    ["toehold_paths_evaluated", "barcode_subsets_evaluated", "matchings_evaluated"],
)
def test_junction_string_v1_search_evaluations_respect_producer_budget(field: str) -> None:
    payload = three_way_junction_review_payload()
    payload["source"]["algorithm"] = "dnadesign.junction.string.v1"
    payload["search"]["locus_count"] = 10
    payload["search"]["barcode_candidates_generated"] = 50
    payload["search"][field] = 100_001

    accepted = ThreeWayJunctionReviewV1.model_validate(payload)

    assert getattr(accepted.search, field) == 100_001

    payload["search"][field] = 100_002
    with pytest.raises(ValidationError, match=rf"{field} must not exceed 100001"):
        ThreeWayJunctionReviewV1.model_validate(payload)

    payload["source"]["algorithm"] = "other.producer.algorithm.v2"
    extensible = ThreeWayJunctionReviewV1.model_validate(payload)
    assert getattr(extensible.search, field) == 100_002


@pytest.mark.parametrize(
    "field",
    ["toehold_seed", "barcode_generation_seed", "barcode_subset_seed", "matching_seed"],
)
@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_junction_string_v1_search_seeds_respect_uint64_domain(field: str, seed: int) -> None:
    payload = three_way_junction_review_payload()
    payload["source"]["algorithm"] = "dnadesign.junction.string.v1"
    payload["search"][field] = seed

    with pytest.raises(ValidationError, match=rf"{field} must be between 0 and"):
        ThreeWayJunctionReviewV1.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["toehold_seed", "barcode_generation_seed", "barcode_subset_seed", "matching_seed"],
)
def test_junction_string_v1_search_seeds_accept_uint64_boundaries(field: str) -> None:
    payload = three_way_junction_review_payload()
    payload["source"]["algorithm"] = "dnadesign.junction.string.v1"

    for seed in (0, (1 << 64) - 1):
        payload["search"][field] = seed
        accepted = ThreeWayJunctionReviewV1.model_validate(payload)
        assert getattr(accepted.search, field) == seed


@pytest.mark.parametrize(
    "field",
    ["toehold_seed", "barcode_generation_seed", "barcode_subset_seed", "matching_seed"],
)
@pytest.mark.parametrize("seed", [True, 1.0, "11"])
def test_search_seeds_require_exact_integers(field: str, seed: object) -> None:
    payload = three_way_junction_review_payload()
    payload["search"][field] = seed

    with pytest.raises(ValidationError, match="valid integer"):
        ThreeWayJunctionReviewV1.model_validate(payload)


def test_search_seed_domain_remains_extensible_for_other_algorithms() -> None:
    payload = three_way_junction_review_payload()
    payload["source"]["algorithm"] = "other.producer.algorithm.v2"
    payload["search"]["toehold_seed"] = -1
    payload["search"]["matching_seed"] = 1 << 64

    accepted = ThreeWayJunctionReviewV1.model_validate(payload)

    assert accepted.search.toehold_seed == -1
    assert accepted.search.matching_seed == 1 << 64


def test_junction_string_v1_requires_one_complement_end_preparation_per_target() -> None:
    payload = _two_junction_review_payload()
    payload["source"]["algorithm"] = "dnadesign.junction.string.v1"

    with pytest.raises(ValidationError, match="must use one complement end preparation"):
        ThreeWayJunctionReviewV1.model_validate(payload)


def test_other_algorithms_may_declare_mixed_complement_end_preparation() -> None:
    payload = _two_junction_review_payload()
    payload["source"]["algorithm"] = "other.producer.algorithm.v2"

    accepted = ThreeWayJunctionReviewV1.model_validate(payload)

    assert {junction.complement_end_preparation for junction in accepted.geometry.junctions} == {
        "vendor_5_prime_phosphate",
        "downstream_phosphorylation",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"study": "private-study"}), "Extra inputs are not permitted"),
        (
            lambda payload: payload["geometry"]["junctions"][0].update({"barcode_complement": "AAAAAAAA"}),
            "barcode_complement",
        ),
        (
            lambda payload: payload["recovery"]["forward"].update({"order_sequence_5to3": "TTTTTTTT"}),
            "order_sequence_5to3",
        ),
        (
            lambda payload: payload["search"].update({"thermodynamic_screening": "passed"}),
            "thermodynamic_screening",
        ),
        (
            lambda payload: payload["search"].update({"toehold_min_distance": 10.0, "toehold_mean_distance": 1.0}),
            "toehold_min_distance must be <= toehold_mean_distance",
        ),
        (
            lambda payload: payload["search"].update({"barcode_min_distance": 10.0, "barcode_mean_distance": 1.0}),
            "barcode_min_distance must be <= barcode_mean_distance",
        ),
        (
            lambda payload: payload["search"].update({"toehold_rank_score": 1.5000001}),
            "less than or equal to 1.5",
        ),
        (
            lambda payload: payload["search"].update({"barcode_candidates_generated": 4}),
            "inferred barcode pool factor must be at least 5",
        ),
        (
            lambda payload: payload["search"].update({"locus_count": 2, "barcode_candidates_generated": 11}),
            "barcode_candidates_generated must be a multiple of locus_count",
        ),
        (
            lambda payload: payload["search"].update({"barcode_forbidden_toehold_k": 5}),
            "barcode_forbidden_toehold_k must not exceed",
        ),
        (
            lambda payload: payload["search"].update({"barcode_forbidden_barcode_k": 9}),
            "barcode_forbidden_barcode_k must not exceed barcode length",
        ),
        (
            lambda payload: payload["search"].update({"barcode_forbidden_barcode_k": 3}),
            "barcode_forbidden_barcode_k must be greater than barcode_forbidden_toehold_k",
        ),
        (
            lambda payload: payload["checks"][1].update({"status": "passed"}),
            "thermodynamic_screening check status",
        ),
        (
            lambda payload: payload["checks"].pop(),
            "exactly one assembly-group-scoped thermodynamic_screening check",
        ),
        (
            lambda payload: payload["checks"][1]["subject"].update({"kind": "target", "id": "target-01"}),
            "exactly one assembly-group-scoped thermodynamic_screening check",
        ),
        (
            lambda payload: payload["checks"].append(
                {
                    "subject": {"kind": "target", "id": "target-01"},
                    "check": "thermodynamic_screening",
                    "status": "not_run",
                    "detail": "duplicate scope",
                }
            ),
            "exactly one assembly-group-scoped thermodynamic_screening check",
        ),
        (
            lambda payload: payload["recovery"].update({"mode": "construct_specific"}),
            "recovery.mode",
        ),
        (
            lambda payload: payload["strands"][0].update({"barcode_bearing_sequence_5to3": "ACGT"}),
            "barcode-bearing sequence",
        ),
        (
            lambda payload: payload["recovery"].update({"extended_bottom_sequence_5to3": "ACGT"}),
            "extended_bottom_sequence_5to3",
        ),
        (
            lambda payload: payload["checks"].append(payload["checks"][0]),
            "check subject and name tuples must be unique",
        ),
        (
            lambda payload: payload["checks"][0]["subject"].update({"id": "other-target"}),
            "target check subject id",
        ),
    ],
)
def test_review_contract_fails_closed_on_semantic_drift(mutate, message: str) -> None:
    payload = three_way_junction_review_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        ThreeWayJunctionReviewV1.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "toehold_min_distance",
        "toehold_mean_distance",
        "toehold_rank_score",
        "barcode_min_distance",
        "barcode_mean_distance",
        "barcode_rank_score",
    ],
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_review_contract_rejects_nonfinite_search_metrics(field: str, value: float) -> None:
    payload = three_way_junction_review_payload()
    payload["search"][field] = value

    with pytest.raises(ValidationError, match="finite number"):
        ThreeWayJunctionReviewV1.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"toehold_min_distance": 1.0, "toehold_mean_distance": 1.0}, "zero pairwise"),
        ({"barcode_min_distance": 1.0, "barcode_mean_distance": 1.0}, "zero pairwise"),
        ({"matching_max_pairwise_lcs": 1}, "zero pairwise"),
        ({"toehold_rank_score": 1.0}, "rank scores of 1.5"),
        ({"barcode_rank_score": 1.0}, "rank scores of 1.5"),
        ({"matchings_evaluated": 2}, "exactly one matching evaluation"),
    ],
)
def test_singleton_assembly_group_requires_exact_v1_pairwise_search_evidence(
    updates: dict[str, float | int],
    message: str,
) -> None:
    payload = three_way_junction_review_payload()
    payload["search"].update(updates)

    with pytest.raises(ValidationError, match=message):
        ThreeWayJunctionReviewV1.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"toehold_min_distance": 8.0000001, "toehold_mean_distance": 8.0000001},
            "toehold distances exceed the v1 sequence-derived maximum",
        ),
        (
            {"barcode_min_distance": 8.0000001, "barcode_mean_distance": 8.0000001},
            "barcode distances exceed the sequence-derived maximum",
        ),
        (
            {"matching_max_pairwise_lcs": 13},
            "matching_max_pairwise_lcs exceeds the combined junction length",
        ),
        (
            {"matchings_evaluated": 3},
            "matchings_evaluated must not exceed the distinct permutation count",
        ),
    ],
)
def test_multi_locus_assembly_group_search_respects_sequence_derived_bounds(
    updates: dict[str, float | int],
    message: str,
) -> None:
    search_payload = three_way_junction_review_payload()["search"]
    search_payload.update({"locus_count": 2, "barcode_candidates_generated": 10, **updates})
    search = AssemblyGroupSearchReview.model_validate(search_payload)

    with pytest.raises(ValueError, match=message):
        search._validate_sequence_bounds(toehold_length=4, barcode_length=8)


def test_barcode_subset_evaluations_do_not_exceed_combination_capacity() -> None:
    search_payload = three_way_junction_review_payload()["search"]
    search_payload.update(
        {
            "locus_count": 2,
            "barcode_candidates_generated": 10,
            "barcode_subsets_evaluated": 45,
        }
    )
    accepted = AssemblyGroupSearchReview.model_validate(search_payload)

    accepted._validate_sequence_bounds(toehold_length=4, barcode_length=8)

    search_payload["barcode_subsets_evaluated"] = 46
    rejected = AssemblyGroupSearchReview.model_validate(search_payload)
    with pytest.raises(ValueError, match="barcode_subsets_evaluated must not exceed the distinct combination count"):
        rejected._validate_sequence_bounds(toehold_length=4, barcode_length=8)


@pytest.mark.parametrize(
    ("second_toehold", "second_barcode", "message"),
    [
        ("ACG", "AACCGGTT", "uniform toehold length"),
        ("ACGT", "AACCGGT", "uniform barcode length"),
    ],
)
def test_review_geometry_requires_uniform_junction_sequence_lengths(
    second_toehold: str,
    second_barcode: str,
    message: str,
) -> None:
    geometry = {
        "fragments": [
            {"fragment_id": "f1", "index": 0, "role": "first", "domain_span": {"start": 0, "end": 1}},
            {"fragment_id": "f2", "index": 1, "role": "internal", "domain_span": {"start": 5, "end": 6}},
            {"fragment_id": "f3", "index": 2, "role": "last", "domain_span": {"start": 10, "end": 11}},
        ],
        "junctions": [
            {
                "junction_id": "j1",
                "toehold_span": {"start": 1, "end": 5},
                "left_fragment_id": "f1",
                "right_fragment_id": "f2",
                "toehold": "ACGT",
                "toehold_complement": _reverse_complement("ACGT"),
                "barcode": "AACCGGTT",
                "barcode_complement": _reverse_complement("AACCGGTT"),
                "complement_nick_sequence_layout_valid": True,
                "complement_end_preparation": "vendor_5_prime_phosphate",
            },
            {
                "junction_id": "j2",
                "toehold_span": {"start": 6, "end": 6 + len(second_toehold)},
                "left_fragment_id": "f2",
                "right_fragment_id": "f3",
                "toehold": second_toehold,
                "toehold_complement": _reverse_complement(second_toehold),
                "barcode": second_barcode,
                "barcode_complement": _reverse_complement(second_barcode),
                "complement_nick_sequence_layout_valid": True,
                "complement_end_preparation": "vendor_5_prime_phosphate",
            },
        ],
    }

    with pytest.raises(ValidationError, match=message):
        AssemblyGeometry.model_validate(geometry)
