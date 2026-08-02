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


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def three_way_junction_review_payload() -> dict[str, object]:
    target = "ACGTACGTTGCATGCAGATTACAGGCTAACCGTACGATCGTTAACCGGTTACGATC"
    reverse_binding = _reverse_complement(target[-8:])
    return {
        "contract_kind": "three_way_junction_review_v1",
        "source": {
            "plan_schema": "dnadesign.trijunction.plan.v1",
            "plan_id": f"sha256:{'a' * 64}",
            "request_sha256": f"sha256:{'b' * 64}",
            "algorithm": "trijunction.v1",
        },
        "target": {
            "target_id": "target-01",
            "pool_id": "pool-01",
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
                    "complement_nick_geometry_valid": True,
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
            "expected_product_sequence_5to3": target,
            "extended_top_sequence_5to3": "GG" + target + _reverse_complement("TT"),
            "extended_bottom_sequence_5to3": "TT" + _reverse_complement(target) + _reverse_complement("GG"),
        },
        "search": {
            "pool_id": "pool-01",
            "toehold_seed": 11,
            "barcode_generation_seed": 12,
            "barcode_subset_seed": 13,
            "matching_seed": 14,
            "locus_count": 1,
            "toehold_paths_evaluated": 20,
            "toehold_min_distance": 4.0,
            "toehold_mean_distance": 4.0,
            "toehold_rank_score": 1.0,
            "barcode_candidates_generated": 25,
            "barcode_forbidden_toehold_k": 3,
            "barcode_forbidden_barcode_k": 4,
            "barcode_subsets_evaluated": 20,
            "barcode_min_distance": 6.0,
            "barcode_mean_distance": 6.0,
            "barcode_rank_score": 1.0,
            "matchings_evaluated": 1,
            "matching_max_pairwise_lcs": 2,
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
                "subject": {"kind": "pool", "id": "pool-01"},
                "check": "thermodynamic_screening",
                "status": "not_run",
                "detail": "not part of this contract",
            },
        ],
    }


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
            lambda payload: payload["checks"][1].update({"status": "passed"}),
            "thermodynamic_screening check status",
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
