"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_review_projection.py

Projection of verified plans into the shared visual-review contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from dnadesign.contracts.visual import ThreeWayJunctionReviewV1
from dnadesign.trijunction.contracts import parse_request
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.presentation.review_contract import (
    render_review_contracts,
    review_contracts,
)
from dnadesign.trijunction.tests.scenarios.factories import scale_request_mapping
from dnadesign.trijunction.tests.test_planner import _request_mapping


def test_review_projection_is_exact_valid_and_structurally_scoped() -> None:
    plan = design_trijunction(parse_request(_request_mapping()))

    reviews = review_contracts(plan)

    assert len(reviews) == 1
    review = reviews[0]
    assert isinstance(review, ThreeWayJunctionReviewV1)
    assert review.source.plan_id == plan.plan_id
    assert review.source.algorithm == "dnadesign.trijunction.string.v1"
    assert review.target.sequence_5to3 == plan.targets[0].reconstructed_target
    assert review.recovery.forward.order_sequence_5to3 == plan.targets[0].recovery.forward_order_sequence
    assert review.recovery.reverse.order_sequence_5to3 == plan.targets[0].recovery.reverse_order_sequence
    assert review.search.thermodynamic_screening == "not_run"
    assert review.search.toehold_paths_evaluated <= 100_001
    assert review.search.barcode_subsets_evaluated <= 100_001
    assert review.search.matchings_evaluated <= 100_001
    assert {(check.subject.kind, check.subject.id) for check in review.checks} == {
        ("pool", plan.targets[0].pool_id),
        ("target", plan.targets[0].target_id),
    }
    assert all(
        junction.toehold_span.start
        == next(item for item in plan.pools[0].junctions if item.junction_id == junction.junction_id).start
        for junction in review.geometry.junctions
    )


def test_review_projection_serializes_as_one_canonical_json_array() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=2,
            target_length=1_000,
            topology="independent",
            oligo_length=96,
            search_range=2,
            barcode_generation_attempts=250_000,
        )
    )
    plan = design_trijunction(request)

    first = render_review_contracts(plan)
    second = render_review_contracts(plan)

    assert first == second
    payload = json.loads(first)
    assert isinstance(payload, list)
    assert [row["target"]["target_id"] for row in payload] == [target.target_id for target in plan.targets]
    for row in payload:
        search = row["search"]
        assert search["barcode_subsets_evaluated"] <= math.comb(
            search["barcode_candidates_generated"],
            search["locus_count"],
        )
    assert first.endswith(b"\n")
    assert b"NaN" not in first


def test_review_projection_rejects_cross_layer_junction_drift() -> None:
    plan = design_trijunction(parse_request(_request_mapping()))
    pool = plan.pools[0]
    junction = pool.junctions[0]
    corrupted_pool = replace(
        pool,
        junctions=(replace(junction, barcode="A" * len(junction.barcode)), *pool.junctions[1:]),
    )
    corrupted = replace(plan, pools=(corrupted_pool, *plan.pools[1:]))

    with pytest.raises(TriJunctionDesignError, match="inconsistent evidence"):
        review_contracts(corrupted)
