"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_sequence_dissimilarity_projection.py

Assembly-group sequence-dissimilarity review projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dnadesign.junction.contracts import parse_request
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.presentation.sequence_review import (
    JunctionSequenceDissimilarityV1,
    render_sequence_dissimilarity_contracts,
    sequence_dissimilarity_contracts,
)
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping
from dnadesign.junction.tests.test_planner import _request_mapping


def test_sequence_dissimilarity_projection_is_group_scoped_and_exact() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=3,
            target_length=360,
            topology="shared",
            nominal_fragment_oligo_length=96,
            search_range=2,
            barcode_generation_attempts=50_000,
        )
    )
    plan = design_junction(request)

    reviews = sequence_dissimilarity_contracts(plan)

    assert len(reviews) == 1
    review = reviews[0]
    assert isinstance(review, JunctionSequenceDissimilarityV1)
    assert review.assembly_group_id == plan.assembly_groups[0].assembly_group_id
    assert review.thermodynamic_screening == "not_run"
    assert [junction.junction_id for junction in review.junctions] == [
        junction.junction_id for junction in plan.assembly_groups[0].junctions
    ]
    assert [junction.toehold_sequence_5to3 for junction in review.junctions] == [
        junction.toehold for junction in plan.assembly_groups[0].junctions
    ]
    assert [junction.barcode_sequence_5to3 for junction in review.junctions] == [
        junction.barcode for junction in plan.assembly_groups[0].junctions
    ]


def test_sequence_dissimilarity_projection_is_canonical_and_one_row_per_group() -> None:
    request = parse_request(
        scale_request_mapping(
            target_count=2,
            target_length=360,
            topology="independent",
            nominal_fragment_oligo_length=96,
            search_range=2,
            barcode_generation_attempts=50_000,
        )
    )
    plan = design_junction(request)

    first = render_sequence_dissimilarity_contracts(plan)
    second = render_sequence_dissimilarity_contracts(plan)
    rows = json.loads(first)

    assert first == second
    assert [row["assembly_group_id"] for row in rows] == sorted(
        group.assembly_group_id for group in plan.assembly_groups
    )
    assert first.endswith(b"\n")


def test_sequence_dissimilarity_projection_rejects_group_identity_drift() -> None:
    plan = design_junction(parse_request(_request_mapping()))
    group = plan.assembly_groups[0]
    corrupted = replace(
        plan,
        assembly_groups=(
            replace(group, junctions=(replace(group.junctions[0], assembly_group_id="wrong"), *group.junctions[1:])),
        ),
    )

    with pytest.raises(JunctionDesignError, match="inconsistent assembly group"):
        sequence_dissimilarity_contracts(corrupted)
