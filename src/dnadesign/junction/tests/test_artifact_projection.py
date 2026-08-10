"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_artifact_projection.py

Pre-search publication-size projection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from dnadesign.junction.contracts.publication.limits import ARTIFACT_BYTE_LIMITS
from dnadesign.junction.contracts.request import parse_request
from dnadesign.junction.design import planner as planner_module
from dnadesign.junction.design.planner import design_junction
from dnadesign.junction.design.resources.artifact_projection import project_artifact_bytes
from dnadesign.junction.errors import JunctionDesignError
from dnadesign.junction.publication import verify as verifier_module
from dnadesign.junction.publication.payloads import render_artifact_bytes
from dnadesign.junction.tests.scenarios.factories import scale_request_mapping
from dnadesign.junction.tests.test_planner import _request_mapping


def _universal_request_mapping() -> dict[str, object]:
    raw = _request_mapping()
    raw["targets"] = [raw["targets"][0]]  # type: ignore[index]
    raw["targets"][0]["recovery_primers"]["mode"] = "universal"  # type: ignore[index]
    return raw


def _max_label_request_mapping() -> dict[str, object]:
    raw = _universal_request_mapping()
    target = raw["targets"][0]  # type: ignore[index]
    target["id"] = "t" * 128
    target["assembly_group_id"] = "p" * 128
    policy = raw["order_policy"]
    for field in (
        "synthesis_scale",
        "barcode_bearing_purification",
        "complement_purification",
        "primer_purification",
    ):
        policy[field] = "é" * 64
    return raw


@pytest.mark.parametrize(
    "raw",
    [
        _request_mapping(),
        _universal_request_mapping(),
        _max_label_request_mapping(),
        scale_request_mapping(
            target_count=3,
            target_length=1_000,
            topology="shared",
            nominal_fragment_oligo_length=200,
            search_range=2,
            barcode_generation_attempts=5_000,
        ),
        scale_request_mapping(
            target_count=3,
            target_length=1_000,
            topology="independent",
            nominal_fragment_oligo_length=200,
            search_range=2,
            barcode_generation_attempts=5_000,
        ),
    ],
    ids=[
        "target-specific-multi-group",
        "universal-single-target",
        "maximum-label-bytes",
        "multi-target-assembly-group",
        "independent-assembly-groups",
    ],
)
def test_realized_payloads_do_not_exceed_pre_search_projection(raw: dict[str, object]) -> None:
    request = parse_request(deepcopy(raw))
    predicted_loci = planner_module._predict_loci(request)
    plan = design_junction(request)

    projection = project_artifact_bytes(request, predicted_loci_by_target=predicted_loci)

    assert set(projection) == {
        "plan",
        "checks",
        "orders",
        "targets",
        "order_sequences",
        "expected_products",
        "review",
        "sequence_dissimilarity",
    }
    for key, projected_bytes in projection.items():
        payload = render_artifact_bytes(key, request, plan)
        try:
            assert len(payload) <= projected_bytes <= ARTIFACT_BYTE_LIMITS[key]
        finally:
            del payload


def test_projected_artifact_refusal_happens_before_search_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = scale_request_mapping(
        target_count=3_000,
        target_length=200,
        topology="independent",
        nominal_fragment_oligo_length=96,
        search_range=1,
        barcode_generation_attempts=500,
    )
    request = parse_request(raw)

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(
        JunctionDesignError,
        match="projected 'checks' artifact.*publication and verification limit",
    ):
        design_junction(request)


def test_planning_and_verification_share_one_artifact_limit_contract() -> None:
    assert verifier_module.ARTIFACT_BYTE_LIMITS is ARTIFACT_BYTE_LIMITS
