"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/tests/test_artifact_projection.py

Pre-search publication-size projection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from dnadesign.trijunction.contracts.publication.limits import ARTIFACT_BYTE_LIMITS
from dnadesign.trijunction.contracts.request import parse_request
from dnadesign.trijunction.design import planner as planner_module
from dnadesign.trijunction.design.planner import design_trijunction
from dnadesign.trijunction.design.resources.artifact_projection import project_artifact_bytes
from dnadesign.trijunction.errors import TriJunctionDesignError
from dnadesign.trijunction.publication import verify as verifier_module
from dnadesign.trijunction.publication.payloads import bundle_payloads
from dnadesign.trijunction.tests.scenarios.factories import scale_request_mapping
from dnadesign.trijunction.tests.test_planner import _request_mapping


def _universal_request_mapping() -> dict[str, object]:
    raw = _request_mapping()
    raw["targets"] = [raw["targets"][0]]  # type: ignore[index]
    raw["targets"][0]["recovery_primers"]["mode"] = "universal"  # type: ignore[index]
    return raw


def _max_label_request_mapping() -> dict[str, object]:
    raw = _universal_request_mapping()
    target = raw["targets"][0]  # type: ignore[index]
    target["id"] = "t" * 128
    target["pool_id"] = "p" * 128
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
            oligo_length=200,
            search_range=2,
            barcode_generation_attempts=5_000,
        ),
        scale_request_mapping(
            target_count=3,
            target_length=1_000,
            topology="independent",
            oligo_length=200,
            search_range=2,
            barcode_generation_attempts=5_000,
        ),
    ],
    ids=[
        "target-specific-multi-pool",
        "universal-single-target",
        "maximum-label-bytes",
        "multi-target-pool",
        "independent-pools",
    ],
)
def test_realized_payloads_do_not_exceed_pre_search_projection(raw: dict[str, object]) -> None:
    request = parse_request(deepcopy(raw))
    predicted_loci = planner_module._predict_loci(request)

    projection = project_artifact_bytes(request, predicted_loci_by_target=predicted_loci)
    payloads = bundle_payloads(request, design_trijunction(request))

    assert set(projection) == {"plan", "checks", "orders", "review"}
    assert all(len(payloads[key]) <= projection[key] <= ARTIFACT_BYTE_LIMITS[key] for key in projection)


def test_projected_artifact_refusal_happens_before_search_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = scale_request_mapping(
        target_count=3_000,
        target_length=200,
        topology="independent",
        oligo_length=96,
        search_range=1,
        barcode_generation_attempts=500,
    )
    request = parse_request(raw)

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate materialization must not begin")

    monkeypatch.setattr(planner_module, "enumerate_loci", fail_if_materialized)

    with pytest.raises(
        TriJunctionDesignError,
        match="projected 'checks' artifact.*publication and verification limit",
    ):
        design_trijunction(request)


def test_planning_and_verification_share_one_artifact_limit_contract() -> None:
    assert verifier_module.ARTIFACT_BYTE_LIMITS is ARTIFACT_BYTE_LIMITS
