"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/_support.py

Shared operator state and regeneration test construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    AcquisitionContribution,
    AcquisitionCoordinate,
    AcquisitionMetricProjection,
    AcquisitionProjection,
    acquisition_projection_payload,
    build_acquisition_projection,
    decision_to_dict,
    evaluate_sensitivity,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.sensitivity import (
    sensitivity_evaluations_to_payload,
)

from .....reporter_response.metastudy.sensitivity_coverage import sensitivity_coverage_receipt_payload
from .._builders import _attempts, _evidence, _ready, evaluate_metastudy
from ..evidence._builders import _complete_sensitivity_evidence, _sensitivity_coverages


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _checked_state_path() -> Path:
    return next(
        parent
        / "docs/studies/rt_lnrna_sponging_construct_triage/contexts/reporter-response-metastudy/metastudy-state.yaml"
        for parent in Path(__file__).resolve().parents
        if (parent / "docs/studies/rt_lnrna_sponging_construct_triage").is_dir()
    )


def _acquisition_projection_payload() -> dict[str, object]:
    metric = AcquisitionMetricProjection(
        estimate=1.0,
        method="median_across_acquisitions",
        acquisition_count=1,
        leave_one_acquisition_out_estimates=(),
    )
    acquisition_id = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0]
    projection = AcquisitionProjection(
        contract_id="rt_lnrna_reporter_response_acquisition_projection.v3",
        selected_reduction=(6.0, 10.0),
        coordinates=(
            AcquisitionCoordinate(
                subject_id="synthetic-subject",
                condition_role="dose",
                metric_space="reference_normalized",
                dose_uM=500.0,
                reduction_id="window-6-10h",
                reduction_digest=_digest("1"),
                observation_policy_digest=_digest("2"),
                acquisition_ids=(acquisition_id,),
                contributions=(
                    AcquisitionContribution(
                        acquisition_id=acquisition_id,
                        profile_id="synthetic-profile",
                        profile_digest=_digest("3"),
                        declared_biological_replicate_ids=(),
                        rfp=None,
                        od600=None,
                        rfp_over_od600=None,
                        normalized_reporter_response=1.0,
                        relative_od=1.0,
                    ),
                ),
                rfp=None,
                od600=None,
                rfp_over_od600=None,
                normalized_reporter_response=metric,
                relative_od=metric,
            ),
        ),
    )
    return acquisition_projection_payload(projection)


def _state_for_external_registry(phd_root: Path) -> dict[str, object]:
    registry = phd_root / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text('{"routes": []}\n', encoding="utf-8")
    primary = _evidence()
    attempts = _attempts(primary)
    decision = evaluate_metastudy(primary, readiness=_ready())
    sensitivity = _complete_sensitivity_evidence(primary)
    coverages = _sensitivity_coverages(sensitivity, attempts)
    projection = build_acquisition_projection(primary, selected_reduction=decision.selected_reduction)
    registry_digest = "sha256:" + hashlib.sha256(registry.read_bytes()).hexdigest()
    readiness = decision.readiness
    payload = {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v7",
        "readiness": {
            "schema_id": "rt_lnrna_reporter_response_readiness_snapshot.v1",
            "source_identity": {
                "route_id": operator_state.ROUTE_ID,
                "route_registry_path": operator_state.ROUTE_REGISTRY_PATH,
                "route_registry_digest": registry_digest,
                "normalized_full_receipt_digest": readiness.receipt_digest,
                "normalization": operator_state.RECEIPT_NORMALIZATION,
            },
            "last_verified": "2026-08-02",
            "selected_experiment_count": readiness.selected_experiment_count,
            "related_experiment_count": len(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
            "related_experiment_ids": list(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids),
            "ready_experiment_count": readiness.ready_experiment_count,
            "ready_experiment_ids": list(readiness.ready_experiment_ids),
            "blocked_experiment_ids": list(readiness.blocked_experiment_ids),
        },
        "decision": decision_to_dict(decision),
        "objective_readiness": asdict(DEFAULT_OBJECTIVE_READINESS),
        "sensitivity_evaluations": sensitivity_evaluations_to_payload(evaluate_sensitivity(sensitivity)),
        "sensitivity_coverage_receipts": [sensitivity_coverage_receipt_payload(coverage) for coverage in coverages],
        "acquisition_projection": acquisition_projection_payload(projection),
    }
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["generation_digest"] = operator_state.canonical_digest(
        {
            key: payload[key]
            for key in (
                "readiness",
                "decision",
                "objective_readiness",
                "sensitivity_evaluations",
                "sensitivity_coverage_receipts",
                "acquisition_projection",
            )
            if key in payload
        }
    )
    return payload


__all__ = [
    "_checked_state_path",
    "_digest",
    "_state_for_external_registry",
]
