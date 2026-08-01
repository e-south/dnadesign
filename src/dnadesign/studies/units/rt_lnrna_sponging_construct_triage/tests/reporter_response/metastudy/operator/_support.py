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
from pathlib import Path

import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    DEFAULT_PROTOCOL,
    AcquisitionContribution,
    AcquisitionCoordinate,
    AcquisitionMetricProjection,
    AcquisitionProjection,
    acquisition_projection_payload,
    protocol_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.contracts import (
    canonical_digest,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)


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
        contract_id="rt_lnrna_reporter_response_acquisition_projection.v2",
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
    payload = yaml.safe_load(_checked_state_path().read_text(encoding="utf-8"))
    payload["schema_id"] = "rt_lnrna_reporter_response_metastudy_state.v6"
    payload["decision"]["condition_ontology_digest"] = DEFAULT_PROTOCOL.condition_ontology_digest
    payload["decision"]["policy_digest"] = protocol_digest()
    attempt_digests: dict[str, str] = {}
    for attempt in payload["decision"]["materialization_attempts"]:
        attempt["attempt_digest"] = canonical_digest(
            {key: value for key, value in attempt.items() if key != "attempt_digest"}
        )
        attempt_digests[attempt["experiment_id"]] = attempt["attempt_digest"]
    payload["objective_readiness"] = {
        "contract_id": DEFAULT_OBJECTIVE_READINESS.contract_id,
        "status": DEFAULT_OBJECTIVE_READINESS.status,
        "objective_id": DEFAULT_OBJECTIVE_READINESS.objective_id,
        "blockers": list(DEFAULT_OBJECTIVE_READINESS.blockers),
    }
    payload["sensitivity_evaluations"] = []
    payload.setdefault("sensitivity_coverage_receipts", [])
    for receipt in payload["sensitivity_coverage_receipts"]:
        receipt["materialization_attempt_digest"] = attempt_digests[receipt["experiment_id"]]
    payload["acquisition_projection"] = _acquisition_projection_payload()
    payload.pop("evidence", None)
    payload["readiness"]["source_identity"]["route_registry_digest"] = (
        "sha256:" + hashlib.sha256(registry.read_bytes()).hexdigest()
    )
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
