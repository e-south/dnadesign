"""Tests compact selection state and forgery resistance."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_OBJECTIVE_READINESS,
    MetastudyContractError,
    acquisition_projection_payload,
    build_acquisition_projection,
    decision_evidence_payload,
    decision_to_dict,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    sensitivity_coverage as sensitivity_coverage_contracts,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)

from .._builders import KINETIC_IDS, _evidence, _ready, evaluate_metastudy
from ..evidence._builders import _complete_sensitivity_evidence, _sensitivity_coverages
from ._support import _digest


def test_selected_source_state_is_compact_and_rejects_phase_ineligible_forgery() -> None:
    evidence = _evidence()
    selected = evaluate_metastudy(evidence, readiness=_ready())
    decision = json.loads(json.dumps(decision_to_dict(selected)))
    readiness = {
        "schema_id": "rt_lnrna_reporter_response_readiness_snapshot.v1",
        "source_identity": {
            "route_id": "rt_lnrna_reporter_response_metastudy",
            "route_registry_path": ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json",
            "route_registry_digest": _digest("a"),
            "normalized_full_receipt_digest": decision["readiness"]["receipt_digest"],
            "normalization": "omit environment-specific reader_command before canonical JSON hashing",
        },
        "last_verified": "2026-08-01",
        "selected_experiment_count": 8,
        "related_experiment_count": 1,
        "related_experiment_ids": ["20251105_retron_Eco1_RT_variants"],
        "ready_experiment_count": 8,
        "ready_experiment_ids": list(KINETIC_IDS),
        "blocked_experiment_ids": [],
    }
    body = {
        "readiness": readiness,
        "decision": decision,
        "objective_readiness": asdict(DEFAULT_OBJECTIVE_READINESS),
        "sensitivity_evaluations": [],
        "sensitivity_coverage_receipts": [
            sensitivity_coverage_contracts.sensitivity_coverage_receipt_payload(row)
            for row in _sensitivity_coverages(
                _complete_sensitivity_evidence(evidence),
                selected.materialization_attempts,
            )
        ],
        "acquisition_projection": acquisition_projection_payload(
            build_acquisition_projection(evidence, selected_reduction=selected.selected_reduction)
        ),
    }
    payload = {
        "schema_id": "rt_lnrna_reporter_response_metastudy_state.v6",
        "generation_digest": operator_state.canonical_digest(body),
        **body,
    }
    operator_state.validate_state_payload(payload)

    with_embedded_evidence = {
        **payload,
        "evidence": json.loads(json.dumps(decision_evidence_payload(evidence, decision=selected))),
    }
    with_embedded_evidence["generation_digest"] = operator_state.canonical_digest(
        {**body, "evidence": with_embedded_evidence["evidence"]}
    )
    with pytest.raises(MetastudyContractError, match="fields do not match"):
        operator_state.validate_state_payload(with_embedded_evidence)

    decision["selected_reduction"] = [12.0, 16.0]
    for evaluation in decision["evaluations"]:
        if evaluation["reduction"] == [12.0, 16.0]:
            evaluation.update(
                worst_experiment_control_separation=1_000.0,
                repeated_anchor_drift=0.0,
                within_acquisition_observation_range=0.0,
                eligible_experiment_count=8,
                anchor_ordered_acquisition_count=5,
                co_measured_anchor_acquisition_count=5,
                loo_same_or_adjacent_fraction=1.0,
                eligible=True,
                blockers=[],
            )
    payload["generation_digest"] = operator_state.canonical_digest(body)

    with pytest.raises(MetastudyContractError, match="descriptive support and phase gates"):
        operator_state.validate_state_payload(payload)
