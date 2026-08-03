"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_state_lineage.py

Sensitivity-receipt lineage checks for source-controlled meta-study state.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    MetastudyContractError,
    operator,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.operator import (
    state as operator_state,
)

from ._support import _digest, _state_for_external_registry


def _redigest(payload: dict[str, object]) -> None:
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


def test_state_validation_rejects_incomplete_sensitivity_coverage_receipt(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["profile_count"] -= 1
    _redigest(payload)
    state_path = tmp_path / "incomplete-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="coordinate counts changed"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_sensitivity_receipt_attempt_drift(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["materialization_attempt_digest"] = _digest("f")
    _redigest(payload)
    state_path = tmp_path / "drifted-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="exact materialization attempt"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


@pytest.mark.parametrize(
    "drift_kind",
    [
        "binding_artifact_id",
        "binding_artifact_digest",
        "subject_roster",
    ],
)
def test_state_validation_rejects_sensitivity_receipt_attempt_lineage_drift(
    tmp_path: Path,
    drift_kind: str,
) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    receipt = payload["sensitivity_coverage_receipts"][0]
    if drift_kind == "binding_artifact_id":
        receipt["evidence_binding_artifact_id"] = "drifted-binding-artifact"
    elif drift_kind == "binding_artifact_digest":
        receipt["evidence_binding_artifact_digest"] = _digest("f")
    else:
        for subject in receipt["expected_subjects"]:
            subject["subject_id"] = f"drifted-{subject['subject_id']}"
    _redigest(payload)
    state_path = tmp_path / f"drifted-{drift_kind}-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="exact materialization attempt"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)
