"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/operator/test_state.py

Owner-aligned operator contract tests.

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

from ._support import (
    _digest,
    _state_for_external_registry,
)


def test_state_validation_rejects_canonical_shaped_digest_for_wrong_route_registry(
    tmp_path: Path,
) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["readiness"]["source_identity"]["route_registry_digest"] = _digest("a")
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
    state_path = tmp_path / "forged-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="route registry digest changed"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_accepts_exact_external_route_registry(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "exact-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert operator.validate_source_controlled_state(state_path, phd_root=phd_root) == payload


def test_state_validation_rejects_incomplete_sensitivity_coverage_receipt(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["profile_count"] -= 1
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
    state_path = tmp_path / "incomplete-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="coordinate counts changed"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_sensitivity_receipt_attempt_drift(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["sensitivity_coverage_receipts"][0]["materialization_attempt_digest"] = _digest("f")
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
    state_path = tmp_path / "drifted-coverage-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="exact materialization attempt"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_fails_closed_without_external_route_registry(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    registry = phd_root / ".agents/skills/retron-assay-study-bridge/references/reader-experiment-routes.json"
    registry.unlink()

    with pytest.raises(MetastudyContractError, match="does not contain the canonical route registry"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_float_experiment_count(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    selected_count = payload["readiness"]["selected_experiment_count"]
    payload["readiness"]["selected_experiment_count"] = float(selected_count)
    payload["decision"]["readiness"]["selected_experiment_count"] = float(selected_count)
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
    state_path = tmp_path / "float-count-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="selected experiment count"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_redigested_noncanonical_decision_limitations(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    payload["decision"]["limitations"] = ["duplicate", "duplicate"]
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
    state_path = tmp_path / "noncanonical-limitations-state.yaml"
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(MetastudyContractError, match="must not contain duplicates"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)


def test_state_validation_rejects_duplicate_yaml_key(tmp_path: Path) -> None:
    phd_root = tmp_path / "phd"
    payload = _state_for_external_registry(phd_root)
    state_path = tmp_path / "duplicate-key-state.yaml"
    state_path.write_text(
        "schema_id: shadowed-value\n" + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(MetastudyContractError, match="duplicate YAML key"):
        operator.validate_source_controlled_state(state_path, phd_root=phd_root)
