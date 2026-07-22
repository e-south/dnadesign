"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_policy.py

Tests for the study-owned response-window observation policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import policy

CONFIG = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/"
    "config/observation_policy.yaml"
)


def test_repository_policy_records_approved_explicit_repeat_sources() -> None:
    result = policy.load_response_window_observation_policy(CONFIG)

    assert result.approval_status == "approved"
    assert result.approved_by == "Eric J. South"
    assert result.approved_at == "2026-07-15T19:33:54-04:00"
    expected_reader_sha256 = (
        "e0065183fe010fe40e88e75da998cff5ed1399e56ff1d2bc1268f9331b573a67"  # pragma: allowlist secret
    )
    expected_bindings_sha256 = (
        "273dbe8ba0b97bef3ae6318e2c7b5b95d4174b79688633fcdf06bf25c38d32f8"  # pragma: allowlist secret
    )
    assert result.reader_bundle_sha256 == expected_reader_sha256
    assert result.candidate_bindings_sha256 == expected_bindings_sha256
    assert len(result.repeat_decisions) == 12
    assert result.repeat_decisions["status"].value_counts().to_dict() == {
        "label_source_selected": 8,
        "label_source_excluded": 4,
    }
    assert result.repeat_decisions["label_source_reader_experiment_id"].notna().sum() == 8
    assert set(result.repeat_decisions["evidence_artifact"]) == {"evidence/repeat_adjudication_4_8h_v1.json"}
    assert result.aggregation.primary_reduction_id == "event_logmean_4_8h_post"
    assert result.aggregation.bootstrap_samples == 2000
    assert result.primary_value_requirement == "exact"
    assert result.nonexact_label_action == "exclude_candidate"
    assert result.value_order == ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


def test_approved_policy_requires_named_and_timestamped_signoff(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["approval"]["status"] = "approved"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="approved_by and approved_at"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_approved_policy_rejects_malformed_approval_timestamp(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["approval"] = {
        "status": "approved",
        "approved_by": "study-reviewer",
        "approved_at": "sometime",
        "rationale": "Repeat comparability was reviewed.",
    }

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="timezone-aware ISO-8601"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_policy_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(CONFIG.read_text(encoding="utf-8") + "\nstudy_id: another-study\n", encoding="utf-8")

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="duplicate key"):
        policy.load_response_window_observation_policy(path)


def test_policy_rejects_semantic_drift_in_label_source_strategy(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["aggregation"]["label_source_strategy"] = "latest_slug_wins"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="aggregation semantics disagree"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_policy_rejects_duplicate_repeat_candidates(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["repeat_decisions"].append(deepcopy(payload["repeat_decisions"][0]))

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="duplicate candidate IDs"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_policy_rejects_unpinned_source_manifests(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["source_manifests"]["reader_bundle_sha256"] = "latest"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="SHA-256 digest"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_final_repeat_decision_requires_typed_evidence(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["repeat_decisions"][0]["status"] = "label_source_selected"
    payload["repeat_decisions"][0]["classification"] = "source_agreement_accepted"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="requires typed evidence"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_final_repeat_decision_verifies_confined_evidence_digest(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    decision = payload["repeat_decisions"][0]
    selected_experiment = decision["reader_experiment_ids"][1]
    decision.update(
        {
            "status": "label_source_selected",
            "classification": "source_agreement_accepted",
            "label_source_reader_experiment_id": selected_experiment,
            "adjudicated_by": "study-reviewer",
            "adjudicated_at": "2026-07-15T12:00:00+00:00",
        }
    )
    evidence = tmp_path / "repeat-review.json"
    evidence.write_text(
        json.dumps(_repeat_evidence_payload(payload, decision), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision["evidence_artifact"] = evidence.name
    decision["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()

    loaded = policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))

    assert loaded.repeat_decisions.iloc[0]["status"] == "label_source_selected"
    assert loaded.repeat_decisions.iloc[0]["label_source_reader_experiment_id"] == decision["reader_experiment_ids"][1]


def test_selected_repeat_requires_declared_label_source(tmp_path: Path) -> None:
    evidence = tmp_path / "repeat-review.json"
    evidence.write_text('{"decision":"label_source_selected"}\n', encoding="utf-8")
    payload = _unreviewed_payload()
    decision = payload["repeat_decisions"][0]
    decision.update(
        {
            "status": "label_source_selected",
            "classification": "source_agreement_accepted",
            "evidence_artifact": evidence.name,
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "adjudicated_by": "study-reviewer",
            "adjudicated_at": "2026-07-15T12:00:00+00:00",
        }
    )

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="requires one Reader experiment"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_v1_policy_is_rejected_without_a_compatibility_shim(tmp_path: Path) -> None:
    payload = _unreviewed_payload()
    payload["schema_id"] = "stress_ethanol_cipro_growth.response_window_observation_policy.v1"
    payload["schema_version"] = "1"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="schema identity disagrees"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _unreviewed_payload() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["approval"] = {
        "status": "review_required",
        "approved_by": None,
        "approved_at": None,
        "rationale": "Repeat candidate sources remain under review.",
    }
    for decision in payload["repeat_decisions"]:
        decision.update(
            {
                "label_source_reader_experiment_id": None,
                "status": "review_required",
                "classification": "unresolved",
                "evidence_artifact": None,
                "evidence_sha256": None,
                "adjudicated_by": None,
                "adjudicated_at": None,
                "reason": "cross_experiment_comparability_not_adjudicated",
            }
        )
    return payload


def _repeat_evidence_payload(payload: dict[str, object], decision: dict[str, object]) -> dict[str, object]:
    value_order = payload["label_identity"]["value_order"]
    component_ranges = {component: float(index + 1) / 10.0 for index, component in enumerate(value_order)}
    maximum = max(component_ranges.values())
    return {
        "schema_id": "stress_ethanol_cipro_growth.repeat_adjudication_evidence.v1",
        "schema_version": "1",
        "study_id": "stress_ethanol_cipro_growth",
        "reader_bundle_sha256": payload["source_manifests"]["reader_bundle_sha256"],
        "primary_reduction_id": payload["label_identity"]["primary_reduction_id"],
        "candidate_reviews": [
            {
                "candidate_id": decision["candidate_id"],
                "reader_experiment_ids": decision["reader_experiment_ids"],
                "label_source_reader_experiment_id": decision["label_source_reader_experiment_id"],
                "status": decision["status"],
                "classification": decision["classification"],
                "comparison_evidence": {
                    "component_ranges": component_ranges,
                    "maximum_component_range": maximum,
                    "maximum_range_components": [
                        component for component, value in component_ranges.items() if value == maximum
                    ],
                },
            }
        ],
    }
