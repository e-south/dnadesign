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
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import policy

CONFIG = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/"
    "config/observation_policy.yaml"
)


def test_repository_policy_is_explicitly_blocked_pending_repeat_review() -> None:
    result = policy.load_response_window_observation_policy(CONFIG)

    assert result.approval_status == "review_required"
    assert result.approved_by is None
    assert result.approved_at is None
    expected_reader_sha256 = (
        "f49ef80d12b76d4e92cb73feaf5423187d898de8c1ee7df0099c9fe0c3fc497d"  # pragma: allowlist secret
    )
    expected_bindings_sha256 = (
        "273dbe8ba0b97bef3ae6318e2c7b5b95d4174b79688633fcdf06bf25c38d32f8"  # pragma: allowlist secret
    )
    assert result.reader_bundle_sha256 == expected_reader_sha256
    assert result.candidate_bindings_sha256 == expected_bindings_sha256
    assert len(result.repeat_decisions) == 12
    assert set(result.repeat_decisions["status"]) == {"review_required"}
    assert set(result.repeat_decisions["classification"]) == {"unresolved"}
    assert result.aggregation.primary_reduction_id == "event_logmean_6_12h_post"
    assert result.aggregation.bootstrap_samples == 2000
    assert result.value_order == ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


def test_approved_policy_requires_named_and_timestamped_signoff(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["approval"]["status"] = "approved"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="approved_by and approved_at"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_approved_policy_rejects_malformed_approval_timestamp(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
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


def test_policy_rejects_semantic_drift_in_experiment_weighting(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["aggregation"]["experiment_weighting"] = "well_count"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="aggregation semantics disagree"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_policy_rejects_duplicate_repeat_candidates(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["repeat_decisions"].append(deepcopy(payload["repeat_decisions"][0]))

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="duplicate candidate IDs"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_policy_rejects_unpinned_source_manifests(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["source_manifests"]["reader_bundle_sha256"] = "latest"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="SHA-256 digest"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_final_repeat_decision_requires_typed_evidence(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["repeat_decisions"][0]["status"] = "comparable"
    payload["repeat_decisions"][0]["classification"] = "assay_context_comparable"

    with pytest.raises(policy.ResponseWindowObservationPolicyError, match="requires typed evidence"):
        policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))


def test_final_repeat_decision_verifies_confined_evidence_digest(tmp_path: Path) -> None:
    evidence = tmp_path / "repeat-review.json"
    evidence.write_text('{"decision":"comparable"}\n', encoding="utf-8")
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decision = payload["repeat_decisions"][0]
    decision.update(
        {
            "status": "comparable",
            "classification": "assay_context_comparable",
            "evidence_artifact": evidence.name,
            "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "adjudicated_by": "study-reviewer",
            "adjudicated_at": "2026-07-15T12:00:00+00:00",
        }
    )

    loaded = policy.load_response_window_observation_policy(_write(tmp_path / "policy.yaml", payload))

    assert loaded.repeat_decisions.iloc[0]["status"] == "comparable"


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
