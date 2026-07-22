"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/artifact_manifest.py

Manifest construction and identity validation for observation bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_contract import (
    RECORD_FILES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    STUDY_ID,
    ResponseWindowObservationArtifactError,
)
from .artifact_io import file_sha256
from .contracts import VALUE_COLUMNS
from .sources import ResponseWindowObservationEvidence

_MANIFEST_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "created_at",
    "policy",
    "source_manifests",
    "observation_contract",
    "records",
}


def build_manifest(
    evidence: ResponseWindowObservationEvidence,
    *,
    staged: Path,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "policy": {
            "policy_id": evidence.policy.policy_id,
            "config_sha256": evidence.policy.config_sha256,
            "approved_by": evidence.policy.approved_by,
            "approved_at": evidence.policy.approved_at,
        },
        "source_manifests": {
            "reader_bundle": {"sha256": evidence.reader_manifest_sha256},
            "candidate_bindings": {"sha256": evidence.candidate_bindings_manifest_sha256},
        },
        "observation_contract": {
            "y_space": evidence.policy.y_space,
            "value_order": list(evidence.policy.value_order),
            "primary_reduction_id": evidence.policy.aggregation.primary_reduction_id,
            "observed_round": evidence.policy.observed_round,
            "batch_id": evidence.policy.batch_id,
            "experiment_unit": "reader_experiment",
            "label_source_strategy": "explicit_policy_selection",
            "singleton_point_estimate": "identity",
            "repeated_point_estimate": "selected_reader_experiment_identity",
            "primary_value_requirement": evidence.policy.primary_value_requirement,
            "nonexact_label_action": evidence.policy.nonexact_label_action,
            "uncertainty_method": "selected_reader_joint_bootstrap",
            "uncertainty_scope": "descriptive_not_population_coverage",
            "event_time_sensitivity": "separate",
            "bootstrap_samples": evidence.policy.aggregation.bootstrap_samples,
            "candidate_count": len(evidence.preview.observations),
        },
        "records": {
            record_id: {
                "path": filename,
                "sha256": file_sha256(staged / filename),
                "row_count": len(frames[record_id]),
                "columns": frames[record_id].columns.tolist(),
            }
            for record_id, filename in RECORD_FILES.items()
        },
    }


def validate_manifest_identity(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ResponseWindowObservationArtifactError("observation manifest fields disagree with the v2 contract.")
    if (
        payload["schema_id"] != SCHEMA_ID
        or str(payload["schema_version"]) != SCHEMA_VERSION
        or payload["study_id"] != STUDY_ID
    ):
        raise ResponseWindowObservationArtifactError("observation manifest identity disagrees.")
    _timestamp(payload["created_at"])
    policy = payload["policy"]
    if not isinstance(policy, dict) or set(policy) != {"policy_id", "config_sha256", "approved_by", "approved_at"}:
        raise ResponseWindowObservationArtifactError("observation policy provenance is malformed.")
    for field in ("policy_id", "approved_by", "approved_at"):
        if not isinstance(policy[field], str) or not policy[field].strip():
            raise ResponseWindowObservationArtifactError(f"observation policy {field} is empty.")
    if not is_sha256(policy["config_sha256"]):
        raise ResponseWindowObservationArtifactError("observation policy config digest is invalid.")
    _validate_sources(payload["source_manifests"])
    _validate_observation_contract(payload["observation_contract"])
    if not isinstance(payload["records"], dict) or set(payload["records"]) != set(RECORD_FILES):
        raise ResponseWindowObservationArtifactError("observation record inventory is incomplete.")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_sources(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"reader_bundle", "candidate_bindings"}:
        raise ResponseWindowObservationArtifactError("observation source manifests are malformed.")
    if any(
        not isinstance(item, dict) or set(item) != {"sha256"} or not is_sha256(item["sha256"])
        for item in value.values()
    ):
        raise ResponseWindowObservationArtifactError("observation source-manifest digest is invalid.")


def _validate_observation_contract(value: object) -> None:
    expected = {
        "y_space",
        "value_order",
        "primary_reduction_id",
        "observed_round",
        "batch_id",
        "experiment_unit",
        "label_source_strategy",
        "singleton_point_estimate",
        "repeated_point_estimate",
        "primary_value_requirement",
        "nonexact_label_action",
        "uncertainty_method",
        "uncertainty_scope",
        "event_time_sensitivity",
        "bootstrap_samples",
        "candidate_count",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ResponseWindowObservationArtifactError("observation contract is malformed.")
    if (
        value["value_order"] != list(VALUE_COLUMNS)
        or value["experiment_unit"] != "reader_experiment"
        or value["label_source_strategy"] != "explicit_policy_selection"
        or value["singleton_point_estimate"] != "identity"
        or value["repeated_point_estimate"] != "selected_reader_experiment_identity"
        or value["primary_value_requirement"] != "exact"
        or value["nonexact_label_action"] != "exclude_candidate"
        or value["uncertainty_method"] != "selected_reader_joint_bootstrap"
        or value["uncertainty_scope"] != "descriptive_not_population_coverage"
        or value["event_time_sensitivity"] != "separate"
    ):
        raise ResponseWindowObservationArtifactError("observation scientific semantics disagree.")
    if (
        isinstance(value["observed_round"], bool)
        or not isinstance(value["observed_round"], int)
        or value["observed_round"] < 0
        or value["y_space"] != "reader_response_window_vector_v1"
        or value["primary_reduction_id"] != "event_logmean_4_8h_post"
        or not isinstance(value["batch_id"], str)
        or not value["batch_id"].strip()
        or isinstance(value["bootstrap_samples"], bool)
        or not isinstance(value["bootstrap_samples"], int)
        or value["bootstrap_samples"] < 100
        or isinstance(value["candidate_count"], bool)
        or not isinstance(value["candidate_count"], int)
        or value["candidate_count"] < 1
    ):
        raise ResponseWindowObservationArtifactError("observation round or batch identity is invalid.")


def _timestamp(value: object) -> None:
    try:
        created_at = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ResponseWindowObservationArtifactError("observation created_at is invalid.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ResponseWindowObservationArtifactError("observation created_at must be timezone-aware.")


__all__ = ["build_manifest", "is_sha256", "validate_manifest_identity"]
