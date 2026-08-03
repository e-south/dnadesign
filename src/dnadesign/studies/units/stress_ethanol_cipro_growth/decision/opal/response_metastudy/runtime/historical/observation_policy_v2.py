"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/historical/observation_policy_v2.py

Project the immutable v2 observation policy used by the frozen behavior replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.contracts import (
    DECISION_COLUMNS,
    VALUE_COLUMNS,
    ResponseWindowAggregationError,
    ResponseWindowAggregationPolicy,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.repeat_adjudication import (
    validate_repeat_adjudications,
)

_SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_observation_policy.v2"
_STUDY_ID = "stress_ethanol_cipro_growth"
_POLICY_ID = "explicit_label_source_response_observations_v1"
_Y_SPACE = "reader_response_window_vector_v1"
_BATCH_ID = "pre_round0_response_corpus_4_8h_v1"
_TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "policy_id",
    "approval",
    "source_manifests",
    "label_identity",
    "aggregation",
    "censoring",
    "unbound_reader_designs",
    "repeat_decisions",
}


class HistoricalObservationPolicyV2Error(ValueError):
    """Raised when frozen observation-policy provenance is missing or changed."""


@dataclass(frozen=True)
class HistoricalObservationPolicyV2:
    config_path: Path
    config_sha256: str
    policy_id: str
    approval_status: str
    approved_by: str
    approved_at: str
    reader_bundle_sha256: str
    candidate_bindings_sha256: str
    aggregation: ResponseWindowAggregationPolicy
    repeat_decisions: pd.DataFrame
    unbound_reader_designs: pd.DataFrame


def load_historical_observation_policy_v2(
    path: Path,
    *,
    expected_sha256: str,
    expected_reader_bundle_sha256: str,
    expected_candidate_bindings_sha256: str,
    expected_approval_sha256: str,
    expected_primary_reduction_id: str,
) -> HistoricalObservationPolicyV2:
    """Verify and project the exact policy snapshot approved for frozen replay."""

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise HistoricalObservationPolicyV2Error(f"historical observation policy is missing: {source_path}")
    raw = source_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise HistoricalObservationPolicyV2Error("historical observation policy digest mismatch.")
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise HistoricalObservationPolicyV2Error(f"historical observation policy is invalid YAML: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise HistoricalObservationPolicyV2Error("historical observation policy fields are incomplete or unexpected.")
    if (
        payload["schema_id"] != _SCHEMA_ID
        or str(payload["schema_version"]) != "2"
        or payload["study_id"] != _STUDY_ID
        or payload["policy_id"] != _POLICY_ID
    ):
        raise HistoricalObservationPolicyV2Error("historical observation policy identity disagrees.")

    approval = _mapping(payload["approval"], "approval")
    if set(approval) != {"status", "approved_by", "approved_at", "rationale"}:
        raise HistoricalObservationPolicyV2Error("historical observation policy approval fields disagree.")
    approval_identity = {key: approval.get(key) for key in ("status", "approved_by", "approved_at")}
    observed_approval_sha256 = hashlib.sha256(
        json.dumps(approval_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if observed_approval_sha256 != expected_approval_sha256:
        raise HistoricalObservationPolicyV2Error("historical observation policy approval identity disagrees.")
    if (
        approval["status"] != "approved"
        or not isinstance(approval["approved_by"], str)
        or not approval["approved_by"].strip()
        or not isinstance(approval["approved_at"], str)
        or not approval["approved_at"].strip()
        or not isinstance(approval["rationale"], str)
        or not approval["rationale"].strip()
    ):
        raise HistoricalObservationPolicyV2Error("historical observation policy approval rationale is empty.")

    sources = _mapping(payload["source_manifests"], "source_manifests")
    expected_sources = {
        "reader_bundle_sha256": expected_reader_bundle_sha256,
        "candidate_bindings_sha256": expected_candidate_bindings_sha256,
    }
    if sources != expected_sources:
        raise HistoricalObservationPolicyV2Error("historical observation policy source identities disagree.")

    label = _mapping(payload["label_identity"], "label_identity")
    expected_label = {
        "y_space": _Y_SPACE,
        "observed_round": 0,
        "batch_id": _BATCH_ID,
        "primary_reduction_id": expected_primary_reduction_id,
        "value_order": list(VALUE_COLUMNS),
    }
    if label != expected_label:
        raise HistoricalObservationPolicyV2Error("historical observation policy label identity disagrees.")

    aggregation = _aggregation(payload["aggregation"], policy_id=_POLICY_ID)
    censoring = _mapping(payload["censoring"], "censoring")
    if censoring != {"primary_value_requirement": "exact", "nonexact_label_action": "exclude_candidate"}:
        raise HistoricalObservationPolicyV2Error("historical observation censoring semantics disagree.")
    repeat_decisions = _repeat_decisions(
        payload["repeat_decisions"],
        evidence_root=source_path.parent,
        reader_bundle_sha256=expected_reader_bundle_sha256,
        primary_reduction_id=expected_primary_reduction_id,
    )
    unbound_reader_designs = _unbound_reader_designs(payload["unbound_reader_designs"])
    return HistoricalObservationPolicyV2(
        config_path=source_path,
        config_sha256=observed_sha256,
        policy_id=_POLICY_ID,
        approval_status=str(approval["status"]),
        approved_by=str(approval["approved_by"]),
        approved_at=str(approval["approved_at"]),
        reader_bundle_sha256=expected_reader_bundle_sha256,
        candidate_bindings_sha256=expected_candidate_bindings_sha256,
        aggregation=aggregation,
        repeat_decisions=repeat_decisions,
        unbound_reader_designs=unbound_reader_designs,
    )


def _aggregation(value: object, *, policy_id: str) -> ResponseWindowAggregationPolicy:
    payload = _mapping(value, "aggregation")
    expected_outer = {
        "experiment_unit": "reader_experiment",
        "label_source_strategy": "explicit_policy_selection",
        "singleton": "identity",
        "repeated": "selected_reader_experiment_identity",
        "event_time_sensitivity": "separate",
    }
    if {key: payload.get(key) for key in expected_outer} != expected_outer or set(payload) != {
        *expected_outer,
        "uncertainty",
    }:
        raise HistoricalObservationPolicyV2Error("historical observation aggregation semantics disagree.")
    uncertainty = _mapping(payload["uncertainty"], "aggregation.uncertainty")
    expected_uncertainty = {
        "method": "selected_reader_joint_bootstrap",
        "experiment_resampling": "none",
        "reader_draw_resampling": "one_joint_draw_per_sample",
        "samples": 2000,
        "confidence_level": 0.90,
        "random_seed": 7,
        "minimum_reader_draws_per_experiment": 100,
    }
    if uncertainty != expected_uncertainty:
        raise HistoricalObservationPolicyV2Error("historical observation uncertainty semantics disagree.")
    return ResponseWindowAggregationPolicy(
        policy_id=policy_id,
        primary_reduction_id="event_logmean_4_8h_post",
        bootstrap_samples=2000,
        confidence_level=0.90,
        random_seed=7,
        minimum_reader_draws_per_experiment=100,
    )


def _repeat_decisions(
    value: object,
    *,
    evidence_root: Path,
    reader_bundle_sha256: str,
    primary_reduction_id: str,
) -> pd.DataFrame:
    if not isinstance(value, list):
        raise HistoricalObservationPolicyV2Error("historical repeat decisions must be a list.")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, f"repeat_decisions[{index}]")
        if set(row) != set(DECISION_COLUMNS):
            raise HistoricalObservationPolicyV2Error("historical repeat-decision fields disagree.")
        rows.append({column: row[column] for column in DECISION_COLUMNS})
    frame = pd.DataFrame.from_records(rows, columns=DECISION_COLUMNS)
    try:
        validate_repeat_adjudications(
            frame,
            evidence_root=evidence_root,
            expected_reader_bundle_sha256=reader_bundle_sha256,
            expected_primary_reduction_id=primary_reduction_id,
        )
    except ResponseWindowAggregationError as exc:
        raise HistoricalObservationPolicyV2Error(str(exc)) from exc
    return frame


def _unbound_reader_designs(value: object) -> pd.DataFrame:
    if not isinstance(value, list):
        raise HistoricalObservationPolicyV2Error("historical unbound Reader designs must be a list.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, f"unbound_reader_designs[{index}]")
        if set(row) != {"design_id", "reason"} or row.get("reason") != "absent_from_study_candidate_bindings":
            raise HistoricalObservationPolicyV2Error("historical unbound Reader design declarations disagree.")
        design_id = row.get("design_id")
        if not isinstance(design_id, str) or not design_id.strip():
            raise HistoricalObservationPolicyV2Error("historical unbound Reader design identity is empty.")
        rows.append({"design_id": design_id, "reason": str(row["reason"])})
    frame = pd.DataFrame.from_records(rows, columns=["design_id", "reason"])
    if frame["design_id"].duplicated().any():
        raise HistoricalObservationPolicyV2Error("historical unbound Reader designs contain duplicates.")
    return frame


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HistoricalObservationPolicyV2Error(f"{context} must be a mapping.")
    return value


__all__ = [
    "HistoricalObservationPolicyV2",
    "HistoricalObservationPolicyV2Error",
    "load_historical_observation_policy_v2",
]
