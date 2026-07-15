"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/policy.py

Parse the checked-in scientific policy for response-window observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from .aggregation import (
    DECISION_COLUMNS,
    REPEAT_STATUSES,
    VALUE_COLUMNS,
    ResponseWindowAggregationPolicy,
)
from .contracts import ResponseWindowAggregationError
from .repeat_adjudication import validate_repeat_adjudications

SCHEMA_ID = "stress_ethanol_cipro_growth.response_window_observation_policy.v1"
SCHEMA_VERSION = "1"
STUDY_ID = "stress_ethanol_cipro_growth"
APPROVAL_STATUSES = frozenset({"review_required", "approved"})

_TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "policy_id",
    "approval",
    "source_manifests",
    "label_identity",
    "aggregation",
    "unbound_reader_designs",
    "repeat_decisions",
}
_APPROVAL_FIELDS = {"status", "approved_by", "approved_at", "rationale"}
_SOURCE_MANIFEST_FIELDS = {
    "reader_bundle_sha256",
    "candidate_bindings_sha256",
}
_LABEL_FIELDS = {
    "y_space",
    "observed_round",
    "batch_id",
    "primary_reduction_id",
    "value_order",
}
_AGGREGATION_FIELDS = {
    "experiment_unit",
    "experiment_weighting",
    "singleton",
    "exactly_two",
    "three_or_more",
    "uncertainty",
    "event_time_sensitivity",
}
_UNCERTAINTY_FIELDS = {
    "method",
    "experiment_resampling",
    "reader_draw_resampling",
    "samples",
    "confidence_level",
    "random_seed",
    "minimum_reader_draws_per_experiment",
}
_EXPECTED_AGGREGATION_SEMANTICS = {
    "experiment_unit": "reader_experiment",
    "experiment_weighting": "equal",
    "singleton": "identity",
    "exactly_two": "midpoint_not_robust",
    "three_or_more": "componentwise_median",
    "event_time_sensitivity": "separate",
}
_EXPECTED_UNCERTAINTY_SEMANTICS = {
    "method": "hierarchical_joint_bootstrap",
    "experiment_resampling": "with_replacement",
    "reader_draw_resampling": "one_joint_draw_per_sampled_experiment",
}


class ResponseWindowObservationPolicyError(ValueError):
    """Raised when the checked-in observation policy is incomplete or ambiguous."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class ResponseWindowObservationPolicy:
    config_path: Path
    config_sha256: str
    policy_id: str
    approval_status: str
    approved_by: str | None
    approved_at: str | None
    approval_rationale: str
    reader_bundle_sha256: str
    candidate_bindings_sha256: str
    y_space: str
    observed_round: int
    batch_id: str
    value_order: tuple[str, ...]
    aggregation: ResponseWindowAggregationPolicy
    repeat_decisions: pd.DataFrame
    unbound_reader_designs: pd.DataFrame


def load_response_window_observation_policy(path: Path) -> ResponseWindowObservationPolicy:
    """Parse one exact policy and reject changes to its scientific semantics."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ResponseWindowObservationPolicyError(f"response-window observation policy not found: {config_path}")
    raw = config_path.read_bytes()
    try:
        payload = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ResponseWindowObservationPolicyError(
            f"could not parse response-window observation policy: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ResponseWindowObservationPolicyError(
            f"response-window observation policy fields must be exactly {sorted(_TOP_LEVEL_FIELDS)}."
        )
    if payload["schema_id"] != SCHEMA_ID or str(payload["schema_version"]) != SCHEMA_VERSION:
        raise ResponseWindowObservationPolicyError("response-window observation policy schema identity disagrees.")
    if payload["study_id"] != STUDY_ID:
        raise ResponseWindowObservationPolicyError("response-window observation policy study identity disagrees.")

    approval = _mapping(payload["approval"], fields=_APPROVAL_FIELDS, label="approval")
    approval_status = _required_text(approval["status"], field="approval.status")
    if approval_status not in APPROVAL_STATUSES:
        raise ResponseWindowObservationPolicyError(f"approval.status must be one of {sorted(APPROVAL_STATUSES)}.")
    approved_by = _optional_text(approval["approved_by"], field="approval.approved_by")
    approved_at = _optional_text(approval["approved_at"], field="approval.approved_at")
    if approved_at is not None:
        _timestamp(approved_at, field="approval.approved_at")
    if approval_status == "approved" and (approved_by is None or approved_at is None):
        raise ResponseWindowObservationPolicyError("approved policy requires approved_by and approved_at.")
    if approval_status == "review_required" and (approved_by is not None or approved_at is not None):
        raise ResponseWindowObservationPolicyError("review_required policy cannot carry approval identity or time.")

    source_manifests = _mapping(
        payload["source_manifests"],
        fields=_SOURCE_MANIFEST_FIELDS,
        label="source_manifests",
    )

    label = _mapping(payload["label_identity"], fields=_LABEL_FIELDS, label="label_identity")
    value_order = tuple(_text_list(label["value_order"], field="label_identity.value_order"))
    if value_order != VALUE_COLUMNS:
        raise ResponseWindowObservationPolicyError(
            f"label value order must be the Reader response-window order {VALUE_COLUMNS}."
        )
    observed_round = _nonnegative_int(label["observed_round"], field="label_identity.observed_round")

    aggregation = _mapping(payload["aggregation"], fields=_AGGREGATION_FIELDS, label="aggregation")
    observed_semantics = {field: aggregation[field] for field in _EXPECTED_AGGREGATION_SEMANTICS}
    if observed_semantics != _EXPECTED_AGGREGATION_SEMANTICS:
        raise ResponseWindowObservationPolicyError(
            "response-window aggregation semantics disagree; experiment weighting, point estimates, and "
            "event-time separation require an explicit schema revision."
        )
    uncertainty = _mapping(
        aggregation["uncertainty"],
        fields=_UNCERTAINTY_FIELDS,
        label="aggregation.uncertainty",
    )
    observed_uncertainty = {field: uncertainty[field] for field in _EXPECTED_UNCERTAINTY_SEMANTICS}
    if observed_uncertainty != _EXPECTED_UNCERTAINTY_SEMANTICS:
        raise ResponseWindowObservationPolicyError(
            "response-window uncertainty semantics disagree; joint hierarchical resampling is required."
        )

    decisions = _repeat_decisions(payload["repeat_decisions"], evidence_root=config_path.parent)
    exclusions = _unbound_reader_designs(payload["unbound_reader_designs"])
    return ResponseWindowObservationPolicy(
        config_path=config_path,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        policy_id=_required_text(payload["policy_id"], field="policy_id"),
        approval_status=approval_status,
        approved_by=approved_by,
        approved_at=approved_at,
        approval_rationale=_required_text(approval["rationale"], field="approval.rationale"),
        reader_bundle_sha256=_required_sha256(
            source_manifests["reader_bundle_sha256"],
            field="source_manifests.reader_bundle_sha256",
        ),
        candidate_bindings_sha256=_required_sha256(
            source_manifests["candidate_bindings_sha256"],
            field="source_manifests.candidate_bindings_sha256",
        ),
        y_space=_required_text(label["y_space"], field="label_identity.y_space"),
        observed_round=observed_round,
        batch_id=_required_text(label["batch_id"], field="label_identity.batch_id"),
        value_order=value_order,
        aggregation=ResponseWindowAggregationPolicy(
            policy_id=_required_text(payload["policy_id"], field="policy_id"),
            primary_reduction_id=_required_text(
                label["primary_reduction_id"], field="label_identity.primary_reduction_id"
            ),
            bootstrap_samples=_positive_int(uncertainty["samples"], field="aggregation.uncertainty.samples"),
            confidence_level=_finite_float(
                uncertainty["confidence_level"], field="aggregation.uncertainty.confidence_level"
            ),
            random_seed=_integer(uncertainty["random_seed"], field="aggregation.uncertainty.random_seed"),
            minimum_reader_draws_per_experiment=_positive_int(
                uncertainty["minimum_reader_draws_per_experiment"],
                field="aggregation.uncertainty.minimum_reader_draws_per_experiment",
            ),
        ),
        repeat_decisions=decisions,
        unbound_reader_designs=exclusions,
    )


def _repeat_decisions(value: object, *, evidence_root: Path) -> pd.DataFrame:
    if not isinstance(value, list):
        raise ResponseWindowObservationPolicyError("repeat_decisions must be a list.")
    rows: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, fields=set(DECISION_COLUMNS), label=f"repeat_decisions[{index}]")
        status = _required_text(row["status"], field=f"repeat_decisions[{index}].status")
        if status not in REPEAT_STATUSES:
            raise ResponseWindowObservationPolicyError(
                f"repeat_decisions[{index}].status must be one of {sorted(REPEAT_STATUSES)}."
            )
        rows.append(
            {
                "candidate_id": _required_text(row["candidate_id"], field=f"repeat_decisions[{index}].candidate_id"),
                "reader_design_ids": _text_list(
                    row["reader_design_ids"], field=f"repeat_decisions[{index}].reader_design_ids"
                ),
                "reader_experiment_ids": _text_list(
                    row["reader_experiment_ids"],
                    field=f"repeat_decisions[{index}].reader_experiment_ids",
                ),
                "status": status,
                "classification": _required_text(
                    row["classification"], field=f"repeat_decisions[{index}].classification"
                ),
                "evidence_artifact": _optional_text(
                    row["evidence_artifact"], field=f"repeat_decisions[{index}].evidence_artifact"
                ),
                "evidence_sha256": _optional_text(
                    row["evidence_sha256"], field=f"repeat_decisions[{index}].evidence_sha256"
                ),
                "adjudicated_by": _optional_text(
                    row["adjudicated_by"], field=f"repeat_decisions[{index}].adjudicated_by"
                ),
                "adjudicated_at": _optional_text(
                    row["adjudicated_at"], field=f"repeat_decisions[{index}].adjudicated_at"
                ),
                "reason": _required_text(row["reason"], field=f"repeat_decisions[{index}].reason"),
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=DECISION_COLUMNS)
    if frame["candidate_id"].duplicated().any():
        raise ResponseWindowObservationPolicyError("repeat decisions contain duplicate candidate IDs.")
    try:
        validate_repeat_adjudications(frame, evidence_root=evidence_root)
    except ResponseWindowAggregationError as exc:
        raise ResponseWindowObservationPolicyError(str(exc)) from exc
    return frame


def _unbound_reader_designs(value: object) -> pd.DataFrame:
    if not isinstance(value, list):
        raise ResponseWindowObservationPolicyError("unbound_reader_designs must be a list.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        row = _mapping(raw, fields={"design_id", "reason"}, label=f"unbound_reader_designs[{index}]")
        reason = _required_text(row["reason"], field=f"unbound_reader_designs[{index}].reason")
        if reason != "absent_from_study_candidate_bindings":
            raise ResponseWindowObservationPolicyError(
                "unbound Reader design reason must be absent_from_study_candidate_bindings."
            )
        rows.append(
            {
                "design_id": _required_text(row["design_id"], field=f"unbound_reader_designs[{index}].design_id"),
                "reason": reason,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=["design_id", "reason"])
    if frame["design_id"].duplicated().any():
        raise ResponseWindowObservationPolicyError("unbound Reader designs contain duplicate IDs.")
    return frame


def _mapping(value: object, *, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ResponseWindowObservationPolicyError(f"{label} fields must be exactly {sorted(fields)}.")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResponseWindowObservationPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _text_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ResponseWindowObservationPolicyError(f"{field} must be a non-empty list.")
    result = [_required_text(item, field=field) for item in value]
    if len(result) != len(set(result)):
        raise ResponseWindowObservationPolicyError(f"{field} contains duplicate values.")
    return result


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResponseWindowObservationPolicyError(f"{field} must be an integer.")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result < 0:
        raise ResponseWindowObservationPolicyError(f"{field} must be nonnegative.")
    return result


def _positive_int(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result < 1:
        raise ResponseWindowObservationPolicyError(f"{field} must be positive.")
    return result


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseWindowObservationPolicyError(f"{field} must be numeric.")
    result = float(value)
    if not pd.notna(result) or result in {float("inf"), float("-inf")}:
        raise ResponseWindowObservationPolicyError(f"{field} must be finite.")
    return result


def _required_sha256(value: object, *, field: str) -> str:
    result = _required_text(value, field=field).lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ResponseWindowObservationPolicyError(f"{field} must be a lowercase SHA-256 digest.")
    return result


def _timestamp(value: str, *, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ResponseWindowObservationPolicyError(f"{field} must be a timezone-aware ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResponseWindowObservationPolicyError(f"{field} must be a timezone-aware ISO-8601 timestamp.")


__all__ = [
    "APPROVAL_STATUSES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "STUDY_ID",
    "ResponseWindowObservationPolicy",
    "ResponseWindowObservationPolicyError",
    "load_response_window_observation_policy",
]
