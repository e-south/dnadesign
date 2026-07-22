"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/repeat_evidence.py

Typed evidence contract for repeated Reader experiment adjudication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifact_io import read_json_object
from .contracts import REPEAT_CLASSIFICATIONS, REPEAT_STATUSES, VALUE_COLUMNS

SCHEMA_ID = "stress_ethanol_cipro_growth.repeat_adjudication_evidence.v1"
SCHEMA_VERSION = "1"
STUDY_ID = "stress_ethanol_cipro_growth"

_TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "reader_bundle_sha256",
    "primary_reduction_id",
    "candidate_reviews",
}
_REVIEW_FIELDS = {
    "candidate_id",
    "reader_experiment_ids",
    "label_source_reader_experiment_id",
    "status",
    "classification",
    "comparison_evidence",
}
_COMPARISON_FIELDS = {
    "component_ranges",
    "maximum_component_range",
    "maximum_range_components",
}


class RepeatEvidenceContractError(ValueError):
    """Raised when repeat-review evidence is malformed or disagrees with its decision."""


def validate_repeat_evidence_artifact(
    path: Path,
    *,
    expected_reader_bundle_sha256: str,
    expected_primary_reduction_id: str,
    candidate_id: str,
    reader_experiment_ids: Sequence[str],
    label_source_reader_experiment_id: str | None,
    status: str,
    classification: str,
) -> None:
    """Verify one candidate decision against a typed, source-bound evidence artifact."""

    source = Path(path)
    try:
        payload = read_json_object(source, label="repeat-adjudication evidence")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepeatEvidenceContractError(f"could not parse repeat-adjudication evidence: {exc}") from exc
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, label="repeat-adjudication evidence")
    if payload["schema_id"] != SCHEMA_ID or str(payload["schema_version"]) != SCHEMA_VERSION:
        raise RepeatEvidenceContractError("repeat-adjudication evidence schema identity disagrees.")
    if payload["study_id"] != STUDY_ID:
        raise RepeatEvidenceContractError("repeat-adjudication evidence study identity disagrees.")
    reader_digest = _sha256(payload["reader_bundle_sha256"], field="reader_bundle_sha256")
    if reader_digest != expected_reader_bundle_sha256:
        raise RepeatEvidenceContractError("repeat-adjudication evidence Reader bundle digest disagrees.")
    primary_reduction_id = _text(payload["primary_reduction_id"], field="primary_reduction_id")
    if primary_reduction_id != expected_primary_reduction_id:
        raise RepeatEvidenceContractError("repeat-adjudication evidence primary reduction disagrees.")

    reviews = _candidate_reviews(payload["candidate_reviews"])
    review = reviews.get(candidate_id)
    if review is None:
        raise RepeatEvidenceContractError(f"repeat-adjudication evidence has no entry for candidate {candidate_id!r}.")
    observed_experiments = tuple(sorted(review["reader_experiment_ids"]))
    expected_experiments = tuple(sorted(str(value) for value in reader_experiment_ids))
    if observed_experiments != expected_experiments:
        raise RepeatEvidenceContractError(
            f"{candidate_id}: repeat-adjudication evidence experiment identities disagree."
        )
    if review["label_source_reader_experiment_id"] != label_source_reader_experiment_id:
        raise RepeatEvidenceContractError(f"{candidate_id}: repeat-adjudication evidence label source disagrees.")
    if review["status"] != status:
        raise RepeatEvidenceContractError(f"{candidate_id}: repeat-adjudication evidence status disagrees.")
    if review["classification"] != classification:
        raise RepeatEvidenceContractError(f"{candidate_id}: repeat-adjudication evidence classification disagrees.")


def _candidate_reviews(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise RepeatEvidenceContractError("repeat-adjudication candidate_reviews must be a non-empty list.")
    result: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(value):
        _require_exact_fields(raw, _REVIEW_FIELDS, label=f"candidate_reviews[{index}]")
        review = raw
        candidate_id = _text(review["candidate_id"], field=f"candidate_reviews[{index}].candidate_id")
        if candidate_id in result:
            raise RepeatEvidenceContractError("repeat-adjudication evidence contains duplicate candidate IDs.")
        experiments = _text_list(
            review["reader_experiment_ids"],
            field=f"candidate_reviews[{index}].reader_experiment_ids",
            minimum_count=2,
        )
        label_source = _optional_text(
            review["label_source_reader_experiment_id"],
            field=f"candidate_reviews[{index}].label_source_reader_experiment_id",
        )
        if label_source is not None and label_source not in experiments:
            raise RepeatEvidenceContractError(
                f"{candidate_id}: repeat-adjudication evidence label source is not a declared experiment."
            )
        status = _text(review["status"], field=f"candidate_reviews[{index}].status")
        if status not in REPEAT_STATUSES:
            raise RepeatEvidenceContractError(
                f"{candidate_id}: repeat-adjudication evidence status is unsupported: {status!r}."
            )
        classification = _text(
            review["classification"],
            field=f"candidate_reviews[{index}].classification",
        )
        if classification not in REPEAT_CLASSIFICATIONS:
            raise RepeatEvidenceContractError(
                f"{candidate_id}: repeat-adjudication evidence classification is unsupported: {classification!r}."
            )
        if status == "label_source_selected" and label_source is None:
            raise RepeatEvidenceContractError(
                f"{candidate_id}: selected repeat evidence requires a Reader experiment label source."
            )
        if status != "label_source_selected" and label_source is not None:
            raise RepeatEvidenceContractError(
                f"{candidate_id}: repeat evidence without selected status cannot name a label source."
            )
        _validate_comparison_evidence(
            review["comparison_evidence"],
            candidate_id=candidate_id,
        )
        result[candidate_id] = {
            "reader_experiment_ids": experiments,
            "label_source_reader_experiment_id": label_source,
            "status": status,
            "classification": classification,
        }
    return result


def _validate_comparison_evidence(value: object, *, candidate_id: str) -> None:
    _require_exact_fields(value, _COMPARISON_FIELDS, label=f"{candidate_id}.comparison_evidence")
    comparison = value
    raw_ranges = comparison["component_ranges"]
    if not isinstance(raw_ranges, Mapping) or set(raw_ranges) != set(VALUE_COLUMNS):
        raise RepeatEvidenceContractError(
            f"{candidate_id}: comparison component ranges must be exactly {list(VALUE_COLUMNS)}."
        )
    ranges = {
        component: _nonnegative_float(raw_ranges[component], field=f"{candidate_id}.component_ranges.{component}")
        for component in VALUE_COLUMNS
    }
    observed_maximum = _nonnegative_float(
        comparison["maximum_component_range"],
        field=f"{candidate_id}.maximum_component_range",
    )
    expected_maximum = max(ranges.values())
    if not math.isclose(observed_maximum, expected_maximum, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RepeatEvidenceContractError(f"{candidate_id}: maximum component range disagrees with component ranges.")
    observed_components = _text_list(
        comparison["maximum_range_components"],
        field=f"{candidate_id}.maximum_range_components",
        minimum_count=1,
    )
    expected_components = sorted(
        component
        for component, component_range in ranges.items()
        if math.isclose(component_range, expected_maximum, rel_tol=1.0e-12, abs_tol=1.0e-12)
    )
    if observed_components != expected_components:
        raise RepeatEvidenceContractError(f"{candidate_id}: maximum-range components disagree with component ranges.")


def _require_exact_fields(value: object, fields: set[str], *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise RepeatEvidenceContractError(f"{label} fields must be exactly {sorted(fields)}.")


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepeatEvidenceContractError(f"{field} must be non-empty text.")
    return value.strip()


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _text_list(value: object, *, field: str, minimum_count: int) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum_count:
        raise RepeatEvidenceContractError(f"{field} must contain at least {minimum_count} values.")
    result = [_text(item, field=field) for item in value]
    if len(result) != len(set(result)):
        raise RepeatEvidenceContractError(f"{field} contains duplicate values.")
    return result


def _sha256(value: object, *, field: str) -> str:
    digest = _text(value, field=field)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RepeatEvidenceContractError(f"{field} must be a lowercase SHA-256 digest.")
    return digest


def _nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepeatEvidenceContractError(f"{field} must be finite and nonnegative.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RepeatEvidenceContractError(f"{field} must be finite and nonnegative.")
    return result


__all__ = [
    "SCHEMA_ID",
    "RepeatEvidenceContractError",
    "validate_repeat_evidence_artifact",
]
