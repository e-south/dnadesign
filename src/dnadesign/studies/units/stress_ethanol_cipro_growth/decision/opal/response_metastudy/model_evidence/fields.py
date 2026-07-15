"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/fields.py

Typed field extraction for model-evidence manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .contracts import ModelEvidenceError

MODEL_FIELDS = (
    "model_id",
    "model_role",
    "representation_id",
    "target_transform",
    "validation",
    "metric_scope",
    "weakest_target_view_response_separation_spearman",
    "weakest_target_view_feasibility_spearman",
    "weakest_required_ordering_spearman",
    "median_channel_spearman",
    "minimum_channel_spearman",
    "response_magnitude_mae",
    "minimum_defined_group_count",
    "target_view_ordering",
)


def required_mapping(source: dict[str, object], field: str) -> dict[str, object]:
    value = source.get(field)
    if not isinstance(value, dict):
        raise ModelEvidenceError(f"{field} must be a mapping.")
    return value


def required_string(source: dict[str, object], field: str) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value:
        raise ModelEvidenceError(f"{field} must be a non-empty string.")
    return value


def enum_string(source: dict[str, object], field: str, allowed: set[str]) -> str:
    value = required_string(source, field)
    if value not in allowed:
        raise ModelEvidenceError(f"{field} must be one of {sorted(allowed)}; found {value!r}.")
    return value


def required_bool(source: dict[str, object], field: str) -> bool:
    value = source.get(field)
    if not isinstance(value, bool):
        raise ModelEvidenceError(f"{field} must be a boolean.")
    return value


def nonnegative_integer(source: dict[str, object], field: str) -> int:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelEvidenceError(f"{field} must be a non-negative integer.")
    return value


def required_number(source: dict[str, object], field: str) -> float:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelEvidenceError(f"{field} must be numeric.")
    return float(value)


def sha256_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ModelEvidenceError(f"{label} must be a lowercase sha256 digest.")
    return value


def model_record(screen: dict[str, object], field: str, *, expected_role: str) -> dict[str, object]:
    source = required_mapping(screen, field)
    missing = sorted(set(MODEL_FIELDS) - set(source))
    if missing:
        raise ModelEvidenceError(f"response_metric_screen.{field} missing fields: {missing}.")
    if source["model_role"] != expected_role:
        raise ModelEvidenceError(
            f"response_metric_screen.{field}.model_role must be {expected_role!r}; found {source['model_role']!r}."
        )
    record = {key: source[key] for key in MODEL_FIELDS}
    if expected_role == "campaign_model":
        record["configured_model_params"] = dict(required_mapping(source, "configured_model_params"))
    return record


def model_screen_records(screen: dict[str, object]) -> list[dict[str, object]]:
    records = screen.get("prespecified_model_screens")
    if not isinstance(records, list) or not records:
        raise ModelEvidenceError("response_metric_screen.prespecified_model_screens must be non-empty.")
    normalized: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ModelEvidenceError(f"prespecified_model_screens[{index}] must be a mapping.")
        missing = sorted(set(MODEL_FIELDS) - set(record))
        if missing:
            raise ModelEvidenceError(f"prespecified_model_screens[{index}] missing fields: {missing}.")
        normalized.append(dict(record))
    return normalized


def fixed_model_definitions(screen: dict[str, object]) -> list[dict[str, object]]:
    definitions = screen.get("fixed_model_definitions")
    if not isinstance(definitions, list) or not definitions:
        raise ModelEvidenceError("response_metric_screen.fixed_model_definitions must be non-empty.")
    if any(not isinstance(row, dict) or not row.get("model_id") for row in definitions):
        raise ModelEvidenceError("fixed_model_definitions rows must be mappings with model_id.")
    model_ids = [str(row["model_id"]) for row in definitions]
    if len(model_ids) != len(set(model_ids)):
        raise ModelEvidenceError("fixed_model_definitions contains duplicate model IDs.")
    return [dict(row) for row in definitions]


__all__ = [
    "enum_string",
    "fixed_model_definitions",
    "model_record",
    "model_screen_records",
    "nonnegative_integer",
    "required_bool",
    "required_mapping",
    "required_number",
    "required_string",
    "sha256_digest",
]
