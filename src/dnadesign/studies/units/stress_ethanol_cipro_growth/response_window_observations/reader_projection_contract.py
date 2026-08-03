"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/reader_projection_contract.py

Define and validate the study projection over canonical Reader records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from dnadesign.studies.core.reader_records import ReaderRecordError

from .display_contract import validate_study_display

READER_EVENT_WINDOW_PROTOCOL_ID = "plate_reader/four_state_event_window"
READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID = "plot:four_state_event_window_diagnostic"
STUDY_PROJECTION_SCHEMA = "stress_ethanol_cipro_growth.reader_response_projection.v5"
STATE_ORDER = ("00", "10", "01", "11")
READER_EVENT_WINDOW_RECORD_CONTRACTS = {
    "wells": ("four_state_event_window/wells", "plate_reader.four_state_event_window.wells.v3"),
    "designs": ("four_state_event_window/designs", "plate_reader.four_state_event_window.designs.v4"),
    "descriptive_resampling_draws": (
        "four_state_event_window/descriptive_resampling_draws",
        "plate_reader.four_state_event_window.descriptive_resampling_draws.v3",
    ),
    "traces": ("four_state_event_window/traces", "plate_reader.four_state_event_window.traces.v3"),
    "events": ("four_state_event_window/events", "plate_reader.four_state_event_window.events.v2"),
}

ReaderResponseProjectionError = ReaderRecordError

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "study_id",
    "projection_id",
    "reader_experiment_id",
    "source_experiment_ids",
    "reference_design_id",
    "primary_reduction_id",
    "state_order",
    "source",
    "event",
    "aggregation",
    "reductions",
    "records",
    "display",
    "display_artifact",
}
_SOURCE_FIELDS = {
    "response_channel",
    "magnitude_channel",
    "growth_channel",
    "reference_design_id",
    "state_column",
    "state_values",
    "state_values_case_sensitive",
}
_EVENT_FIELDS = {
    "event_id",
    "event_kind",
    "segment_column",
    "pre_segment_index",
    "post_segment_index",
    "estimate_method",
    "declaration",
}
_AGGREGATION_FIELDS = {
    "observation_stat",
    "descriptive_resampling_draws",
    "descriptive_interval_mass",
    "random_seed",
    "quality",
}
_QUALITY_FIELDS = {
    "positive_floor",
    "allowed_max_interior_gap_h",
    "required_min_observations_per_state",
}
_REDUCTION_FIELDS = {
    "id",
    "method",
    "response_basis",
    "role",
    "window_start_event_h",
    "window_end_event_h",
    "pre_window_duration_h",
}
_DISPLAY_ARTIFACT_FIELDS = {"record_id", "source_experiment_id", "design_id", "producer_config_digest", "path"}


def validate_projection_payload(payload: object) -> None:
    """Validate exact projection fields and scientific identities."""

    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ReaderResponseProjectionError(
            f"study Reader projection fields must be exactly {sorted(_TOP_LEVEL_FIELDS)}"
        )
    if payload["schema_version"] != STUDY_PROJECTION_SCHEMA:
        raise ReaderResponseProjectionError(f"study Reader projection must use {STUDY_PROJECTION_SCHEMA!r}")
    if payload["study_id"] != "stress_ethanol_cipro_growth":
        raise ReaderResponseProjectionError("study Reader projection identity disagrees")
    for field_name in ("projection_id", "reader_experiment_id", "reference_design_id", "primary_reduction_id"):
        _text(payload[field_name], label=f"projection.{field_name}")
    _validate_source_experiments(payload["source_experiment_ids"])
    state_order = payload["state_order"]
    if (
        not isinstance(state_order, Sequence)
        or isinstance(state_order, (str, bytes))
        or tuple(state_order) != STATE_ORDER
    ):
        raise ReaderResponseProjectionError(f"study Reader projection state order must be {STATE_ORDER}")
    _validate_source(payload["source"], reference_design_id=str(payload["reference_design_id"]))
    _validate_event(payload["event"])
    _validate_aggregation(payload["aggregation"])
    _validate_reductions(payload["reductions"], primary_reduction_id=str(payload["primary_reduction_id"]))
    expected_records = {
        name: {"record_id": record_id, "contract_id": contract_id}
        for name, (record_id, contract_id) in READER_EVENT_WINDOW_RECORD_CONTRACTS.items()
    }
    if payload["records"] != expected_records:
        raise ReaderResponseProjectionError("study Reader projection record contracts disagree with Reader")
    validate_study_display(payload["display"])
    if _display_reference_design(payload["display"]) != payload["reference_design_id"]:
        raise ReaderResponseProjectionError("projection display reference disagrees with reference_design_id")
    if payload["display_artifact"] is not None:
        validate_display_artifact_spec(payload["display_artifact"])


def validate_display_artifact_spec(value: object) -> dict[str, str]:
    """Validate an optional Reader diagnostic pin without resolving it."""

    item = _exact_mapping(value, fields=_DISPLAY_ARTIFACT_FIELDS, label="projection.display_artifact")
    result = {field: _text(item[field], label=f"projection.display_artifact.{field}") for field in item}
    if result["record_id"] != READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID:
        raise ReaderResponseProjectionError(
            f"projection.display_artifact.record_id must be {READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID!r}"
        )
    _sha256_digest(result["producer_config_digest"], label="projection.display_artifact.producer_config_digest")
    path = Path(result["path"])
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in {".png", ".pdf"}:
        raise ReaderResponseProjectionError(
            "projection.display_artifact.path must be a confined outputs-relative PNG or PDF path"
        )
    return result


def _validate_source_experiments(value: object) -> None:
    if not isinstance(value, list) or len(value) != 8:
        raise ReaderResponseProjectionError("projection.source_experiment_ids must contain exactly eight entries")
    values = [_text(item, label="projection.source_experiment_ids[]") for item in value]
    if len(values) != len(set(values)):
        raise ReaderResponseProjectionError("projection.source_experiment_ids contains duplicates")


def _validate_source(value: object, *, reference_design_id: str) -> None:
    source = _exact_mapping(value, fields=_SOURCE_FIELDS, label="projection.source")
    for field in (
        "response_channel",
        "magnitude_channel",
        "growth_channel",
        "reference_design_id",
        "state_column",
    ):
        _text(source[field], label=f"projection.source.{field}")
    if source["reference_design_id"] != reference_design_id:
        raise ReaderResponseProjectionError("projection source reference disagrees with reference_design_id")
    state_values = _exact_mapping(
        source["state_values"],
        fields=set(STATE_ORDER),
        label="projection.source.state_values",
    )
    values = [_text(state_values[state], label=f"projection.source.state_values.{state}") for state in STATE_ORDER]
    if len(set(values)) != len(values):
        raise ReaderResponseProjectionError("projection source state values must be distinct")
    if not isinstance(source["state_values_case_sensitive"], bool):
        raise ReaderResponseProjectionError("projection.source.state_values_case_sensitive must be boolean")


def _validate_event(value: object) -> None:
    event = _exact_mapping(value, fields=_EVENT_FIELDS, label="projection.event")
    for field in _EVENT_FIELDS - {"pre_segment_index", "post_segment_index"}:
        _text(event[field], label=f"projection.event.{field}")
    pre = event["pre_segment_index"]
    post = event["post_segment_index"]
    if type(pre) is not int or type(post) is not int or pre < 0 or post <= pre:
        raise ReaderResponseProjectionError("projection event segment indexes must be ordered nonnegative integers")


def _validate_aggregation(value: object) -> None:
    aggregation = _exact_mapping(value, fields=_AGGREGATION_FIELDS, label="projection.aggregation")
    _text(aggregation["observation_stat"], label="projection.aggregation.observation_stat")
    draws = aggregation["descriptive_resampling_draws"]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ReaderResponseProjectionError("projection.aggregation.descriptive_resampling_draws must be positive")
    random_seed = aggregation["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ReaderResponseProjectionError("projection.aggregation.random_seed must be a nonnegative integer")
    mass = _number(aggregation["descriptive_interval_mass"], label="projection.aggregation.descriptive_interval_mass")
    if not 0.0 < mass <= 1.0:
        raise ReaderResponseProjectionError("projection.aggregation.descriptive_interval_mass must be in (0, 1]")
    quality = _exact_mapping(
        aggregation["quality"],
        fields=_QUALITY_FIELDS,
        label="projection.aggregation.quality",
    )
    for field in ("positive_floor", "allowed_max_interior_gap_h"):
        if _number(quality[field], label=f"projection.aggregation.quality.{field}") <= 0.0:
            raise ReaderResponseProjectionError(f"projection.aggregation.quality.{field} must be positive")
    observations = quality["required_min_observations_per_state"]
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 1:
        raise ReaderResponseProjectionError(
            "projection.aggregation.quality.required_min_observations_per_state must be positive"
        )


def _validate_reductions(value: object, *, primary_reduction_id: str) -> None:
    if not isinstance(value, list) or not value:
        raise ReaderResponseProjectionError("projection.reductions must be a non-empty list")
    ids: list[str] = []
    primary_ids: list[str] = []
    for index, item in enumerate(value):
        reduction = _exact_mapping(item, fields=_REDUCTION_FIELDS, label=f"projection.reductions[{index}]")
        reduction_id = _text(reduction["id"], label=f"projection.reductions[{index}].id")
        ids.append(reduction_id)
        for field in ("method", "response_basis", "role"):
            _text(reduction[field], label=f"projection.reductions[{index}].{field}")
        start = _number(reduction["window_start_event_h"], label=f"projection.reductions[{index}].window_start_event_h")
        end = _number(reduction["window_end_event_h"], label=f"projection.reductions[{index}].window_end_event_h")
        if end <= start:
            raise ReaderResponseProjectionError(f"projection.reductions[{index}] must have a positive window")
        pre_window = reduction["pre_window_duration_h"]
        if reduction["response_basis"] == "post_minus_pre":
            if (
                pre_window is None
                or _number(
                    pre_window,
                    label=f"projection.reductions[{index}].pre_window_duration_h",
                )
                <= 0.0
            ):
                raise ReaderResponseProjectionError(
                    f"projection.reductions[{index}] post_minus_pre requires a positive pre-window"
                )
        elif pre_window is not None:
            raise ReaderResponseProjectionError(
                f"projection.reductions[{index}] pre-window is valid only for post_minus_pre"
            )
        if reduction["role"] == "primary":
            primary_ids.append(reduction_id)
    if len(ids) != len(set(ids)):
        raise ReaderResponseProjectionError("projection.reductions contains duplicate ids")
    if primary_ids != [primary_reduction_id]:
        raise ReaderResponseProjectionError("projection primary reduction identity disagrees with reductions")


def _display_reference_design(value: object) -> str:
    display = _mapping(value, label="projection.display")
    channels = _mapping(display.get("channels"), label="projection.display.channels")
    return _text(channels.get("reference_design_id"), label="projection.display.channels.reference_design_id")


def _exact_mapping(value: object, *, fields: set[str], label: str) -> Mapping[str, object]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        raise ReaderResponseProjectionError(f"{label} fields must be exactly {sorted(fields)}")
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderResponseProjectionError(f"{label} must be an object")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderResponseProjectionError(f"{label} must be a non-empty string")
    return value.strip()


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReaderResponseProjectionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReaderResponseProjectionError(f"{label} must be finite")
    return result


def _sha256_digest(value: object, *, label: str) -> str:
    token = _text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderResponseProjectionError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderResponseProjectionError(f"{label} must be a lowercase sha256 digest")
    return token


__all__ = [
    "READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID",
    "READER_EVENT_WINDOW_PROTOCOL_ID",
    "READER_EVENT_WINDOW_RECORD_CONTRACTS",
    "STATE_ORDER",
    "STUDY_PROJECTION_SCHEMA",
    "ReaderResponseProjectionError",
    "validate_display_artifact_spec",
    "validate_projection_payload",
]
