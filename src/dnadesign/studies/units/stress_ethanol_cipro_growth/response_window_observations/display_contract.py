"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/display_contract.py

Validation of the study-owned display projection over Reader records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

STUDY_DISPLAY_SCHEMA = "stress_ethanol_cipro_growth.reader_response_display.v1"
STATE_ORDER = ("00", "10", "01", "11")


def validate_study_display(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("study Reader projection lacks display semantics.")
    required = {"schema_version", "study_label", "event_label", "state_labels", "channels", "examples"}
    if set(value) != required or value.get("schema_version") != STUDY_DISPLAY_SCHEMA:
        raise ValueError("study Reader display contract disagrees.")
    state_labels = value.get("state_labels")
    channels = value.get("channels")
    examples = value.get("examples")
    if not isinstance(state_labels, Mapping) or set(state_labels) != set(STATE_ORDER):
        raise ValueError("study display must label every response state exactly once.")
    channel_fields = {"response_ratio", "magnitude_ratio", "growth", "reference_design_id"}
    if not isinstance(channels, Mapping) or set(channels) != channel_fields:
        raise ValueError("study display channel contract disagrees.")
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)) or not examples:
        raise ValueError("study display examples must be non-empty.")
    if any(not isinstance(row, Mapping) or set(row) != {"design_id", "label", "role"} for row in examples):
        raise ValueError("study display example contract disagrees.")
    anchors = [row for row in examples if row["role"] == "reference_anchor"]
    if len(anchors) != 1 or anchors[0]["design_id"] != channels["reference_design_id"]:
        raise ValueError("study display reference anchor disagrees with its channel contract.")
    if not any(row["role"] == "response_example" for row in examples):
        raise ValueError("study display must declare at least one response example.")
    return value


def response_example_labels(value: object) -> dict[str, str]:
    display = validate_study_display(value)
    examples = display["examples"]
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise ValueError("Reader display examples must be a sequence.")
    return {
        str(row["design_id"]): str(row["label"])
        for row in examples
        if isinstance(row, Mapping) and row.get("role") == "response_example"
    }


__all__ = ["STUDY_DISPLAY_SCHEMA", "response_example_labels", "validate_study_display"]
