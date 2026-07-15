"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/display_contract.py

Independent validation of Reader bundle display semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

READER_DISPLAY_SCHEMA = "reader.response_window.display.v1"
STATE_ORDER = ("00", "10", "01", "11")


def validate_reader_display(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Reader response-window bundle lacks display semantics.")
    required = {"schema_version", "study_label", "event_label", "state_labels", "channels", "examples"}
    if set(value) != required or value.get("schema_version") != READER_DISPLAY_SCHEMA:
        raise ValueError("Reader response-window display contract disagrees.")
    state_labels = value.get("state_labels")
    channels = value.get("channels")
    examples = value.get("examples")
    if not isinstance(state_labels, dict) or set(state_labels) != set(STATE_ORDER):
        raise ValueError("Reader display must label every response state exactly once.")
    channel_fields = {"response_ratio", "magnitude_ratio", "growth", "reference_design_id"}
    if not isinstance(channels, dict) or set(channels) != channel_fields:
        raise ValueError("Reader display channel contract disagrees.")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Reader display examples must be non-empty.")
    if any(not isinstance(row, dict) or set(row) != {"design_id", "label", "role"} for row in examples):
        raise ValueError("Reader display example contract disagrees.")
    anchors = [row for row in examples if row["role"] == "reference_anchor"]
    if len(anchors) != 1 or anchors[0]["design_id"] != channels["reference_design_id"]:
        raise ValueError("Reader display reference anchor disagrees with its channel contract.")
    if not any(row["role"] == "response_example" for row in examples):
        raise ValueError("Reader display must declare at least one response example.")
    return value


def response_example_labels(value: object) -> dict[str, str]:
    display = validate_reader_display(value)
    examples = display["examples"]
    if not isinstance(examples, list):
        raise ValueError("Reader display examples must be a list.")
    return {
        str(row["design_id"]): str(row["label"])
        for row in examples
        if isinstance(row, dict) and row.get("role") == "response_example"
    }


__all__ = ["READER_DISPLAY_SCHEMA", "response_example_labels", "validate_reader_display"]
