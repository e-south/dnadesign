"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/measurement_selection.py

Load the response-owned model-screen measurement selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

SCHEMA_ID = "stress_ethanol_cipro_growth.response_measurement_selection.v3"
SCHEMA_VERSION = "3"
_TOP_LEVEL_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "selection_id",
    "scope",
    "label_truth_role",
    "measurements",
    "excluded_designs",
}
_MEASUREMENT_FIELDS = {"reader_experiment_id", "design_id"}
_EXCLUSION_FIELDS = {"design_id", "reason"}
_EXCLUSION_REASONS = {"absent_from_study_candidate_bindings"}


class ResponseMeasurementSelectionError(ValueError):
    """Raised when the response-only screen selection violates its contract."""


@dataclass(frozen=True)
class ResponseMeasurementSelection:
    rows: pd.DataFrame
    excluded_designs: pd.DataFrame
    config_path: Path
    selection_id: str
    scope: str
    label_truth_role: str


def load_response_measurement_selection(
    config_path: Path,
    *,
    reader_designs: pd.DataFrame,
    primary_reduction_id: str,
) -> ResponseMeasurementSelection:
    """Load exact Reader pairs and verify that each exists in the primary reduction."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ResponseMeasurementSelectionError(f"Response measurement selection not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ResponseMeasurementSelectionError(
            f"Could not parse response measurement selection {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ResponseMeasurementSelectionError(
            f"Response measurement selection fields must be exactly {sorted(_TOP_LEVEL_FIELDS)}."
        )
    if payload["schema_id"] != SCHEMA_ID or str(payload["schema_version"]) != SCHEMA_VERSION:
        raise ResponseMeasurementSelectionError("Response measurement selection schema identity mismatch.")
    if payload["study_id"] != "stress_ethanol_cipro_growth":
        raise ResponseMeasurementSelectionError("Response measurement selection study_id mismatch.")
    selection_id = _required_text(payload["selection_id"], field="selection_id")
    if payload["scope"] != "model_screen_only" or payload["label_truth_role"] != "none":
        raise ResponseMeasurementSelectionError(
            "Response measurement selection must be model_screen_only with label_truth_role none."
        )
    rows = _measurement_rows(payload["measurements"])
    excluded_designs = _exclusion_rows(payload["excluded_designs"])
    _assert_reader_pairs_exist(rows, reader_designs=reader_designs, primary_reduction_id=primary_reduction_id)
    _assert_complete_design_accounting(
        rows,
        excluded_designs=excluded_designs,
        reader_designs=reader_designs,
        primary_reduction_id=primary_reduction_id,
    )
    return ResponseMeasurementSelection(
        rows=rows,
        excluded_designs=excluded_designs,
        config_path=path,
        selection_id=selection_id,
        scope="model_screen_only",
        label_truth_role="none",
    )


def _measurement_rows(value: object) -> pd.DataFrame:
    if not isinstance(value, list) or not value:
        raise ResponseMeasurementSelectionError("Response measurement selection measurements must be non-empty.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _MEASUREMENT_FIELDS:
            raise ResponseMeasurementSelectionError(
                f"Response measurement {index} fields must be exactly {sorted(_MEASUREMENT_FIELDS)}."
            )
        rows.append(
            {
                "reader_experiment_id": _required_text(
                    raw["reader_experiment_id"], field=f"measurements[{index}].reader_experiment_id"
                ),
                "design_id": _required_text(raw["design_id"], field=f"measurements[{index}].design_id"),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.duplicated().any():
        raise ResponseMeasurementSelectionError("Response measurement selection contains duplicate Reader pairs.")
    if frame["design_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["design_id"].duplicated(keep=False), "design_id"].unique())
        raise ResponseMeasurementSelectionError(
            f"Response model screen must choose at most one experiment for each design: {duplicates[:10]}"
        )
    return frame


def _exclusion_rows(value: object) -> pd.DataFrame:
    if not isinstance(value, list):
        raise ResponseMeasurementSelectionError("Response measurement selection excluded_designs must be a list.")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _EXCLUSION_FIELDS:
            raise ResponseMeasurementSelectionError(
                f"Response exclusion {index} fields must be exactly {sorted(_EXCLUSION_FIELDS)}."
            )
        reason = _required_text(raw["reason"], field=f"excluded_designs[{index}].reason")
        if reason not in _EXCLUSION_REASONS:
            raise ResponseMeasurementSelectionError(
                f"Response exclusion {index} reason must be one of {sorted(_EXCLUSION_REASONS)}."
            )
        rows.append(
            {
                "design_id": _required_text(raw["design_id"], field=f"excluded_designs[{index}].design_id"),
                "reason": reason,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=["design_id", "reason"])
    if frame["design_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["design_id"].duplicated(keep=False), "design_id"].unique())
        raise ResponseMeasurementSelectionError(f"Response exclusions contain duplicate designs: {duplicates[:10]}")
    return frame


def _assert_reader_pairs_exist(
    selected: pd.DataFrame,
    *,
    reader_designs: pd.DataFrame,
    primary_reduction_id: str,
) -> None:
    required = {"experiment_id", "design_id", "reduction_id", "is_reference"}
    missing = sorted(required - set(reader_designs.columns))
    if missing:
        raise ResponseMeasurementSelectionError(f"Reader designs lack selection fields: {missing}")
    primary = reader_designs.loc[
        reader_designs["reduction_id"].astype(str).eq(str(primary_reduction_id))
        & ~reader_designs["is_reference"].astype(bool),
        ["experiment_id", "design_id"],
    ].rename(columns={"experiment_id": "reader_experiment_id"})
    primary = primary.astype(str)
    if primary.duplicated().any():
        raise ResponseMeasurementSelectionError("Reader primary reduction contains duplicate experiment/design rows.")
    available = set(primary.itertuples(index=False, name=None))
    requested = set(selected.itertuples(index=False, name=None))
    missing_pairs = sorted(requested - available)
    if missing_pairs:
        raise ResponseMeasurementSelectionError(
            f"Response measurements are absent from the Reader primary reduction: {missing_pairs[:10]}"
        )


def _assert_complete_design_accounting(
    selected: pd.DataFrame,
    *,
    excluded_designs: pd.DataFrame,
    reader_designs: pd.DataFrame,
    primary_reduction_id: str,
) -> None:
    primary = reader_designs.loc[
        reader_designs["reduction_id"].astype(str).eq(str(primary_reduction_id))
        & ~reader_designs["is_reference"].astype(bool),
        "design_id",
    ].astype(str)
    available = set(primary)
    selected_ids = set(selected["design_id"].astype(str))
    excluded_ids = set(excluded_designs["design_id"].astype(str))
    overlap = sorted(selected_ids & excluded_ids)
    if overlap:
        raise ResponseMeasurementSelectionError(
            f"Response designs cannot be both selected and excluded: {overlap[:10]}"
        )
    unknown = sorted((selected_ids | excluded_ids) - available)
    if unknown:
        raise ResponseMeasurementSelectionError(
            f"Response selection accounts for designs absent from the Reader primary reduction: {unknown[:10]}"
        )
    unaccounted = sorted(available - selected_ids - excluded_ids)
    if unaccounted:
        raise ResponseMeasurementSelectionError(
            f"Response selection leaves Reader primary designs unaccounted for: {unaccounted[:10]}"
        )


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResponseMeasurementSelectionError(f"Response measurement selection {field} must be non-empty text.")
    return value


__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ResponseMeasurementSelection",
    "ResponseMeasurementSelectionError",
    "load_response_measurement_selection",
]
