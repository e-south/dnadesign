"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/sfxi_reference_overlay.py

SFXI reference-overlay helpers for OPAL plot primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...api.sfxi import (
    SFXI_REFERENCE_OVERLAY_PREFIX,
    validate_sfxi_reference_overlay_records,
)
from ..core.utils import ExitCodes, OpalError
from ..storage.parquet_io import read_parquet_df
from ._param_utils import get_bool, get_str


def sfxi_reference_overlay_enabled(params: Mapping[str, Any]) -> bool:
    """Return whether a plot explicitly requested SFXI reference overlays."""

    return get_bool(
        dict(params),
        ["reference_overlay", "include_reference_overlay", "sfxi_ref_overlay", "reference_points"],
        False,
    )


def load_sfxi_reference_overlay(
    *,
    records_path: Path | None,
    params: Mapping[str, Any],
    expected_setpoint_vector: Sequence[float] | None,
) -> pd.DataFrame:
    """Load and validate materialized ``sfxi_ref`` records for plot overlays."""

    if not sfxi_reference_overlay_enabled(params):
        return pd.DataFrame()
    if records_path is None:
        raise OpalError("SFXI reference overlay requested, but no records path is available.", ExitCodes.BAD_ARGS)

    records_path = Path(records_path)
    schema_names = set(pq.read_schema(records_path).names)
    prefix = SFXI_REFERENCE_OVERLAY_PREFIX
    required = {
        "id",
        f"{prefix}objective_name",
        f"{prefix}api_version",
        f"{prefix}state_order",
        f"{prefix}setpoint_vector",
        f"{prefix}metric_id",
        f"{prefix}metric_value",
        f"{prefix}metric_provenance",
        f"{prefix}denom_used",
        f"{prefix}denom_percentile",
        f"{prefix}logic_fidelity",
        f"{prefix}effect_raw",
        f"{prefix}effect_scaled",
        f"{prefix}sfxi",
    }
    optional = {
        f"{prefix}reference_instance_id",
        f"{prefix}collection_id",
        f"{prefix}batch_id",
        f"{prefix}campaign_id",
        f"{prefix}setpoint_name",
        f"{prefix}reference_design_id",
        f"{prefix}sequence_source_id",
    }
    missing = sorted(required - schema_names)
    if missing:
        raise OpalError(
            "SFXI reference overlay requested, but records.parquet is missing required columns: " + ", ".join(missing),
            ExitCodes.CONTRACT_VIOLATION,
        )

    columns = [column for column in [*sorted(required), *sorted(optional)] if column in schema_names]
    frame = read_parquet_df(records_path, columns=columns)
    frame = frame.dropna(subset=[f"{prefix}metric_value"])

    frame = _apply_filter(frame, params, "reference_collection_id", f"{prefix}collection_id")
    frame = _apply_filter(frame, params, "reference_campaign_id", f"{prefix}campaign_id")
    frame = _apply_filter(frame, params, "reference_batch_id", f"{prefix}batch_id")
    frame = _apply_filter(frame, params, "reference_metric_id", f"{prefix}metric_id")
    if frame.empty:
        raise OpalError("SFXI reference overlay filters matched zero rows.", ExitCodes.BAD_ARGS)

    metric_id = get_str(dict(params), ["reference_metric_id"], None)
    validate_sfxi_reference_overlay_records(
        frame.to_dict(orient="records"),
        expected_setpoint_vector=expected_setpoint_vector,
        metric_id=metric_id,
    )
    return frame.reset_index(drop=True)


def reference_y_values(frame: pd.DataFrame, *, y_axis: str, params: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Return SFXI reference-overlay Y values matching a plot Y axis."""

    override = get_str(dict(params), ["reference_y_axis", "reference_y", "reference_metric"], None)
    key = str(override or y_axis or "score").replace(".", "__").strip().lower()
    prefix = SFXI_REFERENCE_OVERLAY_PREFIX
    column_by_axis = {
        "score": f"{prefix}metric_value",
        "view__selection_score": f"{prefix}metric_value",
        "sfxi": f"{prefix}sfxi",
        "metric_value": f"{prefix}metric_value",
        "effect_raw": f"{prefix}effect_raw",
        "obj__effect_raw": f"{prefix}effect_raw",
        "effect_scaled": f"{prefix}effect_scaled",
        "obj__effect_scaled": f"{prefix}effect_scaled",
        "logic_fidelity": f"{prefix}logic_fidelity",
        "obj__logic_fidelity": f"{prefix}logic_fidelity",
    }
    column = column_by_axis.get(key)
    if column is None:
        raise OpalError(
            "SFXI reference overlay cannot infer a reference Y value for "
            f"y_axis={y_axis!r}. Use y_axis score/effect_raw/effect_scaled or set reference_y_axis.",
            ExitCodes.BAD_ARGS,
        )
    if column not in frame.columns:
        raise OpalError(f"SFXI reference overlay missing Y column: {column}", ExitCodes.CONTRACT_VIOLATION)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise OpalError(f"SFXI reference overlay column {column} contains non-finite values.", ExitCodes.BAD_ARGS)
    return values, column


def reference_label_values(frame: pd.DataFrame) -> list[str]:
    """Build concise reference labels for plot legends/tooltips."""

    prefix = SFXI_REFERENCE_OVERLAY_PREFIX
    for column in (
        f"{prefix}reference_instance_id",
        f"{prefix}reference_design_id",
        f"{prefix}sequence_source_id",
        "id",
    ):
        if column in frame.columns:
            return [str(value) for value in frame[column].fillna("").to_list()]
    return ["" for _ in range(len(frame))]


def _apply_filter(frame: pd.DataFrame, params: Mapping[str, Any], param_key: str, column: str) -> pd.DataFrame:
    raw = params.get(param_key)
    if raw is None or column not in frame.columns:
        return frame
    values = [str(item) for item in (raw if isinstance(raw, list) else [raw])]
    return frame[frame[column].astype(str).isin(values)].copy()
