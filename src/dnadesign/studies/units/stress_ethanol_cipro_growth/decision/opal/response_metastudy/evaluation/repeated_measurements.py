"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/repeated_measurements.py

Cross-experiment agreement for repeatedly measured Reader designs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.response_contracts import STRESS_STATE_IDS

_VALUE_COLUMNS = tuple(f"r{state}" for state in STRESS_STATE_IDS) + tuple(f"b{state}" for state in STRESS_STATE_IDS)


def build_repeated_measurement_evidence(
    all_measurements: pd.DataFrame,
    *,
    selected_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return repeated measurements and source-selection sensitivity summaries."""

    required = {"design_id", "reader_experiment_id", *_VALUE_COLUMNS}
    missing = sorted(required - set(all_measurements.columns))
    if missing:
        raise ValueError(f"repeated-measurement input is missing columns: {missing}")
    selected_required = {"id", "design_id", "reader_experiment_id"}
    selected_missing = sorted(selected_required - set(selected_labels.columns))
    if selected_missing:
        raise ValueError(f"selected-label input is missing columns: {selected_missing}")

    measurements = all_measurements.copy()
    measurements["design_id"] = measurements["design_id"].astype(str)
    measurements["reader_experiment_id"] = measurements["reader_experiment_id"].astype(str)
    counts = measurements.groupby("design_id")["reader_experiment_id"].nunique()
    repeated_ids = set(counts.loc[counts.gt(1)].index.astype(str))
    if not repeated_ids:
        raise ValueError("response screen found no repeatedly measured designs.")
    repeated = measurements.loc[measurements["design_id"].isin(repeated_ids)].copy()

    selected = selected_labels.loc[:, ["id", "design_id", "reader_experiment_id"]].copy()
    selected["design_id"] = selected["design_id"].astype(str)
    selected["reader_experiment_id"] = selected["reader_experiment_id"].astype(str)
    if selected["design_id"].duplicated().any():
        duplicates = sorted(selected.loc[selected["design_id"].duplicated(keep=False), "design_id"].unique())
        raise ValueError(f"selected labels map more than one candidate to a Reader design: {duplicates}")
    selected = selected.rename(
        columns={
            "id": "selected_candidate_id",
            "reader_experiment_id": "selected_reader_experiment_id",
        }
    )
    repeated = repeated.merge(selected, on="design_id", how="left", validate="many_to_one")
    repeated["is_selected_label_source"] = repeated["reader_experiment_id"].eq(
        repeated["selected_reader_experiment_id"]
    )

    rows: list[dict[str, object]] = []
    for design_id, frame in repeated.groupby("design_id", sort=True):
        record: dict[str, object] = {
            "design_id": str(design_id),
            "experiment_count": int(frame["reader_experiment_id"].nunique()),
            "selected_candidate_id": _single_or_none(frame["selected_candidate_id"]),
            "selected_reader_experiment_id": _single_or_none(frame["selected_reader_experiment_id"]),
        }
        median_values = frame.loc[:, list(_VALUE_COLUMNS)].median(axis=0)
        selected_rows = frame.loc[frame["is_selected_label_source"].astype(bool)]
        for column in _VALUE_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            record[f"{column}__range"] = float(np.ptp(values))
            record[f"{column}__cross_experiment_median"] = float(median_values[column])
            record[f"{column}__selected_minus_median"] = (
                float(selected_rows[column].iloc[0] - median_values[column])
                if len(selected_rows) == 1
                else float("nan")
            )
        ranges = [float(record[f"{column}__range"]) for column in _VALUE_COLUMNS]
        deltas = [abs(float(record[f"{column}__selected_minus_median"])) for column in _VALUE_COLUMNS]
        record["maximum_channel_range"] = float(max(ranges))
        record["maximum_selected_to_median_abs_difference"] = (
            float(max(deltas)) if all(np.isfinite(deltas)) else float("nan")
        )
        rows.append(record)
    summary = pd.DataFrame.from_records(rows)
    return (
        repeated.sort_values(["design_id", "reader_experiment_id"], kind="mergesort").reset_index(drop=True),
        summary.sort_values("design_id", kind="mergesort").reset_index(drop=True),
    )


def _single_or_none(values: pd.Series) -> str | None:
    present = values.dropna().astype(str).unique().tolist()
    if len(present) > 1:
        raise ValueError(f"repeated-measurement selection metadata disagrees: {present}")
    return present[0] if present else None


__all__ = ["build_repeated_measurement_evidence"]
