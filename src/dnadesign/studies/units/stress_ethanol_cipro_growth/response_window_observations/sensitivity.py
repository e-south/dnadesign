"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/sensitivity.py

Alternate-reduction and event-time sensitivity summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import EVENT_HALF_RANGE_COLUMNS, VALUE_COLUMNS, ResponseWindowAggregationError


def reduction_sensitivity_rows(
    measurements: pd.DataFrame,
    *,
    candidate_ids: list[str],
    primary_reduction_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id in sorted(candidate_ids):
        frame = measurements.loc[measurements["candidate_id"].eq(candidate_id)]
        primary_experiments = set(
            frame.loc[frame["reduction_id"].eq(primary_reduction_id), "reader_experiment_id"].astype(str)
        )
        candidate_rows: list[dict[str, object]] = []
        for reduction_id, reduction in frame.groupby("reduction_id", sort=True):
            observed_experiments = set(reduction["reader_experiment_id"].astype(str))
            if observed_experiments != primary_experiments:
                raise ResponseWindowAggregationError(
                    f"{candidate_id}: reduction {reduction_id!r} experiment coverage disagrees with primary."
                )
            values = np.median(reduction.loc[:, VALUE_COLUMNS].to_numpy(dtype=float), axis=0)
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "reduction_id": str(reduction_id),
                    "reduction_role": str(reduction["reduction_role"].iloc[0]),
                    "experiment_count": len(primary_experiments),
                    **dict(zip(VALUE_COLUMNS, values.tolist(), strict=True)),
                }
            )
        primary = next(row for row in candidate_rows if row["reduction_id"] == primary_reduction_id)
        for row in candidate_rows:
            deltas = []
            for component in VALUE_COLUMNS:
                delta = float(row[component]) - float(primary[component])
                row[f"{component}__delta_from_primary"] = delta
                deltas.append(abs(delta))
            row["maximum_abs_delta_from_primary"] = max(deltas)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def event_time_sensitivity_rows(
    primary_measurements: pd.DataFrame,
    *,
    candidate_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id in sorted(candidate_ids):
        frame = primary_measurements.loc[primary_measurements["candidate_id"].eq(candidate_id)]
        for component, source_column in zip(VALUE_COLUMNS, EVENT_HALF_RANGE_COLUMNS, strict=True):
            values = frame[source_column].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "component": component,
                    "experiment_count": int(frame["reader_experiment_id"].nunique()),
                    "median_event_time_half_range": float(np.median(values)),
                    "maximum_event_time_half_range": float(np.max(values)),
                }
            )
    return pd.DataFrame.from_records(rows)


__all__ = ["event_time_sensitivity_rows", "reduction_sensitivity_rows"]
