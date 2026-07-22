"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/sensitivity.py

Alternate-reduction and event-time sensitivity summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from .contracts import EVENT_HALF_RANGE_COLUMNS, VALUE_COLUMNS, ResponseWindowAggregationError


def reduction_sensitivity_rows(
    measurements: pd.DataFrame,
    *,
    label_sources: pd.DataFrame,
    primary_reduction_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in label_sources.sort_values("candidate_id", kind="mergesort").itertuples(index=False):
        candidate_id = str(source.candidate_id)
        reader_experiment_id = str(source.reader_experiment_id)
        design_id = str(source.design_id)
        frame = measurements.loc[
            measurements["candidate_id"].astype(str).eq(candidate_id)
            & measurements["reader_experiment_id"].astype(str).eq(reader_experiment_id)
            & measurements["design_id"].astype(str).eq(design_id)
        ]
        if frame.empty:
            raise ResponseWindowAggregationError(f"{candidate_id}: selected label source has no reduction rows.")
        candidate_rows: list[dict[str, object]] = []
        for reduction_id, reduction in frame.groupby("reduction_id", sort=True):
            if len(reduction) != 1:
                raise ResponseWindowAggregationError(
                    f"{candidate_id}: selected label source has duplicate rows for reduction {reduction_id!r}."
                )
            values = reduction.loc[:, VALUE_COLUMNS].iloc[0].to_numpy(dtype=float)
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "design_id": design_id,
                    "reader_experiment_id": reader_experiment_id,
                    "reduction_id": str(reduction_id),
                    "reduction_role": str(reduction["reduction_role"].iloc[0]),
                    **dict(zip(VALUE_COLUMNS, values.tolist(), strict=True)),
                }
            )
        try:
            primary = next(row for row in candidate_rows if row["reduction_id"] == primary_reduction_id)
        except StopIteration as exc:
            raise ResponseWindowAggregationError(
                f"{candidate_id}: selected label source lacks primary reduction {primary_reduction_id!r}."
            ) from exc
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
    label_sources: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in label_sources.sort_values("candidate_id", kind="mergesort").itertuples(index=False):
        candidate_id = str(source.candidate_id)
        reader_experiment_id = str(source.reader_experiment_id)
        design_id = str(source.design_id)
        frame = primary_measurements.loc[
            primary_measurements["candidate_id"].astype(str).eq(candidate_id)
            & primary_measurements["reader_experiment_id"].astype(str).eq(reader_experiment_id)
            & primary_measurements["design_id"].astype(str).eq(design_id)
        ]
        if len(frame) != 1:
            raise ResponseWindowAggregationError(
                f"{candidate_id}: selected label source must resolve to one primary sensitivity row."
            )
        for component, source_column in zip(VALUE_COLUMNS, EVENT_HALF_RANGE_COLUMNS, strict=True):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "design_id": design_id,
                    "reader_experiment_id": reader_experiment_id,
                    "component": component,
                    "event_time_half_range": float(frame[source_column].iloc[0]),
                }
            )
    return pd.DataFrame.from_records(rows)


__all__ = ["event_time_sensitivity_rows", "reduction_sensitivity_rows"]
