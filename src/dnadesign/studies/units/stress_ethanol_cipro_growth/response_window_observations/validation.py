"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/validation.py

Fail-closed validation for experiment evidence and repeat decisions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .censoring import ResponseWindowCensoringError, validated_censor_provenance
from .contracts import (
    DECISION_COLUMNS,
    EVENT_HALF_RANGE_COLUMNS,
    REPEAT_STATUSES,
    VALUE_COLUMNS,
    ResponseWindowAggregationError,
)
from .repeat_adjudication import validate_repeat_adjudications


def validated_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "design_id",
        "reader_experiment_id",
        "reduction_id",
        "reduction_role",
        *VALUE_COLUMNS,
        *EVENT_HALF_RANGE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResponseWindowAggregationError(f"response measurements are missing columns: {missing}")
    try:
        result = validated_censor_provenance(frame)
    except ResponseWindowCensoringError as exc:
        raise ResponseWindowAggregationError(str(exc)) from exc
    for column in ("candidate_id", "design_id", "reader_experiment_id", "reduction_id", "reduction_role"):
        invalid = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ResponseWindowAggregationError(f"response measurement field {column!r} contains empty values.")
        result[column] = result[column].astype(str)
    numeric = result.loc[:, VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ResponseWindowAggregationError("response measurements must contain finite eight-component vectors.")
    result.loc[:, VALUE_COLUMNS] = numeric
    event_ranges = result.loc[:, EVENT_HALF_RANGE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(event_ranges.to_numpy(dtype=float)).all() or (event_ranges < 0.0).any().any():
        raise ResponseWindowAggregationError("event-time half ranges must be finite and nonnegative.")
    result.loc[:, EVENT_HALF_RANGE_COLUMNS] = event_ranges
    identity = ["candidate_id", "design_id", "reader_experiment_id", "reduction_id"]
    if result.duplicated(subset=identity).any():
        raise ResponseWindowAggregationError("response measurements contain duplicate experiment/design reductions.")
    if result.groupby("reduction_id")["reduction_role"].nunique().gt(1).any():
        raise ResponseWindowAggregationError("response reduction roles disagree within a reduction id.")
    return result


def validated_repeat_decisions(
    frame: pd.DataFrame,
    *,
    repeated_ids: frozenset[str],
    primary_measurements: pd.DataFrame,
) -> pd.DataFrame:
    if set(frame.columns) != set(DECISION_COLUMNS):
        raise ResponseWindowAggregationError(f"repeat decision fields must be exactly {sorted(DECISION_COLUMNS)}.")
    result = frame.loc[:, DECISION_COLUMNS].copy()
    for column in ("candidate_id", "status", "reason"):
        result[column] = result[column].astype(str)
    if result["candidate_id"].duplicated().any():
        raise ResponseWindowAggregationError("repeat decisions contain duplicate candidate IDs.")
    invalid_status = sorted(set(result["status"]) - REPEAT_STATUSES)
    if invalid_status:
        raise ResponseWindowAggregationError(f"repeat decisions contain unsupported statuses: {invalid_status}")
    if result["reason"].str.strip().eq("").any():
        raise ResponseWindowAggregationError("repeat decision reasons must be non-empty.")
    validate_repeat_adjudications(result)
    declared_ids = frozenset(result["candidate_id"])
    if missing := sorted(repeated_ids - declared_ids):
        raise ResponseWindowAggregationError(f"repeat policy is missing repeated candidates: {missing}")
    if stale := sorted(declared_ids - repeated_ids):
        raise ResponseWindowAggregationError(f"repeat policy declares non-repeated candidates: {stale}")
    for row in result.itertuples(index=False):
        aliases = _normalized_identity_list(
            row.reader_design_ids,
            candidate_id=row.candidate_id,
            field="reader_design_ids",
        )
        observed_aliases = _observed_identities(
            primary_measurements,
            candidate_id=row.candidate_id,
            field="design_id",
        )
        if aliases != observed_aliases:
            raise ResponseWindowAggregationError(
                f"{row.candidate_id}: repeat-policy design aliases disagree; "
                f"expected={observed_aliases}, found={aliases}."
            )
        experiments = _normalized_identity_list(
            row.reader_experiment_ids,
            candidate_id=row.candidate_id,
            field="reader_experiment_ids",
        )
        observed_experiments = _observed_identities(
            primary_measurements,
            candidate_id=row.candidate_id,
            field="reader_experiment_id",
        )
        if experiments != observed_experiments:
            raise ResponseWindowAggregationError(
                f"{row.candidate_id}: repeat-policy experiment identities disagree; "
                f"expected={observed_experiments}, found={experiments}."
            )
        index = result.index[result["candidate_id"].eq(row.candidate_id)][0]
        result.at[index, "reader_design_ids"] = aliases
        result.at[index, "reader_experiment_ids"] = experiments
    return result


def validated_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    primary_measurements: pd.DataFrame,
    primary_reduction_id: str,
    minimum_count: int,
) -> pd.DataFrame:
    required = {"candidate_id", "design_id", "reader_experiment_id", "reduction_id", "draw_index", *VALUE_COLUMNS}
    if missing := sorted(required - set(frame.columns)):
        raise ResponseWindowAggregationError(f"Reader bootstrap draws are missing columns: {missing}")
    columns = ["candidate_id", "design_id", "reader_experiment_id", "reduction_id", "draw_index", *VALUE_COLUMNS]
    result = frame.loc[frame["reduction_id"].astype(str).eq(primary_reduction_id), columns].copy()
    if result.empty:
        raise ResponseWindowAggregationError("Reader bootstrap draws contain no primary-reduction rows.")
    for column in ("candidate_id", "design_id", "reader_experiment_id", "reduction_id"):
        result[column] = result[column].astype(str)
    numeric = result.loc[:, VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ResponseWindowAggregationError("Reader bootstrap draws must contain finite vectors.")
    result.loc[:, VALUE_COLUMNS] = numeric
    raw_index = pd.to_numeric(result["draw_index"], errors="coerce")
    if raw_index.isna().any() or (raw_index < 0).any() or (raw_index != np.floor(raw_index)).any():
        raise ResponseWindowAggregationError("Reader bootstrap draw_index must contain nonnegative integers.")
    result["draw_index"] = raw_index.astype(int)
    key = ["candidate_id", "design_id", "reader_experiment_id", "draw_index"]
    if result.duplicated(subset=key).any():
        raise ResponseWindowAggregationError("Reader bootstrap draws contain duplicate contribution/draw indices.")
    contribution_key = ["candidate_id", "design_id", "reader_experiment_id"]
    expected = set(primary_measurements[contribution_key].itertuples(index=False, name=None))
    observed = set(result[contribution_key].itertuples(index=False, name=None))
    if observed != expected:
        raise ResponseWindowAggregationError(
            "Reader bootstrap contribution coverage disagrees with the primary measurements."
        )
    index_sets = result.groupby(contribution_key, sort=True)["draw_index"].agg(
        lambda values: tuple(sorted(set(int(value) for value in values)))
    )
    if index_sets.nunique() != 1:
        raise ResponseWindowAggregationError("Reader bootstrap draw-index sets must be complete and identical.")
    counts = result.groupby(contribution_key)["draw_index"].nunique()
    if counts.lt(minimum_count).any():
        sample = counts.loc[counts.lt(minimum_count)].head(10).to_dict()
        raise ResponseWindowAggregationError(
            f"Reader bootstrap support is below {minimum_count} draws for candidate experiments: {sample}"
        )
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def _normalized_identity_list(value: object, *, candidate_id: str, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ResponseWindowAggregationError(f"{candidate_id}: {field} must be a non-empty list.")
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _observed_identities(frame: pd.DataFrame, *, candidate_id: str, field: str) -> list[str]:
    return sorted(frame.loc[frame["candidate_id"].eq(candidate_id), field].astype(str).unique().tolist())


__all__ = ["validated_bootstrap_draws", "validated_measurements", "validated_repeat_decisions"]
