"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/selection_views/performance.py

Validates and summarizes observed objective values by selection view and round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {
    "observed_round",
    "candidate_id",
    "selected_for_view_id",
    "objective_view_id",
    "objective_value",
}


@dataclass(frozen=True, slots=True)
class SelectionViewPerformance:
    """Validated candidate observations and their between-candidate summaries."""

    observations: pd.DataFrame
    summary: pd.DataFrame


def selection_view_performance(frame: pd.DataFrame) -> SelectionViewPerformance:
    """Compare each selected cohort under every declared objective view."""

    observations = _validated_observations(frame)
    summary = (
        observations.groupby(
            ["observed_round", "objective_view_id", "selected_for_view_id"],
            sort=True,
            observed=True,
        )["objective_value"]
        .agg(
            candidate_count="size",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )
    summary["selected_for_objective_view"] = summary["objective_view_id"].eq(summary["selected_for_view_id"])
    return SelectionViewPerformance(observations=observations, summary=summary)


def _validated_observations(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("selection-view performance requires a non-empty pandas DataFrame")
    missing = sorted(_REQUIRED_COLUMNS - set(frame))
    if missing:
        raise ValueError(f"selection-view performance is missing columns: {missing}")
    observations = frame.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    for column in ("candidate_id", "selected_for_view_id", "objective_view_id"):
        if (
            observations[column].isna().any()
            or not observations[column]
            .map(lambda value: isinstance(value, str) and value == value.strip() and bool(value))
            .all()
        ):
            raise ValueError(f"selection-view performance {column} values must be exact non-empty strings")
    if observations["observed_round"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("selection-view performance observed_round values must be non-negative integers")
    try:
        rounds = pd.to_numeric(observations["observed_round"], errors="raise").to_numpy(dtype=float)
        objective_values = pd.to_numeric(observations["objective_value"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection-view performance round and objective values must be numeric") from exc
    if not np.isfinite(rounds).all() or not np.equal(rounds, np.floor(rounds)).all() or np.any(rounds < 0):
        raise ValueError("selection-view performance observed_round values must be non-negative integers")
    if not np.isfinite(objective_values).all():
        raise ValueError("selection-view performance objective values must be finite")
    observations["observed_round"] = rounds.astype(int)
    observations["objective_value"] = objective_values
    if observations.duplicated(["observed_round", "candidate_id", "objective_view_id"]).any():
        raise ValueError("selection-view performance contains duplicate candidate/objective rows")

    expected_views: set[str] | None = None
    for observed_round, round_rows in observations.groupby("observed_round", sort=True):
        objective_views = set(round_rows["objective_view_id"])
        selected_views = set(round_rows["selected_for_view_id"])
        if selected_views != objective_views:
            raise ValueError(
                f"selection-view performance round {observed_round} must select candidates for every objective view"
            )
        if expected_views is None:
            expected_views = objective_views
        elif objective_views != expected_views:
            raise ValueError("selection-view performance requires the same objective and selection views across rounds")
        for candidate_id, candidate_rows in round_rows.groupby("candidate_id", sort=True):
            if set(candidate_rows["objective_view_id"]) != objective_views:
                raise ValueError(
                    "selection-view performance requires a complete objective grid for every round/candidate"
                )
            if candidate_rows["selected_for_view_id"].nunique() != 1:
                raise ValueError(
                    f"selection-view performance candidate {candidate_id!r} has inconsistent selection provenance"
                )
    return observations.sort_values(
        ["observed_round", "objective_view_id", "selected_for_view_id", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


__all__ = ["SelectionViewPerformance", "selection_view_performance"]
