"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/aggregation.py

Assemble equal-experiment candidate observations and separate evidence lanes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .censoring import bounded_label_blockers
from .contracts import (
    CANDIDATE_METADATA_COLUMNS,
    DECISION_COLUMNS,
    EVENT_HALF_RANGE_COLUMNS,
    REPEAT_STATUSES,
    STATE_IDS,
    VALUE_COLUMNS,
    ResponseWindowAggregationError,
    ResponseWindowAggregationPolicy,
    ResponseWindowObservationPreview,
)
from .repeat_diagnostics import repeat_diagnostic_rows
from .sensitivity import event_time_sensitivity_rows, reduction_sensitivity_rows
from .uncertainty import hierarchical_candidate_draws, uncertainty_rows
from .validation import validated_bootstrap_draws, validated_measurements, validated_repeat_decisions


def aggregate_response_window_observations(
    measurements: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    *,
    policy: ResponseWindowAggregationPolicy,
    repeat_decisions: pd.DataFrame,
) -> ResponseWindowObservationPreview:
    """Aggregate Reader experiment units without pooling their underlying wells."""

    measured = validated_measurements(measurements)
    primary = measured.loc[measured["reduction_id"].eq(policy.primary_reduction_id)].copy()
    if primary.empty:
        raise ResponseWindowAggregationError(
            f"measurements contain no primary reduction {policy.primary_reduction_id!r}."
        )
    duplicate = primary.duplicated(subset=["candidate_id", "reader_experiment_id"], keep=False)
    if duplicate.any():
        sample = primary.loc[duplicate, ["candidate_id", "reader_experiment_id", "design_id"]].to_dict(orient="records")
        raise ResponseWindowAggregationError(
            "candidate evidence must contain one design row per Reader experiment; "
            f"duplicate candidate/experiment rows={sample[:10]}."
        )

    experiment_counts = primary.groupby("candidate_id", sort=True)["reader_experiment_id"].nunique()
    repeated_ids = frozenset(experiment_counts.loc[experiment_counts.gt(1)].index.astype(str))
    decisions = validated_repeat_decisions(
        repeat_decisions,
        repeated_ids=repeated_ids,
        primary_measurements=primary,
    )
    decision_by_id = decisions.set_index("candidate_id").to_dict(orient="index")
    repeat_diagnostics = repeat_diagnostic_rows(primary, decisions=decisions)
    observations, contributions, included_ids, blockers = _point_estimates(
        primary,
        decision_by_id=decision_by_id,
    )
    blockers.extend(bounded_label_blockers(contributions))
    draws = validated_bootstrap_draws(
        bootstrap_draws,
        primary_measurements=primary,
        primary_reduction_id=policy.primary_reduction_id,
        minimum_count=policy.minimum_reader_draws_per_experiment,
    )
    candidate_draws = hierarchical_candidate_draws(draws, candidate_ids=included_ids, policy=policy)
    return ResponseWindowObservationPreview(
        observations=observations,
        contributions=contributions,
        bootstrap_draws=candidate_draws,
        uncertainty=uncertainty_rows(observations, candidate_draws, policy=policy),
        repeat_diagnostics=repeat_diagnostics,
        reduction_sensitivity=reduction_sensitivity_rows(
            measured,
            candidate_ids=included_ids,
            primary_reduction_id=policy.primary_reduction_id,
        ),
        event_time_sensitivity=event_time_sensitivity_rows(primary, candidate_ids=included_ids),
        blockers=tuple(sorted(blockers)),
    )


def _point_estimates(
    primary: pd.DataFrame,
    *,
    decision_by_id: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    blockers: list[str] = []
    contribution_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    included_ids: list[str] = []
    for candidate_id, frame in primary.groupby("candidate_id", sort=True):
        candidate_id = str(candidate_id)
        experiment_count = int(frame["reader_experiment_id"].nunique())
        decision = decision_by_id.get(candidate_id)
        status = "singleton" if experiment_count == 1 else str(decision["status"])
        included = status in {"singleton", "comparable"}
        if status == "review_required":
            blockers.append(f"{candidate_id}: repeated experiments require a comparable decision")
        contribution = frame.copy()
        contribution["repeat_decision"] = status
        contribution["repeat_decision_reason"] = "single_experiment" if decision is None else str(decision["reason"])
        for column in (
            "classification",
            "evidence_artifact",
            "evidence_sha256",
            "adjudicated_by",
            "adjudicated_at",
        ):
            contribution[f"repeat_{column}"] = None if decision is None else decision[column]
        contribution["included_in_label"] = included
        contribution["experiment_weight"] = 1.0 / experiment_count if included else 0.0
        contribution_frames.append(contribution)
        if not included:
            continue
        point = np.median(frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=float), axis=0)
        rows.append(
            {
                "candidate_id": candidate_id,
                "reader_design_ids": sorted(frame["design_id"].astype(str).unique().tolist()),
                "experiment_count": experiment_count,
                "aggregation_method": _aggregation_method(experiment_count),
                **_candidate_metadata(frame),
                **dict(zip(VALUE_COLUMNS, point.tolist(), strict=True)),
            }
        )
        included_ids.append(candidate_id)
    observations = (
        pd.DataFrame.from_records(
            rows,
            columns=[
                "candidate_id",
                "reader_design_ids",
                "experiment_count",
                "aggregation_method",
                *(column for column in CANDIDATE_METADATA_COLUMNS if column in primary.columns),
                *VALUE_COLUMNS,
            ],
        )
        .sort_values("candidate_id", kind="mergesort")
        .reset_index(drop=True)
    )
    contributions = (
        pd.concat(contribution_frames, ignore_index=True)
        .sort_values(["candidate_id", "reader_experiment_id", "design_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return observations, contributions, included_ids, blockers


def _candidate_metadata(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in CANDIDATE_METADATA_COLUMNS:
        if column not in frame.columns:
            continue
        values = sorted(set(frame[column].dropna().astype(str)))
        if len(values) != 1 or not values[0].strip():
            raise ResponseWindowAggregationError(
                f"candidate {frame['candidate_id'].iloc[0]!r} has non-invariant {column!r} metadata."
            )
        result[column] = values[0]
    return result


def _aggregation_method(experiment_count: int) -> str:
    if experiment_count == 1:
        return "single_experiment"
    if experiment_count == 2:
        return "two_experiment_midpoint"
    return "componentwise_experiment_median"


__all__ = [
    "CANDIDATE_METADATA_COLUMNS",
    "DECISION_COLUMNS",
    "EVENT_HALF_RANGE_COLUMNS",
    "REPEAT_STATUSES",
    "STATE_IDS",
    "VALUE_COLUMNS",
    "ResponseWindowAggregationError",
    "ResponseWindowAggregationPolicy",
    "ResponseWindowObservationPreview",
    "aggregate_response_window_observations",
]
