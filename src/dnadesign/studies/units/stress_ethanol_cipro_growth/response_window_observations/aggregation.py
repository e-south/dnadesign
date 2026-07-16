"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/aggregation.py

Assemble explicit-source candidate observations and separate evidence lanes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from .contracts import (
    DECISION_COLUMNS,
    EVENT_HALF_RANGE_COLUMNS,
    REPEAT_STATUSES,
    STATE_IDS,
    VALUE_COLUMNS,
    ResponseWindowAggregationError,
    ResponseWindowAggregationPolicy,
    ResponseWindowObservationPreview,
)
from .label_sources import build_label_source_observations
from .repeat_diagnostics import repeat_diagnostic_rows
from .sensitivity import event_time_sensitivity_rows, reduction_sensitivity_rows
from .uncertainty import selected_source_candidate_draws, uncertainty_rows
from .validation import validated_bootstrap_draws, validated_measurements, validated_repeat_decisions


def aggregate_response_window_observations(
    measurements: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    *,
    policy: ResponseWindowAggregationPolicy,
    repeat_decisions: pd.DataFrame,
) -> ResponseWindowObservationPreview:
    """Select declared Reader experiment units without pooling their underlying wells."""

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
    observations, contributions, blockers = build_label_source_observations(
        primary,
        decision_by_id=decision_by_id,
    )
    draws = validated_bootstrap_draws(
        bootstrap_draws,
        primary_measurements=primary,
        primary_reduction_id=policy.primary_reduction_id,
        minimum_count=policy.minimum_reader_draws_per_experiment,
    )
    label_sources = contributions.loc[contributions["included_in_label"].astype(bool)].copy()
    candidate_draws = selected_source_candidate_draws(draws, label_sources=label_sources, policy=policy)
    return ResponseWindowObservationPreview(
        observations=observations,
        contributions=contributions,
        bootstrap_draws=candidate_draws,
        uncertainty=uncertainty_rows(observations, candidate_draws, policy=policy),
        repeat_diagnostics=repeat_diagnostics,
        reduction_sensitivity=reduction_sensitivity_rows(
            measured,
            label_sources=label_sources,
            primary_reduction_id=policy.primary_reduction_id,
        ),
        event_time_sensitivity=event_time_sensitivity_rows(primary, label_sources=label_sources),
        blockers=tuple(sorted(blockers)),
    )


__all__ = [
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
