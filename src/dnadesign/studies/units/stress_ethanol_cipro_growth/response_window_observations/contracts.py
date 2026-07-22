"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/contracts.py

Shared scientific contracts for candidate response-window observations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

STATE_IDS = ("00", "10", "01", "11")
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_IDS) + tuple(f"b{state}" for state in STATE_IDS)
EVENT_HALF_RANGE_COLUMNS = tuple(f"{column}_event_half_range" for column in VALUE_COLUMNS)
CANDIDATE_METADATA_COLUMNS = (
    "display_label",
    "sequence_sha256",
    "source_class",
    "design_family",
    "baserender_adapter_kind",
)
DECISION_COLUMNS = (
    "candidate_id",
    "reader_design_ids",
    "reader_experiment_ids",
    "label_source_reader_experiment_id",
    "status",
    "classification",
    "evidence_artifact",
    "evidence_sha256",
    "adjudicated_by",
    "adjudicated_at",
    "reason",
)
REPEAT_STATUSES = frozenset({"label_source_selected", "label_source_excluded", "remeasure_required", "review_required"})
REPEAT_CLASSIFICATIONS = frozenset(
    {
        "source_agreement_accepted",
        "corrected_technical_error",
        "noncomparable_assay_context",
        "plausible_biological_variation",
        "unresolved_source_disagreement",
        "remeasurement_required",
        "unresolved",
    }
)


class ResponseWindowAggregationError(ValueError):
    """Raised when experiment evidence cannot form trusted candidate observations."""


@dataclass(frozen=True)
class ResponseWindowAggregationPolicy:
    """Scientific rules that turn experiment evidence into candidate evidence."""

    policy_id: str
    primary_reduction_id: str
    bootstrap_samples: int
    confidence_level: float
    random_seed: int
    minimum_reader_draws_per_experiment: int

    def __post_init__(self) -> None:
        if not str(self.policy_id).strip() or not str(self.primary_reduction_id).strip():
            raise ResponseWindowAggregationError("aggregation policy identifiers must be non-empty.")
        if isinstance(self.bootstrap_samples, bool) or self.bootstrap_samples < 100:
            raise ResponseWindowAggregationError("selected-source bootstrap requires at least 100 samples.")
        if not math.isfinite(self.confidence_level) or not 0.0 < self.confidence_level < 1.0:
            raise ResponseWindowAggregationError("confidence_level must be finite and between zero and one.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ResponseWindowAggregationError("random_seed must be an integer.")
        if isinstance(self.minimum_reader_draws_per_experiment, bool) or self.minimum_reader_draws_per_experiment < 1:
            raise ResponseWindowAggregationError("minimum Reader draw count must be positive.")


@dataclass(frozen=True)
class ResponseWindowObservationPreview:
    """Candidate observations and their separate uncertainty evidence."""

    observations: pd.DataFrame
    contributions: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    uncertainty: pd.DataFrame
    repeat_diagnostics: pd.DataFrame
    reduction_sensitivity: pd.DataFrame
    event_time_sensitivity: pd.DataFrame
    blockers: tuple[str, ...]


__all__ = [
    "CANDIDATE_METADATA_COLUMNS",
    "DECISION_COLUMNS",
    "EVENT_HALF_RANGE_COLUMNS",
    "REPEAT_STATUSES",
    "REPEAT_CLASSIFICATIONS",
    "STATE_IDS",
    "VALUE_COLUMNS",
    "ResponseWindowAggregationError",
    "ResponseWindowAggregationPolicy",
    "ResponseWindowObservationPreview",
]
