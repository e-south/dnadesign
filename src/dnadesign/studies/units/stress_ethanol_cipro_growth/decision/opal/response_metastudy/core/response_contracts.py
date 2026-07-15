"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/core/response_contracts.py

Study-owned contracts for response-metric evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .contracts import STRESS_STATE_IDS, StressTargetView


@dataclass(frozen=True)
class ResponseMetricScreen:
    """Typed evidence bundle returned by the response screen."""

    event_intervals: pd.DataFrame
    labels: pd.DataFrame
    margins: pd.DataFrame
    stability: pd.DataFrame
    uncertainty: pd.DataFrame
    calibration: pd.DataFrame
    model_screen: pd.DataFrame
    model_group_metrics: pd.DataFrame
    retrospective_enrichment: pd.DataFrame
    enrichment_summary: pd.DataFrame
    campaign_greedy_support: pd.DataFrame
    best_fixed_challenger_greedy_support: pd.DataFrame
    repeated_measurements: pd.DataFrame
    repeated_agreement: pd.DataFrame
    window_evidence: pd.DataFrame


@dataclass(frozen=True)
class ResponseReviewSpec:
    """Non-biological thresholds used to review label/model support."""

    scale_quantile: float
    model_min_within_group_spearman: float
    model_min_defined_group_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale_quantile) or not 0.5 <= self.scale_quantile < 1.0:
            raise ValueError("response review scale_quantile must be in [0.5, 1).")
        if not math.isfinite(self.model_min_within_group_spearman) or not (
            -1.0 <= self.model_min_within_group_spearman <= 1.0
        ):
            raise ValueError("response review model Spearman threshold must be in [-1, 1].")
        if self.model_min_defined_group_count < 1:
            raise ValueError("response review model group count must be positive.")


RESPONSE_REVIEW_SPEC = ResponseReviewSpec(
    scale_quantile=0.90,
    model_min_within_group_spearman=0.30,
    model_min_defined_group_count=6,
)


OR_PRESSURE_TEST_VIEW = StressTargetView(
    id="or",
    label="OR pressure test",
    target_mask=(0.0, 1.0, 1.0, 1.0),
)

RESPONSE_CONTROL_DESIGNS = {
    "ethanol": "pDual-10-spyp",
    "ciprofloxacin": "pDual-10-sulAp",
}


__all__ = [
    "OR_PRESSURE_TEST_VIEW",
    "RESPONSE_REVIEW_SPEC",
    "RESPONSE_CONTROL_DESIGNS",
    "ResponseMetricScreen",
    "ResponseReviewSpec",
    "STRESS_STATE_IDS",
]
