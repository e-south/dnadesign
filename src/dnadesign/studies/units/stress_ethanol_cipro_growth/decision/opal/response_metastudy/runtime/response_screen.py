"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/response_screen.py

Orchestrate the target-state-aligned response-metric screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..core.contracts import StressTargetView
from ..core.response_contracts import (
    OR_PRESSURE_TEST_VIEW,
    RESPONSE_CONTROL_DESIGNS,
    RESPONSE_REVIEW_SPEC,
    ResponseMetricScreen,
)
from ..evaluation.greedy_support import build_greedy_support_evidence
from ..evaluation.model_representations import build_label_representations
from ..evaluation.model_screen import screen_label_models
from ..evaluation.repeated_measurements import build_repeated_measurement_evidence
from ..evaluation.response_magnitude import (
    build_response_separation_rows,
    summarize_response_separation_stability,
)
from ..evaluation.response_uncertainty import estimate_response_calibration_from_reader_draws
from ..evaluation.window_evidence import build_response_window_evidence


def build_response_metric_screen(
    labels: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    all_measurements: pd.DataFrame,
    event_intervals: pd.DataFrame,
    *,
    reader_designs: pd.DataFrame,
    reader_wells: pd.DataFrame,
    reader_traces: pd.DataFrame,
    reference_design_id: str,
    primary_reduction_id: str,
    label_ids: list[str],
    x_train: np.ndarray,
    groups: np.ndarray,
    random_forest_params: Mapping[str, object],
    target_views: tuple[StressTargetView, ...],
) -> ResponseMetricScreen:
    """Build study-owned metric evidence from Reader-owned reduced records."""
    reduction_ids = frozenset(labels["reduction_id"].astype(str).unique())
    if primary_reduction_id not in reduction_ids:
        raise ValueError(f"Reader labels lack primary reduction {primary_reduction_id!r}.")
    promotion_ids = frozenset({primary_reduction_id})
    screened_target_views = (*target_views, OR_PRESSURE_TEST_VIEW)
    margins = build_response_separation_rows(labels, target_views=screened_target_views)
    stability = summarize_response_separation_stability(
        margins,
        primary_reduction_id=primary_reduction_id,
    )
    primary_labels = labels.loc[labels["reduction_id"].astype(str).eq(primary_reduction_id)].copy()
    primary_draws = bootstrap_draws.loc[bootstrap_draws["reduction_id"].astype(str).eq(primary_reduction_id)].copy()
    label_id_set = set(primary_labels["id"].astype(str))
    draw_id_set = set(primary_draws["id"].astype(str))
    if draw_id_set != label_id_set:
        raise ValueError(
            "Reader primary bootstrap candidate universe disagrees with labels: "
            f"missing={sorted(label_id_set - draw_id_set)}, extra={sorted(draw_id_set - label_id_set)}."
        )
    draw_counts = primary_draws.groupby("id")["draw_index"].nunique()
    if draw_counts.empty or draw_counts.nunique() != 1:
        raise ValueError("Reader primary bootstrap draw counts must be complete and identical.")
    bootstrap_samples = int(draw_counts.iloc[0])
    uncertainty = estimate_response_calibration_from_reader_draws(
        primary_labels,
        primary_draws,
        target_views=screened_target_views,
        scale_quantile=RESPONSE_REVIEW_SPEC.scale_quantile,
        expected_bootstrap_samples=bootstrap_samples,
    )
    repeated_measurements, repeated_agreement = build_repeated_measurement_evidence(
        all_measurements,
        selected_labels=primary_labels,
    )
    representations = build_label_representations(
        ids=label_ids,
        response_summaries=labels.loc[labels["reduction_id"].isin(reduction_ids)],
        primary_reduction_id=primary_reduction_id,
        promotion_reduction_ids=promotion_ids,
    )
    model_screen, model_group_metrics, enrichment = screen_label_models(
        x_train,
        groups=groups,
        candidate_ids=label_ids,
        representations=representations,
        target_views=target_views,
        uncertainty_rows=uncertainty.rows,
        scale_quantile=RESPONSE_REVIEW_SPEC.scale_quantile,
        bootstrap_samples=bootstrap_samples,
        random_forest_params=random_forest_params,
    )
    enrichment_summary = summarize_retrospective_enrichment(enrichment)
    window_evidence = build_response_window_evidence(
        labels=labels,
        margin_rows=margins,
        reader_designs=reader_designs,
        reader_wells=reader_wells,
        reader_traces=reader_traces,
        model_screen=model_screen,
        reference_design_id=reference_design_id,
        response_controls=RESPONSE_CONTROL_DESIGNS,
    )
    campaign_greedy_support = build_greedy_support_evidence(
        model_screen,
        enrichment,
        primary_reduction_id=primary_reduction_id,
        model_role="campaign_model",
    )
    challenger_greedy_support = build_greedy_support_evidence(
        model_screen,
        enrichment,
        primary_reduction_id=primary_reduction_id,
        model_role="fixed_challenger",
    )
    return ResponseMetricScreen(
        event_intervals=event_intervals,
        labels=labels,
        margins=margins,
        stability=stability,
        uncertainty=uncertainty.rows,
        calibration=uncertainty.calibration,
        model_screen=model_screen,
        model_group_metrics=model_group_metrics,
        retrospective_enrichment=enrichment,
        enrichment_summary=enrichment_summary,
        campaign_greedy_support=campaign_greedy_support,
        best_fixed_challenger_greedy_support=challenger_greedy_support,
        repeated_measurements=repeated_measurements,
        repeated_agreement=repeated_agreement,
        window_evidence=window_evidence,
    )


def summarize_retrospective_enrichment(enrichment: pd.DataFrame) -> pd.DataFrame:
    """Summarize a retrospective proxy without calling it a hill climb."""

    required = {
        "representation_id",
        "promotion_eligible",
        "model_id",
        "selection_view_id",
        "reader_experiment_id",
        "selection_defined",
        "selected_true_percentile",
        "beats_group_median",
    }
    missing = sorted(required - set(enrichment.columns))
    if missing:
        raise ValueError(f"retrospective enrichment rows missing columns: {missing}")
    return (
        enrichment.groupby(
            ["representation_id", "promotion_eligible", "model_id", "selection_view_id"],
            sort=True,
            as_index=False,
        )
        .agg(
            held_out_group_count=("reader_experiment_id", "nunique"),
            defined_selection_group_count=("selection_defined", "sum"),
            median_selected_true_percentile=("selected_true_percentile", "median"),
            mean_selected_true_percentile=("selected_true_percentile", "mean"),
            fraction_beating_group_median=("beats_group_median", "mean"),
        )
        .reset_index(drop=True)
    )
