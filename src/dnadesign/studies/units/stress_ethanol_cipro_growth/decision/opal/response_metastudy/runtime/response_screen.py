"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/response_screen.py

Orchestrate the induction-aligned response-metric screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..core.contracts import StressTargetView
from ..core.response_contracts import (
    OR_PRESSURE_TEST_VIEW,
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


def build_response_metric_screen(
    labels: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    all_measurements: pd.DataFrame,
    event_intervals: pd.DataFrame,
    *,
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
    greedy_support = build_greedy_support_evidence(
        model_screen,
        enrichment,
        primary_reduction_id=primary_reduction_id,
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
        greedy_support=greedy_support,
        repeated_measurements=repeated_measurements,
        repeated_agreement=repeated_agreement,
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


def response_screen_manifest(screen: ResponseMetricScreen, *, primary_reduction_id: str) -> dict[str, object]:
    """Return the compact promotion posture recorded in the bundle manifest."""

    eligible = screen.model_screen.loc[
        screen.model_screen["promotion_eligible"].astype(bool)
        & screen.model_screen["all_target_view_metrics_finite"].astype(bool)
    ]
    best = eligible.sort_values(
        ["weakest_required_ordering_spearman", "median_channel_spearman"],
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    event_gap = (
        screen.event_intervals["event_interval_end_assay_h"] - screen.event_intervals["event_interval_start_assay_h"]
    )
    primary = screen.stability.loc[screen.stability["reduction_id"].eq(primary_reduction_id)]
    reduction_columns = [
        "reduction_id",
        "screen_role",
        "response_basis",
        "reduction_method",
        "window_start_event_h",
        "window_end_event_h",
    ]
    reduction_rows = (
        screen.labels.loc[:, reduction_columns]
        .drop_duplicates()
        .sort_values(["screen_role", "reduction_id"], kind="mergesort")
    )
    if "bootstrap_samples" not in screen.calibration.columns:
        raise ValueError("response calibration lacks Reader bootstrap provenance.")
    bootstrap_samples = int(screen.calibration["bootstrap_samples"].iloc[0])
    return {
        "status": "screen_complete_not_promoted",
        "primary_reduction_candidate": primary_reduction_id,
        "reader_event_experiment_count": int(len(screen.event_intervals)),
        "reader_event_gap_h": {"min": float(event_gap.min()), "max": float(event_gap.max())},
        "label_count": int(screen.labels["id"].nunique()),
        "reduction_count": int(screen.labels["reduction_id"].nunique()),
        "review_calibration_by_selection_view": {
            str(selection_view_id): {
                str(row.component): {"threshold": float(row.threshold), "scale": float(row.scale)}
                for row in rows.itertuples(index=False)
            }
            for selection_view_id, rows in screen.calibration.groupby("selection_view_id", sort=True)
        },
        "response_screen_protocol": {
            "bootstrap_samples": bootstrap_samples,
            "scale_quantile": RESPONSE_REVIEW_SPEC.scale_quantile,
            "model_min_within_group_spearman": RESPONSE_REVIEW_SPEC.model_min_within_group_spearman,
            "model_min_defined_group_count": RESPONSE_REVIEW_SPEC.model_min_defined_group_count,
            "model_reduction_ids": sorted(screen.labels["reduction_id"].astype(str).unique()),
            "reductions": [
                {
                    "id": str(row.reduction_id),
                    "screen_role": str(row.screen_role),
                    "response_basis": str(row.response_basis),
                    "method": str(row.reduction_method),
                    "window_start_event_h": float(row.window_start_event_h),
                    "window_end_event_h": float(row.window_end_event_h),
                }
                for row in reduction_rows.itertuples(index=False)
            ],
        },
        "zero_constraint_feasible_count_by_selection_view": {
            str(row.selection_view_id): int(row.zero_constraint_feasible_count)
            for row in primary.itertuples(index=False)
        },
        "best_fixed_model_screen": {
            "representation_id": str(best["representation_id"]),
            "model_id": str(best["model_id"]),
            "weakest_target_view_response_separation_spearman": float(
                best["weakest_target_view_response_separation_spearman"]
            ),
            "weakest_target_view_feasibility_spearman": float(best["weakest_target_view_feasibility_spearman"]),
            "weakest_required_ordering_spearman": float(best["weakest_required_ordering_spearman"]),
            "minimum_defined_group_count": int(best["minimum_defined_group_count"]),
            "metric_scope": str(best["metric_scope"]),
            "posture": "challenger_only_no_hyperparameter_promotion",
        },
        "model_support_ready": bool(
            float(best["weakest_target_view_response_separation_spearman"])
            >= RESPONSE_REVIEW_SPEC.model_min_within_group_spearman
            and float(best["weakest_target_view_feasibility_spearman"])
            >= RESPONSE_REVIEW_SPEC.model_min_within_group_spearman
            and int(best["minimum_defined_group_count"]) >= RESPONSE_REVIEW_SPEC.model_min_defined_group_count
        ),
        "repeated_design_count": int(len(screen.repeated_agreement)),
        "maximum_screen_source_to_cross_experiment_median_abs_difference": float(
            screen.repeated_agreement["maximum_selected_to_median_abs_difference"].max()
        ),
        "ratio_domain_policy": "error_at_or_below_declared_positive_floor",
        "prospective_hill_climb_demonstrated": False,
        "greedy_support": screen.greedy_support.to_dict(orient="records"),
        "interpretation": (
            "Signed margins define relative improvement without positive exemplars. Available evidence is "
            "retrospective and does not yet establish whether greedy or mixed next-round selection is superior."
        ),
    }


def write_response_screen_tables(screen: ResponseMetricScreen, tables_dir: Path) -> dict[str, Path]:
    """Write the response-screen tables and return their artifact registrations."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    table_frames = {
        "reader_event_intervals": screen.event_intervals,
        "reader_response_summaries": screen.labels,
        "response_separation_components": screen.margins,
        "response_separation_stability": screen.stability,
        "response_separation_uncertainty": screen.uncertainty,
        "response_separation_review_scales": screen.calibration,
        "label_model_screen": screen.model_screen,
        "label_model_group_metrics": screen.model_group_metrics,
        "retrospective_enrichment": screen.retrospective_enrichment,
        "retrospective_enrichment_summary": screen.enrichment_summary,
        "greedy_support": screen.greedy_support,
        "repeated_design_measurements": screen.repeated_measurements,
        "repeated_design_agreement": screen.repeated_agreement,
    }
    paths: dict[str, Path] = {}
    for table_id, frame in table_frames.items():
        path = tables_dir / f"{table_id}.csv"
        frame.to_csv(path, index=False)
        paths[f"table__{table_id}"] = path
    return paths
