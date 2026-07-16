"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/response_screen_publication.py

Publish response-screen tables and manifest evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from ..core.response_contracts import RESPONSE_REVIEW_SPEC, ResponseMetricScreen
from .model_evidence_manifest import build_model_evidence_manifest


def response_screen_manifest(
    screen: ResponseMetricScreen,
    *,
    primary_reduction_id: str,
    campaign_to_screen_calibration: Mapping[str, object],
    campaign_model_params: Mapping[str, object],
) -> dict[str, object]:
    """Return the compact promotion posture recorded in the bundle manifest."""

    model_evidence = build_model_evidence_manifest(
        screen.model_screen,
        primary_reduction_id=primary_reduction_id,
        campaign_model_params=campaign_model_params,
    )
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
    window_contract = {
        column: _single_string(screen.window_evidence, column)
        for column in ("response_semantics", "window_selection_basis", "model_evidence_use", "trajectory_role")
    }
    return {
        "status": "screen_complete_not_promoted",
        "evidence_timing": "retrospective",
        "primary_reduction_candidate": primary_reduction_id,
        "reader_event_experiment_count": int(len(screen.event_intervals)),
        "reader_event_gap_h": {"min": float(event_gap.min()), "max": float(event_gap.max())},
        "model_screen_candidate_count": int(screen.labels["id"].nunique()),
        "reduction_count": int(screen.labels["reduction_id"].nunique()),
        "response_semantics": window_contract["response_semantics"],
        "window_comparison": {
            "reduction_count": int(len(screen.window_evidence)),
            "window_selection_basis": window_contract["window_selection_basis"],
            "model_evidence_use": window_contract["model_evidence_use"],
            "trajectory_role": window_contract["trajectory_role"],
        },
        "review_calibration_by_selection_view": {
            str(selection_view_id): {
                str(row.component): {"threshold": float(row.threshold), "scale": float(row.scale)}
                for row in rows.itertuples(index=False)
            }
            for selection_view_id, rows in screen.calibration.groupby("selection_view_id", sort=True)
        },
        "campaign_to_screen_calibration": dict(campaign_to_screen_calibration),
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
        **model_evidence,
        "repeated_design_count": int(len(screen.repeated_agreement)),
        "maximum_screen_source_to_cross_experiment_median_abs_difference": float(
            screen.repeated_agreement["maximum_selected_to_median_abs_difference"].max()
        ),
        "ratio_domain_policy": "error_at_or_below_declared_positive_floor",
        "prospective_hill_climb_demonstrated": False,
        "campaign_greedy_support": screen.campaign_greedy_support.to_dict(orient="records"),
        "best_fixed_challenger_greedy_support": screen.best_fixed_challenger_greedy_support.to_dict(orient="records"),
        "interpretation": (
            "Signed margins define relative improvement without positive exemplars. Available evidence is "
            "retrospective and does not yet establish whether greedy or mixed next-round selection is superior. "
            "Campaign-model support and challenger comparisons are reported separately."
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
        "campaign_greedy_support": screen.campaign_greedy_support,
        "best_fixed_challenger_greedy_support": screen.best_fixed_challenger_greedy_support,
        "repeated_design_measurements": screen.repeated_measurements,
        "repeated_design_agreement": screen.repeated_agreement,
        "response_window_evidence": screen.window_evidence,
    }
    paths: dict[str, Path] = {}
    for table_id, frame in table_frames.items():
        path = tables_dir / f"{table_id}.csv"
        frame.to_csv(path, index=False)
        paths[f"table__{table_id}"] = path
    return paths


def _single_string(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ValueError(f"response-window evidence lacks {column!r}.")
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"response-window evidence must declare one {column!r}; found {values}.")
    return values[0]


__all__ = ["response_screen_manifest", "write_response_screen_tables"]
