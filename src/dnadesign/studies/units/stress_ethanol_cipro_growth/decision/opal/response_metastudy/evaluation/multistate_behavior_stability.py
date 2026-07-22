"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_stability.py

Bootstrap rank-stability summaries for multistate behavior evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class BootstrapRankStability:
    summary: pd.DataFrame
    draws: pd.DataFrame


def build_bootstrap_rank_stability(
    observed_scores: pd.DataFrame,
    bootstrap_scores: pd.DataFrame,
) -> BootstrapRankStability:
    """Compare every joint-bootstrap unit rank with its central unit rank."""
    observed_required = {"id", "selection_view_id", "behavior_score", "protocol_id"}
    bootstrap_required = {"id", "selection_view_id", "draw_index", "behavior_score", "protocol_id"}
    _require_columns(observed_scores, observed_required, context="observed behavior scores")
    _require_columns(bootstrap_scores, bootstrap_required, context="bootstrap behavior scores")
    if observed_scores.duplicated(subset=["id", "selection_view_id"]).any():
        raise ValueError("observed behavior scores must contain one row per unit and view.")
    if bootstrap_scores.duplicated(subset=["id", "selection_view_id", "draw_index"]).any():
        raise ValueError("bootstrap behavior scores must contain one row per unit, view, and draw.")

    draw_records: list[dict[str, object]] = []
    for view_id, central in observed_scores.groupby("selection_view_id", sort=True):
        central = central.loc[:, ["id", "behavior_score"]].rename(columns={"behavior_score": "central_score"})
        unit_count = len(central)
        view_draws = bootstrap_scores.loc[bootstrap_scores["selection_view_id"].astype(str).eq(str(view_id))]
        for draw_index, draw in view_draws.groupby("draw_index", sort=True):
            aligned = central.merge(draw.loc[:, ["id", "behavior_score"]], on="id", how="inner", validate="one_to_one")
            if len(aligned) != unit_count:
                raise ValueError(f"bootstrap draw {draw_index} candidate ids disagree for view {view_id!r}.")
            draw_records.append(
                {
                    "selection_view_id": str(view_id),
                    "draw_index": int(draw_index),
                    "candidate_experiment_unit_count": unit_count,
                    "central_draw_spearman": _spearman(aligned["central_score"], aligned["behavior_score"]),
                    "ranking_method": "descending_score_then_ascending_candidate_experiment_unit_id",
                    "tie_semantics": "ordinal_rank_with_id_tiebreak",
                    "evidence_role": ("candidate_experiment_unit_rank_stability_no_label_aggregation_or_allocation"),
                }
            )
    draws = pd.DataFrame.from_records(draw_records)
    draws["central_draw_spearman"] = pd.array(draws["central_draw_spearman"], dtype="Float64")
    draws["correlation_defined"] = draws["central_draw_spearman"].notna()
    summaries: list[dict[str, object]] = []
    for view_id, rows in draws.groupby("selection_view_id", sort=True):
        correlations = rows["central_draw_spearman"].dropna().to_numpy(dtype=float)
        defined_count = len(correlations)
        summaries.append(
            {
                "selection_view_id": str(view_id),
                "bootstrap_draw_count": len(rows),
                "candidate_experiment_unit_count": int(rows["candidate_experiment_unit_count"].iloc[0]),
                "correlation_defined_draw_count": defined_count,
                "correlation_undefined_draw_count": len(rows) - defined_count,
                "central_draw_spearman_median": float(np.median(correlations)) if defined_count else None,
                "central_draw_spearman_q05": float(np.quantile(correlations, 0.05)) if defined_count else None,
                "central_draw_spearman_q95": float(np.quantile(correlations, 0.95)) if defined_count else None,
                "ranking_method": "descending_score_then_ascending_candidate_experiment_unit_id",
                "tie_semantics": "ordinal_rank_with_id_tiebreak",
                "evidence_role": "candidate_experiment_unit_rank_stability_no_label_aggregation_or_allocation",
            }
        )
    summary = pd.DataFrame.from_records(summaries)
    for column in (
        "central_draw_spearman_median",
        "central_draw_spearman_q05",
        "central_draw_spearman_q95",
    ):
        summary[column] = pd.array(summary[column], dtype="Float64")
    return BootstrapRankStability(summary=summary, draws=draws)


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    return float(spearmanr(left.to_numpy(dtype=float), right.to_numpy(dtype=float)).statistic)


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{context} missing columns: {missing}")


__all__ = ["BootstrapRankStability", "build_bootstrap_rank_stability"]
