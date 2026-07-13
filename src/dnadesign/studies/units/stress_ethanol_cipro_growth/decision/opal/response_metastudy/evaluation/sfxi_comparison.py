"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/sfxi_comparison.py

Canonical SFXI comparisons over Reader-owned response-window records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dnadesign.opal import SFXIScoringConfig, score_vec8

from ..core.contracts import StressTargetView

_VEC8_COLUMNS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")


def build_sfxi_comparison_rows(
    summaries: pd.DataFrame,
    *,
    target_views: tuple[StressTargetView, ...],
    logic_threshold: float,
    scaling_percentile: int = 95,
    scaling_min_n: int = 5,
    scaling_eps: float = 1.0e-8,
    intensity_log2_offset_delta: float = 0.0,
) -> pd.DataFrame:
    """Score each measured assay summary with canonical SFXI decomposition."""

    required = {"id", "design_id", "assay_summary_id", "assay_summary_method", *_VEC8_COLUMNS}
    missing = sorted(required - set(summaries.columns))
    if missing:
        raise ValueError(f"assay summary score table missing columns: {missing}")
    if not np.isfinite(logic_threshold) or not 0.0 <= logic_threshold <= 1.0:
        raise ValueError("logic_threshold must be finite and in [0, 1].")
    ids_by_summary: dict[str, tuple[str, ...]] = {}
    for summary_id, frame in summaries.groupby("assay_summary_id", sort=True):
        ids = frame["id"].astype(str)
        if ids.duplicated().any():
            raise ValueError(f"assay summary {summary_id!r} contains duplicate candidate ids.")
        ids_by_summary[str(summary_id)] = tuple(sorted(ids))
    expected_ids = next(iter(ids_by_summary.values()), ())
    if not expected_ids or any(ids != expected_ids for ids in ids_by_summary.values()):
        raise ValueError("assay summaries must contain one identical candidate-id universe.")

    rows: list[dict[str, object]] = []
    for summary_id, frame in summaries.groupby("assay_summary_id", sort=True):
        ordered = frame.copy()
        ordered["id"] = ordered["id"].astype(str)
        ordered = ordered.sort_values("id", kind="mergesort").reset_index(drop=True)
        methods = ordered["assay_summary_method"].astype(str).unique().tolist()
        if len(methods) != 1:
            raise ValueError(f"assay summary {summary_id!r} must declare one method.")
        vec8 = ordered.loc[:, list(_VEC8_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(vec8).all():
            raise ValueError(f"assay summary {summary_id!r} contains non-finite vec8 values.")
        for target_view in target_views:
            result = score_vec8(
                vec8,
                SFXIScoringConfig(
                    setpoint_vector=target_view.target_mask,
                    scaling_percentile=scaling_percentile,
                    scaling_min_n=scaling_min_n,
                    scaling_eps=scaling_eps,
                    intensity_log2_offset_delta=intensity_log2_offset_delta,
                ),
            )
            for index, source_row in ordered.iterrows():
                rows.append(
                    {
                        "id": str(source_row["id"]),
                        "design_id": str(source_row["design_id"]),
                        "assay_summary_id": str(summary_id),
                        "assay_summary_method": methods[0],
                        "selection_view_id": target_view.id,
                        "logic_fidelity": float(result.logic_fidelity[index]),
                        "effect_scaled": float(result.effect_scaled[index]),
                        "sfxi": float(result.sfxi[index]),
                        "denom_used": float(result.denom_used),
                        "passes_logic_threshold": bool(result.logic_fidelity[index] >= logic_threshold),
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_sfxi_comparison(
    metric_rows: pd.DataFrame,
    *,
    baseline_summary_id: str,
) -> pd.DataFrame:
    """Compare each time-window score decomposition with the canonical snapshot."""

    required = {
        "id",
        "assay_summary_id",
        "assay_summary_method",
        "selection_view_id",
        "logic_fidelity",
        "effect_scaled",
        "sfxi",
        "denom_used",
        "passes_logic_threshold",
    }
    missing = sorted(required - set(metric_rows.columns))
    if missing:
        raise ValueError(f"assay metric rows missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for selection_view_id, view_rows in metric_rows.groupby("selection_view_id", sort=True):
        baseline = view_rows.loc[view_rows["assay_summary_id"].astype(str).eq(baseline_summary_id)].copy()
        if baseline.empty:
            raise ValueError(
                f"assay metrics missing baseline {baseline_summary_id!r} for selection view {selection_view_id!r}."
            )
        baseline = baseline.set_index("id")
        for summary_id, summary in view_rows.groupby("assay_summary_id", sort=True):
            methods = summary["assay_summary_method"].astype(str).unique().tolist()
            if len(methods) != 1:
                raise ValueError(f"assay summary {summary_id!r} must declare one method.")
            aligned = summary.set_index("id").loc[baseline.index]
            score_spearman = _spearman(baseline["sfxi"], aligned["sfxi"])
            rows.append(
                {
                    "assay_summary_id": str(summary_id),
                    "assay_summary_method": methods[0],
                    "selection_view_id": str(selection_view_id),
                    "n": int(len(aligned)),
                    "score_spearman_to_snapshot": score_spearman,
                    "logic_spearman_to_snapshot": _spearman(baseline["logic_fidelity"], aligned["logic_fidelity"]),
                    "effect_spearman_to_snapshot": _spearman(baseline["effect_scaled"], aligned["effect_scaled"]),
                    "correlation_defined": bool(np.isfinite(score_spearman)),
                    "logic_support_count": int(aligned["passes_logic_threshold"].astype(bool).sum()),
                    "median_logic_fidelity": float(aligned["logic_fidelity"].median()),
                    "median_effect_scaled": float(aligned["effect_scaled"].median()),
                    "denom_used": float(aligned["denom_used"].iloc[0]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if np.ptp(left_values) == 0.0 or np.ptp(right_values) == 0.0:
        return float("nan")
    return float(spearmanr(left_values, right_values).statistic)


__all__ = ["build_sfxi_comparison_rows", "summarize_sfxi_comparison"]
