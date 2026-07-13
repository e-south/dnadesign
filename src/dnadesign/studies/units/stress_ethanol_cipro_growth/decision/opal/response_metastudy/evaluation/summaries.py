"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/summaries.py

Policy summary table construction for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..core.contracts import PolicySpec, SfxiEvidenceFrame
from .correlations import target_view_score_correlations
from .overlap import overlap_stats
from .selection import select_top_rows


def summarize_policies(
    policies: Iterable[PolicySpec],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored: dict[str, dict[str, pd.DataFrame]],
    *,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in policies:
        top_by_target_view = {
            evidence.target_view.id: select_top_rows(scored[policy.id][evidence.target_view.id], top_k=top_k)
            for evidence in sfxi_evidence
        }
        overlap = overlap_stats(top_by_target_view)
        metric_rows = []
        effective_topk_counts = [len(frame) for frame in top_by_target_view.values()]
        for evidence in sfxi_evidence:
            label = evidence.target_view.id
            top = top_by_target_view[label]
            eligible_count = int(scored[policy.id][label]["eligible"].sum())
            metric_rows.append(
                {
                    "selection_view_id": label,
                    "topk_median_logic": _nan_if_empty(top, "logic_fidelity", "median"),
                    "topk_min_logic": _nan_if_empty(top, "logic_fidelity", "min"),
                    "topk_median_effect": _nan_if_empty(top, "effect_scaled", "median"),
                    "topk_median_off_state_logic": _nan_if_empty(
                        top,
                        "off_state_logic_level",
                        "median",
                    ),
                    "eligible_count": eligible_count,
                }
            )
        score_corr = target_view_score_correlations(sfxi_evidence, scored[policy.id])
        rows.append(
            {
                "policy_id": policy.id,
                "label": policy.label,
                "tier": policy.tier,
                "kind": policy.kind,
                "beta": float(policy.beta),
                "gamma": float(policy.gamma),
                "logic_gate": policy.logic_gate,
                "off_state_logic_eta": float(policy.off_state_logic_eta),
                "logic_tradeoff_weight": _logic_tradeoff_weight(policy),
                "top_k": int(top_k),
                "total_selected_slots": int(sum(effective_topk_counts)),
                "min_effective_topk": int(min(effective_topk_counts)) if effective_topk_counts else 0,
                "unique_topk": int(overlap["unique_topk"]),
                "all_target_views_overlap": int(overlap["all_target_views_overlap"]),
                "pairwise_overlap_total": int(overlap["pairwise_overlap_total"]),
                "min_target_view_median_logic": _nanmin(row["topk_median_logic"] for row in metric_rows),
                "min_target_view_min_logic": _nanmin(row["topk_min_logic"] for row in metric_rows),
                "mean_topk_effect": _nanmean(row["topk_median_effect"] for row in metric_rows),
                "mean_topk_off_state_logic": _nanmean(row["topk_median_off_state_logic"] for row in metric_rows),
                "min_eligible_count": int(min(row["eligible_count"] for row in metric_rows)),
                "mean_pairwise_score_pearson": float(score_corr["pearson"].mean()),
                "mean_pairwise_score_spearman": float(score_corr["spearman"].mean()),
                "plain_rule": policy.plain_rule,
            }
        )
    return pd.DataFrame(rows).sort_values(["tier", "kind", "policy_id"], kind="mergesort")


def _logic_tradeoff_weight(policy: PolicySpec) -> float:
    if policy.kind != "multiplicative":
        return float("nan")
    total = float(policy.beta + policy.gamma)
    return float(policy.beta / total) if total > 0.0 else float("nan")


def _nan_if_empty(frame: pd.DataFrame, column: str, reducer: str) -> float:
    if frame.empty:
        return float("nan")
    if reducer == "median":
        return float(frame[column].median())
    if reducer == "min":
        return float(frame[column].min())
    raise ValueError(f"Unsupported reducer: {reducer}")


def _nanmin(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))
