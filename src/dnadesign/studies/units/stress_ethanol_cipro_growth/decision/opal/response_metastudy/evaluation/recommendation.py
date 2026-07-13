"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/recommendation.py

Recommendation guardrails for SFXI policy review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.contracts import DEFAULT_RECOMMENDATION_THRESHOLDS, RecommendationThresholds
from ..core.policies import CANONICAL_SFXI_POLICY_ID


def choose_recommendation(
    summary: pd.DataFrame,
    *,
    thresholds: RecommendationThresholds = DEFAULT_RECOMMENDATION_THRESHOLDS,
    model_validation_summary: dict[str, object],
) -> dict[str, object]:
    candidates = summary[summary["tier"].isin(["candidate", "sweep"])].copy()
    candidates = candidates[
        (candidates["min_eligible_count"] >= thresholds.min_eligible_count)
        & (candidates["min_effective_topk"] >= thresholds.min_effective_topk)
        & (candidates["min_target_view_median_logic"] >= thresholds.min_target_view_median_logic)
        & (candidates["all_target_views_overlap"] <= thresholds.max_all_target_views_overlap)
        & (candidates["mean_pairwise_score_spearman"] <= thresholds.max_mean_pairwise_score_spearman)
    ]
    metric_guardrails_passed = not candidates.empty
    model_value = float(model_validation_summary["weakest_target_view_median_score_spearman"])
    model_support_passed = bool(
        np.isfinite(model_value) and model_value >= thresholds.min_target_view_cv_score_spearman
    )
    promoted = None
    if metric_guardrails_passed and model_support_passed:
        promoted = candidates.sort_values(
            [
                "all_target_views_overlap",
                "pairwise_overlap_total",
                "min_target_view_median_logic",
                "mean_topk_effect",
            ],
            ascending=[True, True, False, False],
            kind="mergesort",
        ).iloc[0]
        verdict = "policy_candidate_for_calibration_review"
        rationale = (
            "One policy clears the metric and held-out model review guardrails. It remains a calibration-review "
            "candidate, not biological validation or an automatic synthesis authorization."
        )
    elif not model_support_passed:
        verdict = "do_not_promote_policy"
        rationale = (
            "Held-out vec8 predictions do not preserve target-view score ordering well enough to justify policy "
            "promotion."
        )
    else:
        verdict = "do_not_promote_policy"
        rationale = "No policy clears the target-fidelity, overlap, eligibility, and rank-coupling review guardrails."

    comparison_pool = summary[
        (summary["min_eligible_count"] >= thresholds.min_eligible_count)
        & (summary["min_effective_topk"] >= thresholds.min_effective_topk)
        & np.isfinite(summary["min_target_view_median_logic"])
    ]
    if comparison_pool.empty:
        raise ValueError("No policy has finite logic fidelity and a complete top-k for diagnostic comparison.")
    comparison = comparison_pool.sort_values(
        [
            "min_target_view_median_logic",
            "mean_topk_effect",
            "all_target_views_overlap",
            "pairwise_overlap_total",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    canonical = summary[summary["policy_id"] == CANONICAL_SFXI_POLICY_ID].iloc[0]
    return {
        "verdict": verdict,
        "policy_promotion_ready": promoted is not None,
        "promoted_policy_id": str(promoted["policy_id"]) if promoted is not None else None,
        "promoted_policy_label": str(promoted["label"]) if promoted is not None else None,
        "promoted_plain_rule": str(promoted["plain_rule"]) if promoted is not None else None,
        "comparison_policy_id": str(comparison["policy_id"]),
        "comparison_policy_label": str(comparison["label"]),
        "comparison_plain_rule": str(comparison["plain_rule"]),
        "metric_guardrails_passed": metric_guardrails_passed,
        "model_support_passed": model_support_passed,
        "rationale": rationale,
        "canonical_sfxi_policy": {
            "unique_topk": int(canonical["unique_topk"]),
            "all_target_views_overlap": int(canonical["all_target_views_overlap"]),
            "pairwise_overlap_total": int(canonical["pairwise_overlap_total"]),
            "min_effective_topk": int(canonical["min_effective_topk"]),
            "min_eligible_count": int(canonical["min_eligible_count"]),
            "min_target_view_median_logic": float(canonical["min_target_view_median_logic"]),
            "mean_pairwise_score_spearman": float(canonical["mean_pairwise_score_spearman"]),
        },
        "comparison_policy": {
            "unique_topk": int(comparison["unique_topk"]),
            "all_target_views_overlap": int(comparison["all_target_views_overlap"]),
            "pairwise_overlap_total": int(comparison["pairwise_overlap_total"]),
            "min_effective_topk": int(comparison["min_effective_topk"]),
            "min_eligible_count": int(comparison["min_eligible_count"]),
            "min_target_view_median_logic": float(comparison["min_target_view_median_logic"]),
            "mean_pairwise_score_spearman": float(comparison["mean_pairwise_score_spearman"]),
        },
        "thresholds": {
            "min_eligible_count": int(thresholds.min_eligible_count),
            "min_effective_topk": int(thresholds.min_effective_topk),
            "min_target_view_median_logic": float(thresholds.min_target_view_median_logic),
            "max_all_target_views_overlap": int(thresholds.max_all_target_views_overlap),
            "max_mean_pairwise_score_spearman": float(thresholds.max_mean_pairwise_score_spearman),
            "min_target_view_cv_score_spearman": float(thresholds.min_target_view_cv_score_spearman),
        },
    }
