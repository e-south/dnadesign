"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_recommendation.py

Recommendation-contract tests for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    recommendation,
)


def test_metric_passing_policy_is_not_promoted_when_model_support_fails() -> None:
    summary = pd.DataFrame(
        [
            _policy_row("sfxi_beta1_gamma1", tier="canonical", logic=0.25, spearman=0.95),
            _policy_row("candidate", tier="candidate", logic=0.60, spearman=0.50),
        ]
    )

    result = recommendation.choose_recommendation(
        summary,
        model_validation_summary={"weakest_target_view_median_score_spearman": 0.10},
    )

    assert result["policy_promotion_ready"] is False
    assert result["promoted_policy_id"] is None
    assert result["metric_guardrails_passed"] is True
    assert result["model_support_passed"] is False


def _policy_row(policy_id: str, *, tier: str, logic: float, spearman: float) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "label": policy_id,
        "plain_rule": "test rule",
        "tier": tier,
        "min_eligible_count": 2000,
        "min_effective_topk": 6,
        "min_target_view_median_logic": logic,
        "all_target_views_overlap": 0,
        "pairwise_overlap_total": 0,
        "mean_pairwise_score_pearson": spearman,
        "mean_pairwise_score_spearman": spearman,
        "mean_topk_effect": 0.5,
        "unique_topk": 18,
    }
