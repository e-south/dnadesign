"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/pressure_tests.py

Assemble adversarial pressure tests for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from ..core.contracts import RecommendationThresholds
from ..core.policies import CANONICAL_SFXI_POLICY_ID
from .pressure_rows import (
    effect_dominance_rows,
    model_support_row,
    pressure_row,
    setpoint_support_rows,
    upper_bound_status,
)


def build_pressure_tests(
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    canonical_sfxi_validation: dict[str, object],
    recommendation: dict[str, object],
    thresholds: RecommendationThresholds,
    model_validation_summary: dict[str, object],
    setpoint_support: pd.DataFrame,
    intrinsic_tests: pd.DataFrame,
) -> pd.DataFrame:
    canonical = summary[summary["policy_id"] == CANONICAL_SFXI_POLICY_ID].iloc[0]
    comparison = summary[summary["policy_id"] == recommendation["comparison_policy_id"]].iloc[0]
    rows = [
        model_support_row(
            model_validation_summary,
            minimum=thresholds.min_target_view_cv_score_spearman,
        ),
        pressure_row(
            agent="correctness",
            check_id="canonical_sfxi_score_recompute",
            status="pass" if bool(canonical_sfxi_validation["matches_canonical_ledger"]) else "fail",
            severity="blocker",
            premise="Canonical SFXI scoring must reproduce the persisted OPAL ledger before comparisons matter.",
            evidence=f"max_abs_error={float(canonical_sfxi_validation['max_abs_error']):.3g}",
            threshold="max_abs_error <= 1e-12",
            interpretation="Exact recomputation rules out a config-copy explanation for target-view collapse.",
            action="Keep this as a required guard for every metastudy run.",
        ),
        pressure_row(
            agent="metric",
            check_id="canonical_sfxi_topk_collapse",
            status=(
                "fail"
                if int(canonical["all_target_views_overlap"]) > thresholds.max_all_target_views_overlap
                else "pass"
            ),
            severity="high",
            premise=(
                "Different setpoints should not share top candidates unless those candidates satisfy each setpoint."
            ),
            evidence=(
                f"unique_topk={int(canonical['unique_topk'])}; "
                f"all_target_views_overlap={int(canonical['all_target_views_overlap'])}; "
                f"pairwise_overlap_total={int(canonical['pairwise_overlap_total'])}"
            ),
            threshold=f"all_target_views_overlap <= {thresholds.max_all_target_views_overlap}",
            interpretation="Canonical SFXI is too coupled for a distinct-target-view synthesis claim.",
            action="Do not synthesize from the canonical SFXI top-k.",
        ),
        pressure_row(
            agent="metric",
            check_id="canonical_sfxi_logic_guardrail",
            status=(
                "fail"
                if float(canonical["min_target_view_median_logic"]) < thresholds.min_target_view_median_logic
                else "pass"
            ),
            severity="high",
            premise="Selected candidates need enough target-shape fidelity for setpoint-directed selection.",
            evidence=f"min_target_view_median_logic={float(canonical['min_target_view_median_logic']):.3f}",
            threshold=f">= {thresholds.min_target_view_median_logic:.2f}",
            interpretation="The canonical SFXI top-k has weak response-shape fidelity in at least one target view.",
            action="Treat SFXI tuning as metric review before any measured-round handoff.",
        ),
        pressure_row(
            agent="metric",
            check_id="canonical_sfxi_score_coupling",
            status=upper_bound_status(
                float(canonical["mean_pairwise_score_spearman"]),
                thresholds.max_mean_pairwise_score_spearman,
            ),
            severity="high",
            premise="Target-view rank surfaces should not be nearly identical.",
            evidence=f"mean_pairwise_score_spearman={float(canonical['mean_pairwise_score_spearman']):.3f}",
            threshold=f"<= {thresholds.max_mean_pairwise_score_spearman:.2f}",
            interpretation="Canonical SFXI target-view ranks are strongly coupled.",
            action="Require lower coupling together with target-shape fidelity; do not optimize overlap alone.",
        ),
    ]
    rows.extend(effect_dominance_rows(pairwise))
    rows.extend(
        setpoint_support_rows(
            setpoint_support,
            logic_threshold=thresholds.min_target_view_median_logic,
            minimum_count=thresholds.min_effective_topk,
        )
    )
    rows.extend(_comparison_and_claim_rows(comparison, recommendation, thresholds))
    pressure = pd.DataFrame(rows)
    if set(intrinsic_tests.columns) != set(pressure.columns):
        raise ValueError("intrinsic metric tests do not match the pressure-test schema.")
    return pd.concat([intrinsic_tests[pressure.columns], pressure], ignore_index=True)


def _comparison_and_claim_rows(
    comparison: pd.Series,
    recommendation: dict[str, object],
    thresholds: RecommendationThresholds,
) -> list[dict[str, object]]:
    return [
        pressure_row(
            agent="metric",
            check_id="comparison_policy_logic_guardrail",
            status=(
                "fail"
                if float(comparison["min_target_view_median_logic"]) < thresholds.min_target_view_median_logic
                else "pass"
            ),
            severity="medium",
            premise="A shape-ceiling comparison should clear the same logic guardrail used for canonical SFXI.",
            evidence=(
                f"policy_id={comparison['policy_id']}; "
                f"min_target_view_median_logic={float(comparison['min_target_view_median_logic']):.3f}; "
                f"unique_topk={int(comparison['unique_topk'])}; "
                f"all_target_views_overlap={int(comparison['all_target_views_overlap'])}"
            ),
            threshold=f"min_target_view_median_logic >= {thresholds.min_target_view_median_logic:.2f}",
            interpretation="The shape-ceiling comparison still lacks strong target-shape fidelity.",
            action="Use it as a diagnostic comparison, not as a policy recommendation.",
        ),
        pressure_row(
            agent="selection_contract",
            check_id="comparison_policy_effective_topk",
            status="pass" if int(comparison["min_effective_topk"]) >= thresholds.min_effective_topk else "fail",
            severity="medium",
            premise="A policy comparison is usable only when every target view can produce a full eligible top-k.",
            evidence=(
                f"policy_id={comparison['policy_id']}; "
                f"min_effective_topk={int(comparison['min_effective_topk'])}; "
                f"min_eligible_count={int(comparison['min_eligible_count'])}"
            ),
            threshold=f"min_effective_topk >= {thresholds.min_effective_topk}",
            interpretation="The comparison has enough rows; stricter gates may not.",
            action="Keep effective top-k and eligible-count columns in every policy summary.",
        ),
        pressure_row(
            agent="manuscript_reviewer",
            check_id="claim_boundary",
            status="attention",
            severity="medium",
            premise="Predicted OPAL scores support a decision audit, not a biological-response claim.",
            evidence=(
                f"verdict={recommendation['verdict']}; "
                f"policy_promotion_ready={recommendation['policy_promotion_ready']}"
            ),
            threshold="biological claims require measured follow-up labels",
            interpretation="The evidence supports metric review, not successful stress-responsive promoters.",
            action="Use 'predicted next-build candidates' until measured validation exists.",
        ),
        pressure_row(
            agent="information_architecture",
            check_id="threshold_semantics",
            status="attention",
            severity="medium",
            premise="Review guardrails are not biological laws.",
            evidence=(
                f"min_logic={thresholds.min_target_view_median_logic}; "
                f"max_overlap={thresholds.max_all_target_views_overlap}; "
                f"max_rank_coupling={thresholds.max_mean_pairwise_score_spearman}; "
                f"min_cv_score_spearman={thresholds.min_target_view_cv_score_spearman}"
            ),
            threshold="documented as review guardrails",
            interpretation="These thresholds remain provisional until measured-round calibration.",
            action="Keep thresholds centralized and visible in the manifest and report.",
        ),
        pressure_row(
            agent="information_architecture",
            check_id="mutation_boundary",
            status="pass",
            severity="low",
            premise="Metric review must not mutate OPAL state or synthesis handoffs.",
            evidence="outputs are written under workbench/outputs/response_metastudy/latest",
            threshold="generated workbench output only",
            interpretation="The metastudy remains study-owned and read-only.",
            action="Keep generated outputs ignored unless explicitly promoted.",
        ),
    ]
