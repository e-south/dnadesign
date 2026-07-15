"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/report.py

Markdown report writer for the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.contracts import RecommendationThresholds
from ..core.policies import CANONICAL_SFXI_POLICY_ID, primary_policy_ids
from ..core.response_contracts import ResponseMetricScreen
from .model_validation_report import model_validation_summary_table
from .response_metric_report import response_metric_report_lines


def write_report(
    out_dir: Path,
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    candidates: pd.DataFrame,
    overlap_by_k: pd.DataFrame,
    denominator_sensitivity: pd.DataFrame,
    comparison_panel: pd.DataFrame,
    model_validation: pd.DataFrame,
    setpoint_support: pd.DataFrame,
    response_screen: ResponseMetricScreen,
    pressure_tests: pd.DataFrame,
    plot_manifest: pd.DataFrame,
    recommendation: dict[str, object],
    canonical_sfxi_validation: dict[str, object],
    thresholds: RecommendationThresholds,
    primary_reduction_id: str,
) -> Path:
    report_path = out_dir / "report.md"
    canonical = summary[summary["policy_id"] == CANONICAL_SFXI_POLICY_ID].iloc[0]
    comparison = summary[summary["policy_id"] == recommendation["comparison_policy_id"]].iloc[0]
    primary = summary[summary["policy_id"].isin(primary_policy_ids())].copy()
    primary = primary.sort_values(
        ["min_target_view_median_logic", "all_target_views_overlap", "policy_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    pair_canonical = pairwise[
        (pairwise["policy_id"] == CANONICAL_SFXI_POLICY_ID) & (pairwise["metric"] == "between_selection_views")
    ]
    top_shared = (
        candidates[candidates["policy_id"] == CANONICAL_SFXI_POLICY_ID]
        .query("selection_view_count >= 2")
        .sort_values(["selection_view_count", "id"], ascending=[False, True])
    )
    overlap_canonical = overlap_by_k[
        (overlap_by_k["policy_id"] == CANONICAL_SFXI_POLICY_ID) & (overlap_by_k["overlap_type"] == "all_target_views")
    ]
    panel_summary = (
        comparison_panel.groupby(["panel_role", "selection_view_id"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["panel_role", "selection_view_id"], kind="mergesort")
    )
    denom_focus = denominator_sensitivity[denominator_sensitivity["denominator_factor"].isin((0.5, 1.0, 2.0))]
    model_summary = model_validation_summary_table(model_validation)
    support_at_guardrail = setpoint_support[
        (setpoint_support["logic_threshold"] - thresholds.min_target_view_median_logic).abs() < 1.0e-12
    ]
    text = [
        "# Response Metric Metastudy",
        "",
        "This metastudy tests whether Reader time-series summaries, study target masks, and the configured sequence "
        "representation can support a defensible next-build ranking. Canonical SFXI remains a diagnostic baseline. "
        "The metastudy does not rewrite labels, campaign configs, ledgers, or synthesis handoffs.",
        "",
        "## Premise",
        "",
        "A useful selector must preserve reproducible assay information, respond to the declared target mask, and "
        "retain ordering that transfers to held-out Reader experiments.",
        "",
        "## Minimal Data And Score Path",
        "",
        f"- Reader label candidate: `{primary_reduction_id}`, the 6-12 hour post-stress window mean.",
        "- Response state: `r_i = median-well window mean log2(YFP / CFP)`.",
        "- Anchored fluorescence state: `b_i = median design-well window mean log2(YFP / OD600) "
        "- median same-state pDual-10 well mean`.",
        "- Reader fields: `[r00, r10, r01, r11, b00, b10, b01, b11]` in no-stress, ethanol, "
        "ciprofloxacin, and both-stresses order.",
        "- Each stress target view assigns those fixed states to ON and OFF sets.",
        "- Raw requirements: `m_response = min_ON(r) - max_OFF(r)`, `b_on = min_ON(b)`, and `b_off = max_OFF(b)`.",
        "- Selection channel: `S_feasible = min(z_response, z_on, z_off)` after explicit thresholds and positive "
        "scales are promoted.",
        "",
        "Changing the target mask changes which states enter each minimum or maximum; it does not change the Reader "
        "measurements. Scores order candidates within one target view and are not calibrated for numeric comparison "
        "between target views.",
        "",
        "## SFXI Diagnostic Baseline",
        "",
        "- Documentation: `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`.",
        "- Math helpers: `src/dnadesign/opal/src/objectives/sfxi_math.py`.",
        "- Objective plugin: `src/dnadesign/opal/src/objectives/sfxi_v1.py`.",
        "- State order: `[00, 10, 01, 11]`.",
        "- Score: `logic_fidelity^beta * effect_scaled^gamma`.",
        "- Effect scaling: weighted target-state intensity divided by the run's current-round denominator.",
        "",
        "## Verdict",
        "",
        f"- Recommendation: `{recommendation['verdict']}`.",
        f"- Policy promotion ready: `{recommendation['policy_promotion_ready']}`.",
        f"- Promoted policy: `{recommendation['promoted_policy_id'] or 'none'}`.",
        f"- Shape-ceiling comparison: `{recommendation['comparison_policy_id']}`.",
        f"- Comparison rule: {recommendation['comparison_plain_rule']}",
        f"- Rationale: {recommendation['rationale']}",
        "",
        "Do not synthesize from the canonical SFXI selections. Held-out model support and target-shape coverage are "
        "insufficient, and canonical SFXI remains strongly coupled across target views.",
        "",
        "## Canonical SFXI Scoring Evidence",
        "",
        f"- Recompute max absolute score error: {canonical_sfxi_validation['max_abs_error']:.3g}.",
        f"- Canonical top-6 unique candidate IDs: {int(canonical['unique_topk'])} across 18 slots.",
        f"- Canonical all-target-view overlap: {int(canonical['all_target_views_overlap'])}.",
        f"- Canonical pairwise overlap total: {int(canonical['pairwise_overlap_total'])}.",
        "- Canonical weakest target-view median top-6 logic fidelity: "
        f"{float(canonical['min_target_view_median_logic']):.3f}.",
        "- Canonical mean pairwise score Spearman correlation: "
        f"{float(canonical['mean_pairwise_score_spearman']):.3f}.",
        f"- Canonical minimum effective eligible top-k: {int(canonical['min_effective_topk'])}.",
        "",
        "## Held-Out Model Support",
        "",
        "Shuffled five-fold and leave-one-experiment-out retraining test whether the shared vec8 predictor "
        "preserves observed response ordering. Leave-one-experiment-out results gate policy promotion; "
        "shuffled folds remain a descriptive interpolation check.",
        "Held-out predictions use each SFXI source run's persisted denominator so this tests ordering under the "
        "objective that produced the ledger; no denominator is fitted from held-out labels.",
        "",
        _markdown_table(model_summary),
        "",
        "## Assay Time-Course Robustness",
        "",
        "Reader publishes seven event-relative reductions: a primary 6-12 hour post-event log mean, four declared "
        "window checks, a "
        "duration-normalized linear AUC, and a pre-window-delta check. Response uses log2 YFP/CFP. The magnitude "
        "channel is same-state pDual-10-relative log2 YFP/OD600 fluorescence.",
        "",
        "Reader owns event resolution, trajectory reduction, replicate aggregation, and joint bootstrap records. "
        "The study consumes those records and applies target-view masks; it does not reopen PlateReader trajectories.",
        "",
        f"- Reader primary reduction: `{primary_reduction_id}`. It is the only promotion candidate; the other "
        "windows, AUC, and pre-window delta remain equal-footing sensitivity analyses.",
        "- Response reductions are evaluated only through response and fluorescence requirements. They are not "
        "translated into SFXI logic or intensity fields.",
        "- The SFXI source records remain independent evidence and are evaluated only under their declared vec8 "
        "contract.",
        "",
        *response_metric_report_lines(response_screen, primary_reduction_id=primary_reduction_id),
        "## Activation Boundary",
        "",
        "`response_magnitude_feasibility_v1` is implemented but inactive. The Reader response-window bundle now "
        "provides raw four-state response, reference-relative magnitude, event bounds, and joint bootstrap draws. "
        "Activation still requires one candidate-level repeat aggregation rule and an explicit OPAL label/promotion "
        "contract. SFXI vec8 labels use a distinct metric contract and are not accepted for RMF activation. OR "
        "remains a pressure-test mask, not a configured campaign or synthesis allocation.",
        "",
        "## Predicted Setpoint Support",
        "",
        "This table asks whether the fitted SFXI predictor produces enough candidate response shapes near each "
        "setpoint. "
        "A selection rule cannot recover shapes absent from this surface.",
        "",
        _markdown_table(
            support_at_guardrail[
                [
                    "selection_view_id",
                    "logic_threshold",
                    "candidate_count",
                    "max_logic_fidelity",
                    "p99_logic_fidelity",
                ]
            ]
        ),
        "",
        "## Shape-Ceiling Comparison",
        "",
        f"- Comparison top-6 unique candidate IDs: {int(comparison['unique_topk'])} across 18 slots.",
        f"- Comparison all-target-view overlap: {int(comparison['all_target_views_overlap'])}.",
        f"- Comparison pairwise overlap total: {int(comparison['pairwise_overlap_total'])}.",
        "- Comparison weakest target-view median top-6 logic fidelity: "
        f"{float(comparison['min_target_view_median_logic']):.3f}.",
        "- Comparison mean pairwise score Spearman correlation: "
        f"{float(comparison['mean_pairwise_score_spearman']):.3f}.",
        f"- Comparison minimum effective eligible top-k: {int(comparison['min_effective_topk'])}.",
        "",
        "## Policy Comparison Candidate Panel",
        "",
        "`policy_comparison_panel.csv` is a metric-behavior comparison, not a synthesis handoff. It mixes "
        "canonical SFXI high-effect rows, shape-ceiling comparison rows, logic-first rows, OFF-state-penalized "
        "rows, canonical SFXI shared-overlap rows, and target-view-specific provisional rows.",
        "",
        _markdown_table(panel_summary),
        "",
        "## Denominator Sensitivity",
        "",
        "This probe rescales the SFXI denominator over predicted effect raw values and recomputes top-k summaries. "
        "It checks whether intensity scaling is acting as a hidden driver of candidate choice.",
        "",
        _markdown_table(
            denom_focus[
                [
                    "policy_id",
                    "selection_view_id",
                    "denominator_factor",
                    "effective_topk",
                    "median_logic_fidelity",
                    "median_effect_scaled",
                    "median_off_state_logic_level",
                ]
            ]
        ),
        "",
        "## Review Guardrails",
        "",
        f"- Minimum eligible candidates in weakest target view: {thresholds.min_eligible_count}.",
        f"- Minimum effective top-k in each target view: {thresholds.min_effective_topk}.",
        f"- Minimum weakest-target-view median top-k logic fidelity: {thresholds.min_target_view_median_logic:.2f}.",
        f"- Maximum all-target-view top-k overlap: {thresholds.max_all_target_views_overlap}.",
        f"- Maximum mean pairwise score Spearman correlation: {thresholds.max_mean_pairwise_score_spearman:.2f}.",
        "- Minimum weakest-target-view median held-out score Spearman correlation: "
        f"{thresholds.min_target_view_cv_score_spearman:.2f}.",
        "",
        "These are review guardrails for metric design. They are not biological thresholds.",
        "",
        "## Primary Policy Summary",
        "",
        _markdown_table(
            primary[
                [
                    "policy_id",
                    "min_effective_topk",
                    "unique_topk",
                    "all_target_views_overlap",
                    "pairwise_overlap_total",
                    "min_target_view_median_logic",
                    "mean_topk_effect",
                    "mean_pairwise_score_spearman",
                ]
            ]
        ),
        "",
        "## Adversarial Pressure Tests",
        "",
        _markdown_table(
            pressure_tests[
                [
                    "agent",
                    "check_id",
                    "status",
                    "severity",
                    "evidence",
                    "threshold",
                    "action",
                ]
            ]
        ),
        "",
        "## Canonical SFXI Pairwise Correlations",
        "",
        _markdown_table(pair_canonical[["selection_view_a", "selection_view_b", "pearson", "spearman"]]),
        "",
        "## Canonical SFXI Overlap By K",
        "",
        _markdown_table(overlap_canonical[["k", "observed_overlap", "unique_topk"]]),
        "",
        "## Canonical SFXI Shared Top Candidates",
        "",
        _markdown_table(
            top_shared[
                ["id", "selection_view_id", "rank", "logic_fidelity", "effect_scaled", "selection_view_count"]
            ].head(18)
        ),
        "",
        "## Figure Premises And Alt Text",
        "",
        _plot_manifest_tables(plot_manifest),
        "",
        "## Guardrails",
        "",
        "- Treat this as metric design over predictions, not biological validation.",
        "- Separate scalarizer changes from downstream diversity policy.",
        "- Avoid thresholds that only work at one exact value; prefer regions stable across nearby gates or exponents.",
        "- Require a rerun of OPAL or a formal rerank step before any measured-round synthesis handoff.",
        "- Keep the nearest-12-hour snapshot as provenance after any response-window label is promoted.",
        "",
    ]
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rows = [[str(column) for column in frame.columns]]
    for _, row in frame.iterrows():
        rows.append([_format_cell(row[column]) for column in frame.columns])
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    header = "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def _plot_manifest_tables(plot_manifest: pd.DataFrame) -> str:
    if plot_manifest.empty:
        return "_No plots rendered._"
    sections: list[str] = []
    for tier in ("primary_decision", "metric_diagnostic", "screen_appendix"):
        tier_rows = plot_manifest[plot_manifest["tier"] == tier]
        if tier_rows.empty:
            continue
        sections.append(f"### {tier.replace('_', ' ').title()}")
        sections.append(
            _markdown_table(
                tier_rows[
                    [
                        "plot_id",
                        "visual_type",
                        "premise",
                        "decision_value",
                        "rationale",
                        "alt_text",
                        "non_claim_boundary",
                    ]
                ]
            )
        )
        sections.append("")
    return "\n".join(sections).rstrip()


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
