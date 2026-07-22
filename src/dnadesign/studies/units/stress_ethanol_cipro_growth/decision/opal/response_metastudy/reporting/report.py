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
from .plot_vocabulary import representation_label
from .response_metric_report import response_metric_report_sections

EVIDENCE_SECTION_ORDER = (
    "Assay and label evidence",
    "Model screens",
    "RMF comparator",
    "SFXI comparator",
)


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
    observed_sfxi_components: pd.DataFrame,
    observed_sfxi_robustness: pd.DataFrame,
    sfxi_greedy_replay: pd.DataFrame,
    response_screen: ResponseMetricScreen,
    pressure_tests: pd.DataFrame,
    plot_manifest: pd.DataFrame,
    recommendation: dict[str, object],
    canonical_sfxi_validation: dict[str, object],
    thresholds: RecommendationThresholds,
    primary_reduction_id: str,
    label_truth_ready: bool,
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
    observed_sfxi_summary, observed_sfxi_sensitivity = _observed_sfxi_report_evidence(
        observed_sfxi_components,
        observed_sfxi_robustness,
    )
    sfxi_greedy_summary, sfxi_greedy_overlap = _sfxi_greedy_report_evidence(sfxi_greedy_replay)
    primary_reduction_label = representation_label(primary_reduction_id).replace("\n", " ")
    label_truth_summary = (
        "The configured observed-label publication is verified."
        if label_truth_ready
        else "The configured observed-label publication is not yet available."
    )
    response_sections = response_metric_report_sections(
        response_screen,
        primary_reduction_id=primary_reduction_id,
    )
    text = [
        "# Response Assay and Objective Comparison",
        "",
        "This frozen evidence package records assay development, label construction, model screens, "
        "and objective comparisons. It does not contain active campaign state or choose a synthesis batch.",
        "",
        "## Premise",
        "",
        "A response-based campaign needs a reproducible assay summary, candidate-level label provenance, and "
        "evidence showing how well sequence predictions preserve measured ordering.",
        "",
        "## Current route and decision boundary",
        "",
        "The executable route is Reader response-window evidence → study-owned observed labels → the MSRB "
        "protocol → the OPAL campaign selected by "
        "`docs/studies/stress_ethanol_cipro_growth/record/campaign.yaml`. This package supplies assay-development "
        "and comparator evidence only. It does not authorize synthesis.",
        "",
        f"{label_truth_summary} Predictor support remains weak, and prospective hill climbing has not been shown.",
        "",
        "## Assay and label evidence",
        "",
        f"- Reader primary reduction: `{primary_reduction_id}`, the {primary_reduction_label}.",
        "- Response state: `r_i = median-well window mean log2(YFP / CFP)`.",
        "- Referenced signal: `b_i = median design-well window mean log2(YFP / OD600) "
        "- median same-state pDual-10 well mean`.",
        "- Four-state phenotype: `[r00, r10, r01, r11, b00, b10, b01, b11]` in no-stress, ethanol, "
        "ciprofloxacin, and both-stresses order.",
        "- Study-owned target views assign those fixed states to intended-ON and intended-OFF sets.",
        "- Objective-specific transforms do not change the Reader phenotype.",
        "",
        *response_sections.assay_and_labels,
        "## Model screens",
        "",
        "These retrospective screens evaluate earlier response-window/RMF and SFXI predictor formulations. They "
        "do not validate the active MSRB selector; current MSRB evidence is maintained with its study protocol.",
        "",
        *response_sections.historical_model_screens,
        "### SFXI vec8 screen",
        "",
        "Shuffled five-fold and leave-one-experiment-out retraining test whether the retained shared SFXI vec8 "
        "predictor preserves observed ordering. Leave-one-experiment-out results governed the SFXI "
        "promotion decision; shuffled folds are descriptive interpolation checks.",
        "",
        _markdown_table(model_summary),
        "",
        "### SFXI setpoint support",
        "",
        "This table asks whether the fitted SFXI predictor produced candidate shapes near each declared setpoint. "
        "It is not an MSRB support table.",
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
        *response_sections.rmf_comparator,
        "The RMF implementation and its retained screens remain comparator evidence. The active campaign uses "
        "`multistate_response_behavior_v1`; SFXI vec8 labels remain a separate assay contract.",
        "",
        "## SFXI comparator",
        "",
        "SFXI evaluates a distinct Reader vec8 contract. Its analyses remain useful for explaining why target-view "
        "selections collapsed, but they do not supply response-window labels or the active selector.",
        "",
        "### Definition used in this package",
        "",
        "- Documentation: `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`.",
        "- State order: `[00, 10, 01, 11]`.",
        "- Score: `logic_fidelity^beta * effect_scaled^gamma`.",
        "- Effect scaling: weighted target-state intensity divided by the source run's denominator.",
        "",
        "### SFXI greedy selection replay",
        "",
        "The same 35-label random forest scored the same candidate pool under each target view, then selected the "
        "six highest SFXI scores. The replay uses those recorded predictions and selections; it does not fit a new "
        "model.",
        "",
        "Spearman's rank correlation (rho) compares ordering, not numerical accuracy. Values near +1 mean the "
        "SFXI rank follows that component, values near 0 indicate little consistent agreement, and negative "
        "values mean candidates tend to move in opposite rank directions.",
        "",
        _markdown_table(sfxi_greedy_summary),
        "",
        sfxi_greedy_overlap,
        "",
        "The fitted SFXI score followed scaled effect much more closely than logic fidelity in every view. The "
        "substantial candidate reuse shows that changing the target setpoint did not produce three well-separated "
        "selection lists. This supports the study-specific conclusion that canonical SFXI was poorly aligned with "
        "the desired combination of target pattern, ON signal, and OFF suppression.",
        "",
        "### Observed-label decomposition",
        "",
        "This replay evaluates the 35 SFXI vec8 labels under each persisted target view. It does not "
        "translate the active response-window phenotype into SFXI coordinates. Canonical SFXI is the product of "
        "logic fidelity and scaled target-state effect, so the component associations show which factor most "
        "closely followed the measured score ranks in this corpus.",
        "",
        _markdown_table(observed_sfxi_summary),
        "",
        observed_sfxi_sensitivity,
        "",
        "These are corpus-sensitivity checks, not cross-validation or evidence that SFXI is universally "
        "effect-dominated.",
        "",
        "### SFXI verdict",
        "",
        f"- Recommendation: `{recommendation['verdict']}`.",
        f"- Policy promotion ready: `{recommendation['policy_promotion_ready']}`.",
        f"- Promoted policy: `{recommendation['promoted_policy_id'] or 'none'}`.",
        f"- Shape-ceiling comparison: `{recommendation['comparison_policy_id']}`.",
        f"- Comparison rule: {recommendation['comparison_plain_rule']}",
        f"- Rationale: {recommendation['rationale']}",
        "",
        "The canonical SFXI selections were not authorized for synthesis because held-out support and target-shape "
        "coverage were insufficient and selections were strongly coupled across target views.",
        "",
        "### Canonical SFXI evidence",
        "",
        f"- Recompute max absolute score error: {canonical_sfxi_validation['max_abs_error']:.3g}.",
        f"- Unique candidate IDs: {int(canonical['unique_topk'])} across 18 top-six slots.",
        f"- All-view overlap: {int(canonical['all_target_views_overlap'])}.",
        f"- Pairwise overlap total: {int(canonical['pairwise_overlap_total'])}.",
        f"- Weakest-view median top-six logic fidelity: {float(canonical['min_target_view_median_logic']):.3f}.",
        f"- Mean pairwise score Spearman correlation: {float(canonical['mean_pairwise_score_spearman']):.3f}.",
        f"- Minimum effective eligible top-k: {int(canonical['min_effective_topk'])}.",
        "",
        "### Shape-ceiling comparison",
        "",
        f"- Unique candidate IDs: {int(comparison['unique_topk'])} across 18 top-six slots.",
        f"- All-view overlap: {int(comparison['all_target_views_overlap'])}.",
        f"- Pairwise overlap total: {int(comparison['pairwise_overlap_total'])}.",
        f"- Weakest-view median top-six logic fidelity: {float(comparison['min_target_view_median_logic']):.3f}.",
        f"- Mean pairwise score Spearman correlation: {float(comparison['mean_pairwise_score_spearman']):.3f}.",
        f"- Minimum effective eligible top-k: {int(comparison['min_effective_topk'])}.",
        "",
        "### Candidate panel",
        "",
        "`policy_comparison_panel.csv` compares SFXI scoring behavior; it is not a synthesis handoff.",
        "",
        _markdown_table(panel_summary),
        "",
        "### Denominator sensitivity",
        "",
        "This probe rescales the SFXI denominator and recomputes top-k summaries to test whether "
        "intensity scaling drives candidate choice.",
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
        "### Review guardrails",
        "",
        f"- Minimum eligible candidates in weakest target view: {thresholds.min_eligible_count}.",
        f"- Minimum effective top-k in each target view: {thresholds.min_effective_topk}.",
        f"- Minimum weakest-view median top-k logic fidelity: {thresholds.min_target_view_median_logic:.2f}.",
        f"- Maximum all-view top-k overlap: {thresholds.max_all_target_views_overlap}.",
        f"- Maximum mean pairwise score Spearman correlation: {thresholds.max_mean_pairwise_score_spearman:.2f}.",
        "- Minimum weakest-view median held-out score Spearman correlation: "
        f"{thresholds.min_target_view_cv_score_spearman:.2f}.",
        "",
        "These are metric-review guardrails, not biological thresholds.",
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
        "### Adversarial pressure tests",
        "",
        _markdown_table(pressure_tests[["agent", "check_id", "status", "severity", "evidence", "threshold", "action"]]),
        "",
        "### Pairwise correlations, overlap, and shared candidates",
        "",
        _markdown_table(pair_canonical[["selection_view_a", "selection_view_b", "pearson", "spearman"]]),
        "",
        _markdown_table(overlap_canonical[["k", "observed_overlap", "unique_topk"]]),
        "",
        _markdown_table(
            top_shared[
                ["id", "selection_view_id", "rank", "logic_fidelity", "effect_scaled", "selection_view_count"]
            ].head(18)
        ),
        "",
        "## Figure premises and alt text",
        "",
        _plot_manifest_tables(plot_manifest),
        "",
        "## Claim boundaries",
        "",
        "- Treat retrospective objective comparisons as design evidence, not biological validation.",
        "- Keep objective scoring separate from downstream allocation and diversity policy.",
        "- Require a prospectively frozen prediction before interpreting a later measurement as selection evidence.",
        "- Keep Reader reductions and assay evidence unchanged when comparing objectives.",
        "- Preserve the nearest-12-hour snapshot as provenance for the promoted response-window label.",
        "",
    ]
    _validate_evidence_section_order(text)
    report_path.write_text("\n".join(text), encoding="utf-8")
    return report_path


def _validate_evidence_section_order(lines: list[str]) -> None:
    headings = [f"## {section}" for section in EVIDENCE_SECTION_ORDER]
    positions: list[int] = []
    for heading in headings:
        if lines.count(heading) != 1:
            raise ValueError(f"Report must contain exactly one {heading!r} heading")
        positions.append(lines.index(heading))
    if positions != sorted(positions):
        raise ValueError(f"Report evidence sections are out of order: {EVIDENCE_SECTION_ORDER!r}")


def _observed_sfxi_report_evidence(
    components: pd.DataFrame,
    robustness: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    expected_views = {"ethanol", "ciprofloxacin", "and"}
    full = robustness.loc[robustness["sensitivity_scope"].eq("all_observed_labels")].copy()
    if set(full["selection_view_id"].astype(str)) != expected_views or len(full) != len(expected_views):
        raise ValueError("Observed SFXI report requires one full-corpus row per target view.")
    if components.groupby("selection_view_id")["id"].nunique().to_dict() != {
        view_id: 35 for view_id in sorted(expected_views)
    }:
        raise ValueError("Observed SFXI report requires 35 measured labels per target view.")
    full = full[
        ["selection_view_id", "candidate_count", "sfxi_vs_logic_spearman", "sfxi_vs_effect_spearman"]
    ].sort_values("selection_view_id", kind="mergesort")
    full = full.rename(
        columns={
            "selection_view_id": "target_view",
            "candidate_count": "measured_labels",
            "sfxi_vs_logic_spearman": "rank_rho_logic",
            "sfxi_vs_effect_spearman": "rank_rho_scaled_effect",
        }
    )
    deletion = robustness.loc[robustness["sensitivity_scope"].eq("leave_one_experiment_out")]
    es_only = robustness.loc[robustness["sensitivity_scope"].eq("es_designs_only")]
    if deletion.empty or set(es_only["selection_view_id"].astype(str)) != expected_views:
        raise ValueError("Observed SFXI report requires source-deletion and ES-only sensitivity evidence.")
    deletion_count = deletion["excluded_reader_experiment_id"].astype(str).nunique()
    es_count = int(es_only["candidate_count"].min())
    sentence = (
        f"Across {deletion_count} source-experiment deletions, the minimum SFXI-versus-scaled-effect rank "
        f"correlation was {float(deletion['sfxi_vs_effect_spearman'].min()):.3f}. Among the {es_count} ES designs, "
        f"the minimum was {float(es_only['sfxi_vs_effect_spearman'].min()):.3f}."
    )
    return full, sentence


def _sfxi_greedy_report_evidence(replay: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    expected_views = {"ethanol", "ciprofloxacin", "and"}
    required = {
        "selection_view_id",
        "rank",
        "pool_candidate_count",
        "score_vs_effect_spearman",
        "score_vs_logic_spearman",
        "top_k_effect_overlap",
        "total_selection_slots",
        "unique_selected_sequences",
        "selected_in_all_views",
        "pairwise_overlap_total",
    }
    missing = sorted(required - set(replay.columns))
    if missing:
        raise ValueError(f"Historical SFXI greedy report is missing columns: {missing}")
    if set(replay["selection_view_id"].astype(str)) != expected_views:
        raise ValueError("Historical SFXI greedy report requires all three target views.")
    counts = replay.groupby("selection_view_id")["rank"].size()
    if not counts.eq(6).all():
        raise ValueError("Historical SFXI greedy report requires six persisted selections per target view.")
    summary = (
        replay.groupby("selection_view_id", sort=True)
        .first()[
            [
                "pool_candidate_count",
                "score_vs_effect_spearman",
                "score_vs_logic_spearman",
                "top_k_effect_overlap",
            ]
        ]
        .reset_index()
        .rename(
            columns={
                "selection_view_id": "target_view",
                "pool_candidate_count": "predicted_candidates",
                "score_vs_effect_spearman": "rank_rho_scaled_effect",
                "score_vs_logic_spearman": "rank_rho_logic",
                "top_k_effect_overlap": "sfxi_top6_in_effect_top6",
            }
        )
    )
    first = replay.iloc[0]
    overlap = (
        f"Across {int(first['total_selection_slots'])} target-view slots, the replay contains "
        f"{int(first['unique_selected_sequences'])} unique sequences; "
        f"{int(first['selected_in_all_views'])} sequences occur in all three lists."
    )
    return summary, overlap


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
    section_titles = {
        "assay_and_labels": "Assay and label evidence",
        "historical_model_screens": "Model screens",
        "rmf_comparator": "RMF comparator",
        "sfxi_comparator": "SFXI comparator",
    }
    for review_section in (
        "assay_and_labels",
        "historical_model_screens",
        "rmf_comparator",
        "sfxi_comparator",
    ):
        section_rows = plot_manifest[plot_manifest["review_section"] == review_section].sort_values(
            "section_order", kind="mergesort"
        )
        if section_rows.empty:
            continue
        sections.append(f"### {section_titles[review_section]}")
        sections.append(
            _markdown_table(
                section_rows[
                    [
                        "plot_id",
                        "tier",
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
