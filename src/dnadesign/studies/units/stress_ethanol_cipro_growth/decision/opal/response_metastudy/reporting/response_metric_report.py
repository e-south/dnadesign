"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/response_metric_report.py

Markdown sections for the RMF label and predictor screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.response_contracts import ResponseMetricScreen


@dataclass(frozen=True)
class ResponseMetricReportSections:
    assay_and_labels: tuple[str, ...]
    historical_model_screens: tuple[str, ...]
    rmf_comparator: tuple[str, ...]


def response_metric_report_sections(
    screen: ResponseMetricScreen,
    *,
    primary_reduction_id: str,
) -> ResponseMetricReportSections:
    """Separate assay, historical model, and RMF-comparator evidence."""

    event_gap = (
        screen.event_intervals["event_interval_end_assay_h"] - screen.event_intervals["event_interval_start_assay_h"]
    )
    primary = screen.stability.loc[screen.stability["reduction_id"].eq(primary_reduction_id)]
    eligible = screen.model_screen.loc[
        screen.model_screen["promotion_eligible"].astype(bool)
        & screen.model_screen["all_target_view_metrics_finite"].astype(bool)
    ]
    best = eligible.sort_values(
        ["weakest_required_ordering_spearman", "median_channel_spearman"],
        ascending=False,
        kind="mergesort",
    ).head(8)
    enrichment = screen.enrichment_summary.merge(
        best.loc[:, ["representation_id", "model_id"]],
        on=["representation_id", "model_id"],
        how="inner",
    )
    window_assay_columns = [
        "reduction_id",
        "pdual_response_within_experiment_median_range",
        "pdual_magnitude_within_experiment_median_range",
        "pdual_magnitude_cross_experiment_max_state_range",
        "spyp_ethanol_response_separation_median",
        "sulap_ciprofloxacin_response_separation_median",
        "growth_endpoint_od600_q90",
        "event_sensitivity_max_half_range",
        "event_sensitivity_censored_design_state_component_count",
        "repeat_maximum_channel_range",
        "censoring_observability",
    ]
    window_model_columns = [
        "reduction_id",
        "campaign_random_forest_weakest_ordering_spearman",
        "pls4_weakest_ordering_spearman",
        "pls6_weakest_ordering_spearman",
    ]
    assay_and_labels = (
        "### Assay scope",
        "",
        "The response-window comparison uses the same Reader experiments, candidate identities, controls, and "
        "measurement checks for every declared reduction.",
        "",
        f"- Explicit source event bindings: {len(screen.event_intervals)} Reader experiments.",
        f"- Unresolved stress-addition interval: {event_gap.min():.3f}-{event_gap.max():.3f} h.",
        f"- Candidate reductions: {screen.labels['reduction_id'].nunique()} over "
        f"{screen.labels['id'].nunique()} labels.",
        f"- Primary Reader reduction: `{primary_reduction_id}`.",
        "- Event-bound sensitivity is propagated separately from within-experiment observation resampling.",
        "- No Reader record, OPAL label, campaign config, or synthesis handoff is changed by this screen.",
        "",
        "### Equal-footing window evidence",
        "",
        "Every declared reduction uses the same Reader experiments, candidate identities, response controls, "
        "anchor summaries, repeat comparison, and fixed model screen. Reader-published trajectories are consumed "
        "only for OD and measurement observability; they never replace Reader-owned reduced Y. Model results are "
        "diagnostic and are not a window-selection rule.",
        "",
        _markdown_table(screen.window_evidence.loc[:, window_assay_columns]),
        "",
        "The pDual-10 columns report within-experiment observation ranges and the largest state-specific "
        "cross-experiment range of experiment medians. SpyP and sulAp summarize every Reader occurrence, not the "
        "single source chosen for the retrospective label-model screen.",
        "",
        _markdown_table(screen.window_evidence.loc[:, window_model_columns]),
        "",
        "### Repeated-design evidence",
        "",
        f"- Repeatedly measured designs: {len(screen.repeated_agreement)}.",
        f"- Largest chosen screen-source difference from the cross-experiment median: "
        f"{screen.repeated_agreement['maximum_selected_to_median_abs_difference'].max():.3f} log2 units.",
        "- Ratio-domain policy: values at or below the declared positive floor abort materialization.",
        "",
        "The response-owned screen selection makes this retrospective source choice explicit. Repeated "
        "experiments are not independent campaign labels. The study's explicit label-source policy selects one "
        "reviewed experiment for each included repeated candidate while retaining every contributing Reader "
        "record.",
        "",
    )
    historical_model_screens = (
        "### Response-window phenotype screen",
        "",
        "The fixed screen compares a robust-target RF challenger, PLS, and fold-fitted PCA plus ridge. Complete "
        "Reader experiments are held out. Correlations are calculated within each held-out experiment and then "
        "summarized across experiments. These rows identify challengers; they do not promote hyperparameters.",
        "",
        _markdown_table(
            best[
                [
                    "representation_id",
                    "model_id",
                    "median_channel_spearman",
                    "weakest_target_view_response_separation_spearman",
                    "weakest_target_view_feasibility_spearman",
                    "weakest_required_ordering_spearman",
                    "minimum_defined_group_count",
                    "response_magnitude_mae",
                ]
            ]
        ),
        "",
    )
    rmf_comparator = (
        "## RMF comparator",
        "",
        "The signed response separation does not require a positive exemplar. A negative value still gives a "
        "direction for relative improvement. Positive RMF feasibility is a configured comparator target, not an "
        "entry condition for the active MSRB learning probe.",
        "",
        "Every target-ON response is compared with every target-OFF response. Conditional induction and "
        "interaction are separate diagnostics.",
        "",
        "### Retrospective enrichment proxy",
        "",
        _markdown_table(
            enrichment[
                [
                    "representation_id",
                    "model_id",
                    "selection_view_id",
                    "held_out_group_count",
                    "defined_selection_group_count",
                    "median_selected_true_percentile",
                    "fraction_beating_group_median",
                ]
            ]
        ),
        "",
        "This retrospective proxy can reject an unsupported model. It cannot demonstrate a biological hill "
        "climb; that requires predictions frozen before a prospective selection is measured and comparison "
        "against its declared baseline.",
        "",
        "### Evidence for greedy selection",
        "",
        _markdown_table(
            screen.campaign_greedy_support[
                [
                    "selection_view_id",
                    "held_out_group_count",
                    "groups_beating_median",
                    "fraction_beating_group_median",
                    "fraction_ci_low",
                    "fraction_ci_high",
                    "confidence_method",
                    "evidence_posture",
                ]
            ]
        ),
        "",
        "These intervals evaluate the configured campaign random forest, not the strongest retrospective "
        "challenger. The campaign coordinates six sequence-unique slots per view from ordinal RMF rankings; "
        "these intervals are risk evidence rather than slot-allocation authority. Runtime state and synthesis "
        "authorization are outside this metastudy. Fixed-challenger support is retained separately for "
        "descriptive comparison.",
        "",
        "### Provisional reference boundaries and review scales",
        "",
        "All three semantic thresholds are zero: ON response must exceed OFF response, ON fluorescence must be at "
        "least the same-state pDual-10 reference, and OFF fluorescence must not exceed it. Target-view review "
        "scales use the 90th percentile of well-resampling bootstrap variation combined with event-bound sensitivity. "
        "These values make the comparator screen interpretable but are not biological laws.",
        "",
        _markdown_table(screen.calibration),
        "",
        "### Observed constraint support",
        "",
        _markdown_table(
            primary[
                [
                    "selection_view_id",
                    "n",
                    "positive_response_count",
                    "zero_constraint_feasible_count",
                    "median_response_separation",
                    "median_on_magnitude_floor",
                    "median_off_magnitude_ceiling",
                ]
            ]
        ),
        "",
        "The observed corpus can support directional comparison, but the requirements are unevenly supported "
        "across target views.",
        "",
    )
    return ResponseMetricReportSections(
        assay_and_labels=assay_and_labels,
        historical_model_screens=historical_model_screens,
        rmf_comparator=rmf_comparator,
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rows = [[str(column) for column in frame.columns]]
    for _, row in frame.iterrows():
        rows.append([_format_cell(row[column]) for column in frame.columns])
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(widths))) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows[1:]]
    return "\n".join([header, divider, *body])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


__all__ = ["ResponseMetricReportSections", "response_metric_report_sections"]
