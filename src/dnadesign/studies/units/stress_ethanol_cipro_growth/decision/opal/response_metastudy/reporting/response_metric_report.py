"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/response_metric_report.py

Markdown sections for the RMF label and predictor screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from ..core.response_contracts import ResponseMetricScreen


def response_metric_report_lines(screen: ResponseMetricScreen, *, primary_reduction_id: str) -> list[str]:
    """Summarize event alignment, calibration, and learnability without overclaiming."""

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
    return [
        "## Response-Window Label And RMF Screen",
        "",
        "The signed response separation does not require a positive exemplar. A negative value still gives a direction "
        "for relative improvement. Positive feasibility is therefore a configured decision target, not an entry "
        "condition for a prospective next-build round.",
        "",
        f"- Explicit source event bindings: {len(screen.event_intervals)} Reader experiments.",
        f"- Unresolved stress-addition interval: {event_gap.min():.3f}-{event_gap.max():.3f} h.",
        f"- Candidate reductions: {screen.labels['reduction_id'].nunique()} over "
        f"{screen.labels['id'].nunique()} labels.",
        f"- Primary Reader reduction: `{primary_reduction_id}`.",
        "- Event-bound sensitivity is propagated separately from replicate resampling.",
        "- No Reader record, OPAL label, campaign config, or synthesis handoff is changed by this screen.",
        "",
        "### Provisional Reference Boundaries And Review Scales",
        "",
        "All three semantic thresholds are zero: ON response must exceed OFF response, ON fluorescence must be at "
        "least the same-state pDual-10 reference, and OFF fluorescence must not exceed it. Target-view review "
        "scales use the 90th percentile of replicate-bootstrap variation combined with event-bound sensitivity. "
        "These values make the screen interpretable but are not an activated production calibration or biological law.",
        "",
        _markdown_table(screen.calibration),
        "",
        "### Observed Constraint Support",
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
        "The observed corpus can start a directional search, but grouped evidence does not support the same "
        "selection posture for every target view.",
        "",
        "### Repeated-Design Evidence",
        "",
        f"- Repeatedly measured designs: {len(screen.repeated_agreement)}.",
        f"- Largest chosen screen-source difference from the cross-experiment median: "
        f"{screen.repeated_agreement['maximum_selected_to_median_abs_difference'].max():.3f} log2 units.",
        "- Ratio-domain policy: values at or below the declared positive floor abort materialization.",
        "",
        "The response-owned screen selection makes this retrospective source choice explicit. Repeated "
        "experiments are not independent labels; promotion must declare one aggregation rule and retain every "
        "contributing Reader record.",
        "",
        "### Grouped Model Screen",
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
        "### Retrospective Enrichment Proxy",
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
        "This proxy can reject an unsupported model. It cannot demonstrate a biological hill climb because no "
        "prospective selection has yet been built, measured, and compared with its round-0 baseline.",
        "",
        "### Evidence For Greedy Selection",
        "",
        _markdown_table(
            screen.greedy_support[
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
        "All exact intervals include 0.5. The ciprofloxacin point estimate is higher than the ethanol and AND "
        "estimates, but these data do not distinguish any target view from chance or establish calibrated success "
        "probabilities. The configured mechanism under review is greedy top-six per selection view; it remains "
        "inactive, and these intervals are risk evidence rather than slot-allocation authority.",
        "",
    ]


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


__all__ = ["response_metric_report_lines"]
