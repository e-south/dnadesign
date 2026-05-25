"""Markdown renderer for DenseGen axis probe reviews."""

from __future__ import annotations

from typing import Any, Mapping

from .formatting import _fmt, _gate_observed, _gate_threshold


def render_probe_review_markdown(review_manifest: Mapping[str, Any], metrics_payload: Mapping[str, Any]) -> str:
    runs = metrics_payload.get("runs") or []
    round_rows = metrics_payload.get("rounds") or []
    coverage = review_manifest.get("gate_coverage") or {}
    campaign_reviews = review_manifest.get("opal_campaign_reviews") or []
    configured_plots = review_manifest.get("opal_configured_plots") or []
    plot_quality = review_manifest.get("plot_quality") or {}
    next_steps = review_manifest.get("next_steps") or {}
    plots = review_manifest.get("probe_plots") or []
    problems = review_manifest.get("problems") or []
    decision_reasons = review_manifest.get("decision_reasons") or []
    gate_results = review_manifest.get("gate_results") or []
    metric_quality = review_manifest.get("metric_quality") or {}
    round_dynamics = review_manifest.get("round_dynamics") or metrics_payload.get("round_dynamics") or []
    trajectory_qa = review_manifest.get("trajectory_qa") or metrics_payload.get("trajectory_qa") or {}
    trajectory_pairs = trajectory_qa.get("pairs") if isinstance(trajectory_qa, Mapping) else []
    seed_summaries = trajectory_qa.get("seed_summaries") if isinstance(trajectory_qa, Mapping) else []
    definitions = review_manifest.get("metric_definitions") or metrics_payload.get("metric_definitions") or {}
    next_steps = review_manifest.get("next_steps") or {}
    lines = [
        "# DenseGen axis OPAL probe review",
        "",
        "This artifact is the study-specific benchmark layer. OPAL campaign run review artifacts "
        "remain campaign-scoped under each scratch campaign's `outputs/review/` directory.",
        "",
        "## Gate Decision",
        "",
        f"- decision: `{review_manifest.get('decision')}`",
        f"- persisted decision: `{review_manifest.get('persisted_decision')}`",
        f"- status: `{review_manifest.get('status')}`",
        f"- contract problems: `{', '.join(problems) if problems else 'none'}`",
        f"- decision reasons: `{len(decision_reasons)}`",
        f"- run_root: `{review_manifest.get('run_root')}`",
        f"- weak count-aware runs: `{metric_quality.get('weak_count_approx_binomial_p_gt_0_05', 0)}`",
        "",
        "## Coverage",
        "",
        f"- campaigns: `{', '.join(coverage.get('campaigns') or []) or 'none'}`",
        f"- splits: `{', '.join(coverage.get('splits') or []) or 'none'}`",
        f"- positive/null pairs complete: `{coverage.get('positive_null_pairs_complete')}`",
        f"- omitted scored gates: `{', '.join(coverage.get('omitted_scored_gates') or []) or 'none'}`",
        "",
        "## Decision Reasons",
        "",
    ]
    if decision_reasons:
        for reason in decision_reasons:
            lines.append(
                "- `{gate}` `{status}`: {reason}".format(
                    gate=reason.get("gate", "unknown"),
                    status=reason.get("status", "unknown"),
                    reason=reason.get("reason", "no reason recorded"),
                )
            )
    else:
        lines.append("No blocking decision reasons were recorded.")
    lines.extend(["", "## Gate Results", ""])
    if gate_results:
        lines.extend(
            [
                "| gate | status | campaign | split | observed | threshold | reason |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in gate_results:
            lines.append(
                "| `{gate}` | `{status}` | `{campaign}` | `{split}` | {observed} | {threshold} | {reason} |".format(
                    gate=row.get("gate", ""),
                    status=row.get("status", ""),
                    campaign=row.get("campaign", ""),
                    split=row.get("split_id", ""),
                    observed=_fmt(_gate_observed(row)),
                    threshold=_fmt(_gate_threshold(row)),
                    reason=row.get("reason", ""),
                )
            )
    else:
        lines.append("No gate results were recorded.")
    lines.extend(
        [
            "",
            "## Metric Guide",
            "",
            f"- selected target count: {definitions.get('selected_target_count', '')}",
            f"- precision@K: {definitions.get('precision_at_k', '')}",
            f"- prevalence: {definitions.get('target_prevalence', '')}",
            f"- lift: {definitions.get('lift', '')}",
            f"- binomial p>=k: {definitions.get('binomial_tail_p', '')}",
            f"- null lift: {definitions.get('null_lift', '')}",
            f"- trajectory AUC: {definitions.get('trajectory_auc', '')}",
            f"- paired AUC delta: {definitions.get('paired_auc_delta', '')}",
            f"- round metrics: {definitions.get('round', '')}",
            f"- round dynamics: {definitions.get('round_dynamics', '')}",
            "",
            "## Metrics",
            "",
        ]
    )
    if runs:
        lines.extend(
            [
                "| run_key | oracle | split | selected target | prevalence | precision@K | "
                "lift | binom p>=k | selected classes |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in runs:
            classes = row.get("off_target_class_distribution_true") or {}
            class_text = ", ".join(f"{key}:{value}" for key, value in classes.items())
            lines.append(
                (
                    "| `{run_key}` | `{oracle}` | `{split}` | {selected_count} | {prevalence} | "
                    "{precision} | {lift} | {p_value} | {classes} |"
                ).format(
                    run_key=row.get("run_key"),
                    oracle=row.get("oracle_id"),
                    split=row.get("split_id"),
                    selected_count=row.get("selected_target_count_label_true", ""),
                    prevalence=_fmt(row.get("target_class_prevalence_true")),
                    precision=_fmt(row.get("selected_target_precision_at_k_true")),
                    lift=_fmt(row.get("target_lift_at_k_true")),
                    p_value=_fmt(row.get("selected_target_binomial_tail_p_true")),
                    classes=class_text,
                )
            )
    else:
        lines.append("No scored OPAL run metrics are present yet.")
    lines.extend(["", "## Round Metrics", ""])
    if round_rows:
        lines.extend(
            [
                "| run_key | round | selected target | prevalence | precision@K | lift | binom p>=k |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in round_rows:
            lines.append(
                "| `{run_key}` | {round} | {selected_count} | {prevalence} | {precision} | {lift} | {p_value} |".format(
                    run_key=row.get("run_key"),
                    round=row.get("as_of_round"),
                    selected_count=row.get("selected_target_count_label_true", ""),
                    prevalence=_fmt(row.get("target_class_prevalence_true")),
                    precision=_fmt(row.get("selected_target_precision_at_k_true")),
                    lift=_fmt(row.get("target_lift_at_k_true")),
                    p_value=_fmt(row.get("selected_target_binomial_tail_p_true")),
                )
            )
    else:
        lines.append("No round-level OPAL metrics were recorded.")
    lines.extend(["", "## Round Dynamics", ""])
    if round_dynamics:
        lines.extend(
            [
                "| run_key | oracle | first lift | final lift | max lift | max round | status |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in round_dynamics:
            status = (
                "final null above threshold"
                if row.get("null_final_threshold_exceeded")
                else "transient null spike"
                if row.get("null_transient_spike")
                else "ok"
            )
            lines.append(
                "| `{run_key}` | `{oracle}` | {first} | {final} | {max_lift} | {max_round} | {status} |".format(
                    run_key=row.get("run_key"),
                    oracle=row.get("oracle_id"),
                    first=_fmt(row.get("first_lift")),
                    final=_fmt(row.get("final_lift")),
                    max_lift=_fmt(row.get("max_lift")),
                    max_round=row.get("max_round"),
                    status=status,
                )
            )
    else:
        lines.append("No round-dynamics summary was recorded.")
    lines.extend(["", "## Trajectory QA", ""])
    if trajectory_pairs:
        lines.extend(
            [
                "| seed | campaign | split | positive AUC | null AUC | AUC delta | final delta | status |",
                "|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in trajectory_pairs:
            lines.append(
                (
                    "| {seed} | `{campaign}` | `{split}` | {pos_auc} | {null_auc} | "
                    "{auc_delta} | {final_delta} | `{status}` |"
                ).format(
                    seed="" if row.get("seed") is None else row.get("seed"),
                    campaign=row.get("campaign"),
                    split=row.get("split_id"),
                    pos_auc=_fmt(row.get("positive_lift_auc")),
                    null_auc=_fmt(row.get("null_lift_auc")),
                    auc_delta=_fmt(row.get("paired_auc_delta")),
                    final_delta=_fmt(row.get("final_positive_minus_null_lift")),
                    status=row.get("status"),
                )
            )
    else:
        lines.append("No trajectory QA summary was recorded.")
    if seed_summaries:
        lines.extend(["", "Seed-level summaries:", ""])
        lines.extend(
            [
                "| seed | pairs | AUC delta mean | AUC delta min | final delta mean | status |",
                "|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in seed_summaries:
            lines.append(
                "| {seed} | {pairs} | {auc_mean} | {auc_min} | {final_mean} | `{status}` |".format(
                    seed="" if row.get("seed") is None else row.get("seed"),
                    pairs=row.get("pair_count"),
                    auc_mean=_fmt(row.get("paired_auc_delta_mean")),
                    auc_min=_fmt(row.get("paired_auc_delta_min")),
                    final_mean=_fmt(row.get("final_delta_mean")),
                    status=row.get("status"),
                )
            )
    lines.extend(["", "## OPAL Campaign Reviews", ""])
    if campaign_reviews:
        for review in campaign_reviews:
            lines.append(
                "- `{run_key}`: `{status}` review `{review_path}` manifest `{manifest_path}`".format(
                    run_key=review.get("run_key"),
                    status=review.get("status"),
                    review_path=review.get("review_path"),
                    manifest_path=review.get("manifest_path"),
                )
            )
    else:
        lines.append("No scored scratch campaign reviews were required for this run root.")
    lines.extend(["", "## Probe Plots", ""])
    if plots:
        lines.extend(f"- `{path}`" for path in plots)
    else:
        lines.append("No probe aggregate plots were written.")
    lines.extend(
        [
            "",
            "## Configured OPAL Plots",
            "",
            f"- quality status: `{plot_quality.get('status', 'unknown')}`",
            f"- campaign plot indexes: `{plot_quality.get('campaigns_with_plot_index', 0)}` / "
            f"`{plot_quality.get('campaigns_expected', 0)}`",
            f"- configured plot artifacts: `{plot_quality.get('plot_count', 0)}`",
            f"- quality problems: `{plot_quality.get('problem_count', 0)}`",
        ]
    )
    if next_steps.get("configured_plot_refresh_command"):
        lines.extend(
            [
                "- configured plot refresh:",
                f"  `{next_steps['configured_plot_refresh_command']}`",
                "- rerun report after refresh:",
                f"  `{next_steps.get('rerun_report_command')}`",
            ]
        )
    for entry in configured_plots:
        lines.append(
            "- `{run_key}`: `{status}` plots=`{plot_count}` index `{index_path}`".format(
                run_key=entry.get("run_key"),
                status=entry.get("status"),
                plot_count=entry.get("plot_count"),
                index_path=entry.get("index_path"),
            )
        )
    lines.append("")
    return "\n".join(lines)
