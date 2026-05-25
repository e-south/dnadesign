"""HTML renderer for DenseGen axis probe reviews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .formatting import _e, _fmt, _gate_observed, _gate_threshold, _html_document, _metric_card, _rel


def render_probe_review_html(
    review_manifest: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    *,
    base_dir: Path,
) -> str:
    runs = metrics_payload.get("runs") or []
    round_rows_payload = metrics_payload.get("rounds") or []
    coverage = review_manifest.get("gate_coverage") or {}
    campaign_reviews = review_manifest.get("opal_campaign_reviews") or []
    configured_plots = review_manifest.get("opal_configured_plots") or []
    plot_quality = review_manifest.get("plot_quality") or {}
    plots = review_manifest.get("probe_plots") or []
    problems = review_manifest.get("problems") or []
    decision_reasons = review_manifest.get("decision_reasons") or []
    gate_results = review_manifest.get("gate_results") or []
    metric_quality = review_manifest.get("metric_quality") or {}
    round_dynamics_payload = review_manifest.get("round_dynamics") or metrics_payload.get("round_dynamics") or []
    trajectory_qa = review_manifest.get("trajectory_qa") or metrics_payload.get("trajectory_qa") or {}
    trajectory_pairs = trajectory_qa.get("pairs") if isinstance(trajectory_qa, Mapping) else []
    seed_summaries = trajectory_qa.get("seed_summaries") if isinstance(trajectory_qa, Mapping) else []
    definitions = review_manifest.get("metric_definitions") or metrics_payload.get("metric_definitions") or {}
    next_steps = review_manifest.get("next_steps") or {}
    plot_cards = []
    for path in plots:
        src = _rel(path, base_dir=base_dir)
        title = Path(str(path)).stem.replace("_", " ")
        plot_cards.append(
            "<article>"
            f"<h3>{_e(title)}</h3>"
            f'<a href="{_e(src)}"><img src="{_e(src)}" alt="Probe plot: {_e(title)}"></a>'
            "</article>"
        )
    campaign_links = []
    for review in campaign_reviews:
        index_path = review.get("index_path")
        review_path = review.get("review_path")
        href = index_path or review_path
        campaign_links.append(
            "<li>"
            f'<a href="{_e(_rel(href, base_dir=base_dir))}"><code>{_e(review.get("run_key"))}</code></a>'
            f" round {_e(review.get('round_index'))} run <code>{_e(review.get('run_id'))}</code>"
            "</li>"
        )
    metric_rows = []
    for row in runs:
        classes = row.get("off_target_class_distribution_true") or {}
        class_text = (
            ", ".join(f"{key}:{value}" for key, value in classes.items()) if isinstance(classes, Mapping) else ""
        )
        metric_rows.append(
            "<tr>"
            f"<td><code>{_e(row.get('run_key'))}</code></td>"
            f"<td>{_e(row.get('label_family_id'))}</td>"
            f"<td>{_e(row.get('oracle_id'))}</td>"
            f"<td>{_e(row.get('split_id'))}</td>"
            f"<td>{_e(row.get('selected_target_count_label_true'))}</td>"
            f"<td>{_e(_fmt(row.get('target_class_prevalence_true')))}</td>"
            f"<td>{_e(_fmt(row.get('selected_target_precision_at_k_true')))}</td>"
            f"<td>{_e(_fmt(row.get('target_lift_at_k_true')))}</td>"
            f"<td>{_e(_fmt(row.get('selected_target_binomial_tail_p_true')))}</td>"
            f"<td>{_e(class_text)}</td>"
            "</tr>"
        )
    round_rows = []
    for row in round_rows_payload:
        round_rows.append(
            "<tr>"
            f"<td><code>{_e(row.get('run_key'))}</code></td>"
            f"<td>{_e(row.get('label_family_id'))}</td>"
            f"<td>{_e(row.get('as_of_round'))}</td>"
            f"<td>{_e(row.get('selected_target_count_label_true'))}</td>"
            f"<td>{_e(_fmt(row.get('target_class_prevalence_true')))}</td>"
            f"<td>{_e(_fmt(row.get('selected_target_precision_at_k_true')))}</td>"
            f"<td>{_e(_fmt(row.get('target_lift_at_k_true')))}</td>"
            f"<td>{_e(_fmt(row.get('selected_target_binomial_tail_p_true')))}</td>"
            "</tr>"
        )
    reason_items = []
    for reason in decision_reasons:
        reason_items.append(
            "<li>"
            f"<strong>{_e(reason.get('gate'))}</strong> "
            f"<code>{_e(reason.get('status'))}</code>: {_e(reason.get('reason'))}"
            "</li>"
        )
    gate_rows = []
    for row in gate_results:
        gate_rows.append(
            "<tr>"
            f"<td><code>{_e(row.get('gate'))}</code></td>"
            f"<td><code>{_e(row.get('status'))}</code></td>"
            f"<td>{_e(row.get('label_family_id', ''))}</td>"
            f"<td>{_e(row.get('campaign', ''))}</td>"
            f"<td>{_e(row.get('split_id', ''))}</td>"
            f"<td>{_e(_fmt(_gate_observed(row)))}</td>"
            f"<td>{_e(_fmt(_gate_threshold(row)))}</td>"
            f"<td>{_e(row.get('reason'))}</td>"
            "</tr>"
        )
    configured_plot_cards = []
    for entry in configured_plots:
        plot_links = []
        plot_thumbs = []
        for plot in entry.get("plots") or []:
            media_path = next(iter(plot.get("media_paths") or []), None)
            tidy_path = next(iter(plot.get("tidy_csv_paths") or []), None)
            manifest_path = plot.get("manifest_path")
            media_link = (
                f'<a href="{_e(_rel(media_path, base_dir=base_dir))}">media</a>' if media_path else "media missing"
            )
            tidy_link = f' · <a href="{_e(_rel(tidy_path, base_dir=base_dir))}">csv</a>' if tidy_path else ""
            manifest_link = (
                f' · <a href="{_e(_rel(manifest_path, base_dir=base_dir))}">manifest</a>' if manifest_path else ""
            )
            plot_links.append(
                "<li>"
                f"<code>{_e(plot.get('name'))}</code> ({_e(plot.get('kind'))}) "
                f"{media_link}{tidy_link}{manifest_link}"
                "</li>"
            )
            if media_path:
                media_src = _rel(media_path, base_dir=base_dir)
                caption = f"{plot.get('name')} ({plot.get('kind')})"
                plot_thumbs.append(
                    "<figure>"
                    f'<a href="{_e(media_src)}"><img src="{_e(media_src)}" '
                    f'alt="{_e(entry.get("run_key"))}: {_e(caption)}"></a>'
                    f"<figcaption>{_e(caption)}</figcaption>"
                    "</figure>"
                )
        quality = entry.get("quality") or {}
        configured_plot_cards.append(
            "<article>"
            f"<h3><code>{_e(entry.get('run_key'))}</code></h3>"
            f"<p>Status: <code>{_e(entry.get('status'))}</code>; "
            f"quality: <code>{_e(quality.get('status'))}</code>; "
            f"plots: <code>{_e(entry.get('plot_count'))}</code></p>"
            f"<ul>{''.join(plot_links) if plot_links else '<li>No manifest-backed configured plots.</li>'}</ul>"
            f'<div class="plot-thumb-grid">{"".join(plot_thumbs)}</div>'
            "</article>"
        )
    plot_index_text = f"{plot_quality.get('campaigns_with_plot_index', 0)} / "
    plot_index_text += f"{plot_quality.get('campaigns_expected', 0)}"
    configured_plot_cards_html = "".join(configured_plot_cards) or "<p>No configured OPAL plot indexes found.</p>"
    configured_plot_next_step_html = ""
    if next_steps.get("configured_plot_refresh_command"):
        configured_plot_next_step_html = (
            "<p>Refresh configured OPAL plots with "
            f"<code>{_e(next_steps.get('configured_plot_refresh_command'))}</code>, then rerun "
            f"<code>{_e(next_steps.get('rerun_report_command'))}</code>.</p>"
        )
    round_rows_html = "".join(round_rows) or '<tr><td colspan="8">No round-level metrics recorded.</td></tr>'
    round_dynamics_rows = []
    for row in round_dynamics_payload:
        status = (
            "final null above threshold"
            if row.get("null_final_threshold_exceeded")
            else "transient null spike"
            if row.get("null_transient_spike")
            else "ok"
        )
        round_dynamics_rows.append(
            "<tr>"
            f"<td><code>{_e(row.get('run_key'))}</code></td>"
            f"<td>{_e(row.get('label_family_id'))}</td>"
            f"<td>{_e(row.get('oracle_id'))}</td>"
            f"<td>{_e(_fmt(row.get('first_lift')))}</td>"
            f"<td>{_e(_fmt(row.get('final_lift')))}</td>"
            f"<td>{_e(_fmt(row.get('max_lift')))}</td>"
            f"<td>{_e(row.get('max_round'))}</td>"
            f"<td>{_e(status)}</td>"
            "</tr>"
        )
    round_dynamics_html = (
        "".join(round_dynamics_rows)
        if round_dynamics_rows
        else '<tr><td colspan="8">No round-dynamics summary recorded.</td></tr>'
    )
    trajectory_rows = []
    for row in trajectory_pairs or []:
        trajectory_rows.append(
            "<tr>"
            f"<td>{_e(row.get('seed', ''))}</td>"
            f"<td>{_e(row.get('label_family_id', ''))}</td>"
            f"<td>{_e(row.get('campaign'))}</td>"
            f"<td>{_e(row.get('split_id'))}</td>"
            f"<td>{_e(_fmt(row.get('positive_lift_auc')))}</td>"
            f"<td>{_e(_fmt(row.get('null_lift_auc')))}</td>"
            f"<td>{_e(_fmt(row.get('paired_auc_delta')))}</td>"
            f"<td>{_e(_fmt(row.get('final_positive_minus_null_lift')))}</td>"
            f"<td><code>{_e(row.get('status'))}</code></td>"
            "</tr>"
        )
    trajectory_html = "".join(trajectory_rows) or '<tr><td colspan="9">No trajectory QA recorded.</td></tr>'
    seed_rows = []
    for row in seed_summaries or []:
        seed_rows.append(
            "<tr>"
            f"<td>{_e(row.get('seed', ''))}</td>"
            f"<td>{_e(row.get('pair_count'))}</td>"
            f"<td>{_e(_fmt(row.get('paired_auc_delta_mean')))}</td>"
            f"<td>{_e(_fmt(row.get('paired_auc_delta_min')))}</td>"
            f"<td>{_e(_fmt(row.get('final_delta_mean')))}</td>"
            f"<td><code>{_e(row.get('status'))}</code></td>"
            "</tr>"
        )
    seed_html = "".join(seed_rows) if seed_rows else '<tr><td colspan="6">No seed summaries recorded.</td></tr>'
    body = f"""
    <header>
      <p>DenseGen axis probe review</p>
      <h1>{_e(review_manifest.get("decision"))}</h1>
      <p class="lede">Study benchmark layer over OPAL campaign review artifacts.</p>
    </header>
    <main>
      <section class="summary-grid">
        {_metric_card("Status", review_manifest.get("status"))}
        {_metric_card("Contract problems", ", ".join(problems) if problems else "none")}
        {_metric_card("Decision reasons", len(decision_reasons))}
        {_metric_card("Weak K tests", metric_quality.get("weak_count_approx_binomial_p_gt_0_05", 0))}
        {_metric_card("Persisted decision", review_manifest.get("persisted_decision"))}
        {_metric_card("Runs", len(runs))}
        {_metric_card("Campaigns", ", ".join(coverage.get("campaigns") or []) or "none")}
        {_metric_card("Families", ", ".join(coverage.get("label_families") or []) or "none")}
        {_metric_card("Splits", ", ".join(coverage.get("splits") or []) or "none")}
        {_metric_card("Omitted gates", ", ".join(coverage.get("omitted_scored_gates") or []) or "none")}
      </section>
      <section>
        <h2>Coverage Contract</h2>
        <dl>
          <dt>Run root</dt><dd><code>{_e(review_manifest.get("run_root"))}</code></dd>
          <dt>Positive/null complete</dt><dd>{_e(coverage.get("positive_null_pairs_complete"))}</dd>
          <dt>Scope</dt><dd>pre-assay synthetic-oracle benchmark; not a global OPAL readiness claim</dd>
        </dl>
      </section>
      <section>
        <h2>Decision Reasons</h2>
        <ul>{"".join(reason_items) if reason_items else "<li>No blocking decision reasons recorded.</li>"}</ul>
      </section>
      <section>
        <h2>Gate Results</h2>
        <table>
          <thead>
            <tr>
              <th>Gate</th><th>Status</th><th>Family</th><th>Campaign</th><th>Split</th><th>Observed</th><th>Threshold</th><th>Reason</th>
            </tr>
          </thead>
          <tbody>{"".join(gate_rows)}</tbody>
        </table>
      </section>
      <section>
        <h2>Metric Guide</h2>
        <details open>
          <summary>Metric Guide</summary>
          <dl>
            <dt>Selected target count</dt><dd>{_e(definitions.get("selected_target_count", ""))}</dd>
            <dt>Precision@K</dt><dd>{_e(definitions.get("precision_at_k", ""))}</dd>
            <dt>Prevalence</dt><dd>{_e(definitions.get("target_prevalence", ""))}</dd>
            <dt>Lift</dt><dd>{_e(definitions.get("lift", ""))}</dd>
            <dt>Binomial p&gt;=k</dt><dd>{_e(definitions.get("binomial_tail_p", ""))}</dd>
            <dt>Null lift</dt><dd>{_e(definitions.get("null_lift", ""))}</dd>
            <dt>Trajectory AUC</dt><dd>{_e(definitions.get("trajectory_auc", ""))}</dd>
            <dt>Paired AUC delta</dt><dd>{_e(definitions.get("paired_auc_delta", ""))}</dd>
            <dt>Round metrics</dt><dd>{_e(definitions.get("round", ""))}</dd>
            <dt>Round dynamics</dt><dd>{_e(definitions.get("round_dynamics", ""))}</dd>
          </dl>
        </details>
      </section>
      <section>
        <h2>Probe Plots</h2>
        <div class="plot-grid">{"".join(plot_cards) if plot_cards else "<p>No probe aggregate plots written.</p>"}</div>
      </section>
      <section>
        <h2>Configured OPAL Plots</h2>
        <div class="summary-grid">
          {_metric_card("Plot quality", plot_quality.get("status"))}
          {_metric_card("Plot indexes", plot_index_text)}
          {_metric_card("Configured plots", plot_quality.get("plot_count"))}
          {_metric_card("Quality problems", plot_quality.get("problem_count"))}
        </div>
        {configured_plot_next_step_html}
        <div class="plot-grid">{configured_plot_cards_html}</div>
      </section>
      <section>
        <h2>Scored Runs</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Family</th><th>Oracle</th><th>Split</th><th>Selected Target</th>
              <th>Prevalence</th><th>Precision@K</th><th>Lift@K</th>
              <th>Binom p&gt;=k</th><th>Selected Classes</th>
            </tr>
          </thead>
          <tbody>{"".join(metric_rows)}</tbody>
        </table>
      </section>
      <section>
        <h2>Round Metrics</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Family</th><th>Round</th><th>Selected Target</th>
              <th>Prevalence</th><th>Precision@K</th><th>Lift@K</th>
              <th>Binom p&gt;=k</th>
            </tr>
          </thead>
          <tbody>{round_rows_html}</tbody>
        </table>
      </section>
      <section>
        <h2>Round Dynamics</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Family</th><th>Oracle</th><th>First Lift</th><th>Final Lift</th>
              <th>Max Lift</th><th>Max Round</th><th>Status</th>
            </tr>
          </thead>
          <tbody>{round_dynamics_html}</tbody>
        </table>
      </section>
      <section>
        <h2>Trajectory QA</h2>
        <table>
          <thead>
            <tr>
              <th>Seed</th><th>Family</th><th>Campaign</th><th>Split</th><th>Positive AUC</th>
              <th>Null AUC</th><th>AUC Delta</th><th>Final Delta</th><th>Status</th>
            </tr>
          </thead>
          <tbody>{trajectory_html}</tbody>
        </table>
        <table>
          <thead>
            <tr>
              <th>Seed</th><th>Pairs</th><th>AUC Delta Mean</th><th>AUC Delta Min</th>
              <th>Final Delta Mean</th><th>Status</th>
            </tr>
          </thead>
          <tbody>{seed_html}</tbody>
        </table>
      </section>
      <section>
        <h2>Campaign Reviews</h2>
        <ul>{"".join(campaign_links) if campaign_links else "<li>No OPAL campaign reviews written.</li>"}</ul>
      </section>
    </main>
    """
    return _html_document(title="DenseGen axis probe review", body=body)
