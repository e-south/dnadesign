"""Review artifacts for the study-owned DenseGen axis OPAL probe."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from dnadesign.opal import build_campaign_review

from .artifacts import ProbeArtifactLayout
from .constants import NULL_ORACLE_ID, ORACLE_ID
from .decision import _decision_from_metrics
from .status import audit_run_root


def build_probe_review(run_root: Path, *, include_plots: bool = True) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    metrics_payload = _load_metrics(layout.metrics_path)
    audit = audit_run_root(layout.run_root)
    review_decision = _review_decision(metrics_payload)
    review_problems = _review_problems(audit=audit, review_decision=review_decision)
    review_status = "attention" if review_problems else audit.status
    run_manifest = _build_run_manifest(
        layout,
        audit=audit,
        metrics_payload=metrics_payload,
        review_decision=review_decision,
        review_status=review_status,
        review_problems=review_problems,
    )
    campaign_reviews = _build_campaign_reviews(layout, metrics_payload=metrics_payload, include_plots=include_plots)
    plot_paths = _write_probe_plots(layout, metrics_payload=metrics_payload) if include_plots else []
    review_manifest = {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "problems": review_problems,
        "gate_coverage": _gate_coverage(metrics_payload.get("runs") or []),
        "opal_campaign_reviews": campaign_reviews,
        "probe_plots": [str(path) for path in plot_paths],
        "artifacts": {
            "review_markdown": str(layout.review_path),
            "review_html": str(layout.review_index_path),
            "review_manifest": str(layout.review_manifest_path),
            "run_manifest": str(layout.run_manifest_path),
        },
        "run_manifest": str(layout.run_manifest_path),
    }
    _write_json(layout.run_manifest_path, run_manifest)
    _write_json(layout.review_manifest_path, review_manifest)
    layout.review_path.write_text(render_probe_review_markdown(review_manifest, metrics_payload), encoding="utf-8")
    layout.review_index_path.write_text(
        render_probe_review_html(review_manifest, metrics_payload, base_dir=layout.reports_dir),
        encoding="utf-8",
    )
    return {
        "run_root": str(layout.run_root),
        "review": str(layout.review_path),
        "index": str(layout.review_index_path),
        "review_manifest": str(layout.review_manifest_path),
        "run_manifest": str(layout.run_manifest_path),
        "plots": [str(path) for path in plot_paths],
        "opal_campaign_reviews": campaign_reviews,
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "problems": review_problems,
    }


def render_probe_review_markdown(review_manifest: Mapping[str, Any], metrics_payload: Mapping[str, Any]) -> str:
    runs = metrics_payload.get("runs") or []
    coverage = review_manifest.get("gate_coverage") or {}
    campaign_reviews = review_manifest.get("opal_campaign_reviews") or []
    plots = review_manifest.get("probe_plots") or []
    problems = review_manifest.get("problems") or []
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
        f"- problems: `{', '.join(problems) if problems else 'none'}`",
        f"- run_root: `{review_manifest.get('run_root')}`",
        "",
        "## Coverage",
        "",
        f"- campaigns: `{', '.join(coverage.get('campaigns') or []) or 'none'}`",
        f"- splits: `{', '.join(coverage.get('splits') or []) or 'none'}`",
        f"- positive/null pairs complete: `{coverage.get('positive_null_pairs_complete')}`",
        f"- omitted scored gates: `{', '.join(coverage.get('omitted_scored_gates') or []) or 'none'}`",
        "",
        "## Metrics",
        "",
    ]
    if runs:
        lines.extend(
            [
                "| run_key | oracle | split | precision@K true | lift true | selected classes |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for row in runs:
            classes = row.get("off_target_class_distribution_true") or {}
            class_text = ", ".join(f"{key}:{value}" for key, value in classes.items())
            lines.append(
                "| `{run_key}` | `{oracle}` | `{split}` | {precision} | {lift} | {classes} |".format(
                    run_key=row.get("run_key"),
                    oracle=row.get("oracle_id"),
                    split=row.get("split_id"),
                    precision=row.get("selected_target_precision_at_k_true"),
                    lift=row.get("target_lift_at_k_true"),
                    classes=class_text,
                )
            )
    else:
        lines.append("No scored OPAL run metrics are present yet.")
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
    lines.append("")
    return "\n".join(lines)


def render_probe_review_html(
    review_manifest: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    *,
    base_dir: Path,
) -> str:
    runs = metrics_payload.get("runs") or []
    coverage = review_manifest.get("gate_coverage") or {}
    campaign_reviews = review_manifest.get("opal_campaign_reviews") or []
    plots = review_manifest.get("probe_plots") or []
    problems = review_manifest.get("problems") or []
    plot_cards = []
    for path in plots:
        src = _rel(path, base_dir=base_dir)
        plot_cards.append(
            "<article>"
            f"<h3>{_e(Path(str(path)).stem.replace('_', ' '))}</h3>"
            f'<a href="{_e(src)}"><img src="{_e(src)}" alt="{_e(Path(str(path)).stem)}"></a>'
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
            f"<td>{_e(row.get('oracle_id'))}</td>"
            f"<td>{_e(row.get('split_id'))}</td>"
            f"<td>{_e(row.get('selected_target_precision_at_k_true'))}</td>"
            f"<td>{_e(row.get('target_lift_at_k_true'))}</td>"
            f"<td>{_e(class_text)}</td>"
            "</tr>"
        )
    body = f"""
    <header>
      <p>DenseGen axis probe review</p>
      <h1>{_e(review_manifest.get("decision"))}</h1>
      <p class="lede">Study benchmark layer over OPAL campaign review artifacts.</p>
    </header>
    <main>
      <section class="summary-grid">
        {_metric_card("Status", review_manifest.get("status"))}
        {_metric_card("Problems", ", ".join(problems) if problems else "none")}
        {_metric_card("Persisted decision", review_manifest.get("persisted_decision"))}
        {_metric_card("Runs", len(runs))}
        {_metric_card("Campaigns", ", ".join(coverage.get("campaigns") or []) or "none")}
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
        <h2>Probe Plots</h2>
        <div class="plot-grid">{"".join(plot_cards) if plot_cards else "<p>No probe aggregate plots written.</p>"}</div>
      </section>
      <section>
        <h2>Scored Runs</h2>
        <table>
          <thead>
            <tr>
              <th>Run</th><th>Oracle</th><th>Split</th><th>Precision@K</th><th>Lift@K</th><th>Selected Classes</th>
            </tr>
          </thead>
          <tbody>{"".join(metric_rows)}</tbody>
        </table>
      </section>
      <section>
        <h2>Campaign Reviews</h2>
        <ul>{"".join(campaign_links) if campaign_links else "<li>No OPAL campaign reviews written.</li>"}</ul>
      </section>
    </main>
    """
    return _html_document(title="DenseGen axis probe review", body=body)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"metrics.json not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"metrics.json is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("metrics.json must contain a JSON object")
    if not isinstance(payload.get("safety"), dict):
        raise RuntimeError("metrics.json missing object field: safety")
    if not isinstance(payload.get("runs"), list):
        raise RuntimeError("metrics.json missing list field: runs")
    return payload


def _review_decision(metrics_payload: Mapping[str, Any]) -> str:
    safety = metrics_payload.get("safety")
    runs = metrics_payload.get("runs")
    if not isinstance(safety, Mapping) or not isinstance(runs, list):
        raise RuntimeError("metrics.json is missing safety/runs contract fields")
    return _decision_from_metrics([dict(row) for row in runs if isinstance(row, Mapping)], safety)


def _review_problems(*, audit, review_decision: str | None) -> list[str]:
    problems = list(audit.problems)
    if audit.decision and review_decision and audit.decision != review_decision:
        problems.append(f"persisted_decision_mismatch:{audit.decision}!={review_decision}")
    return problems


def _build_campaign_reviews(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
    include_plots: bool,
) -> list[dict[str, Any]]:
    reviewed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in metrics_payload.get("runs") or []:
        if not isinstance(row, Mapping):
            raise RuntimeError("metrics runs entries must be objects")
        run_key = str(row.get("run_key") or "").strip()
        if not run_key or run_key in seen:
            continue
        seen.add(run_key)
        config_path = layout.campaign_config_path(run_key)
        if not config_path.exists():
            raise RuntimeError(f"scratch campaign config missing for scored run {run_key}: {config_path}")
        run_id = str(row.get("run_id") or "").strip() or None
        round_value = row.get("as_of_round")
        round_selector = str(int(round_value)) if round_value is not None else "latest"
        result = build_campaign_review(
            config_path,
            round_selector=round_selector,
            run_id=run_id,
            include_plots=include_plots,
        )
        reviewed.append(
            {
                "run_key": run_key,
                "status": "written",
                "config_path": str(config_path),
                "review_path": str(result.review_path),
                "index_path": str(result.index_path),
                "manifest_path": str(result.manifest_path),
                "plot_paths": [str(path) for path in result.plot_paths],
                "round_index": result.manifest["review_scope"]["round_index"],
                "run_id": result.manifest["review_scope"]["run_id"],
            }
        )
    return reviewed


def _gate_coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    campaigns = sorted({str(row.get("campaign")) for row in runs if row.get("campaign")})
    splits = sorted({str(row.get("split_id")) for row in runs if row.get("split_id")})
    pair_counts: dict[tuple[str, str], set[str]] = {}
    for row in runs:
        key = (str(row.get("campaign")), str(row.get("split_id")))
        pair_counts.setdefault(key, set()).add(str(row.get("oracle_id")))
    positive_null_pairs_complete = all({ORACLE_ID, NULL_ORACLE_ID}.issubset(values) for values in pair_counts.values())
    omitted: list[str] = []
    if "ethanol" not in campaigns:
        omitted.append("ethanol")
    if "dual" not in campaigns:
        omitted.append("dual")
    if "leave_sigma35_variant" not in splits:
        omitted.append("leave_sigma35_variant")
    return {
        "campaigns": campaigns,
        "splits": splits,
        "positive_null_pairs_complete": bool(positive_null_pairs_complete) if runs else False,
        "omitted_scored_gates": omitted,
    }


def _build_run_manifest(
    layout: ProbeArtifactLayout,
    *,
    audit,
    metrics_payload: Mapping[str, Any],
    review_decision: str | None,
    review_status: str,
    review_problems: list[str],
) -> dict[str, Any]:
    inventory = _artifact_inventory(layout.run_root)
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.run_manifest.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "planned_campaign_count": audit.planned_campaign_count,
        "metrics_run_count": len(metrics_payload.get("runs") or []),
        "shared_sidecar_present": audit.shared_sidecar_present,
        "artifact_inventory": inventory,
        "problems": review_problems,
    }


def _artifact_inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"file_count": 0, "total_bytes": 0}
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        total_bytes += int(path.stat().st_size)
    return {"file_count": file_count, "total_bytes": total_bytes}


def _write_probe_plots(
    layout: ProbeArtifactLayout,
    *,
    metrics_payload: Mapping[str, Any],
) -> list[Path]:
    runs = metrics_payload.get("runs") or []
    if not runs:
        return []
    layout.review_plots_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(runs)
    paths = [
        layout.review_plots_dir / "target_lift_and_precision.png",
        layout.review_plots_dir / "selected_class_composition.png",
    ]
    _plot_lift_and_precision(frame, paths[0])
    _plot_class_composition(frame, paths[1])
    return paths


def _plot_lift_and_precision(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    df = frame.copy()
    df["label"] = df["run_key"].astype(str)
    x = range(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].bar(x, pd.to_numeric(df["target_lift_at_k_true"], errors="coerce"), color="#446A8C")
    axes[0].set_ylabel("target lift@K")
    axes[0].set_title("Probe target lift")
    axes[1].bar(x, pd.to_numeric(df["selected_target_precision_at_k_true"], errors="coerce"), color="#7A6B3F")
    axes[1].set_ylabel("precision@K")
    axes[1].set_title("Probe selected target precision")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["label"].tolist(), rotation=35, ha="right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_class_composition(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        dist = row.get("off_target_class_distribution_true") or {}
        if not isinstance(dist, Mapping):
            continue
        out = {"run_key": str(row.get("run_key"))}
        out.update({str(key): int(value) for key, value in dist.items()})
        rows.append(out)
    if not rows:
        raise RuntimeError("class composition plot requires off_target_class_distribution_true metrics")
    wide = pd.DataFrame(rows).fillna(0)
    classes = [col for col in wide.columns if col != "run_key"]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bottom = [0] * len(wide)
    palette = ["#446A8C", "#8C4E4A", "#5D7D4F", "#7A6B3F", "#6C5F7D", "#4F7D75"]
    for index, axis_class in enumerate(classes):
        values = wide[axis_class].astype(int).to_list()
        ax.bar(wide["run_key"].tolist(), values, bottom=bottom, label=axis_class, color=palette[index % len(palette)])
        bottom = [prev + value for prev, value in zip(bottom, values, strict=True)]
    ax.set_ylabel("selected count")
    ax.set_title("Selected class composition")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _rel(path: Any, *, base_dir: Path) -> str:
    return os.path.relpath(str(path), str(base_dir))


def _metric_card(label: str, value: Any) -> str:
    return f'<article class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></article>'


def _html_document(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2528;
      --muted: #667074;
      --line: #d8ddd7;
      --accent: #8c4e4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header, main {{ width: min(1180px, calc(100vw - 40px)); margin: 0 auto; }}
    header {{ padding: 34px 0 16px; }}
    header > p:first-child {{
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      margin: 0 0 6px;
      text-transform: uppercase;
    }}
    .lede {{ color: var(--muted); margin: 8px 0 0; }}
    h1 {{ font-size: clamp(1.8rem, 2.8vw, 3rem); margin: 0; overflow-wrap: anywhere; }}
    h2 {{
      border-bottom: 1px solid var(--line);
      font-size: 1.18rem;
      margin: 30px 0 14px;
      padding-bottom: 8px;
    }}
    code {{ background: #eef1ef; border-radius: 4px; padding: 1px 5px; }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    .metric, .plot-grid article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 4%);
    }}
    .metric {{ min-height: 86px; padding: 14px; }}
    .metric span {{ color: var(--muted); display: block; font-size: 0.78rem; text-transform: uppercase; }}
    .metric strong {{ display: block; font-size: 1.25rem; margin-top: 8px; overflow-wrap: anywhere; }}
    dl {{ display: grid; gap: 8px 16px; grid-template-columns: minmax(160px, max-content) 1fr; }}
    dt {{ color: var(--muted); font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .plot-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .plot-grid article {{ padding: 12px; }}
    .plot-grid h3 {{ font-size: 0.95rem; margin: 0 0 10px; text-transform: capitalize; }}
    img {{ display: block; height: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
    li {{ margin: 6px 0; }}
    @media (max-width: 640px) {{
      header, main {{ width: min(100vw - 24px, 1180px); }}
      dl {{ grid-template-columns: 1fr; }}
      .plot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
