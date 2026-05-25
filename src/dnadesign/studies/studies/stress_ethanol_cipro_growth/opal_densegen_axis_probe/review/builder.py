"""Build DenseGen axis probe review artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import ProbeArtifactLayout
from ..decision import (
    decision_reasons_from_metrics,
    gate_results_from_metrics,
    metric_definitions,
    metric_quality_from_metrics,
    round_dynamics_summary,
    trajectory_qa_summary,
)
from ..status import audit_run_root
from .campaign_reviews import _build_campaign_reviews
from .configured_plots import _build_configured_plot_reviews, _plot_quality_summary, _review_next_steps
from .io import _write_json, _write_jsonl
from .metrics import _enriched_metrics_payload, _gate_coverage, _load_metrics, _review_decision, _review_problems
from .probe_plots import _write_probe_plots
from .rendering import render_probe_review_html, render_probe_review_markdown
from .run_manifest import _build_run_manifest


def build_probe_review(run_root: Path, *, include_plots: bool = True) -> dict[str, Any]:
    layout = ProbeArtifactLayout(Path(run_root).resolve())
    metrics_payload = _load_metrics(layout.metrics_path)
    metrics_payload = _enriched_metrics_payload(metrics_payload)
    audit = audit_run_root(layout.run_root)
    review_decision = _review_decision(metrics_payload)
    safety = metrics_payload.get("safety") if isinstance(metrics_payload.get("safety"), Mapping) else {}
    runs = metrics_payload.get("runs") if isinstance(metrics_payload.get("runs"), list) else []
    rounds = metrics_payload.get("rounds") if isinstance(metrics_payload.get("rounds"), list) else []
    round_rows = [row for row in rounds if isinstance(row, Mapping)]
    gate_results = gate_results_from_metrics(
        [row for row in runs if isinstance(row, Mapping)],
        safety,
        round_metrics=round_rows,
    )
    decision_reasons = decision_reasons_from_metrics(
        [row for row in runs if isinstance(row, Mapping)],
        safety,
        decision=review_decision,
        round_metrics=round_rows,
    )
    metric_quality = metric_quality_from_metrics([row for row in runs if isinstance(row, Mapping)])
    round_dynamics = round_dynamics_summary(round_rows)
    trajectory_qa = trajectory_qa_summary(
        [row for row in runs if isinstance(row, Mapping)],
        round_rows,
    )
    metrics_payload["decision"] = review_decision
    metrics_payload["decision_reasons"] = decision_reasons
    metrics_payload["gate_results"] = gate_results
    metrics_payload["metric_quality"] = metric_quality
    metrics_payload["metric_definitions"] = metric_definitions()
    metrics_payload["round_dynamics"] = round_dynamics
    metrics_payload["trajectory_qa"] = trajectory_qa
    campaign_reviews = _build_campaign_reviews(layout, metrics_payload=metrics_payload, include_plots=include_plots)
    configured_plots = _build_configured_plot_reviews(layout, metrics_payload=metrics_payload)
    plot_quality = _plot_quality_summary(configured_plots)
    review_problems = [
        *_review_problems(audit=audit, review_decision=review_decision),
        *_campaign_review_problems(campaign_reviews),
        *_plot_quality_problems(plot_quality),
    ]
    review_status = "attention" if review_problems else audit.status
    run_manifest = _build_run_manifest(
        layout,
        audit=audit,
        metrics_payload=metrics_payload,
        review_decision=review_decision,
        review_status=review_status,
        review_problems=review_problems,
        decision_reasons=decision_reasons,
        gate_results=gate_results,
        metric_quality=metric_quality,
    )
    next_steps = _review_next_steps(layout=layout, plot_quality=plot_quality)
    plot_paths = (
        _write_probe_plots(layout, metrics_payload=metrics_payload, configured_plots=configured_plots)
        if include_plots
        else []
    )
    review_manifest = {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.v1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "run_root": str(layout.run_root),
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "problems": review_problems,
        "decision_reasons": decision_reasons,
        "gate_results": gate_results,
        "metric_quality": metric_quality,
        "round_dynamics": round_dynamics,
        "trajectory_qa": trajectory_qa,
        "metric_definitions": metric_definitions(),
        "gate_coverage": _gate_coverage(metrics_payload.get("runs") or []),
        "opal_campaign_reviews": campaign_reviews,
        "opal_configured_plots": configured_plots,
        "plot_quality": plot_quality,
        "next_steps": next_steps,
        "probe_plots": [str(path) for path in plot_paths],
        "artifacts": {
            "review_markdown": str(layout.review_path),
            "review_html": str(layout.review_index_path),
            "review_manifest": str(layout.review_manifest_path),
            "run_manifest": str(layout.run_manifest_path),
        },
        "run_manifest": str(layout.run_manifest_path),
    }
    _write_json(layout.metrics_path, metrics_payload)
    if metrics_payload.get("runs"):
        _write_jsonl(layout.reports_dir / "selection_summary.jsonl", metrics_payload["runs"])
    if metrics_payload.get("rounds"):
        _write_jsonl(layout.reports_dir / "round_metrics.jsonl", metrics_payload["rounds"])
    _write_json(layout.run_manifest_path, run_manifest)
    _write_json(layout.review_manifest_path, review_manifest)
    status_payload = audit.to_dict()
    status_payload["decision_reasons"] = decision_reasons
    status_payload["gate_results"] = gate_results
    status_payload["metric_quality"] = metric_quality
    status_payload["round_dynamics"] = round_dynamics
    status_payload["trajectory_qa"] = trajectory_qa
    _write_json(layout.status_path, status_payload)
    layout.review_path.write_text(render_probe_review_markdown(review_manifest, metrics_payload), encoding="utf-8")
    layout.review_index_path.write_text(
        render_probe_review_html(review_manifest, metrics_payload, base_dir=layout.reports_dir),
        encoding="utf-8",
    )
    return {
        "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.v1",
        "run_root": str(layout.run_root),
        "review": str(layout.review_path),
        "index": str(layout.review_index_path),
        "review_manifest": str(layout.review_manifest_path),
        "run_manifest": str(layout.run_manifest_path),
        "plots": [str(path) for path in plot_paths],
        "opal_campaign_reviews": campaign_reviews,
        "opal_configured_plots": configured_plots,
        "plot_quality": plot_quality,
        "next_steps": next_steps,
        "decision": review_decision,
        "persisted_decision": audit.decision,
        "status": review_status,
        "problems": review_problems,
    }


def _campaign_review_problems(campaign_reviews: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for row in campaign_reviews:
        run_key = str(row.get("run_key") or "unknown")
        warnings = row.get("warnings") or []
        stale_artifacts = row.get("stale_artifacts") or []
        if warnings:
            problems.append(f"opal_campaign_review_warnings:{run_key}:{len(warnings)}")
        if stale_artifacts:
            problems.append(f"opal_campaign_review_stale_artifacts:{run_key}:{len(stale_artifacts)}")
    return problems


def _plot_quality_problems(plot_quality: Mapping[str, Any]) -> list[str]:
    if plot_quality.get("status") == "ok":
        return []
    return [
        f"configured_plot_quality:{problem.get('run_key', 'unknown')}:{problem.get('problem', 'unknown')}"
        for problem in plot_quality.get("problems") or []
    ]
