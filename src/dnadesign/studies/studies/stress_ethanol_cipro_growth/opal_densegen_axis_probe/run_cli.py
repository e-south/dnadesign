"""Run-command implementation for the DenseGen axis OPAL probe."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ProbeArtifactLayout
from .axis_oracle import build_axis_oracle, make_permuted_labels
from .constants import NULL_ORACLE_ID, ORACLE_ID, RUN_STAGES
from .decision import (
    _decision_from_metrics,
    _write_decision,
    decision_reasons_from_metrics,
    enrich_metric_rows,
    gate_results_from_metrics,
    metric_definitions,
    metric_quality_from_metrics,
)
from .decision_evaluation import _evaluate_run, _evaluate_run_rounds
from .decision_inputs import (
    _compact_split_metadata,
    _format_plan_text,
    _source_summary,
    _split_metadata_for_all,
)
from .execution import materialize_probe_inputs, run_opal_rounds_for_probe
from .paths import _default_run_root, _repo_root_from, _resolve_repo_path, validate_run_root_policy
from .plan import build_plan
from .plan_fingerprint import build_plan_record, prepare_probe_run_root
from .scratch import _load_candidate_inputs, _write_json
from .source_contract import validate_candidate_x_surface
from .status import _format_status_text, audit_run_root


def _run_probe(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from(Path.cwd())
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or (
        f"{timestamp}_seed{int(args.seed)}_initial{int(args.initial_labels)}_k{int(args.selection_k)}"
    )
    run_root = Path(args.run_root) if args.run_root else _default_run_root(repo_root, run_id)
    run_root = _resolve_repo_path(repo_root, run_root)
    if args.apply:
        validate_run_root_policy(
            repo_root=repo_root,
            run_root=run_root,
            allow_custom=bool(args.allow_custom_run_root),
        )
    splits = tuple(str(args.splits).split(",")) if args.splits else ()
    plan = build_plan(
        run_root=run_root,
        initial_label_count=int(args.initial_labels),
        selection_k=int(args.selection_k),
        max_x_matrix_gib=args.max_x_matrix_gib,
        score_batch_size=args.score_batch_size,
        seed=int(args.seed),
        rounds=int(args.rounds),
        gate=args.gate,
        splits=splits,
        apply=bool(args.apply),
        stop_after=str(args.stop_after),
    )

    candidates, densegen_sidecar = _load_candidate_inputs(repo_root)
    labels = build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)
    x_surface = validate_candidate_x_surface(repo_root, expected_rows=len(labels))
    safety = _source_summary(labels, run_root=run_root, x_surface=x_surface)
    split_metadata = _split_metadata_for_all(labels, plan=plan)
    x_memory_budget = _x_memory_budget_reports(plan=plan, split_metadata=split_metadata, x_surface=x_surface)

    plan_payload = {
        "source_summary": safety,
        "run_root": str(run_root),
        "planned_runs": len(plan.runs),
        "gate": plan.gate,
        "stop_after": plan.stop_after,
        "seed": plan.seed,
        "split_ids": list(plan.splits),
        "rounds": plan.rounds,
        "initial_label_count": plan.initial_label_count,
        "selection_k": plan.selection_k,
        "max_x_matrix_gib": plan.max_x_matrix_gib,
        "score_batch_size": plan.score_batch_size,
        "x_memory_budget": x_memory_budget,
        "splits": _compact_split_metadata(split_metadata),
        "commands": plan.commands,
    }
    plan_record = build_plan_record(plan_payload)
    if args.json and not args.apply:
        print(
            json.dumps(
                {
                    **plan_payload,
                    "plan_fingerprint": plan_record["fingerprint"],
                    "plan_path": str(ProbeArtifactLayout(run_root).probe_plan_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif not args.json:
        print(_format_plan_text(plan=plan, safety=safety, split_metadata=split_metadata))

    if not args.apply:
        return 0

    layout = ProbeArtifactLayout(run_root)
    plan_record = prepare_probe_run_root(
        layout,
        plan_payload=plan_payload,
        replace_run_root=bool(args.replace_run_root),
    )
    null_labels = make_permuted_labels(labels, seed=int(args.seed))
    materialize_probe_inputs(
        repo_root=repo_root,
        plan=plan,
        labels=labels,
        null_labels=null_labels,
        split_metadata=split_metadata,
    )

    metrics: list[dict[str, Any]] = []
    scored = RUN_STAGES.index(plan.stop_after) >= RUN_STAGES.index("run")
    if args.gate != "source":
        labels_by_oracle = {ORACLE_ID: labels, NULL_ORACLE_ID: null_labels}
        labeled_ids_by_run = run_opal_rounds_for_probe(
            repo_root=repo_root,
            plan=plan,
            labels_by_oracle=labels_by_oracle,
            split_metadata=split_metadata,
            machine_readable=bool(args.json),
        )
        if scored:
            for run in plan.runs:
                run_labels = labels if run.oracle_id == ORACLE_ID else null_labels
                run_metadata = dict(split_metadata[run.split_id])
                run_metadata["train_ids"] = sorted(
                    labeled_ids_by_run.get(run.run_key, set(map(str, split_metadata[run.split_id]["train_ids"])))
                )
                metrics.append(
                    _evaluate_run(
                        run=run,
                        positive_labels=labels,
                        run_labels=run_labels,
                        split_metadata=run_metadata,
                    )
                )
    round_metrics = _round_metrics_for_plan(
        scored=scored,
        plan_runs=plan.runs,
        labels=labels,
        null_labels=null_labels,
        split_metadata=split_metadata,
    )
    return _write_run_outputs(
        args=args,
        layout=layout,
        plan_payload=plan_payload,
        plan_record=plan_record,
        safety=safety,
        metrics=metrics,
        round_metrics=round_metrics,
    )


def _round_metrics_for_plan(
    *,
    scored: bool,
    plan_runs,
    labels: pd.DataFrame,
    null_labels: pd.DataFrame,
    split_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not scored:
        return []
    rows: list[dict[str, Any]] = []
    for run in plan_runs:
        run_labels = labels if run.oracle_id == ORACLE_ID else null_labels
        rows.extend(
            _evaluate_run_rounds(
                run=run,
                positive_labels=labels,
                run_labels=run_labels,
                split_metadata=dict(split_metadata[run.split_id]),
            )
        )
    return rows


def _write_run_outputs(
    *,
    args: argparse.Namespace,
    layout: ProbeArtifactLayout,
    plan_payload: dict[str, Any],
    plan_record: dict[str, Any],
    safety: dict[str, Any],
    metrics: list[dict[str, Any]],
    round_metrics: list[dict[str, Any]],
) -> int:
    decision = _decision_from_metrics(metrics, safety)
    enriched_metrics = enrich_metric_rows(metrics)
    round_metric_rows = enrich_metric_rows(round_metrics)
    gate_results = gate_results_from_metrics(enriched_metrics, safety)
    decision_reasons = decision_reasons_from_metrics(enriched_metrics, safety, decision=decision)
    metric_quality = metric_quality_from_metrics(enriched_metrics)
    _write_metrics(
        layout,
        safety,
        enriched_metrics,
        round_metric_rows,
        decision,
        decision_reasons,
        gate_results,
        metric_quality,
    )
    _write_decision(
        path=layout.decision_path,
        decision=decision,
        safety=safety,
        metrics=enriched_metrics,
        quality_counts=safety["quality_counts"],
    )
    audit = audit_run_root(layout.run_root)
    status_payload = audit.to_dict()
    status_payload["decision_reasons"] = decision_reasons
    status_payload["gate_results"] = gate_results
    status_payload["metric_quality"] = metric_quality
    _write_json(layout.status_path, status_payload)
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": "stress_ethanol_cipro_growth.opal_densegen_axis_probe.run.v1",
                    "mode": "apply",
                    "plan": plan_payload,
                    "plan_fingerprint": plan_record["fingerprint"],
                    "plan_path": str(layout.probe_plan_path),
                    "decision": decision,
                    "reports": str(layout.reports_dir),
                    "status": audit.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"decision={decision}")
        print(f"reports={layout.reports_dir}")
        print(_format_status_text(audit))
    return 0


def _write_metrics(
    layout: ProbeArtifactLayout,
    safety: dict[str, Any],
    enriched_metrics: list[dict[str, Any]],
    round_metric_rows: list[dict[str, Any]],
    decision: str,
    decision_reasons: list[dict[str, Any]],
    gate_results: list[dict[str, Any]],
    metric_quality: dict[str, Any],
) -> None:
    _write_json(
        layout.metrics_path,
        {
            "safety": safety,
            "runs": enriched_metrics,
            "rounds": round_metric_rows,
            "decision": decision,
            "metric_definitions": metric_definitions(),
            "decision_reasons": decision_reasons,
            "gate_results": gate_results,
            "metric_quality": metric_quality,
        },
    )
    if enriched_metrics:
        pd.DataFrame(enriched_metrics).to_csv(layout.reports_dir / "selection_summary.csv", index=False)
        (layout.reports_dir / "selection_summary.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in enriched_metrics) + "\n",
            encoding="utf-8",
        )
    if round_metric_rows:
        pd.DataFrame(round_metric_rows).to_csv(layout.reports_dir / "round_metrics.csv", index=False)
        (layout.reports_dir / "round_metrics.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in round_metric_rows) + "\n",
            encoding="utf-8",
        )


def _x_memory_budget_reports(
    *,
    plan,
    split_metadata: dict[str, dict[str, Any]],
    x_surface: dict[str, Any],
) -> list[dict[str, Any]]:
    from dnadesign.opal import enforce_x_matrix_memory_budget

    reports: list[dict[str, Any]] = []
    if RUN_STAGES.index(plan.stop_after) < RUN_STAGES.index("run"):
        return reports
    for split_id, metadata in split_metadata.items():
        train_count = int(len(metadata.get("train_ids") or []))
        eval_count = int(len(metadata.get("eval_ids") or []))
        score_batch_size = int(plan.score_batch_size or 10_000)
        estimate = enforce_x_matrix_memory_budget(
            row_count=train_count + min(eval_count, score_batch_size),
            x_dim=int(x_surface["x_dim"]),
            item_size_bytes=8,
            max_gib=plan.max_x_matrix_gib,
            context=f"DenseGen probe split {split_id} OPAL streaming X batch",
        )
        reports.append(
            {
                "split_id": split_id,
                "scope": "streaming_score_batch",
                "split_rows": int(train_count + eval_count),
                "score_batch_size": int(score_batch_size),
                "rows": int(estimate.row_count),
                "x_dim": int(estimate.x_dim),
                "raw_gib": float(estimate.raw_gib),
                "estimated_gib": float(estimate.estimated_gib),
                "max_gib": float(estimate.max_gib),
            }
        )
    return reports
