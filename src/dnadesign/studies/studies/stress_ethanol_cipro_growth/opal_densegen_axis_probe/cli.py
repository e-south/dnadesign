"""Command-line entrypoint for the study-owned DenseGen axis OPAL probe."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .artifacts import ProbeArtifactLayout
from .constants import (
    DEFAULT_INITIAL_LABELS,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    NULL_ORACLE_ID,
    ORACLE_ID,
    RUN_STAGES,
    SPLITS,
)
from .execution import materialize_probe_inputs, run_opal_rounds_for_probe
from .paths import _default_run_root, _repo_root_from, _resolve_repo_path, validate_run_root_policy
from .status import _format_status_text, audit_run_root


def _run_probe(args: argparse.Namespace) -> int:
    import pandas as pd

    from .axis_oracle import build_axis_oracle, make_permuted_labels
    from .decision import (
        _compact_split_metadata,
        _decision_from_metrics,
        _evaluate_run,
        _evaluate_run_rounds,
        _format_plan_text,
        _source_summary,
        _split_metadata_for_all,
        _write_decision,
        decision_reasons_from_metrics,
        enrich_metric_rows,
        gate_results_from_metrics,
        metric_definitions,
        metric_quality_from_metrics,
    )
    from .plan import build_plan
    from .scratch import _load_candidate_inputs, _write_json
    from .source_contract import validate_candidate_x_surface

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
    splits = tuple(str(args.splits).split(",")) if args.splits else SPLITS
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
        "rounds": plan.rounds,
        "initial_label_count": plan.initial_label_count,
        "selection_k": plan.selection_k,
        "max_x_matrix_gib": plan.max_x_matrix_gib,
        "score_batch_size": plan.score_batch_size,
        "x_memory_budget": x_memory_budget,
        "splits": _compact_split_metadata(split_metadata),
        "commands": plan.commands,
    }
    if args.json and not args.apply:
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
    elif not args.json:
        print(_format_plan_text(plan=plan, safety=safety, split_metadata=split_metadata))

    if not args.apply:
        return 0

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
    labeled_ids_by_run: dict[str, set[str]] = {}
    if args.gate != "source":
        labels_by_oracle = {ORACLE_ID: labels, NULL_ORACLE_ID: null_labels}
        labeled_ids_by_run = run_opal_rounds_for_probe(
            repo_root=repo_root,
            plan=plan,
            labels_by_oracle=labels_by_oracle,
            split_metadata=split_metadata,
            machine_readable=bool(args.json),
        )
        for run in plan.runs:
            if scored:
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
    round_metrics: list[dict[str, Any]] = []
    if scored:
        for run in plan.runs:
            run_labels = labels if run.oracle_id == ORACLE_ID else null_labels
            round_metrics.extend(
                _evaluate_run_rounds(
                    run=run,
                    positive_labels=labels,
                    run_labels=run_labels,
                    split_metadata=dict(split_metadata[run.split_id]),
                )
            )

    decision = _decision_from_metrics(metrics, safety)
    enriched_metrics = enrich_metric_rows(metrics)
    gate_results = gate_results_from_metrics(enriched_metrics, safety)
    decision_reasons = decision_reasons_from_metrics(enriched_metrics, safety, decision=decision)
    metric_quality = metric_quality_from_metrics(enriched_metrics)
    metrics_payload = {
        "safety": safety,
        "runs": enriched_metrics,
        "rounds": enrich_metric_rows(round_metrics),
        "decision": decision,
        "metric_definitions": metric_definitions(),
        "decision_reasons": decision_reasons,
        "gate_results": gate_results,
        "metric_quality": metric_quality,
    }
    layout = ProbeArtifactLayout(run_root)
    _write_json(layout.metrics_path, metrics_payload)
    if enriched_metrics:
        pd.DataFrame(enriched_metrics).to_csv(layout.reports_dir / "selection_summary.csv", index=False)
        (layout.reports_dir / "selection_summary.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in enriched_metrics) + "\n",
            encoding="utf-8",
        )
    if round_metrics:
        round_metric_rows = enrich_metric_rows(round_metrics)
        pd.DataFrame(round_metric_rows).to_csv(layout.reports_dir / "round_metrics.csv", index=False)
        (layout.reports_dir / "round_metrics.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in round_metric_rows) + "\n",
            encoding="utf-8",
        )
    _write_decision(
        path=layout.decision_path,
        decision=decision,
        safety=safety,
        metrics=enriched_metrics,
        quality_counts=safety["quality_counts"],
    )
    audit = audit_run_root(run_root)
    status_payload = audit.to_dict()
    status_payload["decision_reasons"] = decision_reasons
    status_payload["gate_results"] = gate_results
    status_payload["metric_quality"] = metric_quality
    _write_json(layout.status_path, status_payload)
    if args.json:
        schema = "stress_ethanol_cipro_growth.opal_densegen_axis_probe.run.v1"
        payload = dict(
            schema_version=schema,
            mode="apply",
            plan=plan_payload,
            decision=decision,
            reports=str(layout.reports_dir),
            status=audit.to_dict(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"decision={decision}")
        print(f"reports={layout.reports_dir}")
        print(_format_status_text(audit))
    return 0


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
        row_count = train_count + min(eval_count, score_batch_size)
        estimate = enforce_x_matrix_memory_budget(
            row_count=row_count,
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


def _status_probe(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    audit = audit_run_root(run_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_status_text(audit))
    return 1 if audit.status in {"missing", "attention"} else 0


def _report_probe(args: argparse.Namespace) -> int:
    from .review import build_probe_review

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    payload = build_probe_review(run_root, include_plots=bool(args.plots))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("opal_densegen_axis_probe_v0 review written")
        print(f"review={payload['review']}")
        print(f"index={payload['index']}")
        print(f"review_manifest={payload['review_manifest']}")
        print(f"run_manifest={payload['run_manifest']}")
        print(f"decision={payload['decision']}")
        print(f"status={payload['status']}")
    return 0


def _progress_probe(args: argparse.Namespace) -> int:
    from .progress import format_probe_progress_text, summarize_probe_progress

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    payload = summarize_probe_progress(run_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_probe_progress_text(payload))
    return 0


def _plot_probe(args: argparse.Namespace) -> int:
    from .plotting import generate_probe_campaign_plots

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    payload = generate_probe_campaign_plots(run_root, round_selector=str(args.round))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("opal_densegen_axis_probe_v0 configured plots")
        print(f"run_root={payload['run_root']}")
        print(f"campaign_count={payload['campaign_count']}")
        print(f"any_fail={payload['any_fail']}")
        print(f"mpl_config_dir={payload['mpl_config_dir']}")
    return 1 if payload["any_fail"] else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stress ethanol/cipro DenseGen axis OPAL probe.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Plan or execute the scratch OPAL probe.")
    run.add_argument(
        "--initial-labels",
        type=int,
        default=DEFAULT_INITIAL_LABELS,
        help="Initial labeled seed count before OPAL selections are added round over round.",
    )
    run.add_argument(
        "--selection-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of greedy selections made by each OPAL round.",
    )
    run.add_argument(
        "--max-x-matrix-gib",
        type=float,
        default=None,
        help=("Explicit OPAL X matrix memory budget for scratch campaigns. Default uses OPAL safety.max_x_matrix_gib."),
    )
    run.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help="OPAL scoring batch size for scratch campaigns. Lower this on memory-constrained hosts.",
    )
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of synthetic OPAL label/run rounds per scratch campaign.",
    )
    run.add_argument("--splits", default="random_id,leave_sigma35_variant")
    run.add_argument("--gate", choices=["source", "cipro-random", "random-all", "leave-sigma35", "all"], default="all")
    run.add_argument("--run-root", default=None)
    run.add_argument("--run-id", default=None)
    run.add_argument(
        "--allow-custom-run-root",
        action="store_true",
        help="Allow --apply writes to an external scratch root; repo-local writes stay under .var/studies.",
    )
    run.add_argument(
        "--stop-after",
        choices=RUN_STAGES,
        default="status",
        help="Apply path stage limit. Use 'validate' to dogfood configs without scoring the full candidate pool.",
    )
    run.add_argument("--json", action="store_true", help="Emit machine-readable JSON summaries.")
    run.add_argument("--apply", action="store_true")
    status = subparsers.add_parser("status", help="Audit an existing probe run root.")
    status.add_argument("--run-root", required=True)
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON status.")
    report = subparsers.add_parser("report", help="Write review artifacts for an existing probe run root.")
    report.add_argument("--run-root", required=True)
    report.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Write review plots.")
    report.add_argument("--json", action="store_true", help="Emit machine-readable JSON status.")
    progress = subparsers.add_parser("progress", help="Summarize OPAL round-log progress for a probe run root.")
    progress.add_argument("--run-root", required=True)
    progress.add_argument("--json", action="store_true", help="Emit machine-readable JSON progress.")
    plot = subparsers.add_parser(
        "plot",
        help="Generate configured OPAL plots for all scratch campaigns in one Python process.",
    )
    plot.add_argument("--run-root", required=True)
    plot.add_argument("--round", default="all", help="Round selector passed to OPAL plot generation.")
    plot.add_argument("--json", action="store_true", help="Emit machine-readable JSON plot summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_probe(args)
        if args.command == "status":
            return _status_probe(args)
        if args.command == "report":
            return _report_probe(args)
        if args.command == "progress":
            return _progress_probe(args)
        if args.command == "plot":
            return _plot_probe(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    parser.error(f"unsupported command: {args.command}")
    return 2
