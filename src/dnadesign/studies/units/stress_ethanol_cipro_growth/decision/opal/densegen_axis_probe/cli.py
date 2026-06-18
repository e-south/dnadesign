"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/cli.py

Command-line entrypoint for the study-owned DenseGen axis OPAL probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Sequence

from .core.constants import (
    ACTIVE_LABEL_FAMILY_IDS,
    DEFAULT_INITIAL_LABELS,
    DEFAULT_SEED,
    DEFAULT_SUITE_ID,
    DEFAULT_TOP_K,
    RUN_STAGES,
)
from .core.paths import _repo_root_from, _resolve_repo_path
from .reporting.status import _format_status_text, audit_run_root
from .tfbs.cli import add_tfbs_subcommands, handle_tfbs_command


def _status_probe(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    audit = audit_run_root(run_root)
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    else:
        print(_format_status_text(audit))
    return 0 if audit.status == "ok" else 1


def _report_probe(args: argparse.Namespace) -> int:
    from .reporting.review import build_probe_review

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            payload = build_probe_review(run_root, include_plots=bool(args.plots))
    else:
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
    from .reporting.progress import format_probe_progress_text, summarize_probe_progress

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    payload = summarize_probe_progress(run_root, include_opal_progress=bool(args.full))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_probe_progress_text(payload))
    return 0


def _plot_probe(args: argparse.Namespace) -> int:
    from .reporting.plotting import generate_probe_campaign_plots

    repo_root = _repo_root_from(Path.cwd())
    run_root = _resolve_repo_path(repo_root, Path(args.run_root))
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            payload = generate_probe_campaign_plots(
                run_root,
                round_selector=str(args.round),
                name=args.name,
                tags=args.tag,
                quiet=True,
            )
    else:
        payload = generate_probe_campaign_plots(
            run_root,
            round_selector=str(args.round),
            name=args.name,
            tags=args.tag,
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("opal_densegen_axis_probe_v0 configured plots")
        print(f"run_root={payload['run_root']}")
        print(f"campaign_count={payload['campaign_count']}")
        print(f"any_fail={payload['any_fail']}")
        print(f"mpl_config_dir={payload['mpl_config_dir']}")
    return 1 if payload["any_fail"] else 0


def _suite_probe(args: argparse.Namespace) -> int:
    from .reporting.suite_review import build_probe_suite_review

    repo_root = _repo_root_from(Path.cwd())
    run_roots = [_resolve_repo_path(repo_root, Path(root)) for root in args.run_roots]
    out_dir = _resolve_repo_path(repo_root, Path(args.out_dir)) if args.out_dir else None
    payload = build_probe_suite_review(run_roots, out_dir=out_dir)
    if args.opal_notebook:
        if out_dir is None:
            raise RuntimeError("--opal-notebook requires --out-dir so the suite notebook has a stable artifact root")
        from .reporting.suite_notebook import build_probe_suite_opal_notebook

        payload["opal_notebook"] = build_probe_suite_opal_notebook(
            run_roots,
            out_dir=out_dir / "opal_campaign_set",
            round_selector=str(args.round),
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("opal_densegen_axis_probe_v0 suite review")
        print(f"status={payload['status']}")
        if payload.get("artifacts"):
            print(f"review={payload['artifacts']['suite_review_markdown']}")
            print(f"manifest={payload['artifacts']['suite_review']}")
        if payload.get("opal_notebook"):
            print(f"opal_notebook={payload['opal_notebook']['notebook']}")
    return 0 if payload["status"] == "ok" else 1


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
        "--active-label-families",
        default=",".join(ACTIVE_LABEL_FAMILY_IDS),
        help=(
            "Comma-separated active label-family ids to materialize as OPAL campaigns. "
            "Default prepares the DenseGen plan-logic4 and compact TF-count matrices."
        ),
    )
    run.add_argument(
        "--suite",
        default=DEFAULT_SUITE_ID,
        help="Study-owned probe suite manifest id recorded with generated scratch artifacts.",
    )
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
        "--replace-run-root",
        action="store_true",
        help="Delete an existing probe run root before writing a new plan and scratch artifacts.",
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
    progress.add_argument("--full", action="store_true", help="Include full nested OPAL campaign progress payloads.")
    plot = subparsers.add_parser(
        "plot",
        help="Generate configured OPAL plots for all scratch campaigns in one Python process.",
    )
    plot.add_argument("--run-root", required=True)
    plot.add_argument("--round", default="all", help="Round selector passed to OPAL plot generation.")
    plot.add_argument("--name", default=None, help="Run one configured plot by name across scratch campaigns.")
    plot.add_argument("--tag", action="append", default=[], help="Run configured plots with this tag; repeatable.")
    plot.add_argument("--json", action="store_true", help="Emit machine-readable JSON plot summary.")
    suite = subparsers.add_parser("suite", help="Verify and summarize a complete three-seed probe suite.")
    suite.add_argument("--run-root", dest="run_roots", action="append", required=True)
    suite.add_argument("--out-dir", default=None, help="Optional directory for suite_review.json and suite_review.md.")
    suite.add_argument(
        "--opal-notebook",
        action="store_true",
        help="Also write a suite-scope OPAL campaign-set notebook with seed-replicate collection visuals.",
    )
    suite.add_argument("--round", default="all", help="Round selector for the optional OPAL notebook.")
    suite.add_argument("--json", action="store_true", help="Emit machine-readable JSON suite summary.")
    add_tfbs_subcommands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            from .runtime.run_cli import _run_probe

            return _run_probe(args)
        if args.command == "status":
            return _status_probe(args)
        if args.command == "report":
            return _report_probe(args)
        if args.command == "progress":
            return _progress_probe(args)
        if args.command == "plot":
            return _plot_probe(args)
        if args.command == "suite":
            return _suite_probe(args)
        tfbs_result = handle_tfbs_command(args)
        if tfbs_result is not None:
            return tfbs_result
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    parser.error(f"unsupported command: {args.command}")
    return 2
