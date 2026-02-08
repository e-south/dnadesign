"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/analyze.py

Run the analysis pipeline for Cruncher sample runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.analysis.layout import load_summary, report_json_path, report_md_path, summary_path
from dnadesign.cruncher.cli.campaign_targeting import resolve_runtime_targeting
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir
from dnadesign.cruncher.utils.paths import resolve_catalog_root, workspace_state_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

console = Console()


def analyze(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    campaign: str | None = typer.Option(
        None,
        "--campaign",
        "-n",
        help="Campaign name to expand in-memory for this command.",
    ),
    runs: list[str] | None = typer.Option(
        None,
        "--run",
        help="Sample run name or run directory path to analyze (repeatable). Overrides analysis.runs.",
    ),
    latest: bool = typer.Option(False, "--latest", help="Analyze the latest sample run."),
    summary_flag: bool = typer.Option(False, "--summary", help="Print a concise analysis summary after analyze."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    if runs and latest:
        raise typer.BadParameter("Use either --run or --latest, not both.")
    cfg = load_config(config_path)
    try:
        cfg = resolve_runtime_targeting(
            cfg=cfg,
            config_path=config_path,
            command_name="analyze",
            campaign_name=campaign,
        ).cfg
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    if cfg.analysis is None:
        console.print("Error: analysis section is required for analyze.")
        raise typer.Exit(code=1)
    try:
        cache_dir = workspace_state_root(config_path) / "numba_cache"
        ensure_numba_cache_dir(config_path.parent, cache_dir=cache_dir)
        ensure_mpl_cache(resolve_catalog_root(config_path, cfg.catalog.catalog_root))
        from dnadesign.cruncher.app.analyze_workflow import run_analyze

        analysis_runs = run_analyze(
            cfg,
            config_path,
            runs_override=runs or None,
            use_latest=latest,
        )
        for analysis_dir in analysis_runs:
            summary_payload = load_summary(summary_path(analysis_dir), required=True)
            analysis_id = summary_payload.get("analysis_id")
            console.print(f"Analysis outputs → {render_path(analysis_dir, base=config_path.parent)}")
            console.print(f"  summary: {render_path(summary_path(analysis_dir), base=config_path.parent)}")
            report_path = report_md_path(analysis_dir)
            if report_path.exists():
                console.print(f"  report: {render_path(report_path, base=config_path.parent)}")
            console.print(f"  analysis_id: {analysis_id}")
            run_dir = analysis_dir
            config_hint = render_path(config_path)
            console.print("Next steps:")
            console.print(f"  cruncher runs show {render_path(run_dir, base=config_path.parent)} -c {config_hint}")
            console.print(f"  cruncher notebook --latest {render_path(run_dir, base=config_path.parent)}")
            console.print(f"  open {render_path(report_path, base=config_path.parent)}")
            if summary_flag:
                report_json = report_json_path(analysis_dir)
                if report_json.exists():
                    payload = load_summary(report_json, required=True)
                    status = payload.get("status", "unknown")
                    warnings = payload.get("warnings", [])
                    highlights = payload.get("highlights", {})
                    objective = highlights.get("objective", {})
                    diversity = highlights.get("diversity", {})
                    overlap = highlights.get("overlap", {})
                    sampling = highlights.get("sampling", {})
                    console.print("Summary:")
                    console.print(f"  status: {status}  warnings: {len(warnings) if isinstance(warnings, list) else 0}")
                    console.print(
                        f"  best_score_final: {objective.get('best_score_final')}  "
                        f"top_k_median_final: {objective.get('top_k_median_final')}"
                    )
                    console.print(
                        f"  unique_fraction: {diversity.get('unique_fraction')}  n_elites: {diversity.get('n_elites')}"
                    )
                    console.print(
                        f"  overlap_rate_median: {overlap.get('overlap_rate_median')}  "
                        f"overlap_total_bp_median: {overlap.get('overlap_total_bp_median')}"
                    )
                    console.print(
                        f"  acceptance_rate_mh_tail: {sampling.get('acceptance_rate_mh_tail')}  "
                        f"swap_acceptance_rate: {sampling.get('swap_acceptance_rate')}"
                    )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        message = str(exc)
        console.print(f"Error: {message}")
        lowered = message.lower()
        if any(
            marker in lowered
            for marker in (
                "no analysis runs configured",
                "no sample runs found",
                "not found under",
            )
        ):
            has_embedded_hint = "analysis.runs" in lowered and "--run" in lowered and "--latest" in lowered
            if not has_embedded_hint:
                console.print("Hint: set analysis.runs, pass --run, or use --latest.")
        raise typer.Exit(code=1)
