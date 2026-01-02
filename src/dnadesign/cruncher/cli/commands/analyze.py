"""Analyze command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.workflows.analyze.plot_registry import PLOT_SPECS

console = Console()


def _parse_tf_pair(raw: str) -> tuple[str, str]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise typer.BadParameter("--tf-pair must be formatted as TF1,TF2 (comma-separated).")
    return parts[0], parts[1]


def analyze(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (required).",
        metavar="CONFIG",
    ),
    runs: list[str] | None = typer.Option(
        None,
        "--run",
        help="Sample run name to analyze (repeatable). Overrides analysis.runs.",
    ),
    latest: bool = typer.Option(False, "--latest", help="Analyze the latest sample run."),
    tf_pair: str | None = typer.Option(
        None,
        "--tf-pair",
        help="Override analysis.tf_pair as TF1,TF2 (comma-separated).",
    ),
    list_plots: bool = typer.Option(False, "--list-plots", help="List which plots would run and exit."),
) -> None:
    if config is None:
        console.print("Missing CONFIG. Example: cruncher analyze path/to/config.yaml")
        raise typer.Exit(code=1)
    if runs and latest:
        raise typer.BadParameter("Use either --run or --latest, not both.")
    cfg = load_config(config)
    if cfg.analysis is None:
        console.print("Error: analysis section is required for analyze.")
        raise typer.Exit(code=1)
    tf_pair_override = _parse_tf_pair(tf_pair) if tf_pair else None
    if list_plots:
        plan_table = Table(title="Analysis plot plan", header_style="bold")
        plan_table.add_column("Plot")
        plan_table.add_column("Enabled")
        plan_table.add_column("Requires")
        pair_required = tf_pair_override or cfg.analysis.tf_pair
        for spec in PLOT_SPECS:
            enabled = getattr(cfg.analysis.plots, spec.key, False)
            requires = []
            if "trace" in spec.requires:
                requires.append("trace.nc")
            if "tf_pair" in spec.requires:
                requires.append("tf_pair")
            if "elites" in spec.requires:
                requires.append("elites.parquet")
            requires_label = ", ".join(requires) if requires else "-"
            status = "yes" if enabled else "no"
            if enabled and "tf_pair" in spec.requires and not pair_required:
                status = "missing tf_pair"
            plan_table.add_row(spec.label, status, requires_label)
        console.print(plan_table)
        if not (tf_pair_override or cfg.analysis.tf_pair):
            console.print("Hint: pairwise plots require analysis.tf_pair or --tf-pair TF1,TF2.")
        return
    try:
        from dnadesign.cruncher.workflows.analyze_workflow import run_analyze

        analysis_runs = run_analyze(
            cfg,
            config,
            runs_override=runs or None,
            use_latest=latest,
            tf_pair_override=tf_pair_override,
        )
        for analysis_dir in analysis_runs:
            analysis_id = analysis_dir.name
            console.print(f"Analysis outputs → {analysis_dir}")
            console.print(f"  summary: {analysis_dir / 'summary.json'}")
            console.print(f"  analysis_id: {analysis_id}")
            sample_dir = analysis_dir.parent.parent
            run_name = sample_dir.name
            console.print("Next steps:")
            console.print(f"  cruncher runs show {config} {run_name}")
            console.print(f"  cruncher notebook --analysis-id {analysis_id} {sample_dir}")
            console.print(f"  cruncher report {config} {run_name}")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: set analysis.runs, pass --run, or use --latest.")
        raise typer.Exit(code=1)
