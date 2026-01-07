"""Analyze command."""

from __future__ import annotations

from pathlib import Path

import typer
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.analysis_layout import load_summary
from dnadesign.cruncher.workflows.analyze.plot_registry import plot_keys, plot_registry_rows
from rich.console import Console
from rich.table import Table

console = Console()


def _parse_tf_pair(raw: str) -> tuple[str, str]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise typer.BadParameter("--tf-pair must be formatted as TF1,TF2 (comma-separated).")
    return parts[0], parts[1]


def _parse_plot_keys(raw: list[str]) -> list[str]:
    keys: list[str] = []
    for item in raw:
        keys.extend([part.strip() for part in item.split(",") if part.strip()])
    return keys


def analyze(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (required).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
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
    plots: list[str] | None = typer.Option(
        None,
        "--plots",
        help="Override analysis plots by key (repeatable or comma-separated). Use 'all' to enable every plot.",
    ),
    list_plots: bool = typer.Option(False, "--list-plots", help="List which plots would run and exit."),
) -> None:
    plot_keys_override: list[str] | None = None
    if plots:
        plot_keys_override = _parse_plot_keys(plots)
        known = plot_keys()
        unknown = [key for key in plot_keys_override if key != "all" and key not in known]
        if unknown:
            raise typer.BadParameter(f"Unknown plot keys: {', '.join(unknown)}")
        if "all" in plot_keys_override and len(plot_keys_override) > 1:
            raise typer.BadParameter("Use either --plots all or explicit plot keys, not both.")
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    if runs and latest:
        raise typer.BadParameter("Use either --run or --latest, not both.")
    cfg = load_config(config_path)
    if cfg.analysis is None:
        console.print("Error: analysis section is required for analyze.")
        raise typer.Exit(code=1)
    tf_pair_override = _parse_tf_pair(tf_pair) if tf_pair else None
    if list_plots:
        plan_table = Table(title="Analysis plot plan", header_style="bold")
        plan_table.add_column("Key")
        plan_table.add_column("Plot")
        plan_table.add_column("Enabled")
        plan_table.add_column("Requires")
        plan_table.add_column("Outputs")
        pair_required = tf_pair_override or cfg.analysis.tf_pair
        for row in plot_registry_rows(
            enabled=cfg.analysis.plots,
            pair_available=pair_required is not None,
            overrides=plot_keys_override,
        ):
            plan_table.add_row(
                row["key"],
                row["label"],
                row["enabled"],
                row["requires"],
                row["outputs"],
            )
        console.print(plan_table)
        if not (tf_pair_override or cfg.analysis.tf_pair):
            console.print("Hint: pairwise plots require analysis.tf_pair or --tf-pair TF1,TF2.")
        return
    try:
        from dnadesign.cruncher.workflows.analyze_workflow import run_analyze

        analysis_runs = run_analyze(
            cfg,
            config_path,
            runs_override=runs or None,
            use_latest=latest,
            tf_pair_override=tf_pair_override,
            plot_keys_override=plot_keys_override,
        )
        for analysis_dir in analysis_runs:
            summary = load_summary(analysis_dir / "summary.json", required=True)
            analysis_id = summary.get("analysis_id")
            console.print(f"Analysis outputs → {analysis_dir}")
            console.print(f"  summary: {analysis_dir / 'summary.json'}")
            console.print(f"  analysis_id: {analysis_id}")
            sample_dir = analysis_dir.parent
            run_name = sample_dir.name
            console.print("Next steps:")
            console.print(f"  cruncher runs show {config_path} {run_name}")
            console.print(f"  cruncher notebook --latest {sample_dir}")
            console.print(f"  cruncher report {config_path} {run_name}")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: set analysis.runs, pass --run, or use --latest.")
        raise typer.Exit(code=1)
