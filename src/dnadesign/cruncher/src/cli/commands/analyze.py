"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/analyze.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from dnadesign.cruncher.analysis.layout import load_summary, summary_path
from dnadesign.cruncher.analysis.plot_registry import (
    plot_keys,
    plot_registry_rows,
)
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.numba_cache import ensure_numba_cache_dir
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


def _should_show_run_hint(message: str) -> bool:
    lowered = message.lower()
    if "pass --run" in lowered or "use --latest" in lowered:
        return False
    run_markers = (
        "no analysis runs configured",
        "no sample runs found",
        "not found under",
    )
    return any(marker in lowered for marker in run_markers)


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
    runs: list[str] | None = typer.Option(
        None,
        "--run",
        help="Sample run name or run directory path to analyze (repeatable). Overrides analysis.runs.",
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
    scatter_background: bool | None = typer.Option(
        None,
        "--scatter-background/--no-scatter-background",
        help="Toggle random baseline points in pwm__scatter (overrides analysis.scatter_background).",
    ),
    scatter_background_samples: int | None = typer.Option(
        None,
        "--scatter-background-samples",
        help="Number of random baseline sequences for pwm__scatter (defaults to MCMC subsample size).",
    ),
    scatter_background_seed: int | None = typer.Option(
        None,
        "--scatter-background-seed",
        help="Seed for random baseline sequences (overrides analysis.scatter_background_seed).",
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
        ensure_numba_cache_dir(config_path.parent)
        from dnadesign.cruncher.app.analyze_workflow import run_analyze

        analysis_runs = run_analyze(
            cfg,
            config_path,
            runs_override=runs or None,
            use_latest=latest,
            tf_pair_override=tf_pair_override,
            plot_keys_override=plot_keys_override,
            scatter_background_override=scatter_background,
            scatter_background_samples_override=scatter_background_samples,
            scatter_background_seed_override=scatter_background_seed,
        )
        for analysis_dir in analysis_runs:
            summary = load_summary(summary_path(analysis_dir), required=True)
            analysis_id = summary.get("analysis_id")
            console.print(f"Analysis outputs → {render_path(analysis_dir, base=config_path.parent)}")
            console.print(f"  summary: {render_path(summary_path(analysis_dir), base=config_path.parent)}")
            console.print(f"  analysis_id: {analysis_id}")
            sample_dir = analysis_dir.parent
            run_name = sample_dir.name
            config_hint = render_path(config_path)
            console.print("Next steps:")
            console.print(f"  cruncher runs show {run_name} -c {config_hint}")
            console.print(f"  cruncher notebook --latest {render_path(sample_dir, base=config_path.parent)}")
            if latest and not runs:
                console.print(f"  cruncher report --latest -c {config_hint}")
            else:
                console.print(f"  cruncher report {run_name} -c {config_hint}")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        message = str(exc)
        console.print(f"Error: {message}")
        if _should_show_run_hint(message):
            console.print("Hint: set analysis.runs, pass --run, or use --latest.")
        raise typer.Exit(code=1)
