"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/export.py

Export sequence-centric sample-run artifacts for downstream consumption.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.app.export_sequences_service import run_export_sequences
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config

app = typer.Typer(no_args_is_help=True, help="Export sequence-focused artifacts from sample runs.")
console = Console()


@app.command("sequences", help="Export per-TF consensus, elite windows, and TF combinations for sample runs.")
def export_sequences_cmd(
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
        help="Sample run name or run directory path to export (repeatable).",
    ),
    latest: bool = typer.Option(False, "--latest", help="Export from the latest sample run."),
    table_format: str = typer.Option(
        "parquet",
        "--table-format",
        help="Export table format: parquet or csv.",
    ),
    max_combo_size: int | None = typer.Option(
        None,
        "--max-combo-size",
        help="Maximum TF combination size for combo exports (must be >=2).",
    ),
) -> None:
    if runs and latest:
        raise typer.BadParameter("Use either --run or --latest, not both.")
    if max_combo_size is not None and max_combo_size < 2:
        raise typer.BadParameter("--max-combo-size must be >= 2.")
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        results = run_export_sequences(
            cfg,
            config_path,
            runs_override=runs or None,
            use_latest=latest,
            table_format=table_format,
            max_combo_size=max_combo_size,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    workspace_root = config_path.parent
    for result in results:
        console.print(f"Run `{result.run_name}` export → {render_path(result.output_dir, base=workspace_root)}")
        console.print(f"  manifest: {render_path(result.manifest_path, base=workspace_root)}")
        for key in sorted(result.files):
            path = result.files[key]
            rows = result.row_counts.get(key, 0)
            console.print(f"  {key}: {render_path(path, base=workspace_root)}  rows={rows}")
