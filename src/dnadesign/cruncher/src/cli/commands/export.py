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

from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_workspace_root

app = typer.Typer(no_args_is_help=True, help="Export sequence-specificity artifacts from sample runs.")
console = Console()


def run_export_sequences(*args, **kwargs):
    from dnadesign.cruncher.app.export_sequences_service import run_export_sequences as _run_export_sequences

    return _run_export_sequences(*args, **kwargs)


@app.command(
    "sequences",
    help="Export consensus-site and elite sequence tables for sample runs.",
)
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
        "csv",
        "--table-format",
        help="Export table format: parquet or csv.",
    ),
) -> None:
    if runs and latest:
        raise typer.BadParameter("Use either --run or --latest, not both.")
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    try:
        results = run_export_sequences(
            cfg,
            config_path,
            runs_override=runs or None,
            use_latest=latest,
            table_format=table_format,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    workspace_root = resolve_workspace_root(config_path)
    for result in results:
        console.print(f"Run `{result.run_name}` export → {render_path(result.output_dir, base=workspace_root)}")
        console.print(f"  manifest: {render_path(result.manifest_path, base=workspace_root)}")
        for key in sorted(result.files):
            path = result.files[key]
            rows = result.row_counts.get(key, 0)
            console.print(f"  {key}: {render_path(path, base=workspace_root)}  rows={rows}")
