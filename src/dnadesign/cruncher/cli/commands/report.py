"""Report command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.config.load import load_config

console = Console()


def report(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (required).",
        metavar="CONFIG",
    ),
    run_name: str | None = typer.Argument(
        None,
        help="Run directory name (see `cruncher runs list`).",
        metavar="RUN",
    ),
) -> None:
    if config is None or run_name is None:
        console.print("Missing CONFIG/RUN. Example: cruncher report path/to/config.yaml sample_run")
        raise typer.Exit(code=1)
    cfg = load_config(config)
    try:
        from dnadesign.cruncher.workflows.report_workflow import run_report

        run_report(cfg, config, run_name)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher sample first, then cruncher report <config> <run>.")
        raise typer.Exit(code=1)
