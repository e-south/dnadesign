"""Report command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.workflows.report_workflow import run_report

app = typer.Typer(no_args_is_help=True, help="Summarize a sample run into report.json and report.md.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG"),
    batch: str = typer.Argument(..., help="Run directory name (see `cruncher runs list`).", metavar="RUN"),
) -> None:
    cfg = load_config(config)
    try:
        run_report(cfg, config, batch)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher sample first, then cruncher report <config> <run>.")
        raise typer.Exit(code=1)
