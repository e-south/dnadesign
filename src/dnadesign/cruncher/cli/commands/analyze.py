"""Analyze command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.config.load import load_config

app = typer.Typer(no_args_is_help=True, help="Generate diagnostics and plots for completed sample runs.")
console = Console()


@app.callback(invoke_without_command=True)
def main(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    try:
        from dnadesign.cruncher.workflows.analyze_workflow import run_analyze

        run_analyze(cfg, config)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher sample first, then cruncher analyze <config>.")
        raise typer.Exit(code=1)
