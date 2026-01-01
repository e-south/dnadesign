"""Parse command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.config.load import load_config

app = typer.Typer(no_args_is_help=True, help="Load cached motifs and render PWM logos.")
console = Console()


@app.callback(invoke_without_command=True)
def main(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    try:
        from dnadesign.cruncher.workflows.parse_workflow import run_parse

        run_parse(cfg, config)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch + lock before parse, and verify cache paths.")
        raise typer.Exit(code=1)
