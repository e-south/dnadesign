"""Parse command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.config.load import load_config

console = Console()


def parse(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (required).",
        metavar="CONFIG",
    ),
) -> None:
    if config is None:
        console.print("Missing CONFIG. Example: cruncher parse path/to/config.yaml")
        raise typer.Exit(code=1)
    cfg = load_config(config)
    try:
        from dnadesign.cruncher.workflows.parse_workflow import run_parse

        run_parse(cfg, config)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch + lock before parse, and verify cache paths.")
        raise typer.Exit(code=1)
