"""Notebook CLI helpers."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.services.notebook_service import generate_notebook

app = typer.Typer(no_args_is_help=True, help="Generate an optional analysis notebook (marimo).")
console = Console()


@app.callback(invoke_without_command=True)
def generate_notebook_cmd(
    run_dir: Path = typer.Argument(..., help="Path to a sample run directory.", metavar="RUN_DIR"),
    analysis_id: str | None = typer.Option(None, "--analysis-id", help="Analysis ID to load."),
    latest: bool = typer.Option(False, "--latest", help="Use the latest analysis run."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing notebook file."),
) -> None:
    try:
        notebook_path = generate_notebook(run_dir, analysis_id=analysis_id, latest=latest, force=force)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(f"Notebook created → {notebook_path}")
    console.print("Open with: marimo edit " + str(notebook_path))
    console.print("Read-only app: marimo run " + str(notebook_path))
