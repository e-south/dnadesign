"""Notebook CLI helpers."""

from __future__ import annotations

from pathlib import Path

import typer
from dnadesign.cruncher.services.notebook_service import generate_notebook
from rich.console import Console

console = Console()


def notebook(
    run_dir: Path = typer.Argument(..., help="Path to a sample run directory.", metavar="RUN_DIR"),
    analysis_id: str | None = typer.Option(None, "--analysis-id", help="Analysis ID to load."),
    latest: bool = typer.Option(False, "--latest", help="Use the latest analysis run."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing notebook file."),
    strict: bool = typer.Option(
        True,
        "--strict/--lenient",
        help="Fail if summary.json or plot_manifest.json is missing or invalid.",
    ),
) -> None:
    """Generate a marimo notebook scaffold for an analysis run."""
    try:
        notebook_path = generate_notebook(
            run_dir,
            analysis_id=analysis_id,
            latest=latest,
            force=force,
            strict=strict,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(f"Notebook created → {notebook_path}")
    console.print("Open with: marimo edit " + str(notebook_path))
    console.print("Read-only app: marimo run " + str(notebook_path))
