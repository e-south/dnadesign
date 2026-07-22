"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/snapback_visual.py

Visual-only Snapback CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.cli.commands.snapback_presenters import console
from dnadesign.cruncher.cli.commands.snapback_services import run_snapback_visual


def visual_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.visual.snapback.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace the existing visual-only output root.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the visual report as JSON."),
) -> None:
    try:
        run_dir, report = run_snapback_visual(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Snapback visual outputs -> {run_dir}")
        console.print(f"Status -> {report.status}")
        console.print(f"Plot -> {report.plot_path}")
        console.print(
            "Product -> "
            f"upstream={report.upstream_context_nt} "
            f"stem={report.stem_sequence} "
            f"cap={report.cap_sequence} "
            f"foldback={report.foldback_sequence}"
        )


__all__ = ["visual_cmd"]
