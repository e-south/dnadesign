"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/prune.py

Registration for infer prune CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...prune import prune_usr_overlay
from ..common import raise_cli_error


def register(app: typer.Typer) -> None:
    @app.command("prune", help="Prune infer write-back overlays from a USR dataset.")
    def prune(
        dataset: str = typer.Option(..., "--usr", help="USR dataset identifier."),
        usr_root: Path = typer.Option(..., "--usr-root", help="USR datasets root path."),
        mode: str = typer.Option("archive", "--mode", help="Prune mode: archive | delete."),
    ) -> None:
        try:
            summary = prune_usr_overlay(dataset=dataset, usr_root=usr_root, mode=mode)
            typer.echo(f"dataset: {summary['dataset']}")
            typer.echo(f"namespace: {summary['namespace']}")
            typer.echo(f"mode: {summary['mode']}")
            typer.echo(f"removed: {summary['removed']}")
            archived_path = summary.get("archived_path")
            if archived_path:
                typer.echo(f"archived_path: {archived_path}")
        except Exception as error:
            raise_cli_error(error)
