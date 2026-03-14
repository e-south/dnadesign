"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/workspace.py

Workspace commands for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...errors import ConstructError
from ...workspace import init_workspace, workspace_root_with_source

workspace_app = typer.Typer(no_args_is_help=True, help="Workspace scaffolding for construct.")


@workspace_app.command("where")
def where(root: str | None = typer.Option(None, "--root", help="Override workspace root.")) -> None:
    workspace_root, source = workspace_root_with_source(root)
    typer.echo(f"workspace_root: {workspace_root}")
    typer.echo(f"workspace_root_source: {source}")


@workspace_app.command("init")
def init(
    workspace_id: str = typer.Option(..., "--id", help="Workspace directory name."),
    root: str | None = typer.Option(None, "--root", help="Override workspace root."),
) -> None:
    try:
        workspace_dir = init_workspace(workspace_id=workspace_id, root=root)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc

    config_path = workspace_dir / "config.yaml"
    typer.echo(f"workspace: {workspace_dir}")
    typer.echo("profile: default")
    typer.echo(f"config: {config_path}")
    typer.echo(f"Next: construct validate config --config {config_path}")
    typer.echo(
        "Then: replace scaffold inputs with canonical sequences and seed "
        "the anchor dataset into outputs/usr_datasets"
    )
    typer.echo(f"Then: construct validate config --config {config_path} --runtime")
    typer.echo(f"Then: construct run --config {config_path} --dry-run")
