"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/workspace.py

Registration for infer workspace CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...workspace import init_workspace, resolve_workspace_root, resolve_workspace_template
from ..common import raise_cli_error


def register(app: typer.Typer) -> None:
    workspace_app = typer.Typer(no_args_is_help=True, help="Workspace discovery and scaffold utilities.")
    app.add_typer(workspace_app, name="workspace")

    @workspace_app.command("where", help="Show effective workspace and template roots.")
    def workspace_where(
        root: Optional[Path] = typer.Option(None, "--root", help="Explicit workspace root."),
        profile: str = typer.Option("local", "--profile", help="Template profile: local | usr-pressure."),
    ) -> None:
        try:
            workspace_root, source = resolve_workspace_root(root)
            template_path = resolve_workspace_template(None, profile=profile)
            typer.echo(f"workspace_root: {workspace_root}")
            typer.echo(f"workspace_root_source: {source}")
            typer.echo(f"workspace_profile: {profile}")
            typer.echo(f"workspace_template: {template_path}")
        except Exception as error:
            raise_cli_error(error)

    @workspace_app.command("init", help="Create a workspace with config.yaml and infer output folders.")
    def workspace_init(
        workspace_id: str = typer.Option(..., "--id", "-i", help="Workspace identifier (directory name)."),
        root: Optional[Path] = typer.Option(None, "--root", help="Workspace root path."),
        template: Optional[Path] = typer.Option(
            None,
            "--template",
            help="Config template path (overrides --profile default).",
        ),
        profile: str = typer.Option("local", "--profile", help="Template profile: local | usr-pressure."),
    ) -> None:
        try:
            workspace_dir = init_workspace(workspace_id=workspace_id, root=root, template=template, profile=profile)
            config_path = workspace_dir / "config.yaml"
            typer.echo(f"Workspace initialized: {workspace_dir}")
            typer.echo(f"config: {config_path}")
            typer.echo(f"profile: {profile}")
            if template is None and profile == "usr-pressure":
                typer.echo("Review ingest.dataset and ingest.root in config.yaml before running.")
            typer.echo("Next:")
            typer.echo(f"  - uv run infer validate config --config {config_path}")
            typer.echo(f"  - uv run infer run --config {config_path} --dry-run")
        except Exception as error:
            raise_cli_error(error)
