"""
Workspace-oriented Snapback CLI commands.
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.cli.commands.snapback_presenters import console, echo_scaffold_line
from dnadesign.cruncher.cli.commands.snapback_services import init_snapback_workspace, snapback_workspace_path


def init_workspace_cmd(
    workspace: str | None = typer.Argument(
        None,
        metavar="WORKSPACE",
        help="Workspace directory name. Creates <root>/WORKSPACE unless --output is used.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Parent Cruncher workspaces directory for WORKSPACE. Defaults to the standard Cruncher workspaces root.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Explicit snapback workspace root to create. Overrides WORKSPACE/--root.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing scaffold created by this command.",
    ),
) -> None:
    try:
        if output is not None and workspace is not None:
            raise typer.BadParameter("Use either WORKSPACE [--root] or --output, not both.")
        if output is not None and root is not None:
            raise typer.BadParameter("Use either --output or --root, not both.")
        if output is None and workspace is None:
            raise typer.BadParameter("Provide WORKSPACE or --output.")
        target_root = output if output is not None else snapback_workspace_path(workspace, root=root)
        result = init_snapback_workspace(target_root, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    echo_scaffold_line("Snapback workspace scaffold", result.workspace_root)
    echo_scaffold_line("README", result.readme_path)
    echo_scaffold_line("Runbook", result.runbook_path)
    echo_scaffold_line("Runbook config", result.runbook_config_path)
    echo_scaffold_line("Example spec", result.example_spec_path)
    echo_scaffold_line("Example solve spec", result.example_solve_spec_path)
    echo_scaffold_line("Catalog", result.catalog_path)


__all__ = ["init_workspace_cmd"]
