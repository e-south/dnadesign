"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/cli/commands/workspace.py

Workspace CLI commands for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ..common import emit, fail, resolve_format

app = typer.Typer(help="Workspace commands for latentdna.")


@app.command("where")
def where(
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    from ...services.workspace_service import workspace_where

    emit(workspace_where(), format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("init")
def init(
    workspace: str = typer.Option(..., "--workspace"),
    template: str = typer.Option(
        "minimal", "--template", help="Built-in template name or absolute template directory."
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    from ...services.workspace_service import init_workspace
    from ..previews import preview_workspace_init

    resolved_format = resolve_format(json_output=json_output, format_name=format_name)
    try:
        payload = (
            preview_workspace_init(workspace=workspace, template=template)
            if dry_run
            else init_workspace(workspace=workspace, template=template)
        )
    except Exception as exc:
        fail(exc)
    if resolved_format == "text" and not dry_run:
        payload = {
            **payload,
            "template": template,
            "next": f"uv run latentdna validate workspace --workspace {workspace}",
        }
    emit(payload, format_name=resolved_format, quiet=quiet)


@app.command("list")
def list_cmd(
    root: str | None = typer.Option(None, "--root"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        from ...services.workspace_service import list_workspaces

        payload = {"workspaces": list_workspaces(root=root)}
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("show")
def show(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        from ...services.workspace_service import show_workspace

        payload = show_workspace(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("snapshot")
def snapshot(
    workspace: str = typer.Option(..., "--workspace"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build the snapshot payload without writing workspace_snapshot.json.",
    ),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        from ...services.workspace_snapshot_service import workspace_snapshot

        payload = workspace_snapshot(workspace, write=not dry_run)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("refresh")
def refresh(
    workspace: str = typer.Option(..., "--workspace"),
    target: list[str] | None = typer.Option(None, "--target"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        from ...services.workspace_service import refresh_workspace

        payload = refresh_workspace(workspace, targets=target, dry_run=dry_run)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
