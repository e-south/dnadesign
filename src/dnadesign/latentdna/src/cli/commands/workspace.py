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

from ...services.workspace_service import (
    init_workspace,
    list_workspaces,
    refresh_workspace,
    show_workspace,
    workspace_where,
)
from ...services.workspace_snapshot_service import workspace_snapshot
from ..common import emit, fail, resolve_format
from ..previews import preview_workspace_init

app = typer.Typer(help="Workspace commands for latentdna.")


@app.command("where")
def where(
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    emit(workspace_where(), format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("init")
def init(
    workspace: str = typer.Option(..., "--workspace"),
    template: str = typer.Option("minimal", "--template"),
    from_study_dir: str | None = typer.Option(None, "--from-study-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    resolved_format = resolve_format(json_output=json_output, format_name=format_name)
    try:
        payload = (
            preview_workspace_init(workspace=workspace, template=template, from_study_dir=from_study_dir)
            if dry_run
            else init_workspace(workspace=workspace, template=template, from_study_dir=from_study_dir)
        )
    except Exception as exc:
        fail(exc)
    if resolved_format == "text" and not dry_run:
        payload = {
            **payload,
            "template": template,
            **({"from_study_dir": from_study_dir} if from_study_dir is not None else {}),
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
        payload = show_workspace(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("snapshot")
def snapshot(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = workspace_snapshot(workspace)
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
        payload = refresh_workspace(workspace, targets=target, dry_run=dry_run)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
