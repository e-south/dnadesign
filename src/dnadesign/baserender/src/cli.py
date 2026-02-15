"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/cli.py

Baserender vNext CLI for Sequence Rows v3 job configs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .cli_actions import (
    discover_workspaces_action,
    init_workspace_action,
    list_style_presets_action,
    normalize_job_action,
    run_job_action,
    show_style_action,
    validate_job_action,
)
from .core import BaseRenderError
from .workspace import default_workspaces_root

app = typer.Typer(help="Baserender vNext CLI")
job_app = typer.Typer(help="Sequence Rows v3 job commands")
style_app = typer.Typer(help="Style commands")
workspace_app = typer.Typer(help="Workspace commands")
app.add_typer(job_app, name="job")
app.add_typer(style_app, name="style")
app.add_typer(workspace_app, name="workspace")


def _exit_cli_error(exc: Exception) -> None:
    typer.echo(f"ERROR: {exc}", err=True)
    raise typer.Exit(code=2) from exc


@job_app.command("validate")
def job_validate(
    job: str | None = typer.Argument(None, help="Path to Sequence Rows v3 job YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yaml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """Validate a Sequence Rows v3 job config."""
    try:
        parsed = validate_job_action(job, workspace, workspace_root)
    except BaseRenderError as exc:
        _exit_cli_error(exc)
    typer.echo(f"OK: {parsed.path}")


@job_app.command("run")
def job_run(
    job: str | None = typer.Argument(None, help="Path to Sequence Rows v3 job YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yaml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """Run a Sequence Rows v3 job config."""
    try:
        report = run_job_action(job, workspace, workspace_root)
    except BaseRenderError as exc:
        _exit_cli_error(exc)

    typer.echo("Run complete")
    for key, value in report.outputs.items():
        typer.echo(f"- {key}: {value}")


@job_app.command("normalize")
def job_normalize(
    job: str | None = typer.Argument(None, help="Path to Sequence Rows v3 job YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yaml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
    out: Path = typer.Option(..., "--out", help="Output path for normalized YAML."),
) -> None:
    """Normalize and rewrite a Sequence Rows v3 job config with absolute resolved paths."""
    try:
        written = normalize_job_action(
            job,
            workspace,
            workspace_root,
            out=out,
        )
    except BaseRenderError as exc:
        _exit_cli_error(exc)
    typer.echo(f"Wrote: {written}")


@style_app.command("list")
def style_list() -> None:
    """List available style presets."""
    presets = list_style_presets_action()
    for name in presets:
        typer.echo(name)


@style_app.command("show")
def style_show(
    preset: str = typer.Argument(..., help="Preset name or path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of YAML."),
) -> None:
    """Show a style preset file."""
    try:
        data = show_style_action(preset)
    except BaseRenderError as exc:
        _exit_cli_error(exc)

    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        typer.echo(yaml.safe_dump(data, sort_keys=False))


@workspace_app.command("list")
def workspace_list(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """List discovered baserender workspaces."""
    try:
        workspaces = discover_workspaces_action(root)
    except BaseRenderError as exc:
        _exit_cli_error(exc)

    if not workspaces:
        target = (default_workspaces_root() if root is None else root).resolve()
        typer.echo(f"No workspaces found under: {target}")
        raise typer.Exit(code=1)

    typer.echo("name\tjob\troot")
    for workspace in workspaces:
        typer.echo(f"{workspace.name}\t{workspace.job_path}\t{workspace.root}")


@workspace_app.command("init")
def workspace_init(
    name: str = typer.Argument(..., help="Workspace directory name."),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """Create a workspace scaffold with job.yaml, inputs/, and outputs/."""
    try:
        workspace = init_workspace_action(name, root)
    except BaseRenderError as exc:
        _exit_cli_error(exc)

    typer.echo(f"Workspace created: {workspace.root}")
    typer.echo(f"- job: {workspace.job_path}")
    typer.echo(f"- inputs: {workspace.inputs_dir}")
    typer.echo(f"- outputs: {workspace.outputs_dir}")


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
