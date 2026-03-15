"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/workspace.py

Workspace commands for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...errors import ConstructError
from ...runtime import preflight_from_config, run_from_config
from ...workspace import (
    doctor_workspace_registry,
    init_workspace,
    load_workspace_registry,
    project_root,
    resolve_workspace_project,
    workspace_registry_path,
    workspace_root_with_source,
    workspace_template_with_source,
)
from ._render import echo_run_result, echo_validate_result

workspace_app = typer.Typer(no_args_is_help=True, help="Workspace scaffolding for construct.")


def _construct_command_prefix(*, workspace_dir: Path | None = None) -> str:
    repo_root = project_root()
    if workspace_dir is not None:
        try:
            workspace_dir.resolve().relative_to(repo_root)
        except ValueError:
            return f"uv run --project {repo_root} construct"
    return "uv run construct"


@workspace_app.command("where")
def where(
    root: str | None = typer.Option(None, "--root", help="Override workspace root."),
    profile: str = typer.Option("blank", "--profile", help="Workspace profile: blank | promoter-swap-demo."),
) -> None:
    try:
        workspace_root, source = workspace_root_with_source(root)
        template_path, template_source = workspace_template_with_source(profile)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc
    typer.echo(f"workspace_root: {workspace_root}")
    typer.echo(f"workspace_root_source: {source}")
    typer.echo(f"workspace_profile: {profile}")
    typer.echo(f"workspace_template_source: {template_source}")
    if template_path is not None:
        typer.echo(f"workspace_template: {template_path}")


@workspace_app.command("init")
def init(
    workspace_id: str = typer.Option(..., "--id", help="Workspace directory name."),
    root: str | None = typer.Option(None, "--root", help="Override workspace root."),
    profile: str = typer.Option("blank", "--profile", help="Workspace profile: blank | promoter-swap-demo."),
) -> None:
    try:
        workspace_dir = init_workspace(workspace_id=workspace_id, root=root, profile=profile)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc

    config_path = workspace_dir / "config.yaml"
    registry_path = workspace_registry_path(workspace_dir)
    command_prefix = _construct_command_prefix(workspace_dir=workspace_dir)
    typer.echo(f"workspace: {workspace_dir}")
    typer.echo(f"profile: {profile}")
    typer.echo(f"workspace_registry: {registry_path}")
    if config_path.exists():
        typer.echo(f"config: {config_path}")
        typer.echo(f"Next: {command_prefix} workspace show --workspace {workspace_dir}")
        typer.echo(f"Next: {command_prefix} validate config --config {config_path}")
    else:
        typer.echo("config: choose one of the packaged config.*.yaml files in this workspace")
        typer.echo(f"Next: {command_prefix} workspace show --workspace {workspace_dir}")
    if profile == "promoter-swap-demo":
        typer.echo(
            f"Then: {command_prefix} seed promoter-swap-demo "
            f"--root {workspace_dir / 'outputs' / 'usr_datasets'} "
            f"--manifest {workspace_dir / 'inputs' / 'seed_manifest.yaml'}"
        )
        typer.echo("Then: run ./runbook.sh --mode dry-run --config <chosen-config>")
    else:
        typer.echo("Then: update construct.workspace.yaml with your project inventory and USR root choice.")
        typer.echo(f"Then: edit {workspace_dir / 'inputs' / 'import_manifest.template.yaml'} for your own inputs.")
        typer.echo(
            f"Optional demo bootstrap: {command_prefix} seed promoter-swap-demo "
            f"--root {workspace_dir / 'outputs' / 'usr_datasets'} "
            f"--manifest {workspace_dir / 'inputs' / 'seed_manifest.yaml'}"
        )


@workspace_app.command("show")
def show(
    workspace: str = typer.Option(".", "--workspace", help="Workspace directory or construct.workspace.yaml path."),
) -> None:
    try:
        registry, registry_path = load_workspace_registry(workspace)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc

    payload = registry.workspace
    typer.echo(f"workspace_registry: {registry_path}")
    typer.echo(f"workspace_id: {payload.id}")
    typer.echo(f"profile: {payload.profile}")
    typer.echo(f"description: {payload.description}")
    typer.echo(f"shared_usr_root: {payload.roots.shared_usr_root} (repo-relative hint)")
    typer.echo(f"workspace_usr_root: {payload.roots.workspace_usr_root} (workspace-relative default)")
    typer.echo(f"projects_total: {len(payload.projects)}")
    for project in payload.projects:
        typer.echo(
            "project: "
            f"id={project.id} "
            f"config={project.config} "
            f"flow={project.flow} "
            f"input_dataset={project.input_dataset} "
            f"template_dataset={project.template_dataset or ''} "
            f"output_dataset={project.output_dataset}"
        )


@workspace_app.command("doctor")
def doctor(
    workspace: str = typer.Option(".", "--workspace", help="Workspace directory or construct.workspace.yaml path."),
) -> None:
    try:
        report = doctor_workspace_registry(workspace)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc

    typer.echo(f"workspace_registry: {report.registry_path}")
    typer.echo(f"workspace_id: {report.workspace_id}")
    typer.echo(f"profile: {report.profile}")
    typer.echo(f"projects_checked: {report.projects_checked}")
    typer.echo(f"issues_total: {len(report.issues)}")
    for issue in report.issues:
        typer.echo(f"{issue.severity}: project={issue.project_id} {issue.message}")
    if report.issues:
        raise typer.Exit(1)
    typer.echo("workspace_doctor: ok")


@workspace_app.command("validate-project")
def validate_project(
    workspace: str = typer.Option(".", "--workspace", help="Workspace directory or construct.workspace.yaml path."),
    project: str = typer.Option(..., "--project", help="Workspace project id from construct.workspace.yaml."),
    runtime: bool = typer.Option(
        False,
        "--runtime",
        help="Resolve template and input dataset, then report the planned runtime summary.",
    ),
) -> None:
    try:
        resolution = resolve_workspace_project(workspace, project_id=project)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(2) from exc

    preflight = None
    if runtime:
        try:
            preflight = preflight_from_config(resolution.config_path)
        except ConstructError as exc:
            typer.echo(f"Error: {exc}")
            raise typer.Exit(1) from exc
    echo_validate_result(config_path=resolution.config_path, loaded=resolution.config, preflight=preflight)


@workspace_app.command("run-project")
def run_project(
    workspace: str = typer.Option(".", "--workspace", help="Workspace directory or construct.workspace.yaml path."),
    project: str = typer.Option(..., "--project", help="Workspace project id from construct.workspace.yaml."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and build outputs without writing USR data.",
    ),
) -> None:
    try:
        resolution = resolve_workspace_project(workspace, project_id=project)
        result = run_from_config(resolution.config_path, dry_run=dry_run)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1) from exc
    echo_run_result(result)
