"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/yiu.py

CLI entrypoints for the payload-centric YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.cli.yiu_presenter import (
    print_render_outcome,
    print_show_outcome,
    print_validation_report,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, render, and inspect payload-centric YIU workflows.",
)
console = Console()


def validate_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.validate import validate_yiu_spec as _validate_yiu_spec

    return _validate_yiu_spec(*args, **kwargs)


def render_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.render import render_yiu_spec as _render_yiu_spec

    return _render_yiu_spec(*args, **kwargs)


def render_yiu_spec_outcome(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.render import render_yiu_spec_outcome as _render_yiu_spec_outcome

    return _render_yiu_spec_outcome(*args, **kwargs)


def show_yiu_bundle(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow.show import show_yiu_bundle as _show_yiu_bundle

    return _show_yiu_bundle(*args, **kwargs)


def init_yiu_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import init_yiu_workspace as _init_yiu_workspace

    return _init_yiu_workspace(*args, **kwargs)


def yiu_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import yiu_workspace_path as _yiu_workspace_path

    return _yiu_workspace_path(*args, **kwargs)


@app.command("init-workspace", help="Scaffold a payload-centric YIU workspace.")
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
        help="Explicit YIU workspace root to create. Overrides WORKSPACE/--root.",
    ),
    force_overwrite: bool = typer.Option(False, "--force-overwrite", help="Replace an existing workspace root."),
) -> None:
    try:
        if output is not None and workspace is not None:
            raise typer.BadParameter("Use either WORKSPACE [--root] or --output, not both.")
        if output is not None and root is not None:
            raise typer.BadParameter("Use either --output or --root, not both.")
        if output is None and workspace is None:
            raise typer.BadParameter("Provide WORKSPACE or --output.")
        target_root = output if output is not None else yiu_workspace_path(workspace, root=root)
        result = init_yiu_workspace(target_root, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"YIU workspace scaffold -> {result.workspace_root}")
    console.print(f"Runbook -> {result.runbook_path}")
    console.print(f"Runbook doc -> {result.runbook_doc_path}")
    console.print(f"Spec -> {result.spec_path}")


@app.command("validate", help="Validate a payload-centric YIU spec and print the normalized payload summary.")
def validate_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    json_output: bool = typer.Option(False, "--json", help="Print the validation report as JSON."),
) -> None:
    try:
        report = validate_yiu_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return
    print_validation_report(console, report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("render", help="Validate a payload-centric YIU spec, publish its bundle, and optionally render its views.")
def render_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic bundle directory."
    ),
    emit_renders: bool = typer.Option(
        False,
        "--emit-renders",
        help="Immediately render every published BaseRender job after the bundle is written.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the render report as JSON."),
) -> None:
    try:
        outcome = render_yiu_spec_outcome(spec, force_overwrite=force_overwrite, emit_renders=emit_renders)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    payload = outcome.model_dump(mode="json")
    report = outcome.report
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    print_render_outcome(console, outcome, emit_renders=emit_renders)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("show", help="Show payload-centric summary for one YIU bundle.")
def show_cmd(
    bundle: Path = typer.Option(..., "--bundle", help="Path to a published YIU payload bundle."),
    json_output: bool = typer.Option(False, "--json", help="Print the normalized bundle summary as JSON."),
    verbose: bool = typer.Option(False, "--verbose", help="Include split-row debug details in the output."),
) -> None:
    try:
        outcome = show_yiu_bundle(bundle, verbose=verbose)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(outcome.model_dump(mode="json", exclude_unset=True), indent=2))
        return
    print_show_outcome(console, outcome, verbose=verbose)
