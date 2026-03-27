"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/yiu.py

CLI entrypoints for the YIU hairpin oligo processing workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, trace, design, and inspect YIU hairpin oligo processing workflows.",
)
console = Console()


def validate_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow import validate_yiu_spec as _validate_yiu_spec

    return _validate_yiu_spec(*args, **kwargs)


def run_yiu_design(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow import run_yiu_design as _run_yiu_design

    return _run_yiu_design(*args, **kwargs)


def run_yiu_trace(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow import run_yiu_trace as _run_yiu_trace

    return _run_yiu_trace(*args, **kwargs)


def yiu_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow import yiu_show_payload as _yiu_show_payload

    return _yiu_show_payload(*args, **kwargs)


def init_yiu_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import init_yiu_workspace as _init_yiu_workspace

    return _init_yiu_workspace(*args, **kwargs)


def yiu_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workspace_service import yiu_workspace_path as _yiu_workspace_path

    return _yiu_workspace_path(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"YIU spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    console.print(f"Protocol -> {report.protocol}")
    console.print(f"Sequence mode -> {report.sequence_mode}")
    console.print(f"Validation mode -> {report.validation_mode}")
    console.print(f"States -> {len(report.states)}")
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


@app.command("init-workspace", help="Scaffold a YIU workflow workspace.")
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
    console.print(f"Spec -> {result.spec_path}")
    console.print(f"Restriction catalog -> {result.restriction_catalog_path}")
    console.print(f"Nickase catalog -> {result.nickase_catalog_path}")
    console.print(f"Adapter catalog -> {result.adapter_catalog_path}")


@app.command("validate", help="Validate a YIU protocol spec and emit a deterministic step-trace report.")
def validate_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        report = validate_yiu_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("design", help="Write explicit YIU run artifacts for a validated or unsatisfied spec.")
def design_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic run directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        run_dir, report = run_yiu_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"YIU outputs -> {run_dir}")
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("trace", help="Materialize the modeled YIU state graph without solve-mode search.")
def trace_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic run directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        run_dir, report = run_yiu_trace(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"YIU trace outputs -> {run_dir}")
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("show", help="Show paths and summary for a YIU run directory.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a YIU run directory under outputs/yiu/explicit/."),
) -> None:
    try:
        payload = yiu_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"YIU run -> {payload['spec_name']}")
    console.print(f"Run dir -> {payload['run_dir']}")
    console.print(f"Status -> {payload['status']}: {payload['status_message']}")
    if payload.get("protocol_template"):
        console.print(f"Protocol template -> {payload['protocol_template']}")
    elif payload.get("protocol"):
        console.print(f"Protocol -> {payload['protocol']}")
    if payload.get("view_contract_version") is not None:
        console.print(f"View contract -> {payload['view_contract_version']}")
    console.print(f"Manifest -> {payload['manifest_path']}")
    console.print(f"Status file -> {payload['status_path']}")
    console.print(f"Report -> {payload['report_path']}")
    console.print(f"Trace -> {payload['trace_path']}")
    console.print(f"Trace manifest -> {payload['trace_manifest_path']}")
    console.print(f"Published views manifest -> {payload['published_views_manifest_path']}")
    console.print(f"Published views -> {payload['published_views_dir']}")
