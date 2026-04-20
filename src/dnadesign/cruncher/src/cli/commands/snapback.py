"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/snapback.py

CLI entrypoints for v2 snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, design, solve, and inspect single-nick snapback workflows.",
)
console = Console()


def validate_snapback_spec(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec as _validate_snapback_spec

    return _validate_snapback_spec(*args, **kwargs)


def run_snapback_design(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import run_snapback_design as _run_snapback_design

    return _run_snapback_design(*args, **kwargs)


def run_snapback_solve(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve as _run_snapback_solve

    return _run_snapback_solve(*args, **kwargs)


def snapback_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import snapback_show_payload as _snapback_show_payload

    return _snapback_show_payload(*args, **kwargs)


def init_snapback_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import init_snapback_workspace as _init_snapback_workspace

    return _init_snapback_workspace(*args, **kwargs)


def snapback_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import snapback_workspace_path as _snapback_workspace_path

    return _snapback_workspace_path(*args, **kwargs)


def _print_report(report) -> None:
    console.print(f"Snapback spec -> {report.spec_name}")
    console.print(f"Status -> {report.status}")
    if report.candidate is not None:
        console.print(
            "Intended nick -> "
            f"{report.candidate.intended_nick.variant_id}@{report.candidate.nick_boundary} "
            f"({report.candidate.intended_site.orientation})"
        )
        console.print(f"Released prefix nt -> {report.candidate.released_prefix_nt}")
        console.print(f"Cap nt -> {report.candidate.cap_nt}")
        console.print(f"Added nt -> {report.candidate.added_nt}")
        console.print(f"Terminal ligatable duplex bp -> {report.candidate.terminal_ligatable_duplex_bp}")
        console.print(f"Max uninterrupted duplex bp -> {report.candidate.max_uninterrupted_duplex_bp}")
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _print_solve_report(report) -> None:
    console.print(f"Snapback solve spec -> {report.spec_path}")
    console.print(f"Status -> {report.status}")
    if report.solve_id:
        console.print(f"Solve id -> {report.solve_id}")
    if report.run_dir:
        console.print(f"Outputs -> {report.run_dir}")
    console.print(
        "Search -> "
        f"nodes={report.metadata.visited_search_node_count}, "
        f"enumerated={report.metadata.enumerated_candidate_count}, "
        f"accepted={report.metadata.accepted_candidate_count}, "
        f"materialized={report.metadata.materialized_hit_count}"
    )
    for code, warning in zip(report.metadata.warning_codes, report.metadata.warnings, strict=False):
        console.print(f"Warning -> {code}: {warning}")
    if report.hits:
        console.print("Hits:")
        for hit in report.hits:
            line = (
                f"  - rank {hit.rank}: {hit.hit_id} "
                f"{hit.variant_id}@{hit.nick_boundary} cap={hit.cap_sequence} "
                f"added_nt={hit.added_nt}"
            )
            if hit.materialized_run_dir is not None:
                line += f" run={hit.materialized_run_dir}"
            console.print(line)
    if report.issues:
        console.print("Issues:")
        for issue in report.issues:
            console.print(f"  - {issue.code}: {issue.message}")


def _echo_scaffold_line(label: str, value: str | Path) -> None:
    typer.echo(f"{label} -> {value}")


@app.command("init-workspace", help="Scaffold a snapback workspace with v2 explicit and solve examples.")
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
    _echo_scaffold_line("Snapback workspace scaffold", result.workspace_root)
    _echo_scaffold_line("README", result.readme_path)
    _echo_scaffold_line("Manifest", result.manifest_path)
    _echo_scaffold_line("Example spec", result.example_spec_path)
    _echo_scaffold_line("Example solve spec", result.example_solve_spec_path)
    _echo_scaffold_line("Catalog", result.catalog_path)


@app.command("validate", help="Validate a v2 explicit snapback spec and emit a deterministic report.")
def validate_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.yaml.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    try:
        report = validate_snapback_spec(spec)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("design", help="Materialize one v2 explicit snapback design bundle.")
def design_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing run directory if it already exists.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the run payload as JSON."),
) -> None:
    try:
        validation_report = validate_snapback_spec(spec)
        if validation_report.status == "invalid_catalog":
            if json_output:
                typer.echo(validation_report.model_dump_json(indent=2))
            else:
                _print_report(validation_report)
            raise typer.Exit(code=1)
        run_dir, report = run_snapback_design(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(
            json.dumps(
                {"run_dir": str(run_dir), "status": report.status, "spec_name": report.spec_name},
                indent=2,
            )
        )
    else:
        console.print(f"Outputs -> {run_dir}")
        _print_report(report)
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("solve", help="Search for concrete snapback designs that satisfy a v2 solve spec.")
def solve_cmd(
    spec: Path = typer.Option(
        ...,
        "--spec",
        help="Path to <workspace>/configs/snapback/<name>.snapback.solve.yaml.",
    ),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Replace an existing deterministic solve run directory.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the solve report as JSON."),
) -> None:
    try:
        run_dir, report = run_snapback_solve(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        console.print(f"Snapback solve outputs -> {run_dir}")
        _print_solve_report(report)
    if report.status not in {"satisfied", "search_truncated"}:
        raise typer.Exit(code=1)


@app.command("show", help="Read a snapback design or solve bundle and print a path-oriented summary.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a snapback run directory."),
    json_output: bool = typer.Option(False, "--json", help="Print the show payload as JSON."),
) -> None:
    try:
        payload = snapback_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"Snapback bundle -> {payload['spec_name']}")
    console.print(f"Kind -> {payload['kind']}")
    console.print(f"Status -> {payload['status']}")
    if payload["kind"] == "explicit":
        console.print(f"Manifest -> {payload['manifest_path']}")
        console.print(f"Status file -> {payload['status_path']}")
        console.print(f"Report JSON -> {payload['report_json']}")
        console.print(f"Report Markdown -> {payload['report_md']}")
    else:
        console.print(f"Solve manifest -> {payload['solve_manifest']}")
        console.print(f"Solve status -> {payload['solve_status']}")
        console.print(f"Solve report -> {payload['solve_report']}")
