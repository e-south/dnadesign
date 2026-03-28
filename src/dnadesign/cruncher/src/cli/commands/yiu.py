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
    help="Scaffold, validate, trace, design, solve, and inspect YIU hairpin oligo processing workflows.",
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


def run_yiu_solve(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_solve_workflow import run_yiu_solve as _run_yiu_solve

    return _run_yiu_solve(*args, **kwargs)


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
    console.print(f"Runbook doc -> {result.runbook_doc_path}")
    console.print(f"Spec -> {result.spec_path}")
    console.print(f"Solve spec -> {result.solve_spec_path}")
    console.print(f"Compat specs -> {', '.join(str(path) for path in result.compat_spec_paths)}")
    console.print(f"Enzyme catalog -> {result.enzyme_catalog_path}")
    console.print(f"Oligo-parts catalog -> {result.oligo_parts_catalog_path}")
    console.print(f"Backbone catalog -> {result.backbone_catalog_path}")


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


@app.command("design", help="Write explicit YIU run artifacts. Operationally identical to trace in this tranche.")
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


@app.command(
    "trace",
    help="Materialize the modeled YIU state graph. Operationally identical to design in this tranche.",
)
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


@app.command("solve", help="Search for concrete YIU source sequences that satisfy a YIU solve spec.")
def solve_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.solve.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic solve directory."
    ),
    max_hits: int | None = typer.Option(None, "--max-hits", help="Override search.max_hits for this run."),
    materialize_top_k: int | None = typer.Option(
        None,
        "--materialize-top-k",
        help="Override search.materialize_top_k for this run.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full solve report as JSON."),
) -> None:
    try:
        run_dir, report = run_yiu_solve(
            spec,
            force_overwrite=force_overwrite,
            max_hits=max_hits,
            materialize_top_k=materialize_top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"YIU solve outputs -> {run_dir}")
        console.print(f"Status -> {report.status}")
        console.print(f"Solve id -> {report.solve_id}")
        console.print(f"Accepted hits -> {report.metadata.accepted_candidate_count}")
        console.print(f"Returned hits -> {report.metadata.returned_hit_count}")
        console.print(f"Materialized hits -> {report.metadata.materialized_hit_count}")
        if report.metadata.search_truncated:
            console.print("Search truncated -> True")
        if report.metadata.accepted_pool_truncated:
            console.print("Accepted pool truncated -> True")
        if report.metadata.warning_codes:
            console.print(f"Warning codes -> {', '.join(report.metadata.warning_codes)}")
    if report.status != "solved":
        raise typer.Exit(code=1)


@app.command("show", help="Show paths and summary for a YIU run directory.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a YIU run directory under outputs/yiu/explicit/."),
    json_output: bool = typer.Option(False, "--json", help="Print the normalized bundle inventory as JSON."),
) -> None:
    try:
        payload = yiu_show_payload(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"YIU run -> {payload['spec_name']}")
    console.print(f"Bundle kind -> {payload['bundle_kind']}")
    console.print(f"Run id -> {payload['run_id']}")
    console.print(f"Run dir -> {payload['run_dir']}")
    if payload.get("solve_id"):
        console.print(f"Solve id -> {payload['solve_id']}")
    console.print(f"Status -> {payload['status']}: {payload['status_message']}")
    if payload.get("protocol_template"):
        console.print(f"Protocol template -> {payload['protocol_template']}")
    elif payload.get("protocol"):
        console.print(f"Protocol -> {payload['protocol']}")
    if payload.get("template_alias_used"):
        console.print(f"Template alias -> {payload['template_alias_used']}")
    if payload.get("template_alias_status"):
        console.print(f"Template alias status -> {payload['template_alias_status']}")
    if payload.get("view_contract_version") is not None:
        console.print(f"View contract -> {payload['view_contract_version']}")
    if payload.get("step_count") is not None:
        console.print(f"Step count -> {payload['step_count']}")
    if payload.get("state_count") is not None:
        console.print(f"State count -> {payload['state_count']}")
    if payload.get("issue_count") is not None:
        console.print(f"Issue count -> {payload['issue_count']}")
    if payload.get("accepted_candidate_count") is not None:
        console.print(f"Accepted hits -> {payload['accepted_candidate_count']}")
    if payload.get("returned_hit_count") is not None:
        console.print(f"Returned hits -> {payload['returned_hit_count']}")
    if payload.get("materialized_hit_count") is not None:
        console.print(f"Materialized hits -> {payload['materialized_hit_count']}")
    if payload.get("search_truncated") is not None:
        console.print(f"Search truncated -> {payload['search_truncated']}")
    if payload.get("accepted_pool_truncated") is not None:
        console.print(f"Accepted pool truncated -> {payload['accepted_pool_truncated']}")
    if payload.get("warning_codes"):
        console.print(f"Warning codes -> {', '.join(str(code) for code in payload['warning_codes'])}")
    if payload.get("final_state_kind"):
        console.print(f"Final state kind -> {payload['final_state_kind']}")
    console.print(f"View count -> {payload['emitted_view_count']}")
    console.print(f"Job count -> {payload['emitted_job_count']}")
    console.print(f"Render count -> {payload['emitted_render_count']}")
    console.print(f"Manifest -> {payload['manifest_path']}")
    console.print(f"Status file -> {payload['status_path']}")
    console.print(f"Report -> {payload['report_path']}")
    if payload.get("trace_path"):
        console.print(f"Trace -> {payload['trace_path']}")
    if payload.get("trace_manifest_path"):
        console.print(f"Trace manifest -> {payload['trace_manifest_path']}")
    if payload.get("published_views_dir"):
        console.print(f"Published views -> {payload['published_views_dir']}")
    if payload.get("visual_manifest_path"):
        console.print(f"Visual manifest -> {payload['visual_manifest_path']}")
    if payload.get("published_jobs_dir"):
        console.print(f"Published jobs -> {payload['published_jobs_dir']}")
    if payload.get("published_renders_dir"):
        console.print(f"Published renders -> {payload['published_renders_dir']}")
    if payload.get("first_hit_path"):
        console.print(f"First hit -> {payload['first_hit_path']}")
    if payload.get("top_hit_bundle_paths"):
        for hit_path in payload["top_hit_bundle_paths"]:
            console.print(f"Top hit bundle -> {hit_path}")
    if payload.get("top_hit_ids"):
        console.print(f"Top hits -> {', '.join(str(item) for item in payload['top_hit_ids'])}")
