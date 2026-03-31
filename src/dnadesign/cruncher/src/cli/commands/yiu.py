"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/yiu.py

CLI entrypoints for the YIU hairpin oligo processing workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    no_args_is_help=True,
    help="Scaffold, validate, trace, solve, render, and inspect YIU hairpin oligo processing workflows.",
)
console = Console()


def validate_yiu_spec(*args, **kwargs):
    from dnadesign.cruncher.app.yiu_workflow import validate_yiu_spec as _validate_yiu_spec

    return _validate_yiu_spec(*args, **kwargs)


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


def ensure_workspace_mpl_cache(*args, **kwargs):
    from dnadesign.cruncher.viz.mpl import ensure_workspace_mpl_cache as _ensure_workspace_mpl_cache

    return _ensure_workspace_mpl_cache(*args, **kwargs)


def ensure_mpl_cache(*args, **kwargs):
    from dnadesign.cruncher.viz.mpl import ensure_mpl_cache as _ensure_mpl_cache

    return _ensure_mpl_cache(*args, **kwargs)


def infer_workspace_root_from_output_artifact(*args, **kwargs):
    from dnadesign.cruncher.viz.mpl import (
        infer_workspace_root_from_output_artifact as _infer_workspace_root_from_output_artifact,
    )

    return _infer_workspace_root_from_output_artifact(*args, **kwargs)


def _render_status(*, job_count: int, rendered_count: int) -> str:
    if job_count <= 0:
        return "not_requested"
    if rendered_count <= 0:
        return "missing"
    if rendered_count >= job_count:
        return "rendered"
    return "partial"


def _format_hard_invariant_summary(summary: object) -> str | None:
    if not isinstance(summary, dict):
        return None
    total = summary.get("total")
    guaranteed = summary.get("guaranteed")
    impossible = summary.get("impossible")
    state_id = summary.get("state_id")
    if total is None or guaranteed is None or impossible is None:
        return None
    base = f"{guaranteed}/{total} guaranteed"
    if impossible:
        base = f"{base}, {impossible} impossible"
    if state_id:
        return f"{base} in {state_id}"
    return base


def _format_visual_render_summary(summary: object) -> str | None:
    if not isinstance(summary, dict):
        return None
    render_status = summary.get("render_status")
    render_count = summary.get("render_count")
    view_count = summary.get("view_count")
    if render_status is None or render_count is None or view_count is None:
        return None
    return f"{render_status} ({render_count}/{view_count})"


def run_yiu_render(run: Path) -> dict[str, object]:
    resolved = Path(run).expanduser().resolve()
    visual_inventory_path = resolved / "visual_inventory.json"
    if not visual_inventory_path.exists():
        raise FileNotFoundError(f"visual inventory not found: {visual_inventory_path}")
    payload = json.loads(visual_inventory_path.read_text(encoding="utf-8"))
    views = payload.get("views")
    if not isinstance(views, list):
        raise ValueError("visual_inventory.json must define a 'views' list")
    workspace_root = infer_workspace_root_from_output_artifact(visual_inventory_path)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
    else:
        ensure_mpl_cache(resolved)

    baserender = importlib.import_module("dnadesign.baserender")

    render_paths: list[str] = []
    rendered_count = 0
    job_count = 0
    render_timestamp: str | None = None
    for entry in views:
        if not isinstance(entry, dict):
            continue
        contract_relpath = entry.get("view_contract_path")
        render_relpath = entry.get("render_artifact_path")
        if not contract_relpath or not render_relpath:
            entry["render_requested"] = False
            entry["render_completed"] = False
            entry["last_rendered_at"] = None
            continue
        resolved_contract_path = (resolved / str(contract_relpath)).resolve()
        resolved_render_path = (resolved / str(render_relpath)).resolve()
        job_count += 1
        baserender.run_job(
            {
                "version": 3,
                "results_root": ".",
                "input": {
                    "kind": "json",
                    "path": str(resolved_contract_path),
                    "adapter": {"kind": str(entry.get("contract_kind") or "sequence_evidence_map_v1")},
                    "alphabet": "iupac_dna",
                },
                "render": {
                    "renderer": str(
                        entry.get("renderer_kind") or payload.get("renderer_kind") or "nucleotide_evidence_map"
                    ),
                    "style": {"preset": None, "overrides": {}},
                },
                "outputs": [{"kind": "images", "path": str(resolved_render_path), "fmt": "pdf"}],
                "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
            },
            kind="render_job_v3",
            caller_root=resolved,
        )
        if resolved_render_path.exists():
            render_timestamp = datetime.now(timezone.utc).isoformat()
            entry["render_requested"] = True
            entry["render_completed"] = True
            entry["last_rendered_at"] = render_timestamp
            render_paths.append(str(resolved_render_path))
            rendered_count += 1
        else:
            entry["render_requested"] = True
            entry["render_completed"] = False
            entry["last_rendered_at"] = None
    payload["render_count"] = rendered_count
    payload["render_status"] = _render_status(job_count=job_count, rendered_count=rendered_count)
    payload["last_rendered_at"] = render_timestamp if rendered_count > 0 else None
    visual_inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "run_dir": str(resolved),
        "visual_inventory_path": str(visual_inventory_path.resolve()),
        "job_count": job_count,
        "rendered_count": rendered_count,
        "render_paths": render_paths,
    }


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


@app.command(
    "trace",
    help="Materialize the modeled YIU state graph and explicit bundle.",
)
def trace_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic run directory."
    ),
    emit_renders: bool = typer.Option(
        False,
        "--emit-renders",
        help="Immediately render every published BaseRender job after the trace bundle is written.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    if emit_renders and json_output:
        raise typer.BadParameter("--emit-renders cannot be combined with --json.")
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
        if emit_renders:
            render_payload = run_yiu_render(run_dir)
            typer.echo(f"Rendered jobs -> {render_payload['job_count']}")
            for render_path in render_payload.get("render_paths", []):
                typer.echo(f"Render -> {render_path}")
    if report.status != "satisfied":
        raise typer.Exit(code=1)


@app.command("render", help="Run every published BaseRender job listed in visual_inventory.json for one YIU bundle.")
def render_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a YIU explicit or solve run directory."),
) -> None:
    try:
        payload = run_yiu_render(run)
    except Exception as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    visual_inventory_path = payload.get(
        "visual_inventory_path",
        str(Path(run).expanduser().resolve() / "visual_inventory.json"),
    )
    render_paths = payload.get("render_paths", [])
    typer.echo(f"YIU render run -> {payload['run_dir']}")
    typer.echo(f"Visual inventory -> {visual_inventory_path}")
    typer.echo(f"Rendered jobs -> {payload['job_count']}")
    for render_path in render_paths:
        typer.echo(f"Render -> {render_path}")


@app.command("solve", help="Search for a canonical satisfying YIU source sequence within the bounded solve window.")
def solve_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/yiu/<name>.yiu.solve.yaml."),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Replace an existing deterministic solve directory."
    ),
    emit_renders: bool = typer.Option(
        False,
        "--emit-renders",
        help="Immediately render every published BaseRender job after the solve bundle is written.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full solve report as JSON."),
) -> None:
    if emit_renders and json_output:
        raise typer.BadParameter("--emit-renders cannot be combined with --json.")
    try:
        run_dir, report = run_yiu_solve(spec, force_overwrite=force_overwrite)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        console.print(f"YIU solve outputs -> {run_dir}")
        console.print(f"Status -> {report.status}")
        console.print(f"Solve id -> {report.solve_id}")
        console.print(f"Satisfying solutions -> {report.satisfying_solution_count}")
        console.print(f"Exhaustive search -> {report.metadata.exhaustive_search}")
        if report.selected_solution_path:
            console.print(f"Selected solution -> {report.selected_solution_path}")
        if report.metadata.warning_codes:
            console.print(f"Warning codes -> {', '.join(report.metadata.warning_codes)}")
        if emit_renders:
            render_payload = run_yiu_render(run_dir)
            typer.echo(f"Rendered jobs -> {render_payload['job_count']}")
            for render_path in render_payload.get("render_paths", []):
                typer.echo(f"Render -> {render_path}")
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
    console.print(f"Bundle kind -> {payload['bundle_kind']}")
    console.print(f"Run id -> {payload['run_id']}")
    console.print(f"Run dir -> {payload['run_dir']}")
    console.print(f"Protocol template -> {payload.get('protocol_template')}")
    if payload["bundle_kind"] == "solve":
        if payload.get("canonical_template_id"):
            console.print(f"Template id -> {payload['canonical_template_id']}")
        console.print(f"Solve status -> {payload.get('solve_status')}")
        if payload.get("exhaustive_search") is not None:
            console.print(f"Exhaustive search -> {payload['exhaustive_search']}")
        console.print(f"Satisfying solutions -> {payload.get('satisfying_solution_count')}")
        console.print(f"Comparison solutions -> {payload.get('comparison_solution_count')}")
        if payload.get("selected_canonical_solution_path"):
            console.print(f"Selected solution -> {payload['selected_canonical_solution_path']}")
    else:
        console.print(f"Schema version -> {payload.get('schema_version')}")
        console.print(f"State count -> {payload.get('state_count')}")
        console.print(f"Explicit final state -> {payload.get('explicit_final_state')}")
    hard_invariants = _format_hard_invariant_summary(payload.get("hard_invariant_summary"))
    if hard_invariants is not None:
        console.print(f"Hard invariants -> {hard_invariants}")
    visual_renders = _format_visual_render_summary(payload.get("visual_render_summary"))
    if visual_renders is not None:
        console.print(f"Visual renders -> {visual_renders}")
    if payload.get("visual_inventory_path"):
        console.print(f"Visual inventory -> {payload['visual_inventory_path']}")
    key_artifact_paths = payload.get("key_artifact_paths")
    if isinstance(key_artifact_paths, dict):
        for label, artifact_path in key_artifact_paths.items():
            if artifact_path:
                console.print(f"{label.replace('_', ' ').title()} -> {artifact_path}")
