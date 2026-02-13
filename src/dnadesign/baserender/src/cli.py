"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/cli.py

Baserender vNext CLI (Job v3 only).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .api import run_job_v3
from .config import list_style_presets, load_job_v3, resolve_preset_path, validate_job_v3
from .core import BaseRenderError
from .workspace import default_workspaces_root, discover_workspaces, init_workspace, resolve_workspace_job_path

app = typer.Typer(help="Baserender vNext CLI")
job_app = typer.Typer(help="Job v3 commands")
style_app = typer.Typer(help="Style commands")
workspace_app = typer.Typer(help="Workspace commands")
app.add_typer(job_app, name="job")
app.add_typer(style_app, name="style")
app.add_typer(workspace_app, name="workspace")


def _resolve_job_spec(job: str | None, workspace: str | None, workspace_root: Path | None) -> str:
    if (job is None and workspace is None) or (job is not None and workspace is not None):
        raise BaseRenderError("Provide exactly one of <job> or --workspace")
    if workspace is not None:
        return str(resolve_workspace_job_path(workspace, root=workspace_root))
    assert job is not None
    return job


@job_app.command("validate")
def job_validate(
    job: str | None = typer.Argument(None, help="Path to Job v3 YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """Validate a Job v3 config."""
    try:
        parsed = validate_job_v3(
            _resolve_job_spec(job, workspace, workspace_root),
            caller_root=Path.cwd(),
        )
    except BaseRenderError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"OK: {parsed.path}")


@job_app.command("run")
def job_run(
    job: str | None = typer.Argument(None, help="Path to Job v3 YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
) -> None:
    """Run a Job v3 config."""
    try:
        report = run_job_v3(
            _resolve_job_spec(job, workspace, workspace_root),
            caller_root=Path.cwd(),
        )
    except BaseRenderError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)

    typer.echo("Run complete")
    for key, value in report.outputs.items():
        typer.echo(f"- {key}: {value}")


@job_app.command("normalize")
def job_normalize(
    job: str | None = typer.Argument(None, help="Path to Job v3 YAML (or job name)."),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace name containing job.yml."),
    workspace_root: Path | None = typer.Option(
        None,
        "--workspace-root",
        help="Workspace root directory (default: <cwd>/workspaces).",
    ),
    out: Path = typer.Option(..., "--out", help="Output path for normalized YAML."),
) -> None:
    """Normalize and rewrite a Job v3 config with absolute resolved paths."""
    try:
        parsed = load_job_v3(
            _resolve_job_spec(job, workspace, workspace_root),
            caller_root=Path.cwd(),
        )
    except BaseRenderError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)

    data = {
        "version": 3,
        "results_root": str(parsed.results_root),
        "input": {
            "kind": parsed.input.kind,
            "path": str(parsed.input.path),
            "adapter": {
                "kind": parsed.input.adapter.kind,
                "columns": dict(parsed.input.adapter.columns),
                "policies": dict(parsed.input.adapter.policies),
            },
            "alphabet": parsed.input.alphabet,
            "limit": parsed.input.limit,
            "sample": (
                None
                if parsed.input.sample is None
                else {
                    "mode": parsed.input.sample.mode,
                    "n": parsed.input.sample.n,
                    "seed": parsed.input.sample.seed,
                }
            ),
        },
        "selection": (
            None
            if parsed.selection is None
            else {
                "path": str(parsed.selection.path),
                "match_on": parsed.selection.match_on,
                "column": parsed.selection.column,
                "overlay_column": parsed.selection.overlay_column,
                "keep_order": parsed.selection.keep_order,
                "on_missing": parsed.selection.on_missing,
            }
        ),
        "pipeline": {
            "plugins": [
                (spec.name if not spec.params else {spec.name: dict(spec.params)}) for spec in parsed.pipeline.plugins
            ]
        },
        "render": {
            "renderer": parsed.render.renderer,
            "style": {
                "preset": parsed.render.style_preset,
                "overrides": dict(parsed.render.style_overrides),
            },
        },
        "outputs": [
            (
                {
                    "kind": "images",
                    "dir": str(cfg.dir),
                    "fmt": cfg.fmt,
                }
                if cfg.kind == "images"
                else {
                    "kind": "video",
                    "path": str(cfg.path),
                    "fmt": cfg.fmt,
                    "fps": cfg.fps,
                    "frames_per_record": cfg.frames_per_record,
                    "pauses": dict(cfg.pauses),
                    "width_px": cfg.width_px,
                    "height_px": cfg.height_px,
                    "aspect": cfg.aspect_ratio,
                    "total_duration": cfg.total_duration,
                }
            )
            for cfg in parsed.outputs
        ],
        "run": {
            "strict": parsed.run.strict,
            "fail_on_skips": parsed.run.fail_on_skips,
            "emit_report": parsed.run.emit_report,
            "report_path": str(parsed.run.report_path) if parsed.run.report_path else None,
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    typer.echo(f"Wrote: {out}")


@style_app.command("list")
def style_list() -> None:
    """List available style presets."""
    presets = list_style_presets()
    for name in presets:
        typer.echo(name)


@style_app.command("show")
def style_show(
    preset: str = typer.Argument(..., help="Preset name or path."),
    as_json: bool = typer.Option(False, "--json", help="Output JSON instead of YAML."),
) -> None:
    """Show a style preset file."""
    try:
        path = resolve_preset_path(preset)
        if path is None:
            raise ValueError("Preset path is null")
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)

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
        workspaces = discover_workspaces(root=root)
    except BaseRenderError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)

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
    """Create a workspace scaffold with job.yml, inputs/, outputs/, and reports/."""
    try:
        workspace = init_workspace(name, root=root)
    except BaseRenderError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"Workspace created: {workspace.root}")
    typer.echo(f"- job: {workspace.job_path}")
    typer.echo(f"- inputs: {workspace.inputs_dir}")
    typer.echo(f"- outputs: {workspace.outputs_dir}")


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()
