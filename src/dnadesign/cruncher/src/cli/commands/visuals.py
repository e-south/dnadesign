"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/visuals.py

Thin Cruncher wrapper over the public BaseRender job API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from dnadesign.cruncher.viz.mpl import (
    ensure_mpl_cache,
    ensure_workspace_mpl_cache,
    infer_workspace_root_from_output_artifact,
)

app = typer.Typer(no_args_is_help=True, help="Validate or run published visual jobs through BaseRender's public API.")
console = Console()


def _load_baserender(job: Path):
    workspace_root = infer_workspace_root_from_output_artifact(job)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
    else:
        ensure_mpl_cache(job.parent)

    import dnadesign.baserender as baserender

    return baserender


@app.command("validate", help="Validate a published render job through dnadesign.baserender.validate_job.")
def validate_cmd(
    job: Path = typer.Option(..., "--job", help="Path to a RenderJobV3 YAML file."),
) -> None:
    try:
        baserender = _load_baserender(job)
        parsed = baserender.validate_job(job, kind="render_job_v3", caller_root=job.parent)
    except Exception as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Render job -> {parsed.path}")
    console.print("Render job kind -> render_job_v3")
    console.print(f"Renderer -> {parsed.render.renderer}")
    console.print(f"Results root -> {parsed.results_root}")


@app.command("run", help="Run a published render job through dnadesign.baserender.run_job.")
def run_cmd(
    job: Path = typer.Option(..., "--job", help="Path to a RenderJobV3 YAML file."),
) -> None:
    try:
        baserender = _load_baserender(job)
        report = baserender.run_job(job, kind="render_job_v3", caller_root=job.parent)
    except Exception as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Rendered job -> {job}")
    for key, value in sorted(report.outputs.items()):
        console.print(f"{key} -> {value}")
