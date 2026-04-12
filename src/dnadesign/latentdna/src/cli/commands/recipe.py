"""
Recipe CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.recipe_service import run_recipe, validate_recipe
from ..common import emit, fail, resolve_format
from ..previews import preview_recipe_run

app = typer.Typer(help="Recipe commands for latentdna.")


@app.command("validate")
def validate(
    recipe_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = validate_recipe(workspace, recipe_id).model_dump(mode="json")
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("run")
def run(
    recipe_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = (
            preview_recipe_run(workspace, recipe_id, force=force)
            if dry_run
            else run_recipe(workspace, recipe_id, force=force).model_dump(mode="json")
        )
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
