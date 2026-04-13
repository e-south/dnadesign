"""
Inspection CLI commands for latentdna.
"""

from __future__ import annotations

import typer

from ...services.inspection_service import (
    inspect_alignment,
    inspect_artifacts,
    inspect_landmarks,
    inspect_missingness,
    inspect_notebook_health,
    inspect_plots,
    inspect_source,
    inspect_views,
)
from ..common import emit, fail, resolve_format

app = typer.Typer(help="Inspection commands for latentdna.")


@app.command("source")
def source(
    source_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_source(workspace, source_id)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("views")
def views(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_views(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("alignment")
def alignment(
    alignment_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_alignment(workspace, alignment_id)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("landmarks")
def landmarks(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_landmarks(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("missingness")
def missingness(
    source_id: str = typer.Argument(...),
    workspace: str = typer.Option(..., "--workspace"),
    column: list[str] = typer.Option([], "--column"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_missingness(workspace, source_id, columns=list(column) or None)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("artifacts")
def artifacts(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_artifacts(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("plots")
def plots(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_plots(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)


@app.command("notebook-health")
def notebook_health(
    workspace: str = typer.Option(..., "--workspace"),
    format_name: str = typer.Option("text", "--format"),
    json_output: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    try:
        payload = inspect_notebook_health(workspace)
    except Exception as exc:
        fail(exc)
    emit(payload, format_name=resolve_format(json_output=json_output, format_name=format_name), quiet=quiet)
