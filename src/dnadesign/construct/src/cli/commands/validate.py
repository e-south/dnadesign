"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/validate.py

Validation command surfaces for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...api import load_job_config, preflight_from_config
from ...errors import ConstructError

validate_app = typer.Typer(no_args_is_help=True, help="Validation commands for construct.")


@validate_app.command("config")
def validate_config(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, help="Construct job YAML."),
    runtime: bool = typer.Option(
        False,
        "--runtime",
        help="Resolve template and input dataset, then report the planned runtime summary.",
    ),
) -> None:
    try:
        loaded, config_path = load_job_config(config)
    except ConstructError as exc:
        typer.echo(f"Error: {exc}")
        raise typer.Exit(1) from exc
    preflight = None
    if runtime:
        try:
            preflight = preflight_from_config(config)
        except ConstructError as exc:
            typer.echo(f"Error: {exc}")
            raise typer.Exit(1) from exc
    typer.echo(f"Config OK: {config_path}")
    typer.echo(f"job_id: {loaded.job.id}")
    typer.echo(f"input_dataset: {loaded.job.input.dataset}")
    typer.echo(f"output_dataset: {loaded.job.output.dataset}")
    if preflight is None:
        return
    typer.echo(f"template_id: {preflight.template_id}")
    typer.echo(f"template_length: {preflight.template_length}")
    typer.echo(f"template_circular: {str(preflight.template_circular).lower()}")
    typer.echo(f"rows_total: {preflight.records_total}")
    for row in preflight.planned_rows:
        typer.echo(
            "row: "
            f"input_id={row.input_id} "
            f"anchor_length={row.anchor_length} "
            f"full_construct_length={row.full_construct_length} "
            f"output_length={row.output_length}"
        )
