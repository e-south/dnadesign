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
from ._render import echo_validate_result

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
    echo_validate_result(config_path=config_path, loaded=loaded, preflight=preflight)
