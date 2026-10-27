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

from ...contracts.errors import ConstructError
from ...interfaces.api import load_job_config, preflight_from_config
from ._errors import exit_with_error
from ._format import validate_output_format
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
    usr_root: Path | None = typer.Option(
        None,
        "--usr-root",
        help="Absolute operator-managed USR root used for runtime validation.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_requested = str(output_format or "").strip().lower()
    try:
        format_norm = validate_output_format(output_format)
        if usr_root is not None and not runtime:
            raise ConstructError("--usr-root requires --runtime.")
        loaded, config_path = load_job_config(config)
    except (ConstructError, OSError) as exc:
        exit_with_error(exc, code=1, output_format=format_requested)
    preflight = None
    if runtime:
        try:
            preflight = preflight_from_config(config, usr_root=usr_root)
        except (ConstructError, OSError) as exc:
            exit_with_error(exc, code=1, output_format=format_norm)
    echo_validate_result(config_path=config_path, loaded=loaded, preflight=preflight, output_format=format_norm)
