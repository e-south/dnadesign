"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/run.py

construct run command implementation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...api import run_from_config
from ...errors import ConstructError
from ._errors import exit_with_error
from ._render import echo_run_result


def run(
    config: Path = typer.Option(..., "--config", exists=True, readable=True, help="Construct job YAML."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and build outputs without writing USR data.",
    ),
) -> None:
    try:
        result = run_from_config(config, dry_run=dry_run)
    except (ConstructError, OSError) as exc:
        exit_with_error(exc, code=1)

    echo_run_result(result)
