"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/app.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys

import typer

from .registry import discover_commands, install_registered_commands

# single Typer app exposed as entrypoint in pyproject.toml
app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="OPAL — Optimization with Active Learning.",
)


@app.callback()
def _root_callback(
    debug: bool = typer.Option(
        True, "--debug/--no-debug", help="Print full tracebacks on internal errors."
    )
) -> None:
    """Root callback sets global debug behavior."""
    os.environ["OPAL_DEBUG"] = "1" if debug else "0"


def _build() -> typer.Typer:
    discover_commands()
    install_registered_commands(app)
    return app


def main() -> None:
    _build()
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("Aborted by user (Ctrl-C).", err=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
