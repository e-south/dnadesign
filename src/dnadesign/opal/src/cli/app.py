"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/app.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys

import typer

from ..core.pretty import maybe_install_rich_traceback
from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter
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
        False,
        "--debug/--no-debug",
        help="Print full tracebacks on internal errors (OPAL_DEBUG=1).",
    ),
    color: bool = typer.Option(
        True,
        "--color/--no-color",
        help="Enable/disable styled output (Rich). JSON output is never styled.",
    ),
) -> None:
    """
    Root callback sets global debug + CLI styling behavior via env flags.
    """
    os.environ["OPAL_DEBUG"] = "1" if debug else "0"

    # Rich styling is opt-in via environment; default on for TTY.
    os.environ["OPAL_CLI_RICH"] = "1" if color else "0"
    os.environ["OPAL_CLI_MARKUP"] = "1" if color else "0"

    # Pretty tracebacks if Rich is enabled
    if color:
        maybe_install_rich_traceback()
    # Silence PyArrow sysctlbyname warnings on macOS (no-op elsewhere)
    maybe_install_pyarrow_sysctl_filter()


def _build() -> typer.Typer:
    discover_commands()
    install_registered_commands(app)
    return app


def main() -> None:
    maybe_install_pyarrow_sysctl_filter()
    _build()
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("Aborted by user (Ctrl-C).", err=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
