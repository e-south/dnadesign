"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/app.py

Infer CLI root app and top-level callback wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

import typer

from ..bootstrap import initialize_registry
from .commands import register_all
from .console import rich_tracebacks, setup_console_logging

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Model-agnostic sequence inference CLI.",
)


@app.callback()
def _root(
    log_level: str = typer.Option(
        os.environ.get("INFER_LOG_LEVEL", "INFO"),
        "--log-level",
        help="Console log level.",
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON logs."),
    trace: bool = typer.Option(False, "--trace", help="Rich tracebacks on errors."),
) -> None:
    initialize_registry()
    setup_console_logging(log_level, json_logs)
    rich_tracebacks(enabled=trace)


register_all(app)
