"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/app.py

Root OPS CLI application with lazily imported subcommand modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from .dispatch import LazyGroup

app = typer.Typer(
    cls=LazyGroup,
    no_args_is_help=True,
    help=(
        "Cross-tool orchestration commands for deterministic batch plans. "
        "Start with `uv run ops catalog list --simple` to browse routes from the terminal."
    ),
)


@app.callback()
def root_callback() -> None:
    """Root OPS CLI callback."""


def main() -> None:
    app()


__all__ = ["app", "main"]
