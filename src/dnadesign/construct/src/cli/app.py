"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/app.py

Construct CLI root app wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from .commands import register_all

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Create contextualized or multi-part DNA constructs and write them into USR datasets.",
)

register_all(app)
