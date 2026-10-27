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
    help=(
        "Realize template/part constructs into USR datasets, or compose declared "
        "sequence products into explicit artifact bundles."
    ),
)

register_all(app)
