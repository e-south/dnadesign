"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/cli/commands/__init__.py

Command registration for construct CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from .run import run
from .seed import seed_app
from .validate import validate_app
from .workspace import workspace_app


def register_all(app: typer.Typer) -> None:
    app.command("run")(run)
    app.add_typer(seed_app, name="seed")
    app.add_typer(validate_app, name="validate")
    app.add_typer(workspace_app, name="workspace")
