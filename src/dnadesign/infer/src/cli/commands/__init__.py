"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/__init__.py

CLI command-group registration for infer.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from .adapters import register as register_adapters_commands
from .extract import register as register_extract_command
from .generate import register as register_generate_command
from .presets import register as register_presets_commands
from .run import register as register_run_command
from .validate import register as register_validate_commands
from .workspace import register as register_workspace_commands


def register_all(app: typer.Typer) -> None:
    register_run_command(app)
    register_extract_command(app)
    register_generate_command(app)
    register_presets_commands(app)
    register_adapters_commands(app)
    register_validate_commands(app)
    register_workspace_commands(app)
