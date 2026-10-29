"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/commands.py

Cluster CLI command registration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer
from rich.console import Console

from .commands_analysis import register_analyze_command
from .commands_fit import register_fit_command, register_sweep_command
from .commands_table import register_delete_columns_command, register_intra_similarity_command
from .commands_umap import register_umap_command


def register_all(app: typer.Typer, *, console: Console) -> None:
    register_fit_command(app, console=console)
    register_delete_columns_command(app, console=console)
    register_umap_command(app, console=console)
    register_sweep_command(app, console=console)
    register_analyze_command(app, console=console)
    register_intra_similarity_command(app, console=console)


__all__ = ["register_all"]
