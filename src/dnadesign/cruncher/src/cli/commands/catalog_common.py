"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog_common.py

Shared helpers for catalog CLI command modules.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config

console = Console()


def load_config_or_exit(config_path: Path):
    try:
        return load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


def render_pwm_matrix(table_title: str, pwm_matrix: list[list[float]]) -> Table:
    table = Table(title=table_title, header_style="bold")
    table.add_column("Pos", justify="right")
    table.add_column("A")
    table.add_column("C")
    table.add_column("G")
    table.add_column("T")
    for idx, row in enumerate(pwm_matrix, start=1):
        table.add_row(str(idx), *(f"{val:.3f}" for val in row))
    return table
