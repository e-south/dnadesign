"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/presets.py

Registration for infer presets CLI command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...presets import list_presets, load_preset
from ..console import console, render_preset_detail, render_presets_table


def register(app: typer.Typer) -> None:
    presets_app = typer.Typer(no_args_is_help=True, help="Presets registry.")
    app.add_typer(presets_app, name="presets")

    @presets_app.command("list", help="List available presets.")
    def presets_list() -> None:
        render_presets_table(list_presets())

    @presets_app.command("show", help="Show a preset details.")
    def presets_show(preset: str = typer.Argument(...)) -> None:
        try:
            render_preset_detail(load_preset(preset))
        except KeyError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=2) from error
