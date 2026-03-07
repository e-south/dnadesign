"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/adapters.py

Registration for infer adapter utility CLI command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from ...engine import clear_adapter_cache
from ...registry import list_fns, list_models
from ..console import console, render_adapters_table, render_functions_table


def register(app: typer.Typer) -> None:
    adapters_app = typer.Typer(no_args_is_help=True, help="Adapter utilities.")
    app.add_typer(adapters_app, name="adapters")

    @adapters_app.command("list", help="List registered model_ids and capabilities.")
    def adapters_list() -> None:
        render_adapters_table(list_models())

    @adapters_app.command("fns", help="List registered namespaced functions.")
    def adapters_fns() -> None:
        render_functions_table(list_fns())

    @adapters_app.command("cache-clear", help="Clear in-process adapter cache.")
    def adapters_cache_clear() -> None:
        clear_adapter_cache()
        console.print("[green]✔ Adapter cache cleared.[/green]")
