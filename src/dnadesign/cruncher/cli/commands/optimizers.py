"""Optimizer registry command."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.core.optimizers.registry import list_optimizer_specs

app = typer.Typer(no_args_is_help=True, help="List available MCMC optimizers.")
console = Console()


@app.command("list", help="List available optimizer kernels.")
def list_optimizers() -> None:
    table = Table(title="Available optimizers", header_style="bold")
    table.add_column("Name")
    table.add_column("Description")
    for spec in list_optimizer_specs():
        table.add_row(spec.name, spec.description)
    console.print(table)
