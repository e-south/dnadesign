"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/optimizers.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="List available MCMC optimizers.")
console = Console()


@app.command("list", help="List available optimizer kernels.")
def list_optimizers() -> None:
    from dnadesign.cruncher.core.optimizers.registry import list_optimizer_specs

    table = Table(title="Available optimizers", header_style="bold")
    table.add_column("Name")
    table.add_column("Description")
    for spec in list_optimizer_specs():
        table.add_row(spec.name, spec.description)
    console.print(table)
