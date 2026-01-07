"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/workspaces.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

import typer
from dnadesign.cruncher.cli.config_resolver import discover_workspaces, workspace_search_roots
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="List discoverable cruncher workspaces.")
console = Console()


@app.command("list", help="List discoverable workspaces and their configs.")
def list_workspaces() -> None:
    workspaces = discover_workspaces()
    if not workspaces:
        roots = workspace_search_roots()
        roots_rendered = "\n".join(f"- {root}" for root in roots) if roots else "- (none)"
        console.print("No workspaces discovered.")
        console.print("Workspace discovery searched:")
        console.print(roots_rendered)
        console.print(f"Hint: set CRUNCHER_WORKSPACE_ROOTS=/path/a{os.pathsep}/path/b to add roots.")
        raise typer.Exit(code=1)

    table = Table(title="Workspaces", header_style="bold")
    table.add_column("Index", justify="right")
    table.add_column("Name")
    table.add_column("Config")
    table.add_column("Root")
    table.add_column("Catalog (.cruncher)")
    for idx, workspace in enumerate(workspaces, start=1):
        catalog_flag = "yes" if workspace.catalog_path.exists() else "no"
        table.add_row(
            str(idx),
            workspace.name,
            str(workspace.config_path),
            str(workspace.root),
            catalog_flag,
        )
    console.print(table)
