"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/setup/list_workspaces_cmd.py

Setup list-workspaces command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_setup_list_workspaces_command(
    setup_app: typer.Typer,
    *,
    list_workspaces_handler: Callable[..., None],
) -> None:
    @setup_app.command("list-workspaces")
    def list_workspaces(
        tool: str = typer.Option(..., "--tool", help="Tool name (for example: densegen)."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        list_workspaces_handler(
            tool=tool,
            json_output=json_output,
        )
