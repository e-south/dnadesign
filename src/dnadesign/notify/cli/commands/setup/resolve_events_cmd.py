"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/setup/resolve_events_cmd.py

Setup resolve-events command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_setup_resolve_events_command(
    setup_app: typer.Typer,
    *,
    resolve_events_handler: Callable[..., None],
) -> None:
    @setup_app.command("resolve-events")
    def resolve_events(
        tool: str = typer.Option(..., "--tool", help="Tool name for resolver mode."),
        config: Path | None = typer.Option(None, "--config", "-c", help="Tool config path for resolver mode."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            help="Workspace name for resolver mode (shorthand for tool workspace config path).",
        ),
        print_policy: bool = typer.Option(
            False,
            "--print-policy",
            help="Print only the default policy for the resolved tool/config.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        resolve_events_handler(
            tool=tool,
            config=config,
            workspace=workspace,
            print_policy=print_policy,
            json_output=json_output,
        )
