"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/setup/list_workspaces_cmd.py

Execution logic for notify setup list-workspaces command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import typer

from ....errors import NotifyError


def run_setup_list_workspaces_command(
    *,
    tool: str,
    json_output: bool,
    list_tool_workspaces_fn: Callable[..., list[str]],
) -> None:
    try:
        names = list_tool_workspaces_fn(tool=tool, search_start=Path.cwd())
        if json_output:
            typer.echo(json.dumps({"ok": True, "tool": str(tool).strip(), "workspaces": names}, sort_keys=True))
            return
        if not names:
            typer.echo("No workspaces found.")
            return
        for name in names:
            typer.echo(name)
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
