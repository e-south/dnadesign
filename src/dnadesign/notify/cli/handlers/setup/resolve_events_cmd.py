"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/setup/resolve_events_cmd.py

Execution logic for notify setup resolve-events command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import typer

from ....errors import NotifyConfigError, NotifyError


def run_setup_resolve_events_command(
    *,
    tool: str,
    config: Path | None,
    workspace: str | None,
    print_policy: bool,
    json_output: bool,
    normalize_tool_name_fn: Callable[[str | None], str | None],
    resolve_tool_workspace_config_fn: Callable[..., Path],
    resolve_tool_events_path_fn: Callable[..., tuple[Path, str | None]],
) -> None:
    try:
        if print_policy and json_output:
            raise NotifyConfigError("pass either --print-policy or --json, not both")
        if (config is None) == (workspace is None):
            raise NotifyConfigError("pass exactly one of --config or --workspace")
        tool_name = normalize_tool_name_fn(tool)
        if tool_name is None:
            raise NotifyConfigError("tool must be a non-empty string")
        if config is not None:
            config_path = config.expanduser().resolve()
        else:
            workspace_name = str(workspace or "").strip()
            if not workspace_name:
                raise NotifyConfigError("workspace must be a non-empty string")
            config_path = resolve_tool_workspace_config_fn(
                tool=tool_name,
                workspace=workspace_name,
                search_start=Path.cwd(),
            )
        events_path, default_policy = resolve_tool_events_path_fn(tool=tool_name, config=config_path)

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "ok": True,
                        "tool": tool_name,
                        "config": str(config_path),
                        "events": str(events_path),
                        "policy": default_policy,
                    },
                    sort_keys=True,
                )
            )
            return
        if print_policy:
            if default_policy is not None:
                typer.echo(default_policy)
            return
        typer.echo(str(events_path))
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
