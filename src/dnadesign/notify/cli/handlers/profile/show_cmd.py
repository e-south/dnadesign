"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/profile/show_cmd.py

Execution logic for notify profile show command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError


def run_profile_show_command(
    *,
    profile: Path,
    read_profile_fn: Callable[[Path], dict[str, Any]],
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        data = read_profile_fn(profile_path)
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
