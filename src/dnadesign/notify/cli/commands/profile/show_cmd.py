"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/profile/show_cmd.py

Profile show command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_profile_show_command(
    profile_app: typer.Typer,
    *,
    show_handler: Callable[..., None],
) -> None:
    @profile_app.command("show")
    def profile_show(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
    ) -> None:
        show_handler(profile=profile)
