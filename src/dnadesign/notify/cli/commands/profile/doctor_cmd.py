"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/profile/doctor_cmd.py

Profile doctor command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_profile_doctor_command(
    profile_app: typer.Typer,
    *,
    doctor_handler: Callable[..., None],
) -> None:
    @profile_app.command("doctor")
    def profile_doctor(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        doctor_handler(profile=profile, json_output=json_output)
