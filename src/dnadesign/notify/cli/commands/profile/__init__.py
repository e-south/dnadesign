"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/profile/__init__.py

Profile command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer

from .doctor_cmd import register_profile_doctor_command
from .init_cmd import register_profile_init_command
from .show_cmd import register_profile_show_command
from .wizard_cmd import register_profile_wizard_command


def register_profile_commands(
    profile_app: typer.Typer,
    *,
    init_handler: Callable[..., None],
    wizard_handler: Callable[..., None],
    show_handler: Callable[..., None],
    doctor_handler: Callable[..., None],
) -> None:
    register_profile_init_command(profile_app, init_handler=init_handler)
    register_profile_wizard_command(profile_app, wizard_handler=wizard_handler)
    register_profile_show_command(profile_app, show_handler=show_handler)
    register_profile_doctor_command(profile_app, doctor_handler=doctor_handler)


__all__ = ["register_profile_commands"]
