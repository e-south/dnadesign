"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/__init__.py

Notify CLI command registration helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .delivery import register_send_command
from .profile import register_profile_commands
from .runtime import register_spool_drain_command, register_usr_events_watch_command
from .setup import register_setup_commands

__all__ = [
    "register_profile_commands",
    "register_send_command",
    "register_setup_commands",
    "register_spool_drain_command",
    "register_usr_events_watch_command",
]
