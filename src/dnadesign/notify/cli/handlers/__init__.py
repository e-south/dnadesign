"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/__init__.py

Execution entrypoints for notify CLI command groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .profile import (
    run_profile_doctor_command,
    run_profile_init_command,
    run_profile_show_command,
    run_profile_wizard_command,
)
from .runtime import run_spool_drain_command, run_usr_events_watch_command
from .send import run_send_command
from .setup import (
    run_setup_list_workspaces_command,
    run_setup_resolve_events_command,
    run_setup_slack_command,
    run_setup_webhook_command,
)

__all__ = [
    "run_profile_doctor_command",
    "run_profile_init_command",
    "run_profile_show_command",
    "run_profile_wizard_command",
    "run_spool_drain_command",
    "run_send_command",
    "run_setup_list_workspaces_command",
    "run_setup_resolve_events_command",
    "run_setup_slack_command",
    "run_setup_webhook_command",
    "run_usr_events_watch_command",
]
