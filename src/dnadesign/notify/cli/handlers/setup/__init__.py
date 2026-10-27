"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/setup/__init__.py

Execution entrypoints for notify setup commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .list_workspaces_cmd import run_setup_list_workspaces_command
from .resolve_events_cmd import run_setup_resolve_events_command
from .slack_cmd import run_setup_slack_command
from .webhook_cmd import run_setup_webhook_command

__all__ = [
    "run_setup_list_workspaces_command",
    "run_setup_resolve_events_command",
    "run_setup_slack_command",
    "run_setup_webhook_command",
]
