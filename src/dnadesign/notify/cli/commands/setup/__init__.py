"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/setup/__init__.py

Setup command registration for observer-only Notify onboarding flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer

from .list_workspaces_cmd import register_setup_list_workspaces_command
from .resolve_events_cmd import register_setup_resolve_events_command
from .slack_cmd import register_setup_slack_command
from .webhook_cmd import register_setup_webhook_command


def register_setup_commands(
    setup_app: typer.Typer,
    *,
    slack_handler: Callable[..., None],
    webhook_handler: Callable[..., None],
    resolve_events_handler: Callable[..., None],
    list_workspaces_handler: Callable[..., None],
) -> None:
    register_setup_slack_command(setup_app, slack_handler=slack_handler)
    register_setup_webhook_command(setup_app, webhook_handler=webhook_handler)
    register_setup_resolve_events_command(setup_app, resolve_events_handler=resolve_events_handler)
    register_setup_list_workspaces_command(setup_app, list_workspaces_handler=list_workspaces_handler)


__all__ = ["register_setup_commands"]
