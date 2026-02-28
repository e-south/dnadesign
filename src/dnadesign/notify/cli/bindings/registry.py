"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/registry.py

Command registration helpers for notify CLI bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from ..commands import (
    register_profile_commands,
    register_send_command,
    register_setup_commands,
    register_spool_drain_command,
    register_usr_events_watch_command,
)


def register_notify_cli_bindings(
    *,
    app: typer.Typer,
    usr_events_app: typer.Typer,
    spool_app: typer.Typer,
    profile_app: typer.Typer,
    setup_app: typer.Typer,
    usr_events_watch_handler: Callable[..., object],
    spool_drain_handler: Callable[..., object],
    send_handler: Callable[..., object],
    profile_init_handler: Callable[..., object],
    profile_wizard_handler: Callable[..., object],
    profile_show_handler: Callable[..., object],
    profile_doctor_handler: Callable[..., object],
    setup_slack_handler: Callable[..., object],
    setup_webhook_handler: Callable[..., object],
    setup_resolve_events_handler: Callable[..., object],
    setup_list_workspaces_handler: Callable[..., object],
) -> None:
    register_usr_events_watch_command(usr_events_app, watch_handler=usr_events_watch_handler)
    register_spool_drain_command(spool_app, drain_handler=spool_drain_handler)
    register_send_command(app, send_handler=send_handler)
    register_profile_commands(
        profile_app,
        init_handler=profile_init_handler,
        wizard_handler=profile_wizard_handler,
        show_handler=profile_show_handler,
        doctor_handler=profile_doctor_handler,
    )
    register_setup_commands(
        setup_app,
        slack_handler=setup_slack_handler,
        webhook_handler=setup_webhook_handler,
        resolve_events_handler=setup_resolve_events_handler,
        list_workspaces_handler=setup_list_workspaces_handler,
    )
