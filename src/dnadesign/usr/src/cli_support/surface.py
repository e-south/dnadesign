"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/cli_surface.py

Typer app construction and top-level help text for the USR CLI surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import typer

_ROOT_HELP = (
    "USR datasets CLI.\n\n"
    "Task-first runbook: src/dnadesign/usr/docs/operations/workflow-map.md\n"
    "Sync contract: pull/push default to verify=hash with strict sidecar checks."
)


@dataclass(frozen=True)
class CliApps:
    app: typer.Typer
    remotes_app: typer.Typer
    legacy_app: typer.Typer
    maintenance_app: typer.Typer
    densegen_app: typer.Typer
    dev_app: typer.Typer
    namespace_app: typer.Typer
    events_app: typer.Typer
    state_app: typer.Typer


def build_cli_apps(*, show_dev_commands: bool) -> CliApps:
    app = typer.Typer(add_completion=True, no_args_is_help=True, help=_ROOT_HELP)
    remotes_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Manage SSH remotes.")
    legacy_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Legacy dataset utilities.")
    maintenance_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Dataset maintenance utilities.")
    densegen_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Densegen-specific utilities.")
    dev_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Developer utilities (unstable).")
    namespace_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Manage namespace registry.")
    events_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inspect dataset events.")
    state_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Record state utilities.")

    app.add_typer(remotes_app, name="remotes")
    app.add_typer(legacy_app, name="legacy")
    app.add_typer(maintenance_app, name="maintenance")
    app.add_typer(densegen_app, name="densegen")
    app.add_typer(namespace_app, name="namespace")
    app.add_typer(events_app, name="events")
    app.add_typer(state_app, name="state")
    if show_dev_commands:
        app.add_typer(dev_app, name="dev")

    return CliApps(
        app=app,
        remotes_app=remotes_app,
        legacy_app=legacy_app,
        maintenance_app=maintenance_app,
        densegen_app=densegen_app,
        dev_app=dev_app,
        namespace_app=namespace_app,
        events_app=events_app,
        state_app=state_app,
    )
