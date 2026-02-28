"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/__init__.py

Typer application router for notify command groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import typer

from .bindings import register_notify_cli_bindings

app = typer.Typer(help="Send notifications for dnadesign runs via webhooks.")
usr_events_app = typer.Typer(help="Consume USR JSONL events and emit notifications.")
spool_app = typer.Typer(help="Drain spooled notifications.")
profile_app = typer.Typer(help="Manage reusable notify profiles.")
setup_app = typer.Typer(help="Observer-only setup helpers for notify profiles.")

app.add_typer(usr_events_app, name="usr-events")
app.add_typer(spool_app, name="spool")
app.add_typer(profile_app, name="profile")
app.add_typer(setup_app, name="setup")

register_notify_cli_bindings(
    app=app,
    usr_events_app=usr_events_app,
    spool_app=spool_app,
    profile_app=profile_app,
    setup_app=setup_app,
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
