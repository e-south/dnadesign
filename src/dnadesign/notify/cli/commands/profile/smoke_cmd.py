"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/profile/smoke_cmd.py

Profile smoke command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_profile_smoke_command(
    profile_app: typer.Typer,
    *,
    smoke_handler: Callable[..., None],
) -> None:
    @profile_app.command("smoke")
    def profile_smoke(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
        tool: str = typer.Option(..., "--tool", help="Source tool name (for example densegen or infer_evo2)."),
        config: Path = typer.Option(..., "--config", "-c", help="Tool config path used to resolve events path."),
        cursor: Path = typer.Option(..., "--cursor", help="Cursor file path."),
        spool_dir: Path = typer.Option(..., "--spool-dir", help="Spool directory path."),
        policy: str = typer.Option(..., "--policy", help="Workflow policy name."),
        secret_source: str = typer.Option("file", "--secret-source", help="Webhook source for setup slack."),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference URI used for setup slack (for example file:///...).",
        ),
        store_webhook: bool = typer.Option(
            False,
            "--store-webhook/--no-store-webhook",
            help="Persist literal webhook values when secret source is literal.",
        ),
        tls_ca_bundle: Path | None = typer.Option(
            None,
            "--tls-ca-bundle",
            help="CA bundle file for webhook HTTPS calls.",
        ),
        only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
        dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run the watcher step."),
        advance_cursor_on_dry_run: bool = typer.Option(
            False,
            "--advance-cursor-on-dry-run/--no-advance-cursor-on-dry-run",
            help="Persist cursor during dry-run watcher step.",
        ),
    ) -> None:
        smoke_handler(
            profile=profile,
            tool=tool,
            config=config,
            cursor=cursor,
            spool_dir=spool_dir,
            policy=policy,
            secret_source=secret_source,
            secret_ref=secret_ref,
            store_webhook=store_webhook,
            tls_ca_bundle=tls_ca_bundle,
            only_tools=only_tools,
            dry_run=dry_run,
            advance_cursor_on_dry_run=advance_cursor_on_dry_run,
        )
