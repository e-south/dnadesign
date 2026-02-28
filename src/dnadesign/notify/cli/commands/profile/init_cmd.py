"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/profile/init_cmd.py

Profile init command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_profile_init_command(
    profile_app: typer.Typer,
    *,
    init_handler: Callable[..., None],
) -> None:
    @profile_app.command("init")
    def profile_init(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
        provider: str = typer.Option(..., help="Provider: generic|slack|discord."),
        url_env: str = typer.Option(..., "--url-env", help="Environment variable holding webhook URL."),
        events: Path = typer.Option(..., "--events", help="USR events JSONL path."),
        cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
        only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
        only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
        spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory for failed payload spool files."),
        include_args: bool = typer.Option(
            False,
            "--include-args/--no-include-args",
            help="Whether to include event args in payload meta.",
        ),
        include_context: bool = typer.Option(
            False,
            "--include-context/--no-include-context",
            help="Whether to include host/cwd context in payload.",
        ),
        include_raw_event: bool = typer.Option(
            False,
            "--include-raw-event/--no-include-raw-event",
            help="Whether to include the full USR event blob in payload meta.",
        ),
        progress_step_pct: int | None = typer.Option(
            None,
            "--progress-step-pct",
            help="DenseGen progress heartbeat threshold as percentage points (1-100).",
        ),
        progress_min_seconds: float | None = typer.Option(
            None,
            "--progress-min-seconds",
            help="Minimum spacing between DenseGen progress heartbeats in seconds.",
        ),
        tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
        policy: str | None = typer.Option(
            None,
            "--policy",
            help="Workflow policy defaults: densegen|infer_evo2|generic.",
        ),
        force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
    ) -> None:
        init_handler(
            profile=profile,
            provider=provider,
            url_env=url_env,
            events=events,
            cursor=cursor,
            only_actions=only_actions,
            only_tools=only_tools,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
            tls_ca_bundle=tls_ca_bundle,
            policy=policy,
            force=force,
        )
