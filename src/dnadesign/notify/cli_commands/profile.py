"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_commands/profile.py

Profile command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_profile_commands(
    profile_app: typer.Typer,
    *,
    init_handler: Callable[..., None],
    wizard_handler: Callable[..., None],
    show_handler: Callable[..., None],
    doctor_handler: Callable[..., None],
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
            policy=policy,
            force=force,
        )

    @profile_app.command("wizard")
    def profile_wizard(
        profile: Path = typer.Option(
            Path("outputs/notify/generic/profile.json"),
            "--profile",
            help="Path to profile JSON file.",
        ),
        provider: str = typer.Option("slack", help="Provider: generic|slack|discord."),
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
        policy: str | None = typer.Option(
            None,
            "--policy",
            help="Workflow policy defaults: densegen|infer_evo2|generic.",
        ),
        secret_source: str = typer.Option(
            "auto",
            "--secret-source",
            help=(
                "Webhook source. auto requires keychain/secretservice; "
                "use env with --url-env for environment-based wiring."
            ),
        ),
        url_env: str | None = typer.Option(
            None,
            "--url-env",
            help="Environment variable name for webhook URL (default: NOTIFY_WEBHOOK with --secret-source env).",
        ),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help=(
                "Secret reference for keychain/secretservice mode: "
                "keychain://service/account or secretservice://service/account."
            ),
        ),
        webhook_url: str | None = typer.Option(
            None,
            "--webhook-url",
            help="Webhook URL to store securely (avoid shell history in shared environments).",
        ),
        store_webhook: bool = typer.Option(
            True,
            "--store-webhook/--no-store-webhook",
            help="Store webhook URL in the selected secure secret backend.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
        force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
    ) -> None:
        wizard_handler(
            profile=profile,
            provider=provider,
            events=events,
            cursor=cursor,
            only_actions=only_actions,
            only_tools=only_tools,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            policy=policy,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            json_output=json_output,
            force=force,
        )

    @profile_app.command("show")
    def profile_show(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
    ) -> None:
        show_handler(profile=profile)

    @profile_app.command("doctor")
    def profile_doctor(
        profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        doctor_handler(profile=profile, json_output=json_output)
