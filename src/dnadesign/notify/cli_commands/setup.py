"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_commands/setup.py

Setup command registration for observer-only Notify onboarding flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_setup_commands(
    setup_app: typer.Typer,
    *,
    slack_handler: Callable[..., None],
    webhook_handler: Callable[..., None],
    resolve_events_handler: Callable[..., None],
    list_workspaces_handler: Callable[..., None],
) -> None:
    @setup_app.command("slack")
    def setup_slack(
        profile: Path = typer.Option(
            Path("outputs/notify/generic/profile.json"),
            "--profile",
            help="Path to profile JSON file.",
        ),
        events: Path | None = typer.Option(None, "--events", help="USR .events.log path for existing runs."),
        tool: str | None = typer.Option(
            None,
            "--tool",
            help="Tool name for resolver mode (for example: densegen).",
        ),
        config: Path | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Tool config path for resolver mode.",
        ),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            help="Workspace name for resolver mode (shorthand for tool workspace config path).",
        ),
        policy: str | None = typer.Option(
            None,
            "--policy",
            help="Workflow policy defaults: densegen|infer_evo2|generic.",
        ),
        cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
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
        secret_source: str = typer.Option(
            "auto",
            "--secret-source",
            help="Webhook source: auto|env|keychain|secretservice|file.",
        ),
        url_env: str | None = typer.Option(
            None,
            "--url-env",
            help="Environment variable name for webhook URL (default: NOTIFY_WEBHOOK with --secret-source env).",
        ),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference for keychain/secretservice/file mode.",
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
        slack_handler(
            profile=profile,
            events=events,
            tool=tool,
            config=config,
            workspace=workspace,
            policy=policy,
            cursor=cursor,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
            tls_ca_bundle=tls_ca_bundle,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            json_output=json_output,
            force=force,
        )

    @setup_app.command("webhook")
    def setup_webhook(
        name: str = typer.Option(
            "default",
            "--name",
            help="Logical secret name used for default secure references.",
        ),
        secret_source: str = typer.Option(
            "auto",
            "--secret-source",
            help="Webhook source: auto|env|keychain|secretservice|file.",
        ),
        url_env: str | None = typer.Option(
            None,
            "--url-env",
            help="Environment variable name for webhook URL (default: NOTIFY_WEBHOOK with --secret-source env).",
        ),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference for keychain/secretservice/file mode.",
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
    ) -> None:
        webhook_handler(
            name=name,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            json_output=json_output,
        )

    @setup_app.command("resolve-events")
    def resolve_events(
        tool: str = typer.Option(..., "--tool", help="Tool name for resolver mode."),
        config: Path | None = typer.Option(None, "--config", "-c", help="Tool config path for resolver mode."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            help="Workspace name for resolver mode (shorthand for tool workspace config path).",
        ),
        print_policy: bool = typer.Option(
            False,
            "--print-policy",
            help="Print only the default policy for the resolved tool/config.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        resolve_events_handler(
            tool=tool,
            config=config,
            workspace=workspace,
            print_policy=print_policy,
            json_output=json_output,
        )

    @setup_app.command("list-workspaces")
    def list_workspaces(
        tool: str = typer.Option(..., "--tool", help="Tool name (for example: densegen)."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    ) -> None:
        list_workspaces_handler(
            tool=tool,
            json_output=json_output,
        )
