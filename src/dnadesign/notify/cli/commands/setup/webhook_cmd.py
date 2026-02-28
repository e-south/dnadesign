"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/setup/webhook_cmd.py

Setup webhook command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

import typer


def register_setup_webhook_command(
    setup_app: typer.Typer,
    *,
    webhook_handler: Callable[..., None],
) -> None:
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
