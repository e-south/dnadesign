"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli_commands/send.py

Send command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_send_command(
    app: typer.Typer,
    *,
    send_handler: Callable[..., None],
) -> None:
    @app.command("send")
    def send(
        status: str = typer.Option(..., help="Status: success|failure|started|running."),
        tool: str = typer.Option(..., help="Tool name (densegen, infer, opal, etc.)."),
        run_id: str = typer.Option(..., help="Run identifier."),
        provider: str = typer.Option(..., help="Provider: generic|slack|discord."),
        url: str | None = typer.Option(None, help="Webhook URL."),
        url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
        ),
        tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
        message: str | None = typer.Option(None, help="Optional message."),
        meta: Path | None = typer.Option(None, help="Path to JSON metadata file."),
        timeout: float = typer.Option(10.0, help="HTTP timeout (seconds)."),
        retries: int = typer.Option(0, help="Number of retries on failure."),
        dry_run: bool = typer.Option(False, help="Print payload and exit without sending."),
    ) -> None:
        send_handler(
            status=status,
            tool=tool,
            run_id=run_id,
            provider=provider,
            url=url,
            url_env=url_env,
            secret_ref=secret_ref,
            tls_ca_bundle=tls_ca_bundle,
            message=message,
            meta=meta,
            timeout=timeout,
            retries=retries,
            dry_run=dry_run,
        )
