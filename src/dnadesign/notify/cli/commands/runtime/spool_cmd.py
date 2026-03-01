"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/commands/runtime/spool_cmd.py

Spool drain command registration for Notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer


def register_spool_drain_command(
    spool_app: typer.Typer,
    *,
    drain_handler: Callable[..., None],
) -> None:
    @spool_app.command("drain")
    def spool_drain(
        spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory containing spooled payload files."),
        provider: str | None = typer.Option(None, help="Override provider: generic|slack|discord."),
        url: str | None = typer.Option(None, help="Webhook URL."),
        url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
        secret_ref: str | None = typer.Option(
            None,
            "--secret-ref",
            help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
        ),
        tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
        profile: Path | None = typer.Option(None, "--profile", help="Path to profile JSON file."),
        connect_timeout: float = typer.Option(5.0, help="HTTP connect timeout seconds."),
        read_timeout: float = typer.Option(10.0, help="HTTP read timeout seconds."),
        retry_max: int = typer.Option(3, "--retry-max", help="Max retries for each delivery."),
        retry_base_seconds: float = typer.Option(0.5, "--retry-base-seconds", help="Base retry delay in seconds."),
        fail_fast: bool = typer.Option(False, "--fail-fast", help="Abort on first failed spool item."),
    ) -> None:
        drain_handler(
            spool_dir=spool_dir,
            provider=provider,
            url=url,
            url_env=url_env,
            secret_ref=secret_ref,
            tls_ca_bundle=tls_ca_bundle,
            profile=profile,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_max=retry_max,
            retry_base_seconds=retry_base_seconds,
            fail_fast=fail_fast,
        )
