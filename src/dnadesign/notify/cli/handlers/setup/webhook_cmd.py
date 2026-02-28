"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/setup/webhook_cmd.py

Execution logic for notify setup webhook command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Callable

import typer

from ....errors import NotifyError


def run_setup_webhook_command(
    *,
    name: str,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    json_output: bool,
    resolve_cli_optional_string_fn: Callable[..., str | None],
    resolve_webhook_config_fn: Callable[..., dict[str, str]],
    secret_backend_available_fn: Callable[[str], bool],
    resolve_secret_ref_fn: Callable[[str], str],
    store_secret_ref_fn: Callable[[str, str], str],
) -> None:
    try:
        name_value = resolve_cli_optional_string_fn(field="name", cli_value=name)
        if name_value is None:
            name_value = "default"
        secret_name = "".join(char if char.isalnum() else "-" for char in str(name_value)).strip("-")
        if not secret_name:  # pragma: allowlist secret
            secret_name = "default"  # pragma: allowlist secret

        webhook_config = resolve_webhook_config_fn(
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            secret_name=secret_name,
            secret_backend_available_fn=secret_backend_available_fn,
            resolve_secret_ref_fn=resolve_secret_ref_fn,
            store_secret_ref_fn=store_secret_ref_fn,
        )

        payload = {
            "ok": True,
            "name": secret_name,
            "webhook": webhook_config,
        }
        if json_output:
            typer.echo(json.dumps(payload, sort_keys=True))
            return
        typer.echo("Webhook reference configured.")
        typer.echo(f"  source: {webhook_config['source']}")
        typer.echo(f"  ref: {webhook_config['ref']}")
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
