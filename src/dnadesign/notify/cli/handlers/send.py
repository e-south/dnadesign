"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/send.py

Execution logic for the notify send command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer

from ...errors import NotifyError


def run_send_command(
    *,
    status: str,
    tool: str,
    run_id: str,
    provider: str,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    message: str | None,
    meta: Path | None,
    timeout: float,
    retries: int,
    dry_run: bool,
    load_meta_fn: Callable[[Path | None], dict[str, Any]],
    resolve_webhook_url_fn: Callable[[str | None, str | None, str | None], str],
    validate_provider_webhook_url_fn: Callable[[str, str], None],
    build_payload_fn: Callable[[str, str, str, str | None, dict[str, Any] | None], dict[str, Any]],
    format_for_provider_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    resolve_tls_ca_bundle_fn: Callable[[str, Path | None], Path | None],
    post_json_fn: Callable[[str, dict[str, Any], float, int, Path | None], None],
) -> None:
    try:
        webhook_url = resolve_webhook_url_fn(url=url, url_env=url_env, secret_ref=secret_ref)
        validate_provider_webhook_url_fn(provider=provider, webhook_url=webhook_url)
        meta_data = load_meta_fn(meta)
        payload = build_payload_fn(status=status, tool=tool, run_id=run_id, message=message, meta=meta_data)
        formatted = format_for_provider_fn(provider, payload)
        if dry_run:
            typer.echo(json.dumps(formatted, indent=2, sort_keys=True))
            return
        tls_ca_bundle_value = resolve_tls_ca_bundle_fn(webhook_url=webhook_url, tls_ca_bundle=tls_ca_bundle)
        post_json_fn(webhook_url, formatted, timeout, retries, tls_ca_bundle_value)
        typer.echo("Notification sent.")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
