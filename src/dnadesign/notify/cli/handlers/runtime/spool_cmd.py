"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/runtime/spool_cmd.py

Execution logic for notify spool drain runtime command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError


def run_spool_drain_command(
    *,
    spool_dir: Path | None,
    provider: str | None,
    url: str | None,
    url_env: str | None,
    secret_ref: str | None,
    tls_ca_bundle: Path | None,
    profile: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    fail_fast: bool,
    run_spool_drain_fn: Callable[..., None],
    read_profile_fn: Callable[[Path], dict[str, Any]],
    resolve_cli_optional_string_fn: Callable[..., str | None],
    resolve_path_value_fn: Callable[..., Path],
    resolve_profile_webhook_source_fn: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_optional_path_value_fn: Callable[..., Path | None],
    resolve_webhook_url_fn: Callable[..., str],
    resolve_tls_ca_bundle_fn: Callable[..., Path | None],
    validate_provider_webhook_url_fn: Callable[..., None],
    format_for_provider_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    post_with_backoff_fn: Callable[..., None],
) -> None:
    try:
        run_spool_drain_fn(
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
            read_profile=read_profile_fn,
            resolve_cli_optional_string=resolve_cli_optional_string_fn,
            resolve_path_value=resolve_path_value_fn,
            resolve_profile_webhook_source=resolve_profile_webhook_source_fn,
            resolve_optional_path_value=resolve_optional_path_value_fn,
            resolve_webhook_url=resolve_webhook_url_fn,
            resolve_tls_ca_bundle=resolve_tls_ca_bundle_fn,
            validate_provider_webhook_url=validate_provider_webhook_url_fn,
            format_for_provider=format_for_provider_fn,
            post_with_backoff=post_with_backoff_fn,
        )
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
