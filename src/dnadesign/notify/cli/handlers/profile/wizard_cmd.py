"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/profile/wizard_cmd.py

Execution logic for notify profile wizard command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import click
import typer

from ....errors import NotifyError


def run_profile_wizard_command(
    *,
    ctx: typer.Context,
    profile: Path,
    provider: str,
    events: Path,
    cursor: Path | None,
    only_actions: str | None,
    only_tools: str | None,
    spool_dir: Path | None,
    include_args: bool,
    include_context: bool,
    include_raw_event: bool,
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tls_ca_bundle: Path | None,
    policy: str | None,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    json_output: bool,
    force: bool,
    default_profile_path: Path,
    resolve_profile_path_for_wizard_fn: Callable[..., Path],
    create_wizard_profile_flow_fn: Callable[..., dict[str, Any]],
    resolve_usr_events_path_fn: Callable[..., Path],
    ensure_private_directory_fn: Callable[..., None],
    secret_backend_available_fn: Callable[[str], bool],
    resolve_secret_ref_fn: Callable[[str], str],
    store_secret_ref_fn: Callable[[str, str], str],
    write_profile_file_fn: Callable[[Path, dict[str, Any], bool], None],
) -> None:
    try:
        profile_value = profile
        profile_source = ctx.get_parameter_source("profile")
        default_profile_selected = profile_source == click.core.ParameterSource.DEFAULT
        if default_profile_selected and profile_value == default_profile_path:
            profile_value = resolve_profile_path_for_wizard_fn(profile=profile_value, policy=policy)
        result = create_wizard_profile_flow_fn(
            profile=profile_value,
            provider=provider,
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
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            force=force,
            events_require_exists=True,
            events_source=None,
            resolve_events_path=lambda path, required: resolve_usr_events_path_fn(path, require_exists=required),
            ensure_private_directory_fn=ensure_private_directory_fn,
            secret_backend_available_fn=secret_backend_available_fn,
            resolve_secret_ref_fn=resolve_secret_ref_fn,
            store_secret_ref_fn=store_secret_ref_fn,
            write_profile_file_fn=lambda path, payload, overwrite: write_profile_file_fn(path, payload, overwrite),
        )
        if json_output:
            typer.echo(json.dumps({"ok": True, **result}, sort_keys=True))
            return
        typer.echo(f"Profile written: {result['profile']}")
        for line in result["next_steps"]:
            typer.echo(line)
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
