"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/handlers/profile/doctor_cmd.py

Execution logic for notify profile doctor command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer

from ....errors import NotifyError


def run_profile_doctor_command(
    *,
    profile: Path,
    json_output: bool,
    read_profile_fn: Callable[[Path], dict[str, Any]],
    resolve_profile_webhook_source_fn: Callable[[dict[str, Any]], tuple[str | None, str | None]],
    resolve_path_value_fn: Callable[..., Path],
    resolve_profile_events_source_fn: Callable[..., tuple[str, Path] | None],
    resolve_usr_events_path_fn: Callable[..., Path],
    resolve_optional_path_value_fn: Callable[..., Path | None],
    probe_path_writable_fn: Callable[[Path], None],
    resolve_webhook_url_fn: Callable[[str | None, str | None, str | None], str],
    validate_provider_webhook_url_fn: Callable[[str, str], None],
    resolve_tls_ca_bundle_fn: Callable[[str, Path | None], Path | None],
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        data = read_profile_fn(profile_path)
        profile_url_env, profile_secret_ref = resolve_profile_webhook_source_fn(data)

        events_path = resolve_path_value_fn(
            field="events",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        profile_events_source = resolve_profile_events_source_fn(profile_data=data, profile_path=profile_path)
        events_exists = events_path.exists()
        if events_exists:
            resolve_usr_events_path_fn(events_path)
        elif profile_events_source is not None:
            resolve_usr_events_path_fn(events_path, require_exists=False)
        else:
            resolve_usr_events_path_fn(events_path)

        cursor_path = resolve_optional_path_value_fn(
            field="cursor",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        if cursor_path is not None:
            probe_path_writable_fn(cursor_path.parent)

        spool_path = resolve_optional_path_value_fn(
            field="spool_dir",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        if spool_path is not None:
            probe_path_writable_fn(spool_path)

        webhook_url = resolve_webhook_url_fn(url=None, url_env=profile_url_env, secret_ref=profile_secret_ref)
        validate_provider_webhook_url_fn(provider=str(data.get("provider") or ""), webhook_url=webhook_url)
        profile_tls_ca_bundle = resolve_optional_path_value_fn(
            field="tls_ca_bundle",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        resolved_tls_ca_bundle = resolve_tls_ca_bundle_fn(
            webhook_url=webhook_url,
            tls_ca_bundle=profile_tls_ca_bundle,
        )

        if json_output:
            payload: dict[str, Any] = {
                "ok": True,
                "profile": str(profile_path),
                "provider": str(data.get("provider")),
                "events": str(events_path),
                "events_exists": bool(events_exists),
            }
            if cursor_path is not None:
                payload["cursor"] = str(cursor_path)
            if spool_path is not None:
                payload["spool_dir"] = str(spool_path)
            if resolved_tls_ca_bundle is not None:
                payload["tls_ca_bundle"] = str(resolved_tls_ca_bundle)
            typer.echo(json.dumps(payload, sort_keys=True))
            return
        if events_exists:
            typer.echo("Profile wiring OK.")
        else:
            typer.echo("Profile wiring OK. events file not created yet; start watcher with --wait-for-events.")
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)
