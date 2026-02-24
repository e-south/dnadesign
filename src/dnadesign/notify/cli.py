"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli.py

Command-line notifier for dnadesign batch runs and USR event streams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import typer

from .cli_commands import (
    register_profile_commands,
    register_send_command,
    register_setup_commands,
    register_spool_drain_command,
    register_usr_events_watch_command,
)
from .cli_commands.providers import format_for_provider
from .cli_resolve import resolve_cli_optional_string as _resolve_cli_optional_string
from .cli_resolve import resolve_existing_file_path as _resolve_existing_file_path
from .cli_resolve import resolve_optional_path_value as _resolve_optional_path_value
from .cli_resolve import resolve_optional_string_value as _resolve_optional_string_value
from .cli_resolve import resolve_path_value as _resolve_path_value
from .cli_resolve import resolve_string_value as _resolve_string_value
from .cli_resolve import resolve_usr_events_path as _resolve_usr_events_path_raw
from .cli_runtime import run_spool_drain, run_usr_events_watch
from .errors import NotifyConfigError, NotifyDeliveryError, NotifyError
from .event_transforms import event_message as _event_message
from .event_transforms import event_meta as _event_meta
from .event_transforms import status_for_action as _status_for_action
from .event_transforms import validate_usr_event as _validate_usr_event_data
from .events_source import normalize_tool_name as _normalize_setup_tool_name
from .events_source import resolve_tool_events_path as _resolve_tool_events_path
from .http import post_json
from .payload import build_payload
from .profile_flows import create_wizard_profile as _create_wizard_profile_flow
from .profile_flows import resolve_profile_path_for_setup as _resolve_profile_path_for_setup
from .profile_flows import resolve_profile_path_for_wizard as _resolve_profile_path_for_wizard
from .profile_flows import resolve_setup_events as _resolve_setup_events
from .profile_flows import resolve_webhook_config as _resolve_webhook_config
from .profile_schema import PROFILE_VERSION
from .profile_schema import read_profile as _read_profile
from .profile_schema import resolve_profile_events_source as _resolve_profile_events_source
from .profile_schema import resolve_profile_webhook_source as _resolve_profile_webhook_source
from .secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from .spool_ops import ensure_private_directory as _ensure_private_directory
from .usr_events_watch import watch_usr_events_loop
from .validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url
from .workflow_policy import DEFAULT_PROFILE_PATH as _DEFAULT_PROFILE_PATH
from .workflow_policy import default_profile_path_for_tool as _default_profile_path_for_tool
from .workflow_policy import policy_defaults as _policy_defaults_for
from .workflow_policy import resolve_workflow_policy as _resolve_workflow_policy
from .workspace_source import list_tool_workspaces as _list_tool_workspaces
from .workspace_source import resolve_tool_workspace_config_path as _resolve_tool_workspace_config_path

app = typer.Typer(help="Send notifications for dnadesign runs via webhooks.")
usr_events_app = typer.Typer(help="Consume USR JSONL events and emit notifications.")
spool_app = typer.Typer(help="Drain spooled notifications.")
profile_app = typer.Typer(help="Manage reusable notify profiles.")
setup_app = typer.Typer(help="Observer-only setup helpers for notify profiles.")
app.add_typer(usr_events_app, name="usr-events")
app.add_typer(spool_app, name="spool")
app.add_typer(profile_app, name="profile")
app.add_typer(setup_app, name="setup")


@lru_cache(maxsize=1)
def _usr_event_version() -> int:
    module = importlib.import_module("dnadesign.usr.event_schema")
    version = getattr(module, "USR_EVENT_VERSION", None)
    if not isinstance(version, int):
        raise NotifyConfigError("USR event schema is invalid: USR_EVENT_VERSION must be an integer")
    return int(version)


def _load_meta(meta_path: Path | None) -> dict[str, Any]:
    if meta_path is None:
        return {}
    if not meta_path.exists():
        raise NotifyError(f"meta file not found: {meta_path}")
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise NotifyError(f"meta file is not valid JSON: {meta_path}") from exc
    if not isinstance(data, dict):
        raise NotifyError("meta file must contain a JSON object")
    return data


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_usr_events_path(events_path: Path, *, require_exists: bool = True) -> Path:
    return _resolve_usr_events_path_raw(
        events_path,
        require_exists=require_exists,
        validate_usr_event=_validate_usr_event,
    )


def _probe_path_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".notify-write-probe.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()


def _write_profile_file(profile_path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if profile_path.exists() and not force:
        raise NotifyConfigError(f"profile already exists: {profile_path}. Pass --force to overwrite.")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = profile_path.with_suffix(profile_path.suffix + ".tmp")
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(profile_path)
    try:
        profile_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on profile: {profile_path}") from exc


def _validate_usr_event(event: dict[str, Any], *, allow_unknown_version: bool) -> None:
    _validate_usr_event_data(
        event,
        expected_version=_usr_event_version(),
        allow_unknown_version=allow_unknown_version,
    )


def _post_with_backoff(
    webhook_url: str,
    formatted_payload: dict[str, Any],
    *,
    tls_ca_bundle: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
) -> None:
    total_timeout = float(connect_timeout) + float(read_timeout)
    retries = max(int(retry_max), 0)
    base_delay = max(float(retry_base_seconds), 0.0)
    attempt = 0
    while True:
        try:
            post_json(
                webhook_url,
                formatted_payload,
                timeout=total_timeout,
                retries=0,
                tls_ca_bundle=tls_ca_bundle,
            )
            return
        except NotifyDeliveryError:
            if attempt >= retries:
                raise
            delay = base_delay * (2**attempt)
            jitter = random.uniform(0.0, base_delay) if base_delay > 0 else 0.0
            time.sleep(delay + jitter)
            attempt += 1


def _send_impl(
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
) -> None:
    try:
        webhook_url = resolve_webhook_url(url=url, url_env=url_env, secret_ref=secret_ref)
        validate_provider_webhook_url(provider=provider, webhook_url=webhook_url)
        meta_data = _load_meta(meta)
        payload = build_payload(
            status=status,
            tool=tool,
            run_id=run_id,
            message=message,
            meta=meta_data,
        )
        formatted = format_for_provider(provider, payload)
        if dry_run:
            typer.echo(json.dumps(formatted, indent=2, sort_keys=True))
            return
        tls_ca_bundle_value = resolve_tls_ca_bundle(webhook_url=webhook_url, tls_ca_bundle=tls_ca_bundle)
        post_json(
            webhook_url,
            formatted,
            timeout=timeout,
            retries=retries,
            tls_ca_bundle=tls_ca_bundle_value,
        )
        typer.echo("Notification sent.")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _profile_init_impl(
    profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
    provider: str = typer.Option(..., help="Provider: generic|slack|discord."),
    url_env: str = typer.Option(..., "--url-env", help="Environment variable holding webhook URL."),
    events: Path = typer.Option(..., "--events", help="USR .events.log JSONL path."),
    cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
    only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
    only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
    spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory for failed payload spool files."),
    include_args: bool = typer.Option(False, "--include-args/--no-include-args"),
    include_context: bool = typer.Option(False, "--include-context/--no-include-context"),
    include_raw_event: bool = typer.Option(False, "--include-raw-event/--no-include-raw-event"),
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
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Workflow policy defaults: densegen|infer_evo2|generic.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        events_path = _resolve_usr_events_path(events)
        policy_name = _resolve_workflow_policy(policy=policy)

        cursor_path = cursor.expanduser().resolve() if cursor is not None else (events_path.parent / "notify.cursor")
        payload: dict[str, Any] = {
            "profile_version": PROFILE_VERSION,
            "provider": str(provider).strip(),
            "events": str(events_path),
            "cursor": str(cursor_path),
            "include_args": bool(include_args),
            "include_context": bool(include_context),
            "include_raw_event": bool(include_raw_event),
            "webhook": {"source": "env", "ref": str(url_env).strip()},
        }
        if progress_step_pct is not None:
            progress_step_pct_value = int(progress_step_pct)
            if progress_step_pct_value < 1 or progress_step_pct_value > 100:
                raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
            payload["progress_step_pct"] = progress_step_pct_value
        if progress_min_seconds is not None:
            progress_min_seconds_value = float(progress_min_seconds)
            if progress_min_seconds_value <= 0.0:
                raise NotifyConfigError("progress_min_seconds must be a positive number")
            payload["progress_min_seconds"] = progress_min_seconds_value
        if only_actions is not None:
            payload["only_actions"] = str(only_actions).strip()
        if only_tools is not None:
            payload["only_tools"] = str(only_tools).strip()
        if spool_dir is not None:
            payload["spool_dir"] = str(spool_dir.expanduser().resolve())
        if tls_ca_bundle is not None:
            payload["tls_ca_bundle"] = str(_resolve_existing_file_path(field="tls_ca_bundle", path_value=tls_ca_bundle))
        if policy_name is not None:
            payload["policy"] = policy_name
            for key, value in _policy_defaults_for(policy_name).items():
                payload.setdefault(key, value)

        if not payload["provider"]:
            raise NotifyConfigError("provider must be a non-empty string")
        if not payload["webhook"]["ref"]:
            raise NotifyConfigError("url_env must be a non-empty string")

        _write_profile_file(profile_path, payload, force=force)
        typer.echo(f"Profile written: {profile_path}")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _profile_wizard_impl(
    ctx: typer.Context,
    profile: Path = typer.Option(_DEFAULT_PROFILE_PATH, "--profile", help="Path to profile JSON file."),
    provider: str = typer.Option("slack", help="Provider: generic|slack|discord."),
    events: Path = typer.Option(..., "--events", help="USR .events.log JSONL path."),
    cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
    only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
    only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
    spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory for failed payload spool files."),
    include_args: bool = typer.Option(False, "--include-args/--no-include-args"),
    include_context: bool = typer.Option(False, "--include-context/--no-include-context"),
    include_raw_event: bool = typer.Option(False, "--include-raw-event/--no-include-raw-event"),
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
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Workflow policy defaults: densegen|infer_evo2|generic.",
    ),
    secret_source: str = typer.Option(
        "auto",
        "--secret-source",
        help="Webhook source: auto|env|keychain|secretservice|file.",
    ),
    url_env: str | None = typer.Option(
        None,
        "--url-env",
        help="Environment variable holding webhook URL (default: NOTIFY_WEBHOOK for --secret-source env).",
    ),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
    ),
    webhook_url: str | None = typer.Option(None, "--webhook-url", help="Webhook URL to store in secure backend."),
    store_webhook: bool = typer.Option(
        True,
        "--store-webhook/--no-store-webhook",
        help="Store webhook URL in the selected secure secret backend.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        profile_value = profile
        profile_source = ctx.get_parameter_source("profile")
        default_profile_selected = profile_source == click.core.ParameterSource.DEFAULT
        if default_profile_selected and profile_value == _DEFAULT_PROFILE_PATH:
            profile_value = _resolve_profile_path_for_wizard(profile=profile_value, policy=policy)
        result = _create_wizard_profile_flow(
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
            resolve_events_path=lambda p, required: _resolve_usr_events_path(p, require_exists=required),
            ensure_private_directory_fn=_ensure_private_directory,
            secret_backend_available_fn=is_secret_backend_available,
            resolve_secret_ref_fn=resolve_secret_ref,
            store_secret_ref_fn=store_secret_ref,
            write_profile_file_fn=lambda path, payload, overwrite: _write_profile_file(
                path,
                payload,
                force=overwrite,
            ),
        )
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "ok": True,
                        **result,
                    },
                    sort_keys=True,
                )
            )
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


def _profile_show_impl(
    profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        data = _read_profile(profile_path)
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _profile_doctor_impl(
    profile: Path = typer.Option(..., "--profile", help="Path to profile JSON file."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        data = _read_profile(profile_path)
        profile_url_env, profile_secret_ref = _resolve_profile_webhook_source(data)

        events_path = _resolve_path_value(
            field="events",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        profile_events_source = _resolve_profile_events_source(profile_data=data, profile_path=profile_path)
        events_exists = events_path.exists()
        if events_exists:
            _resolve_usr_events_path(events_path)
        elif profile_events_source is not None:
            _resolve_usr_events_path(events_path, require_exists=False)
        else:
            _resolve_usr_events_path(events_path)

        cursor_path = _resolve_optional_path_value(
            field="cursor",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        if cursor_path is not None:
            _probe_path_writable(cursor_path.parent)

        spool_path = _resolve_optional_path_value(
            field="spool_dir",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        if spool_path is not None:
            _probe_path_writable(spool_path)

        webhook_url = resolve_webhook_url(url=None, url_env=profile_url_env, secret_ref=profile_secret_ref)
        validate_provider_webhook_url(provider=str(data.get("provider") or ""), webhook_url=webhook_url)
        profile_tls_ca_bundle = _resolve_optional_path_value(
            field="tls_ca_bundle",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
        resolved_tls_ca_bundle = resolve_tls_ca_bundle(webhook_url=webhook_url, tls_ca_bundle=profile_tls_ca_bundle)

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


def _setup_slack_impl(
    profile: Path = typer.Option(_DEFAULT_PROFILE_PATH, "--profile", help="Path to profile JSON file."),
    events: Path | None = typer.Option(None, "--events", help="USR .events.log path for existing runs."),
    tool: str | None = typer.Option(None, "--tool", help="Tool name for resolver mode."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Tool config path for resolver mode."),
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
    include_args: bool = typer.Option(False, "--include-args/--no-include-args"),
    include_context: bool = typer.Option(False, "--include-context/--no-include-context"),
    include_raw_event: bool = typer.Option(False, "--include-raw-event/--no-include-raw-event"),
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
        help="Environment variable holding webhook URL (default: NOTIFY_WEBHOOK for --secret-source env).",
    ),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
    ),
    webhook_url: str | None = typer.Option(None, "--webhook-url", help="Webhook URL to store in secure backend."),
    store_webhook: bool = typer.Option(
        True,
        "--store-webhook/--no-store-webhook",
        help="Store webhook URL in the selected secure secret backend.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        setup_resolution = _resolve_setup_events(
            events=events,
            tool=tool,
            config=config,
            workspace=workspace,
            policy=policy,
            search_start=Path.cwd(),
            resolve_tool_events_path_fn=_resolve_tool_events_path,
            resolve_tool_workspace_config_fn=_resolve_tool_workspace_config_path,
            normalize_tool_name_fn=_normalize_setup_tool_name,
        )
        resolved_config: Path | None = None
        if setup_resolution.events_source is not None:
            config_value = setup_resolution.events_source.get("config")
            if config_value is not None:
                resolved_config = Path(str(config_value))

        profile_value = _resolve_profile_path_for_setup(
            profile=profile,
            tool_name=setup_resolution.tool_name,
            policy=setup_resolution.policy,
            config=resolved_config,
        )

        result = _create_wizard_profile_flow(
            profile=profile_value,
            provider="slack",
            events=setup_resolution.events_path,
            cursor=cursor,
            only_actions=None,
            only_tools=None,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
            tls_ca_bundle=tls_ca_bundle,
            policy=setup_resolution.policy,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            force=force,
            events_require_exists=setup_resolution.events_require_exists,
            events_source=setup_resolution.events_source,
            resolve_events_path=lambda p, required: _resolve_usr_events_path(p, require_exists=required),
            ensure_private_directory_fn=_ensure_private_directory,
            secret_backend_available_fn=is_secret_backend_available,
            resolve_secret_ref_fn=resolve_secret_ref,
            store_secret_ref_fn=store_secret_ref,
            write_profile_file_fn=lambda path, payload, overwrite: _write_profile_file(
                path,
                payload,
                force=overwrite,
            ),
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


def _setup_webhook_impl(
    name: str = typer.Option("default", "--name", help="Logical secret name used for default secure references."),
    secret_source: str = typer.Option(
        "auto",
        "--secret-source",
        help="Webhook source: auto|env|keychain|secretservice|file.",
    ),
    url_env: str | None = typer.Option(
        None,
        "--url-env",
        help="Environment variable holding webhook URL (default: NOTIFY_WEBHOOK for --secret-source env).",
    ),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
    ),
    webhook_url: str | None = typer.Option(None, "--webhook-url", help="Webhook URL to store in secure backend."),
    store_webhook: bool = typer.Option(
        True,
        "--store-webhook/--no-store-webhook",
        help="Store webhook URL in the selected secure secret backend.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
) -> None:
    try:
        name_value = _resolve_cli_optional_string(field="name", cli_value=name)
        if name_value is None:
            name_value = "default"
        secret_name = "".join(char if char.isalnum() else "-" for char in str(name_value)).strip("-")
        if not secret_name:  # pragma: allowlist secret
            secret_name = "default"  # pragma: allowlist secret

        webhook_config = _resolve_webhook_config(
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            secret_name=secret_name,
            secret_backend_available_fn=is_secret_backend_available,
            resolve_secret_ref_fn=resolve_secret_ref,
            store_secret_ref_fn=store_secret_ref,
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


def _setup_resolve_events_impl(
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
    try:
        if print_policy and json_output:
            raise NotifyConfigError("pass either --print-policy or --json, not both")
        if (config is None) == (workspace is None):
            raise NotifyConfigError("pass exactly one of --config or --workspace")
        tool_name = _normalize_setup_tool_name(tool)
        if tool_name is None:
            raise NotifyConfigError("tool must be a non-empty string")
        if config is not None:
            config_path = config.expanduser().resolve()
        else:
            workspace_name = str(workspace or "").strip()
            if not workspace_name:
                raise NotifyConfigError("workspace must be a non-empty string")
            config_path = _resolve_tool_workspace_config_path(
                tool=tool_name,
                workspace=workspace_name,
                search_start=Path.cwd(),
            )
        events_path, default_policy = _resolve_tool_events_path(tool=tool_name, config=config_path)

        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "ok": True,
                        "tool": tool_name,
                        "config": str(config_path),
                        "events": str(events_path),
                        "policy": default_policy,
                    },
                    sort_keys=True,
                )
            )
            return
        if print_policy:
            if default_policy is not None:
                typer.echo(default_policy)
            return
        typer.echo(str(events_path))
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _setup_list_workspaces_impl(
    tool: str = typer.Option(..., "--tool", help="Tool name (for example: densegen)."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON output."),
) -> None:
    try:
        names = _list_tool_workspaces(tool=tool, search_start=Path.cwd())
        if json_output:
            typer.echo(json.dumps({"ok": True, "tool": str(tool).strip(), "workspaces": names}, sort_keys=True))
            return
        if not names:
            typer.echo("No workspaces found.")
            return
        for name in names:
            typer.echo(name)
    except NotifyError as exc:
        if json_output:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _usr_events_watch_impl(
    provider: str | None = typer.Option(None, help="Provider: generic|slack|discord."),
    url: str | None = typer.Option(None, help="Webhook URL."),
    url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account, secretservice://service/account, or file:///path.",
    ),
    tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
    events: Path | None = typer.Option(None, "--events", help="USR .events.log JSONL path."),
    profile: Path | None = typer.Option(None, "--profile", help="Path to profile JSON file."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Tool config path for auto-profile mode. "
            "Use either --config or --workspace. "
            "When set with --tool and without --profile/--events, notify auto-loads "
            "profile from <config-dir>/outputs/notify/<tool>/profile.json."
        ),
    ),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help=(
            "Workspace name for auto-profile mode (shorthand for tool workspace config path). "
            "Use with --tool and without --profile/--events."
        ),
    ),
    cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
    follow: bool = typer.Option(False, "--follow", help="Follow events file for new lines."),
    wait_for_events: bool = typer.Option(
        False,
        "--wait-for-events",
        help="When following, wait for events file creation instead of failing immediately.",
    ),
    idle_timeout: float | None = typer.Option(
        None,
        "--idle-timeout",
        help="Exit after this many seconds without new events while following.",
    ),
    poll_interval_seconds: float = typer.Option(
        0.2,
        "--poll-interval-seconds",
        help="Polling interval for follow/wait loops (seconds).",
    ),
    stop_on_terminal_status: bool = typer.Option(
        False,
        "--stop-on-terminal-status",
        help="Exit after the first event mapped to success or failure status.",
    ),
    on_truncate: str = typer.Option(
        "error",
        "--on-truncate",
        help="Behavior when cursor offset exceeds file size: error|restart.",
    ),
    only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
    only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
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
    on_invalid_event: str = typer.Option(
        "error",
        "--on-invalid-event",
        help="Behavior for malformed event lines: error|skip.",
    ),
    allow_unknown_version: bool = typer.Option(
        False,
        "--allow-unknown-version",
        help="Allow unknown event_version values.",
    ),
    tool: str | None = typer.Option(
        None,
        help=(
            "Override tool name. Also required with --config/--workspace for auto-profile mode "
            "(profile path namespace)."
        ),
    ),
    run_id: str | None = typer.Option(None, help="Override run id."),
    message: str | None = typer.Option(None, help="Override message."),
    include_args: bool | None = typer.Option(None, "--include-args/--no-include-args"),
    include_context: bool | None = typer.Option(None, "--include-context/--no-include-context"),
    include_raw_event: bool | None = typer.Option(None, "--include-raw-event/--no-include-raw-event"),
    connect_timeout: float = typer.Option(5.0, help="HTTP connect timeout seconds."),
    read_timeout: float = typer.Option(10.0, help="HTTP read timeout seconds."),
    retry_max: int = typer.Option(3, "--retry-max", help="Max retries for each delivery."),
    retry_base_seconds: float = typer.Option(0.5, "--retry-base-seconds", help="Base retry delay in seconds."),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Abort on first unsent event."),
    spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Write failed payloads to spool directory."),
    dry_run: bool = typer.Option(False, help="Print formatted payloads instead of posting."),
    advance_cursor_on_dry_run: bool = typer.Option(
        True,
        "--advance-cursor-on-dry-run/--no-advance-cursor-on-dry-run",
        help="When --dry-run is enabled, persist cursor offsets (default: enabled).",
    ),
) -> None:
    try:
        run_usr_events_watch(
            provider=provider,
            url=url,
            url_env=url_env,
            secret_ref=secret_ref,
            tls_ca_bundle=tls_ca_bundle,
            events=events,
            profile=profile,
            config=config,
            workspace=workspace,
            cursor=cursor,
            follow=follow,
            wait_for_events=wait_for_events,
            idle_timeout=idle_timeout,
            poll_interval_seconds=poll_interval_seconds,
            stop_on_terminal_status=stop_on_terminal_status,
            on_truncate=on_truncate,
            only_actions=only_actions,
            only_tools=only_tools,
            progress_step_pct=progress_step_pct,
            progress_min_seconds=progress_min_seconds,
            on_invalid_event=on_invalid_event,
            allow_unknown_version=allow_unknown_version,
            tool=tool,
            run_id=run_id,
            message=message,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_max=retry_max,
            retry_base_seconds=retry_base_seconds,
            fail_fast=fail_fast,
            spool_dir=spool_dir,
            dry_run=dry_run,
            advance_cursor_on_dry_run=advance_cursor_on_dry_run,
            read_profile=_read_profile,
            resolve_string_value=_resolve_string_value,
            resolve_path_value=_resolve_path_value,
            resolve_optional_path_value=_resolve_optional_path_value,
            resolve_optional_string_value=_resolve_optional_string_value,
            resolve_profile_events_source=_resolve_profile_events_source,
            normalize_tool_name=_normalize_setup_tool_name,
            resolve_tool_events_path=_resolve_tool_events_path,
            resolve_tool_workspace_config=_resolve_tool_workspace_config_path,
            resolve_usr_events_path=_resolve_usr_events_path,
            resolve_profile_webhook_source=_resolve_profile_webhook_source,
            default_profile_path_for_tool=_default_profile_path_for_tool,
            resolve_cli_optional_string=_resolve_cli_optional_string,
            resolve_webhook_url=resolve_webhook_url,
            resolve_tls_ca_bundle=resolve_tls_ca_bundle,
            validate_provider_webhook_url=validate_provider_webhook_url,
            split_csv=_split_csv,
            watch_usr_events_loop=watch_usr_events_loop,
            validate_usr_event=_validate_usr_event,
            status_for_action=_status_for_action,
            event_message=_event_message,
            event_meta=_event_meta,
            post_with_backoff=_post_with_backoff,
        )
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _spool_drain_impl(
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
    try:
        run_spool_drain(
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
            read_profile=_read_profile,
            resolve_cli_optional_string=_resolve_cli_optional_string,
            resolve_path_value=_resolve_path_value,
            resolve_profile_webhook_source=_resolve_profile_webhook_source,
            resolve_optional_path_value=_resolve_optional_path_value,
            resolve_webhook_url=resolve_webhook_url,
            resolve_tls_ca_bundle=resolve_tls_ca_bundle,
            validate_provider_webhook_url=validate_provider_webhook_url,
            format_for_provider=format_for_provider,
            post_with_backoff=_post_with_backoff,
        )
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


register_usr_events_watch_command(usr_events_app, watch_handler=_usr_events_watch_impl)
register_spool_drain_command(spool_app, drain_handler=_spool_drain_impl)
register_send_command(app, send_handler=_send_impl)
register_profile_commands(
    profile_app,
    init_handler=_profile_init_impl,
    wizard_handler=_profile_wizard_impl,
    show_handler=_profile_show_impl,
    doctor_handler=_profile_doctor_impl,
)
register_setup_commands(
    setup_app,
    slack_handler=_setup_slack_impl,
    webhook_handler=_setup_webhook_impl,
    resolve_events_handler=_setup_resolve_events_impl,
    list_workspaces_handler=_setup_list_workspaces_impl,
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
