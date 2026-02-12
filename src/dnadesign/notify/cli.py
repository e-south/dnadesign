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
from .errors import NotifyConfigError, NotifyDeliveryError, NotifyError
from .events_source import normalize_tool_name as _normalize_setup_tool_name
from .events_source import resolve_tool_events_path as _resolve_tool_events_path
from .http import post_json
from .payload import build_payload
from .profile_ops import sanitize_profile_name as _sanitize_profile_name
from .profile_ops import wizard_next_steps as _wizard_next_steps
from .profile_schema import PROFILE_VERSION
from .profile_schema import read_profile as _read_profile
from .profile_schema import resolve_profile_events_source as _resolve_profile_events_source
from .profile_schema import resolve_profile_webhook_source as _resolve_profile_webhook_source
from .secrets import is_secret_backend_available, store_secret_ref
from .spool_ops import ensure_private_directory as _ensure_private_directory
from .usr_events_watch import watch_usr_events_loop
from .validation import resolve_tls_ca_bundle, resolve_webhook_url
from .workflow_policy import DEFAULT_PROFILE_PATH as _DEFAULT_PROFILE_PATH
from .workflow_policy import DEFAULT_WEBHOOK_ENV as _DEFAULT_WEBHOOK_ENV
from .workflow_policy import default_profile_path_for_tool as _default_profile_path_for_tool
from .workflow_policy import policy_defaults as _policy_defaults_for
from .workflow_policy import resolve_workflow_policy as _resolve_workflow_policy

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
    module = importlib.import_module("dnadesign.usr.src.event_schema")
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


def _event_meta(
    event: dict[str, Any],
    *,
    include_args: bool,
    include_raw_event: bool,
    include_context: bool,
) -> dict[str, Any]:
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    meta = {
        "usr_event_version": event.get("event_version"),
        "usr_action": event.get("action"),
        "usr_dataset_name": dataset.get("name"),
        "usr_fingerprint": event.get("fingerprint"),
        "usr_registry_hash": event.get("registry_hash"),
        "usr_timestamp": event.get("timestamp_utc"),
    }
    if include_context:
        meta["usr_dataset_root"] = dataset.get("root")
    if include_args:
        meta["usr_args"] = event.get("args")
    if include_raw_event:
        meta["usr_event"] = event
    return meta


def _status_for_action(action: str, *, event: dict[str, Any] | None = None) -> str:
    action_norm = str(action or "").strip().lower()
    if not action_norm:
        return "running"
    if "fail" in action_norm or "error" in action_norm:
        return "failure"
    if action_norm == "init":
        return "started"
    if action_norm in {"materialize", "compact_overlay", "overlay_compact", "registry_freeze"}:
        return "running"
    return "running"


def _event_message(
    event: dict[str, Any],
    *,
    run_id: str,
    duration_seconds: float | None,
) -> str:
    action = str(event.get("action") or "event")
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = dataset.get("name") or "unknown-dataset"
    metrics_raw = event.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}

    rows_written = metrics.get("rows_written")
    if rows_written is not None:
        return f"{action} on {dataset_name} (rows_written={rows_written})"
    return f"{action} on {dataset_name}"


def _validate_usr_event(event: dict[str, Any], *, allow_unknown_version: bool) -> None:
    if not isinstance(event, dict):
        raise NotifyConfigError("event line must decode to a JSON object")
    if "event_version" not in event:
        raise NotifyConfigError("event missing required 'event_version'")
    version = event.get("event_version")
    if not isinstance(version, int):
        raise NotifyConfigError("event_version must be an integer")
    expected_version = _usr_event_version()
    if version != expected_version and not allow_unknown_version:
        raise NotifyConfigError(f"unknown event_version={version}; expected {expected_version}")
    action = event.get("action")
    if not isinstance(action, str) or not action.strip():
        raise NotifyConfigError("event missing required 'action'")
    dataset = event.get("dataset")
    if dataset is not None and not isinstance(dataset, dict):
        raise NotifyConfigError("event field 'dataset' must be an object when provided")
    actor = event.get("actor")
    if actor is not None and not isinstance(actor, dict):
        raise NotifyConfigError("event field 'actor' must be an object when provided")


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
        tls_ca_bundle_value = resolve_tls_ca_bundle(webhook_url=webhook_url, tls_ca_bundle=tls_ca_bundle)
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
    tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Workflow policy defaults: densegen|infer_evo2|generic.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        events_path = _resolve_usr_events_path(events)
        policy_name = _resolve_workflow_policy(policy=policy)

        cursor_path = cursor or (events_path.parent / "notify.cursor")
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
        if only_actions is not None:
            payload["only_actions"] = str(only_actions).strip()
        if only_tools is not None:
            payload["only_tools"] = str(only_tools).strip()
        if spool_dir is not None:
            payload["spool_dir"] = str(spool_dir)
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

        _write_profile_file(profile, payload, force=force)
        typer.echo(f"Profile written: {profile}")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _create_wizard_profile(
    *,
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
    tls_ca_bundle: Path | None,
    policy: str | None,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    force: bool,
    events_require_exists: bool = True,
    events_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    profile_path = profile.expanduser().resolve()
    events_path = _resolve_usr_events_path(events, require_exists=events_require_exists)
    events_exists = events_path.exists()
    provider_value = str(provider).strip()
    if not provider_value:
        raise NotifyConfigError("provider must be a non-empty string")
    policy_name = _resolve_workflow_policy(policy=policy)

    mode = str(secret_source or "").strip().lower()
    if not mode:
        raise NotifyConfigError("secret_source must be a non-empty string")
    if mode == "auto":
        if is_secret_backend_available("keychain"):
            mode = "keychain"
        elif is_secret_backend_available("secretservice"):
            mode = "secretservice"
        else:
            raise NotifyConfigError(
                "secret_source=auto requires keychain or secretservice on this system. "
                "Pass --secret-source env to opt into environment-variable webhook storage."
            )
    if mode not in {"env", "keychain", "secretservice"}:
        raise NotifyConfigError("secret_source must be one of: auto, env, keychain, secretservice")

    webhook_config: dict[str, str]
    if mode == "env":
        env_name = _resolve_cli_optional_string(field="url_env", cli_value=url_env)
        if env_name is None:
            env_name = _DEFAULT_WEBHOOK_ENV
        webhook_config = {"source": "env", "ref": env_name}
    else:
        if not is_secret_backend_available(mode):
            raise NotifyConfigError(f"secret backend '{mode}' is not available on this system")
        secret_value = _resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
        if secret_value is None:
            secret_value = f"{mode}://dnadesign.notify/{_sanitize_profile_name(profile_path)}"
        webhook_config = {"source": "secret_ref", "ref": secret_value}
        if store_webhook:
            webhook_value = _resolve_cli_optional_string(field="webhook_url", cli_value=webhook_url)
            if webhook_value is None:
                webhook_value = str(typer.prompt("Webhook URL", hide_input=True)).strip()
            if not webhook_value:
                raise NotifyConfigError("webhook_url is required when --store-webhook is enabled")
            store_secret_ref(secret_value, webhook_value)

    default_cursor = profile_path.parent / "cursor"
    default_spool = profile_path.parent / "spool"
    cursor_value = cursor or default_cursor
    spool_value = spool_dir or default_spool
    try:
        _ensure_private_directory(cursor_value.parent, label="cursor directory")
        _ensure_private_directory(spool_value, label="spool_dir")
    except NotifyConfigError as exc:
        raise NotifyConfigError(
            f"{exc}. Pass --cursor and --spool-dir to writable paths "
            "if the default profile-scoped paths are restricted."
        ) from exc

    payload: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "provider": provider_value,
        "events": str(events_path),
        "cursor": str(cursor_value),
        "spool_dir": str(spool_value),
        "include_args": bool(include_args),
        "include_context": bool(include_context),
        "include_raw_event": bool(include_raw_event),
        "webhook": webhook_config,
    }
    if tls_ca_bundle is not None:
        payload["tls_ca_bundle"] = str(_resolve_existing_file_path(field="tls_ca_bundle", path_value=tls_ca_bundle))
    if only_actions is not None:
        payload["only_actions"] = str(only_actions).strip()
    if only_tools is not None:
        payload["only_tools"] = str(only_tools).strip()
    if policy_name is not None:
        payload["policy"] = policy_name
        for key, value in _policy_defaults_for(policy_name).items():
            payload.setdefault(key, value)
    if events_source is not None:
        payload["events_source"] = dict(events_source)

    _write_profile_file(profile_path, payload, force=force)
    next_steps = _wizard_next_steps(
        profile_path=profile_path,
        webhook_config=webhook_config,
        events_exists=events_exists,
    )
    return {
        "profile": str(profile_path),
        "provider": provider_value,
        "events": str(events_path),
        "cursor": str(cursor_value),
        "spool_dir": str(spool_value),
        "policy": policy_name,
        "webhook": webhook_config,
        "next_steps": next_steps,
        "events_exists": events_exists,
    }


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
    tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Workflow policy defaults: densegen|infer_evo2|generic.",
    ),
    secret_source: str = typer.Option(
        "auto",
        "--secret-source",
        help="Webhook source: auto|env|keychain|secretservice.",
    ),
    url_env: str | None = typer.Option(
        None,
        "--url-env",
        help="Environment variable holding webhook URL (default: NOTIFY_WEBHOOK for --secret-source env).",
    ),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account or secretservice://service/account.",
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
            policy_namespace = _resolve_workflow_policy(policy=policy)
            if policy_namespace is None:
                raise NotifyConfigError(
                    "default profile path is ambiguous in wizard mode; "
                    "pass --policy or --profile to select a profile namespace"
                )
            profile_value = _default_profile_path_for_tool(policy_namespace)
        result = _create_wizard_profile(
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
            tls_ca_bundle=tls_ca_bundle,
            policy=policy,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            force=force,
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
    tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
    secret_source: str = typer.Option(
        "auto",
        "--secret-source",
        help="Webhook source: auto|env|keychain|secretservice.",
    ),
    url_env: str | None = typer.Option(
        None,
        "--url-env",
        help="Environment variable holding webhook URL (default: NOTIFY_WEBHOOK for --secret-source env).",
    ),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account or secretservice://service/account.",
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
        tool_name: str | None = None
        has_events = events is not None
        has_tool = tool is not None or config is not None
        if has_events and has_tool:
            raise NotifyConfigError("pass either --events or --tool with --config, not both")
        if not has_events and not has_tool:
            raise NotifyConfigError("pass either --events or --tool with --config")

        events_path: Path
        events_source: dict[str, str] | None = None
        policy_value = policy
        events_require_exists = True

        if has_events:
            events_path = events if events is not None else Path("")
        else:
            if tool is None or config is None:
                raise NotifyConfigError("resolver mode requires both --tool and --config")
            config_path = config.expanduser().resolve()
            events_path, default_policy = _resolve_tool_events_path(tool=tool, config=config_path)
            tool_name = _normalize_setup_tool_name(tool)
            events_source = {
                "tool": str(tool_name),
                "config": str(config_path),
            }
            if policy_value is None:
                policy_value = default_policy
            events_require_exists = False

        profile_value = profile
        if profile_value == _DEFAULT_PROFILE_PATH:
            namespace = tool_name
            if namespace is None:
                policy_namespace = _resolve_workflow_policy(policy=policy_value)
                if policy_namespace is None:
                    raise NotifyConfigError(
                        "default profile path is ambiguous in --events mode; "
                        "pass --policy or --profile to select a profile namespace"
                    )
                namespace = policy_namespace
            profile_value = _default_profile_path_for_tool(namespace)

        result = _create_wizard_profile(
            profile=profile_value,
            provider="slack",
            events=events_path,
            cursor=cursor,
            only_actions=None,
            only_tools=None,
            spool_dir=spool_dir,
            include_args=include_args,
            include_context=include_context,
            include_raw_event=include_raw_event,
            tls_ca_bundle=tls_ca_bundle,
            policy=policy_value,
            secret_source=secret_source,
            url_env=url_env,
            secret_ref=secret_ref,
            webhook_url=webhook_url,
            store_webhook=store_webhook,
            force=force,
            events_require_exists=events_require_exists,
            events_source=events_source,
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


def _setup_resolve_events_impl(
    tool: str = typer.Option(..., "--tool", help="Tool name for resolver mode."),
    config: Path = typer.Option(..., "--config", "-c", help="Tool config path for resolver mode."),
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
        config_path = config.expanduser().resolve()
        tool_name = _normalize_setup_tool_name(tool)
        if tool_name is None:
            raise NotifyConfigError("tool must be a non-empty string")
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


def _usr_events_watch_impl(
    provider: str | None = typer.Option(None, help="Provider: generic|slack|discord."),
    url: str | None = typer.Option(None, help="Webhook URL."),
    url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account or secretservice://service/account.",
    ),
    tls_ca_bundle: Path | None = typer.Option(None, "--tls-ca-bundle", help="CA bundle file for HTTPS webhooks."),
    events: Path | None = typer.Option(None, "--events", help="USR .events.log JSONL path."),
    profile: Path | None = typer.Option(None, "--profile", help="Path to profile JSON file."),
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
    tool: str | None = typer.Option(None, help="Override tool name."),
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
        if idle_timeout is not None and float(idle_timeout) <= 0:
            raise NotifyConfigError("idle_timeout must be > 0 when provided")
        if float(poll_interval_seconds) <= 0:
            raise NotifyConfigError("poll_interval_seconds must be > 0")
        profile_path = profile.expanduser().resolve() if profile is not None else None
        profile_data = _read_profile(profile_path) if profile_path is not None else {}
        provider_value = _resolve_string_value(field="provider", cli_value=provider, profile_data=profile_data)
        events_path = _resolve_path_value(
            field="events",
            cli_value=events,
            profile_data=profile_data,
            profile_path=profile_path,
        )
        if events is None:
            profile_events_source = _resolve_profile_events_source(profile_data=profile_data, profile_path=profile_path)
            if profile_events_source is not None:
                source_tool, source_config = profile_events_source
                resolved_events_path, _default_policy = _resolve_tool_events_path(
                    tool=source_tool,
                    config=source_config,
                )
                events_path = _resolve_usr_events_path(resolved_events_path, require_exists=False)
        cursor_path = _resolve_optional_path_value(
            field="cursor",
            cli_value=cursor,
            profile_data=profile_data,
            profile_path=profile_path,
        )
        only_actions_value = _resolve_optional_string_value(
            field="only_actions",
            cli_value=only_actions,
            profile_data=profile_data,
        )
        only_tools_value = _resolve_optional_string_value(
            field="only_tools",
            cli_value=only_tools,
            profile_data=profile_data,
        )
        spool_dir_value = _resolve_optional_path_value(
            field="spool_dir",
            cli_value=spool_dir,
            profile_data=profile_data,
            profile_path=profile_path,
        )
        include_args_value = include_args
        if include_args_value is None:
            include_args_profile = profile_data.get("include_args")
            include_args_value = bool(include_args_profile) if include_args_profile is not None else False
        include_context_value = include_context
        if include_context_value is None:
            include_context_profile = profile_data.get("include_context")
            include_context_value = bool(include_context_profile) if include_context_profile is not None else False
        include_raw_event_value = include_raw_event
        if include_raw_event_value is None:
            include_raw_event_profile = profile_data.get("include_raw_event")
            include_raw_event_value = (
                bool(include_raw_event_profile) if include_raw_event_profile is not None else False
            )
        profile_url_env, profile_secret_ref = _resolve_profile_webhook_source(profile_data)
        url_env_value = _resolve_cli_optional_string(field="url_env", cli_value=url_env)
        if url_env_value is None:
            url_env_value = profile_url_env
        secret_ref_value = _resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
        if secret_ref_value is None:
            secret_ref_value = profile_secret_ref
        profile_tls_ca_bundle = _resolve_optional_path_value(
            field="tls_ca_bundle",
            cli_value=tls_ca_bundle,
            profile_data=profile_data,
            profile_path=profile_path,
        )

        webhook_url: str | None = None
        resolved_tls_ca_bundle: Path | None = None
        if not dry_run:
            webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
            resolved_tls_ca_bundle = resolve_tls_ca_bundle(
                webhook_url=webhook_url,
                tls_ca_bundle=profile_tls_ca_bundle,
            )
        action_filter = set(_split_csv(only_actions_value))
        tool_filter = set(_split_csv(only_tools_value))
        on_invalid_event_mode = str(on_invalid_event or "").strip().lower()
        if on_invalid_event_mode not in {"error", "skip"}:
            raise NotifyConfigError(f"unsupported on-invalid-event mode '{on_invalid_event}'")

        watch_usr_events_loop(
            events_path=events_path,
            cursor_path=cursor_path,
            on_truncate=on_truncate,
            follow=follow,
            wait_for_events=wait_for_events,
            idle_timeout_seconds=idle_timeout,
            poll_interval_seconds=poll_interval_seconds,
            should_advance_cursor=(not dry_run) or bool(advance_cursor_on_dry_run),
            on_invalid_event_mode=on_invalid_event_mode,
            allow_unknown_version=allow_unknown_version,
            action_filter=action_filter,
            tool_filter=tool_filter,
            tool=tool,
            run_id=run_id,
            provider_value=provider_value,
            message=message,
            include_args_value=bool(include_args_value),
            include_context_value=bool(include_context_value),
            include_raw_event_value=bool(include_raw_event_value),
            dry_run=dry_run,
            stop_on_terminal_status=stop_on_terminal_status,
            webhook_url=webhook_url,
            resolved_tls_ca_bundle=resolved_tls_ca_bundle,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_max=retry_max,
            retry_base_seconds=retry_base_seconds,
            fail_fast=fail_fast,
            spool_dir_value=spool_dir_value,
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
        help="Secret reference: keychain://service/account or secretservice://service/account.",
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
        profile_path = profile.expanduser().resolve() if profile is not None else None
        profile_data = _read_profile(profile_path) if profile_path is not None else {}
        provider_override = _resolve_cli_optional_string(field="provider", cli_value=provider)
        profile_provider_value = profile_data.get("provider")
        profile_provider: str | None = None
        if profile_provider_value is not None:
            if not isinstance(profile_provider_value, str) or not profile_provider_value.strip():
                raise NotifyConfigError("profile field 'provider' must be a non-empty string")
            profile_provider = profile_provider_value.strip()
        spool_dir_value = _resolve_path_value(
            field="spool_dir",
            cli_value=spool_dir,
            profile_data=profile_data,
            profile_path=profile_path,
        )
        profile_url_env, profile_secret_ref = _resolve_profile_webhook_source(profile_data)
        url_env_value = _resolve_cli_optional_string(field="url_env", cli_value=url_env)
        if url_env_value is None:
            url_env_value = profile_url_env
        secret_ref_value = _resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
        if secret_ref_value is None:
            secret_ref_value = profile_secret_ref
        profile_tls_ca_bundle = _resolve_optional_path_value(
            field="tls_ca_bundle",
            cli_value=tls_ca_bundle,
            profile_data=profile_data,
            profile_path=profile_path,
        )
        webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
        resolved_tls_ca_bundle = resolve_tls_ca_bundle(
            webhook_url=webhook_url,
            tls_ca_bundle=profile_tls_ca_bundle,
        )
        if not spool_dir_value.exists():
            raise NotifyConfigError(f"spool directory not found: {spool_dir_value}")
        failed = 0
        for path in sorted(spool_dir_value.glob("*.json")):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise NotifyConfigError(f"invalid spool payload file: {path}")
                spool_provider_value = body.get("provider")
                spool_provider: str | None = None
                if spool_provider_value is not None:
                    if not isinstance(spool_provider_value, str) or not spool_provider_value.strip():
                        raise NotifyConfigError(f"invalid spool payload provider value: {path}")
                    spool_provider = spool_provider_value.strip()
                provider_value = provider_override or spool_provider or profile_provider
                if not provider_value:
                    raise NotifyConfigError(
                        f"spool payload missing provider: {path}. Pass --provider or include provider in spool files."
                    )
                formatted = format_for_provider(provider_value, payload)
                _post_with_backoff(
                    webhook_url,
                    formatted,
                    tls_ca_bundle=resolved_tls_ca_bundle,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    retry_max=retry_max,
                    retry_base_seconds=retry_base_seconds,
                )
                path.unlink()
            except NotifyError:
                failed += 1
                if fail_fast:
                    raise
        if failed:
            raise NotifyDeliveryError(f"{failed} spool file(s) failed delivery")
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
    resolve_events_handler=_setup_resolve_events_impl,
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
