"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli.py

Command-line notifier for dnadesign batch runs and USR event streams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import time
from pathlib import Path
from typing import Any, Iterable

import typer

from dnadesign.usr import USR_EVENT_VERSION

from .cli_commands import register_profile_commands, register_spool_drain_command, register_usr_events_watch_command
from .cli_commands.providers import format_for_provider
from .errors import NotifyConfigError, NotifyDeliveryError, NotifyError
from .http import post_json
from .payload import build_payload
from .secrets import is_secret_backend_available, store_secret_ref
from .validation import resolve_webhook_url

app = typer.Typer(help="Send notifications for dnadesign runs via webhooks.")
usr_events_app = typer.Typer(help="Consume USR JSONL events and emit notifications.")
spool_app = typer.Typer(help="Drain spooled notifications.")
profile_app = typer.Typer(help="Manage reusable notify profiles.")
app.add_typer(usr_events_app, name="usr-events")
app.add_typer(spool_app, name="spool")
app.add_typer(profile_app, name="profile")

PROFILE_VERSION = 2
_PROFILE_V1_REQUIRED_KEYS = {"provider", "url_env", "events"}
_PROFILE_V1_ALLOWED_KEYS = {
    "profile_version",
    "provider",
    "url_env",
    "events",
    "cursor",
    "only_actions",
    "only_tools",
    "spool_dir",
    "include_args",
    "include_context",
    "include_raw_event",
    "preset",
}
_PROFILE_V2_REQUIRED_KEYS = {"provider", "events", "webhook"}
_PROFILE_V2_ALLOWED_KEYS = {
    "profile_version",
    "provider",
    "events",
    "webhook",
    "cursor",
    "only_actions",
    "only_tools",
    "spool_dir",
    "include_args",
    "include_context",
    "include_raw_event",
    "preset",
}
_DENSEGEN_PROFILE_PRESET = {
    "only_actions": "densegen_health,densegen_flush_failed,materialize",
    "only_tools": "densegen",
    "include_args": False,
    "include_context": False,
    "include_raw_event": False,
}
_WEBHOOK_SOURCES = {"env", "secret_ref"}


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


def _validate_webhook_config(data: dict[str, Any]) -> None:
    webhook = data.get("webhook")
    if not isinstance(webhook, dict):
        raise NotifyConfigError("profile field 'webhook' must be an object")
    unknown = sorted(set(webhook.keys()) - {"source", "ref"})
    if unknown:
        raise NotifyConfigError(f"profile field 'webhook' contains unsupported keys: {', '.join(unknown)}")
    source = webhook.get("source")
    ref = webhook.get("ref")
    if not isinstance(source, str) or not source.strip():
        raise NotifyConfigError("profile field 'webhook.source' must be a non-empty string")
    source_norm = source.strip().lower()
    if source_norm not in _WEBHOOK_SOURCES:
        allowed = ", ".join(sorted(_WEBHOOK_SOURCES))
        raise NotifyConfigError(f"profile field 'webhook.source' must be one of: {allowed}")
    if not isinstance(ref, str) or not ref.strip():
        raise NotifyConfigError("profile field 'webhook.ref' must be a non-empty string")


def _read_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        raise NotifyConfigError(f"profile file not found: {profile_path}")
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotifyConfigError(f"profile file is not valid JSON: {profile_path}") from exc
    if not isinstance(data, dict):
        raise NotifyConfigError("profile file must contain a JSON object")

    if "url" in data:
        raise NotifyConfigError("profile must not store plain webhook URLs; use url_env")

    version = data.get("profile_version")
    if version == 1:
        unknown = sorted(set(data.keys()) - _PROFILE_V1_ALLOWED_KEYS)
        if unknown:
            raise NotifyConfigError(f"profile contains unsupported keys: {', '.join(unknown)}")
        for key in _PROFILE_V1_REQUIRED_KEYS:
            value = data.get(key)
            if not isinstance(value, str) or not value.strip():
                raise NotifyConfigError(f"profile missing required non-empty string field '{key}'")
    elif version == 2:
        unknown = sorted(set(data.keys()) - _PROFILE_V2_ALLOWED_KEYS)
        if unknown:
            raise NotifyConfigError(f"profile contains unsupported keys: {', '.join(unknown)}")
        for key in _PROFILE_V2_REQUIRED_KEYS:
            value = data.get(key)
            if key == "webhook":
                continue
            if not isinstance(value, str) or not value.strip():
                raise NotifyConfigError(f"profile missing required non-empty string field '{key}'")
        _validate_webhook_config(data)
    else:
        raise NotifyConfigError(f"profile_version must be 1 or {PROFILE_VERSION}; found {version!r}")

    for key in ("events", "cursor", "spool_dir"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise NotifyConfigError(f"profile field '{key}' must be a non-empty string path when provided")

    for key in ("only_actions", "only_tools", "preset"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise NotifyConfigError(f"profile field '{key}' must be a non-empty string when provided")

    include_args = data.get("include_args")
    if include_args is not None and not isinstance(include_args, bool):
        raise NotifyConfigError("profile field 'include_args' must be a boolean when provided")
    include_context = data.get("include_context")
    if include_context is not None and not isinstance(include_context, bool):
        raise NotifyConfigError("profile field 'include_context' must be a boolean when provided")
    include_raw_event = data.get("include_raw_event")
    if include_raw_event is not None and not isinstance(include_raw_event, bool):
        raise NotifyConfigError("profile field 'include_raw_event' must be a boolean when provided")
    return data


def _resolve_cli_optional_string(*, field: str, cli_value: str | None) -> str | None:
    if cli_value is None:
        return None
    value = str(cli_value).strip()
    if not value:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return value


def _resolve_profile_webhook_source(profile_data: dict[str, Any]) -> tuple[str | None, str | None]:
    if not profile_data:
        return None, None
    version = profile_data.get("profile_version")
    if version == 1:
        env_value = profile_data.get("url_env")
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip(), None
        raise NotifyConfigError("profile missing required non-empty string field 'url_env'")
    if version == 2:
        webhook = profile_data.get("webhook")
        if not isinstance(webhook, dict):
            raise NotifyConfigError("profile field 'webhook' must be an object")
        source = str(webhook.get("source") or "").strip().lower()
        ref = str(webhook.get("ref") or "").strip()
        if source not in _WEBHOOK_SOURCES:
            allowed = ", ".join(sorted(_WEBHOOK_SOURCES))
            raise NotifyConfigError(f"profile field 'webhook.source' must be one of: {allowed}")
        if not ref:
            raise NotifyConfigError("profile field 'webhook.ref' must be a non-empty string")
        if source == "env":
            return ref, None
        return None, ref
    raise NotifyConfigError(f"profile_version must be 1 or {PROFILE_VERSION}; found {version!r}")


def _resolve_string_value(*, field: str, cli_value: str | None, profile_data: dict[str, Any]) -> str:
    if cli_value is not None:
        value = str(cli_value).strip()
        if value:
            return value
        raise NotifyConfigError(f"{field} must be a non-empty string")
    profile_value = profile_data.get(field)
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value
    raise NotifyConfigError(f"{field} is required; pass --{field.replace('_', '-')} or provide it in --profile")


def _resolve_optional_string_value(*, field: str, cli_value: str | None, profile_data: dict[str, Any]) -> str | None:
    if cli_value is not None:
        value = str(cli_value).strip()
        if not value:
            raise NotifyConfigError(f"{field} must be a non-empty string when provided")
        return value
    profile_value = profile_data.get(field)
    if profile_value is None:
        return None
    if isinstance(profile_value, str) and profile_value.strip():
        return profile_value
    raise NotifyConfigError(f"profile field '{field}' must be a non-empty string when provided")


def _resolve_path_value(
    *,
    field: str,
    cli_value: Path | None,
    profile_data: dict[str, Any],
    profile_path: Path | None,
) -> Path:
    if cli_value is not None:
        return cli_value
    profile_value = profile_data.get(field)
    if isinstance(profile_value, str) and profile_value.strip():
        resolved = Path(profile_value)
        if profile_path is not None and not resolved.is_absolute():
            return profile_path.parent / resolved
        return resolved
    raise NotifyConfigError(f"{field} is required; pass --{field.replace('_', '-')} or provide it in --profile")


def _resolve_optional_path_value(
    *,
    field: str,
    cli_value: Path | None,
    profile_data: dict[str, Any],
    profile_path: Path | None,
) -> Path | None:
    if cli_value is not None:
        return cli_value
    profile_value = profile_data.get(field)
    if profile_value is None:
        return None
    if isinstance(profile_value, str) and profile_value.strip():
        resolved = Path(profile_value)
        if profile_path is not None and not resolved.is_absolute():
            return profile_path.parent / resolved
        return resolved
    raise NotifyConfigError(f"profile field '{field}' must be a non-empty string path when provided")


def _events_path_hint() -> str:
    return (
        "Find the USR .events.log path with `uv run dense inspect run --usr-events-path` from the DenseGen "
        "workspace, or use `uv run dense inspect run --usr-events-path -c <config.yaml>`."
    )


def _validate_usr_events_probe(events_path: Path) -> None:
    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = str(line).strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise NotifyConfigError(
                    f"events must be a USR .events.log JSONL file; first non-empty line is not JSON "
                    f"(line {line_no}) in {events_path}. {_events_path_hint()}"
                ) from exc
            try:
                _validate_usr_event(event, allow_unknown_version=True)
            except NotifyConfigError as exc:
                raise NotifyConfigError(
                    f"events file does not look like USR .events.log (line {line_no}) in {events_path}: "
                    f"{exc}. {_events_path_hint()}"
                ) from exc
            return


def _resolve_usr_events_path(events_path: Path) -> Path:
    resolved = events_path.expanduser().resolve()
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        raise NotifyConfigError(
            f"events must point to a USR .events.log JSONL file, not a config file: {resolved}. {_events_path_hint()}"
        )
    if not resolved.exists():
        raise NotifyConfigError(f"events file not found: {resolved}. {_events_path_hint()}")
    if not resolved.is_file():
        raise NotifyConfigError(f"events path is not a file: {resolved}")
    _validate_usr_events_probe(resolved)
    return resolved


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


def _sanitize_profile_name(profile_path: Path) -> str:
    stem = str(profile_path.stem or "").strip()
    cleaned = "".join(char if char.isalnum() else "-" for char in stem).strip("-")
    return cleaned or "default"


def _state_root_for_profile(profile_path: Path) -> Path:
    state_home = str(os.environ.get("XDG_STATE_HOME", "")).strip()
    root = Path(state_home).expanduser() if state_home else (Path.home() / ".local" / "state")
    return root / "dnadesign" / "notify" / _sanitize_profile_name(profile_path)


def _wizard_next_steps(*, profile_path: Path, webhook_config: dict[str, str]) -> list[str]:
    profile_arg = str(profile_path)
    steps = [
        "Next steps:",
        f"  1) uv run notify profile doctor --profile {profile_arg}",
        f"  2) uv run notify usr-events watch --profile {profile_arg} --dry-run",
        f"  3) uv run notify usr-events watch --profile {profile_arg} --follow",
    ]
    source = str(webhook_config.get("source") or "").strip().lower()
    ref = str(webhook_config.get("ref") or "").strip()
    if source == "env" and ref:
        steps.append(f"  env: export {ref}=<your_webhook_url>")
    return steps


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
    if action_norm == "densegen_health" and isinstance(event, dict):
        args = event.get("args")
        if isinstance(args, dict):
            status = str(args.get("status") or "").strip().lower()
            if status in {"completed", "complete", "success", "succeeded"}:
                return "success"
            if status in {"failed", "failure", "error"}:
                return "failure"
            if status in {"started", "start"}:
                return "started"
    if "fail" in action_norm or "error" in action_norm:
        return "failure"
    if action_norm == "init":
        return "started"
    if action_norm in {"materialize", "compact_overlay", "overlay_compact", "registry_freeze"}:
        return "running"
    return "running"


def _event_message(event: dict[str, Any]) -> str:
    action = str(event.get("action") or "event")
    dataset_raw = event.get("dataset")
    dataset = dataset_raw if isinstance(dataset_raw, dict) else {}
    dataset_name = dataset.get("name") or "unknown-dataset"
    metrics_raw = event.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    args_raw = event.get("args")
    args = args_raw if isinstance(args_raw, dict) else {}

    if action == "densegen_health":
        status = str(args.get("status") or "running")
        rows_written_session = metrics.get("rows_written_session")
        run_quota = metrics.get("run_quota")
        quota_progress = metrics.get("quota_progress_pct")
        compression = metrics.get("compression_ratio")
        flush_count = metrics.get("flush_count")
        plan = args.get("plan")
        input_name = args.get("input_name")
        library_index = args.get("sampling_library_index")
        parts = [f"densegen {status} on {dataset_name}"]
        if rows_written_session is not None and run_quota is not None:
            parts.append(f"rows={rows_written_session}/{run_quota}")
        elif rows_written_session is not None:
            parts.append(f"rows={rows_written_session}")
        if quota_progress is not None:
            parts.append(f"quota={float(quota_progress):.1f}%")
        if plan:
            parts.append(f"plan={plan}")
        if input_name:
            parts.append(f"input={input_name}")
        if library_index is not None:
            parts.append(f"library={library_index}")
        if flush_count is not None:
            parts.append(f"flushes={flush_count}")
        if compression is not None:
            parts.append(f"cr={float(compression):.3f}")
        return " | ".join(parts)

    if action == "densegen_flush_failed":
        error_type = args.get("error_type")
        orphan_count = metrics.get("orphan_artifacts")
        parts = [f"{action} on {dataset_name}"]
        if error_type:
            parts.append(f"error_type={error_type}")
        if orphan_count is not None:
            parts.append(f"orphan_artifacts={orphan_count}")
        return " | ".join(parts)

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
    if version != USR_EVENT_VERSION and not allow_unknown_version:
        raise NotifyConfigError(f"unknown event_version={version}; expected {USR_EVENT_VERSION}")
    action = event.get("action")
    if not isinstance(action, str) or not action.strip():
        raise NotifyConfigError("event missing required 'action'")
    dataset = event.get("dataset")
    if dataset is not None and not isinstance(dataset, dict):
        raise NotifyConfigError("event field 'dataset' must be an object when provided")
    actor = event.get("actor")
    if actor is not None and not isinstance(actor, dict):
        raise NotifyConfigError("event field 'actor' must be an object when provided")


def _load_cursor_offset(cursor_path: Path | None) -> int:
    if cursor_path is None or not cursor_path.exists():
        return 0
    raw = cursor_path.read_text(encoding="utf-8").strip()
    if not raw:
        return 0
    try:
        offset = int(raw)
    except ValueError as exc:
        raise NotifyConfigError(f"cursor file must contain an integer byte offset: {cursor_path}") from exc
    if offset < 0:
        raise NotifyConfigError(f"cursor offset must be >= 0: {cursor_path}")
    return offset


def _save_cursor_offset(cursor_path: Path | None, offset: int) -> None:
    if cursor_path is None:
        return
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(str(int(offset)), encoding="utf-8")
    try:
        cursor_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on cursor file: {cursor_path}") from exc


def _iter_file_lines(
    events_path: Path,
    *,
    start_offset: int,
    on_truncate: str,
    follow: bool,
) -> Iterable[tuple[int, str]]:
    if not events_path.exists():
        raise NotifyConfigError(f"events file not found: {events_path}")
    mode = str(on_truncate or "").strip().lower()
    if mode not in {"error", "restart"}:
        raise NotifyConfigError(f"unsupported on-truncate mode '{on_truncate}'")
    size = int(events_path.stat().st_size)
    offset = int(start_offset)
    if offset > size:
        if mode == "restart":
            offset = 0
        else:
            raise NotifyConfigError(
                f"cursor offset exceeds events file size: offset={offset} size={size}. "
                "If the file was truncated or rotated, pass --on-truncate restart."
            )

    handle = events_path.open("r", encoding="utf-8")
    try:
        handle.seek(offset)
        while True:
            line = handle.readline()
            if line:
                yield handle.tell(), line
                continue
            if not follow:
                return
            time.sleep(0.2)
            if not events_path.exists():
                continue
            try:
                path_stat = events_path.stat()
            except FileNotFoundError:
                continue
            handle_stat = os.fstat(handle.fileno())
            handle_pos = int(handle.tell())
            path_changed = (int(path_stat.st_dev), int(path_stat.st_ino)) != (
                int(handle_stat.st_dev),
                int(handle_stat.st_ino),
            )
            truncated = handle_pos > int(path_stat.st_size)
            if not path_changed and not truncated:
                continue
            if mode == "error":
                reason = (
                    "events file was replaced while following"
                    if path_changed
                    else f"events file was truncated while following (pos={handle_pos} size={int(path_stat.st_size)})"
                )
                raise NotifyConfigError(f"{reason}. Pass --on-truncate restart to resume from start.")
            handle.close()
            handle = events_path.open("r", encoding="utf-8")
            handle.seek(0)
    finally:
        if not handle.closed:
            handle.close()


def _spool_payload(
    spool_dir: Path,
    *,
    provider: str,
    payload: dict[str, Any],
) -> Path:
    _ensure_private_directory(spool_dir, label="spool_dir")
    body = {"provider": provider, "payload": payload}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    file_name = f"{int(time.time() * 1000)}-{digest}.json"
    out_path = spool_dir / file_name
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(raw, encoding="utf-8")
    try:
        tmp_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on spool temp file: {tmp_path}") from exc
    tmp_path.replace(out_path)
    try:
        out_path.chmod(0o600)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on spool file: {out_path}") from exc
    return out_path


def _ensure_private_directory(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to create {label}: {path}") from exc
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise NotifyConfigError(f"failed to set secure permissions on {label}: {path}") from exc
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise NotifyConfigError(
            f"{label} must not be group/world-accessible (expected mode 700): {path} (mode={oct(mode)})"
        )


def _post_with_backoff(
    webhook_url: str,
    formatted_payload: dict[str, Any],
    *,
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
            post_json(webhook_url, formatted_payload, timeout=total_timeout, retries=0)
            return
        except NotifyDeliveryError:
            if attempt >= retries:
                raise
            delay = base_delay * (2**attempt)
            jitter = random.uniform(0.0, base_delay) if base_delay > 0 else 0.0
            time.sleep(delay + jitter)
            attempt += 1


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
        help="Secret reference: keychain://service/account or secretservice://service/account.",
    ),
    message: str | None = typer.Option(None, help="Optional message."),
    meta: Path | None = typer.Option(None, help="Path to JSON metadata file."),
    timeout: float = typer.Option(10.0, help="HTTP timeout (seconds)."),
    retries: int = typer.Option(0, help="Number of retries on failure."),
    dry_run: bool = typer.Option(False, help="Print payload and exit without sending."),
) -> None:
    try:
        webhook_url = resolve_webhook_url(url=url, url_env=url_env, secret_ref=secret_ref)
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
        post_json(webhook_url, formatted, timeout=timeout, retries=retries)
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
    preset: str | None = typer.Option(None, "--preset", help="Optional preset profile: densegen."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        events_path = _resolve_usr_events_path(events)
        preset_norm = str(preset or "").strip().lower()
        if preset_norm and preset_norm != "densegen":
            raise NotifyConfigError(f"unsupported preset '{preset}'. Supported values: densegen")

        cursor_path = cursor or (events_path.parent / "notify.cursor")
        payload: dict[str, Any] = {
            "profile_version": 1,
            "provider": str(provider).strip(),
            "url_env": str(url_env).strip(),
            "events": str(events_path),
            "cursor": str(cursor_path),
            "include_args": bool(include_args),
            "include_context": bool(include_context),
            "include_raw_event": bool(include_raw_event),
        }
        if only_actions is not None:
            payload["only_actions"] = str(only_actions).strip()
        if only_tools is not None:
            payload["only_tools"] = str(only_tools).strip()
        if spool_dir is not None:
            payload["spool_dir"] = str(spool_dir)
        if preset_norm == "densegen":
            payload["preset"] = "densegen"
            for key, value in _DENSEGEN_PROFILE_PRESET.items():
                payload.setdefault(key, value)

        if not payload["provider"]:
            raise NotifyConfigError("provider must be a non-empty string")
        if not payload["url_env"]:
            raise NotifyConfigError("url_env must be a non-empty string")

        _write_profile_file(profile, payload, force=force)
        typer.echo(f"Profile written: {profile}")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _profile_wizard_impl(
    profile: Path = typer.Option(Path("outputs/notify.profile.json"), "--profile", help="Path to profile JSON file."),
    provider: str = typer.Option("slack", help="Provider: generic|slack|discord."),
    events: Path = typer.Option(..., "--events", help="USR .events.log JSONL path."),
    cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
    only_actions: str | None = typer.Option(None, "--only-actions", help="Comma-separated action filter."),
    only_tools: str | None = typer.Option(None, "--only-tools", help="Comma-separated actor tool filter."),
    spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory for failed payload spool files."),
    include_args: bool = typer.Option(False, "--include-args/--no-include-args"),
    include_context: bool = typer.Option(False, "--include-context/--no-include-context"),
    include_raw_event: bool = typer.Option(False, "--include-raw-event/--no-include-raw-event"),
    preset: str | None = typer.Option(None, "--preset", help="Optional preset profile: densegen."),
    secret_source: str = typer.Option(
        "auto",
        "--secret-source",
        help="Webhook source: auto|env|keychain|secretservice.",
    ),
    url_env: str | None = typer.Option(None, "--url-env", help="Environment variable holding webhook URL."),
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
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile file."),
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        events_path = _resolve_usr_events_path(events)
        provider_value = str(provider).strip()
        if not provider_value:
            raise NotifyConfigError("provider must be a non-empty string")
        preset_norm = str(preset or "").strip().lower()
        if preset_norm and preset_norm != "densegen":
            raise NotifyConfigError(f"unsupported preset '{preset}'. Supported values: densegen")

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
            env_name = _resolve_cli_optional_string(field="url_env", cli_value=url_env) or "DENSEGEN_WEBHOOK"
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

        state_root = _state_root_for_profile(profile_path)
        default_cursor = state_root / "notify.cursor"
        default_spool = state_root / "spool"
        cursor_value = cursor or default_cursor
        spool_value = spool_dir or default_spool
        try:
            _ensure_private_directory(cursor_value.parent, label="cursor directory")
            _ensure_private_directory(spool_value, label="spool_dir")
        except NotifyConfigError as exc:
            raise NotifyConfigError(
                f"{exc}. Pass --cursor and --spool-dir to writable paths if the default state root is restricted."
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
        if only_actions is not None:
            payload["only_actions"] = str(only_actions).strip()
        if only_tools is not None:
            payload["only_tools"] = str(only_tools).strip()
        if preset_norm == "densegen":
            payload["preset"] = "densegen"
            for key, value in _DENSEGEN_PROFILE_PRESET.items():
                payload.setdefault(key, value)

        _write_profile_file(profile_path, payload, force=force)
        typer.echo(f"Profile written: {profile_path}")
        for line in _wizard_next_steps(profile_path=profile_path, webhook_config=webhook_config):
            typer.echo(line)
    except NotifyError as exc:
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
) -> None:
    try:
        profile_path = profile.expanduser().resolve()
        data = _read_profile(profile_path)
        profile_url_env, profile_secret_ref = _resolve_profile_webhook_source(data)
        _ = resolve_webhook_url(url=None, url_env=profile_url_env, secret_ref=profile_secret_ref)

        events_path = _resolve_path_value(
            field="events",
            cli_value=None,
            profile_data=data,
            profile_path=profile_path,
        )
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

        typer.echo("Profile wiring OK.")
    except NotifyError as exc:
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
    events: Path | None = typer.Option(None, "--events", help="USR .events.log JSONL path."),
    profile: Path | None = typer.Option(None, "--profile", help="Path to profile JSON file."),
    cursor: Path | None = typer.Option(None, "--cursor", help="Cursor file storing byte offset."),
    follow: bool = typer.Option(False, "--follow", help="Follow events file for new lines."),
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
) -> None:
    try:
        profile_path = profile.expanduser().resolve() if profile is not None else None
        profile_data = _read_profile(profile_path) if profile_path is not None else {}
        provider_value = _resolve_string_value(field="provider", cli_value=provider, profile_data=profile_data)
        events_path = _resolve_path_value(
            field="events",
            cli_value=events,
            profile_data=profile_data,
            profile_path=profile_path,
        )
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

        webhook_url: str | None = None
        if not dry_run:
            webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
        action_filter = set(_split_csv(only_actions_value))
        tool_filter = set(_split_csv(only_tools_value))
        start_offset = _load_cursor_offset(cursor_path)
        failed_unsent = 0
        on_invalid_event_mode = str(on_invalid_event or "").strip().lower()
        if on_invalid_event_mode not in {"error", "skip"}:
            raise NotifyConfigError(f"unsupported on-invalid-event mode '{on_invalid_event}'")

        for next_offset, line in _iter_file_lines(
            events_path,
            start_offset=start_offset,
            on_truncate=on_truncate,
            follow=follow,
        ):
            text = line.strip()
            if not text:
                _save_cursor_offset(cursor_path, next_offset)
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                if on_invalid_event_mode == "skip":
                    typer.echo(f"Skipping invalid event line: event line is not valid JSON: {exc}")
                    _save_cursor_offset(cursor_path, next_offset)
                    continue
                raise NotifyConfigError(f"event line is not valid JSON: {exc}") from exc
            try:
                _validate_usr_event(event, allow_unknown_version=allow_unknown_version)
            except NotifyConfigError as exc:
                if on_invalid_event_mode == "skip":
                    typer.echo(f"Skipping invalid event line: {exc}")
                    _save_cursor_offset(cursor_path, next_offset)
                    continue
                raise

            action = str(event.get("action"))
            if action_filter and action not in action_filter:
                _save_cursor_offset(cursor_path, next_offset)
                continue

            actor_raw = event.get("actor")
            actor = actor_raw if isinstance(actor_raw, dict) else {}
            actor_tool = actor.get("tool")
            if tool_filter and actor_tool not in tool_filter:
                _save_cursor_offset(cursor_path, next_offset)
                continue
            tool_name = tool or actor_tool
            if not tool_name:
                raise NotifyConfigError("event missing actor.tool; provide --tool to override")
            run_value = run_id or actor.get("run_id")
            if not run_value:
                raise NotifyConfigError("event missing actor.run_id; provide --run-id to override")

            payload = build_payload(
                status=_status_for_action(action, event=event),
                tool=tool_name,
                run_id=run_value,
                message=message or _event_message(event),
                meta=_event_meta(
                    event,
                    include_args=bool(include_args_value),
                    include_raw_event=bool(include_raw_event_value),
                    include_context=bool(include_context_value),
                ),
                timestamp=event.get("timestamp_utc"),
                host=(actor.get("host") if bool(include_context_value) else None),
                cwd=(
                    ((event.get("dataset") or {}) if isinstance(event.get("dataset"), dict) else {}).get("root")
                    if bool(include_context_value)
                    else None
                ),
                version=event.get("version"),
            )
            formatted = format_for_provider(provider_value, payload)
            if dry_run:
                typer.echo(json.dumps(formatted, sort_keys=True))
                _save_cursor_offset(cursor_path, next_offset)
                continue

            sent_or_spooled = False
            try:
                if webhook_url is None:
                    raise NotifyConfigError("webhook URL is required when not running in --dry-run mode")
                _post_with_backoff(
                    webhook_url,
                    formatted,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    retry_max=retry_max,
                    retry_base_seconds=retry_base_seconds,
                )
                sent_or_spooled = True
            except NotifyDeliveryError:
                if spool_dir_value is not None:
                    _spool_payload(spool_dir_value, provider=provider_value, payload=payload)
                    sent_or_spooled = True
                elif fail_fast:
                    raise
                else:
                    failed_unsent += 1

            if sent_or_spooled:
                _save_cursor_offset(cursor_path, next_offset)

        if failed_unsent:
            raise NotifyDeliveryError(f"{failed_unsent} event(s) failed delivery and were not spooled")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def _spool_drain_impl(
    spool_dir: Path | None = typer.Option(None, "--spool-dir", help="Directory containing spooled payload files."),
    provider: str | None = typer.Option(None, help="Provider: generic|slack|discord."),
    url: str | None = typer.Option(None, help="Webhook URL."),
    url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
    secret_ref: str | None = typer.Option(
        None,
        "--secret-ref",
        help="Secret reference: keychain://service/account or secretservice://service/account.",
    ),
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
        provider_value = _resolve_string_value(field="provider", cli_value=provider, profile_data=profile_data)
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
        webhook_url = resolve_webhook_url(url=url, url_env=url_env_value, secret_ref=secret_ref_value)
        if not spool_dir_value.exists():
            raise NotifyConfigError(f"spool directory not found: {spool_dir_value}")
        failed = 0
        for path in sorted(spool_dir_value.glob("*.json")):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                payload = body.get("payload")
                if not isinstance(payload, dict):
                    raise NotifyConfigError(f"invalid spool payload file: {path}")
                formatted = format_for_provider(provider_value, payload)
                _post_with_backoff(
                    webhook_url,
                    formatted,
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
register_profile_commands(
    profile_app,
    init_handler=_profile_init_impl,
    wizard_handler=_profile_wizard_impl,
    show_handler=_profile_show_impl,
    doctor_handler=_profile_doctor_impl,
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
