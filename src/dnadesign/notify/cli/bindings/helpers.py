"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/helpers.py

Shared utility helpers for notify CLI binding modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def usr_event_version(
    *,
    import_module_fn: Callable[[str], Any],
    notify_config_error_cls: type[Exception],
) -> int:
    module = import_module_fn("dnadesign.usr.event_schema")
    version = getattr(module, "USR_EVENT_VERSION", None)
    if not isinstance(version, int):
        raise notify_config_error_cls("USR event schema is invalid: USR_EVENT_VERSION must be an integer")
    return int(version)


def load_meta(
    meta_path: Path | None,
    *,
    notify_error_cls: type[Exception],
) -> dict[str, Any]:
    if meta_path is None:
        return {}
    if not meta_path.exists():
        raise notify_error_cls(f"meta file not found: {meta_path}")
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise notify_error_cls(f"meta file is not valid JSON: {meta_path}") from exc
    if not isinstance(data, dict):
        raise notify_error_cls("meta file must contain a JSON object")
    return data


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_usr_events_path(
    events_path: Path,
    *,
    require_exists: bool,
    resolve_usr_events_path_raw_fn: Callable[..., Path],
    validate_usr_event_fn: Callable[..., None],
) -> Path:
    return resolve_usr_events_path_raw_fn(
        events_path,
        require_exists=require_exists,
        validate_usr_event=validate_usr_event_fn,
    )


def probe_path_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".notify-write-probe.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()


def write_profile_file(
    profile_path: Path,
    payload: dict[str, Any],
    *,
    force: bool,
    notify_config_error_cls: type[Exception],
) -> None:
    if profile_path.exists() and not force:
        raise notify_config_error_cls(f"profile already exists: {profile_path}. Pass --force to overwrite.")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = profile_path.with_suffix(profile_path.suffix + ".tmp")
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(profile_path)
    try:
        profile_path.chmod(0o600)
    except OSError as exc:
        raise notify_config_error_cls(f"failed to set secure permissions on profile: {profile_path}") from exc


def validate_usr_event(
    event: dict[str, Any],
    *,
    allow_unknown_version: bool,
    validate_usr_event_data_fn: Callable[..., None],
    usr_event_version_fn: Callable[[], int],
) -> None:
    validate_usr_event_data_fn(
        event,
        expected_version=usr_event_version_fn(),
        allow_unknown_version=allow_unknown_version,
    )


def post_with_backoff(
    webhook_url: str,
    formatted_payload: dict[str, Any],
    *,
    tls_ca_bundle: Path | None,
    connect_timeout: float,
    read_timeout: float,
    retry_max: int,
    retry_base_seconds: float,
    post_json_fn: Callable[..., None],
    notify_delivery_error_cls: type[Exception],
    sleep_fn: Callable[[float], None],
    jitter_fn: Callable[[float, float], float],
) -> None:
    total_timeout = float(connect_timeout) + float(read_timeout)
    retries = max(int(retry_max), 0)
    base_delay = max(float(retry_base_seconds), 0.0)
    attempt = 0
    while True:
        try:
            post_json_fn(
                webhook_url,
                formatted_payload,
                timeout=total_timeout,
                retries=0,
                tls_ca_bundle=tls_ca_bundle,
            )
            return
        except notify_delivery_error_cls:
            if attempt >= retries:
                raise
            delay = base_delay * (2**attempt)
            jitter = jitter_fn(0.0, base_delay) if base_delay > 0 else 0.0
            sleep_fn(delay + jitter)
            attempt += 1
