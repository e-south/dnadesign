"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/flow_profile.py

Profile path resolution and profile materialization helpers for notify setup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..delivery.secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from ..errors import NotifyConfigError
from ..runtime.spool import ensure_private_directory
from .flow_types import _validate_progress_min_seconds, _validate_progress_step_pct
from .flow_webhook import resolve_webhook_config
from .ops import sanitize_profile_name, wizard_next_steps
from .policy import DEFAULT_PROFILE_PATH, default_profile_path_for_tool, policy_defaults, resolve_workflow_policy
from .resolve import resolve_existing_file_path
from .schema import PROFILE_VERSION


def resolve_profile_path_for_wizard(*, profile: Path, policy: str | None) -> Path:
    if profile != DEFAULT_PROFILE_PATH:
        return profile
    policy_namespace = resolve_workflow_policy(policy=policy)
    if policy_namespace is None:
        raise NotifyConfigError(
            "default profile path is ambiguous in wizard mode; pass --policy or --profile to select a profile namespace"
        )
    return default_profile_path_for_tool(policy_namespace)


def resolve_profile_path_for_setup(
    *,
    profile: Path,
    tool_name: str | None,
    policy: str | None,
    config: Path | None,
) -> Path:
    if profile != DEFAULT_PROFILE_PATH:
        return profile
    namespace = tool_name
    if namespace is None:
        policy_namespace = resolve_workflow_policy(policy=policy)
        if policy_namespace is None:
            raise NotifyConfigError(
                "default profile path is ambiguous in --events mode; "
                "pass --policy or --profile to select a profile namespace"
            )
        namespace = policy_namespace
    if config is not None and tool_name is not None:
        config_path = config.expanduser().resolve()
        return config_path.parent / default_profile_path_for_tool(namespace)
    return default_profile_path_for_tool(namespace)


def create_wizard_profile(
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
    progress_step_pct: int | None,
    progress_min_seconds: float | None,
    tls_ca_bundle: Path | None,
    policy: str | None,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    force: bool,
    events_require_exists: bool,
    events_source: dict[str, str] | None,
    resolve_events_path: Callable[[Path, bool], Path],
    ensure_private_directory_fn: Callable[[Path, str], None] = ensure_private_directory,
    secret_backend_available_fn: Callable[[str], bool] = is_secret_backend_available,
    resolve_secret_ref_fn: Callable[[str], str] = resolve_secret_ref,
    store_secret_ref_fn: Callable[[str, str], None] = store_secret_ref,
    write_profile_file_fn: Callable[[Path, dict[str, Any], bool], None] | None = None,
) -> dict[str, Any]:
    if write_profile_file_fn is None:
        raise NotifyConfigError("write_profile_file_fn is required")

    profile_path = profile.expanduser().resolve()
    events_path = resolve_events_path(events, events_require_exists)
    events_exists = events_path.exists()
    provider_value = str(provider).strip()
    if not provider_value:
        raise NotifyConfigError("provider must be a non-empty string")
    policy_name = resolve_workflow_policy(policy=policy)
    progress_step_pct_value = _validate_progress_step_pct(progress_step_pct)
    progress_min_seconds_value = _validate_progress_min_seconds(progress_min_seconds)

    webhook_config = resolve_webhook_config(
        secret_source=secret_source,
        url_env=url_env,
        secret_ref=secret_ref,
        webhook_url=webhook_url,
        store_webhook=store_webhook,
        secret_name=sanitize_profile_name(profile_path),
        secret_backend_available_fn=secret_backend_available_fn,
        resolve_secret_ref_fn=resolve_secret_ref_fn,
        store_secret_ref_fn=store_secret_ref_fn,
    )

    default_cursor = profile_path.parent / "cursor"
    default_spool = profile_path.parent / "spool"
    cursor_value = cursor or default_cursor
    spool_value = spool_dir or default_spool
    try:
        ensure_private_directory_fn(cursor_value.parent, label="cursor directory")
        ensure_private_directory_fn(spool_value, label="spool_dir")
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
        payload["tls_ca_bundle"] = str(resolve_existing_file_path(field="tls_ca_bundle", path_value=tls_ca_bundle))
    if only_actions is not None:
        payload["only_actions"] = str(only_actions).strip()
    if only_tools is not None:
        payload["only_tools"] = str(only_tools).strip()
    if policy_name is not None:
        payload["policy"] = policy_name
        for key, value in policy_defaults(policy_name).items():
            payload.setdefault(key, value)
    if events_source is not None:
        payload["events_source"] = dict(events_source)
    if progress_step_pct_value is not None:
        payload["progress_step_pct"] = int(progress_step_pct_value)
    if progress_min_seconds_value is not None:
        payload["progress_min_seconds"] = float(progress_min_seconds_value)

    write_profile_file_fn(profile_path, payload, force)
    next_steps = wizard_next_steps(
        profile_path=profile_path,
        webhook_config=webhook_config,
        events_exists=events_exists,
        events_source=events_source,
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
