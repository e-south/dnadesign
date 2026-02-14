"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profile_flows.py

Setup/profile orchestration helpers for notify profile creation flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from .cli_resolve import resolve_cli_optional_string, resolve_existing_file_path
from .errors import NotifyConfigError
from .events_source import normalize_tool_name, resolve_tool_events_path
from .profile_ops import sanitize_profile_name, wizard_next_steps
from .profile_schema import PROFILE_VERSION
from .secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from .spool_ops import ensure_private_directory
from .workflow_policy import (
    DEFAULT_PROFILE_PATH,
    DEFAULT_WEBHOOK_ENV,
    default_profile_path_for_tool,
    policy_defaults,
    resolve_workflow_policy,
)
from .workspace_source import resolve_tool_workspace_config_path


@dataclass(frozen=True)
class SetupEventsResolution:
    events_path: Path
    events_source: dict[str, str] | None
    policy: str | None
    tool_name: str | None
    events_require_exists: bool


def _default_file_secret_path(secret_name: str) -> Path:
    return (Path(__file__).resolve().parent / ".secrets" / f"{secret_name}.webhook").resolve()


def _validate_progress_step_pct(value: int | None) -> int | None:
    if value is None:
        return None
    step = int(value)
    if step < 1 or step > 100:
        raise NotifyConfigError("progress_step_pct must be an integer between 1 and 100")
    return step


def _validate_progress_min_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    minimum = float(value)
    if minimum <= 0.0:
        raise NotifyConfigError("progress_min_seconds must be a positive number")
    return minimum


def resolve_webhook_config(
    *,
    secret_source: str,
    url_env: str | None,
    secret_ref: str | None,
    webhook_url: str | None,
    store_webhook: bool,
    secret_name: str,
    secret_backend_available_fn: Callable[[str], bool] = is_secret_backend_available,
    resolve_secret_ref_fn: Callable[[str], str] = resolve_secret_ref,
    store_secret_ref_fn: Callable[[str, str], None] = store_secret_ref,
) -> dict[str, str]:
    mode = str(secret_source or "").strip().lower()
    if not mode:
        raise NotifyConfigError("secret_source must be a non-empty string")
    if mode not in {"auto", "env", "keychain", "secretservice", "file"}:
        raise NotifyConfigError("secret_source must be one of: auto, env, keychain, secretservice, file")

    if mode == "env":
        env_name = resolve_cli_optional_string(field="url_env", cli_value=url_env)
        if env_name is None:
            env_name = DEFAULT_WEBHOOK_ENV
        return {"source": "env", "ref": env_name}

    provided_secret_ref = resolve_cli_optional_string(field="secret_ref", cli_value=secret_ref)
    secret_refs: list[str]
    if provided_secret_ref is not None:
        secret_refs = [provided_secret_ref]
    elif mode == "auto":
        candidate_modes: list[str] = []
        for candidate in ("keychain", "secretservice", "file"):
            if secret_backend_available_fn(candidate):
                candidate_modes.append(candidate)
        if not candidate_modes:
            raise NotifyConfigError(
                "secret_source=auto requires keychain, secretservice, or file backend availability. "
                "Pass --secret-source env to opt into environment-variable webhook storage."
            )
        secret_refs = []
        for candidate in candidate_modes:
            if candidate == "file":
                secret_path = _default_file_secret_path(secret_name)
                secret_refs.append(secret_path.as_uri())
            else:
                secret_refs.append(f"{candidate}://dnadesign.notify/{secret_name}")
    else:
        if not secret_backend_available_fn(mode):
            raise NotifyConfigError(f"secret backend '{mode}' is not available on this system")
        if mode == "file":
            secret_path = _default_file_secret_path(secret_name)
            secret_refs = [secret_path.as_uri()]
        else:
            secret_refs = [f"{mode}://dnadesign.notify/{secret_name}"]

    if not store_webhook:
        return {"source": "secret_ref", "ref": secret_refs[0]}

    webhook_value = resolve_cli_optional_string(field="webhook_url", cli_value=webhook_url)
    last_error: NotifyConfigError | None = None
    for secret_value in secret_refs:
        webhook_config = {"source": "secret_ref", "ref": secret_value}
        if webhook_value is None:
            try:
                _ = resolve_secret_ref_fn(secret_value)
                return webhook_config
            except NotifyConfigError:
                webhook_value = str(typer.prompt("Webhook URL", hide_input=True)).strip()
        if not webhook_value:
            raise NotifyConfigError("webhook_url is required when --store-webhook is enabled")
        try:
            store_secret_ref_fn(secret_value, webhook_value)
            return webhook_config
        except NotifyConfigError as exc:
            last_error = exc
            if len(secret_refs) == 1:
                raise
            continue

    if last_error is not None:
        raise last_error
    raise NotifyConfigError("failed to resolve webhook secret configuration")


def resolve_setup_events(
    *,
    events: Path | None,
    tool: str | None,
    config: Path | None,
    workspace: str | None,
    policy: str | None,
    search_start: Path | None = None,
    resolve_tool_events_path_fn: Callable[..., tuple[Path, str | None]] = resolve_tool_events_path,
    resolve_tool_workspace_config_fn: Callable[..., Path] = resolve_tool_workspace_config_path,
    normalize_tool_name_fn: Callable[[str | None], str | None] = normalize_tool_name,
) -> SetupEventsResolution:
    has_events = events is not None
    has_tool = tool is not None or config is not None or workspace is not None
    if has_events and has_tool:
        raise NotifyConfigError("pass either --events or --tool with --config/--workspace, not both")
    if not has_events and not has_tool:
        raise NotifyConfigError("pass either --events or --tool with --config/--workspace")

    if has_events:
        events_path = events if events is not None else Path("")
        return SetupEventsResolution(
            events_path=events_path,
            events_source=None,
            policy=policy,
            tool_name=None,
            events_require_exists=True,
        )

    has_config = config is not None
    has_workspace = workspace is not None
    if tool is None or has_config == has_workspace:
        raise NotifyConfigError("resolver mode requires --tool with exactly one of --config or --workspace")
    if has_workspace:
        config_path = resolve_tool_workspace_config_fn(
            tool=tool,
            workspace=str(workspace),
            search_start=search_start,
        )
        if not isinstance(config_path, Path):
            config_path = Path(config_path)
        config_path = config_path.expanduser().resolve()
    else:
        config_path = config.expanduser().resolve() if config is not None else Path("")
    events_path, default_policy = resolve_tool_events_path_fn(tool=tool, config=config_path)
    tool_name = normalize_tool_name_fn(tool)
    events_source = {
        "tool": str(tool_name),
        "config": str(config_path),
    }
    policy_value = policy if policy is not None else default_policy
    return SetupEventsResolution(
        events_path=events_path,
        events_source=events_source,
        policy=policy_value,
        tool_name=tool_name,
        events_require_exists=False,
    )


def resolve_profile_path_for_wizard(*, profile: Path, policy: str | None) -> Path:
    if profile != DEFAULT_PROFILE_PATH:
        return profile
    policy_namespace = resolve_workflow_policy(policy=policy)
    if policy_namespace is None:
        raise NotifyConfigError(
            "default profile path is ambiguous in wizard mode; "
            "pass --policy or --profile to select a profile namespace"
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
        payload["tls_ca_bundle"] = str(
            resolve_existing_file_path(field="tls_ca_bundle", path_value=tls_ca_bundle)
        )
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
