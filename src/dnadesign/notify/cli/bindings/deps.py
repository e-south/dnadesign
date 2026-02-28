"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/deps.py

Dependency exports and helper adapters for notify CLI bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...delivery.http import post_json
from ...delivery.payload import build_payload
from ...delivery.secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from ...delivery.validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url
from ...errors import NotifyConfigError, NotifyDeliveryError, NotifyError
from ...events.source import normalize_tool_name as _normalize_setup_tool_name
from ...events.source import resolve_tool_events_path as _resolve_tool_events_path
from ...events.transforms import event_message as _event_message
from ...events.transforms import event_meta as _event_meta
from ...events.transforms import status_for_action as _status_for_action
from ...events.transforms import validate_usr_event as _validate_usr_event_data
from ...profiles.flows import create_wizard_profile as _create_wizard_profile_flow
from ...profiles.flows import resolve_profile_path_for_setup as _resolve_profile_path_for_setup
from ...profiles.flows import resolve_profile_path_for_wizard as _resolve_profile_path_for_wizard
from ...profiles.flows import resolve_setup_events as _resolve_setup_events
from ...profiles.flows import resolve_webhook_config as _resolve_webhook_config
from ...profiles.policy import DEFAULT_PROFILE_PATH as _DEFAULT_PROFILE_PATH
from ...profiles.policy import default_profile_path_for_tool as _default_profile_path_for_tool
from ...profiles.policy import policy_defaults as _policy_defaults_for
from ...profiles.policy import resolve_workflow_policy as _resolve_workflow_policy
from ...profiles.schema import read_profile as _read_profile
from ...profiles.schema import resolve_profile_events_source as _resolve_profile_events_source
from ...profiles.schema import resolve_profile_webhook_source as _resolve_profile_webhook_source
from ...profiles.workspace import list_tool_workspaces as _list_tool_workspaces
from ...profiles.workspace import resolve_tool_workspace_config_path as _resolve_tool_workspace_config_path
from ...runtime.runner import run_spool_drain, run_usr_events_watch
from ...runtime.spool import ensure_private_directory as _ensure_private_directory
from ...runtime.watch import watch_usr_events_loop
from ..commands.delivery.providers import format_for_provider
from ..handlers import (
    run_profile_doctor_command,
    run_profile_init_command,
    run_profile_show_command,
    run_profile_wizard_command,
    run_send_command,
    run_setup_list_workspaces_command,
    run_setup_resolve_events_command,
    run_setup_slack_command,
    run_setup_webhook_command,
    run_spool_drain_command,
    run_usr_events_watch_command,
)
from ..resolve import resolve_cli_optional_string as _resolve_cli_optional_string
from ..resolve import resolve_existing_file_path as _resolve_existing_file_path
from ..resolve import resolve_optional_path_value as _resolve_optional_path_value
from ..resolve import resolve_optional_string_value as _resolve_optional_string_value
from ..resolve import resolve_path_value as _resolve_path_value
from ..resolve import resolve_string_value as _resolve_string_value
from ..resolve import resolve_usr_events_path as _resolve_usr_events_path_raw
from . import helpers

DEPENDENCY_EXPORTS = (
    "build_payload",
    "format_for_provider",
    "is_secret_backend_available",
    "post_json",
    "resolve_secret_ref",
    "resolve_tls_ca_bundle",
    "resolve_webhook_url",
    "run_profile_doctor_command",
    "run_profile_init_command",
    "run_profile_show_command",
    "run_profile_wizard_command",
    "run_send_command",
    "run_setup_list_workspaces_command",
    "run_setup_resolve_events_command",
    "run_setup_slack_command",
    "run_setup_webhook_command",
    "run_spool_drain",
    "run_spool_drain_command",
    "run_usr_events_watch",
    "run_usr_events_watch_command",
    "store_secret_ref",
    "time",
    "validate_provider_webhook_url",
    "watch_usr_events_loop",
    "_DEFAULT_PROFILE_PATH",
    "_create_wizard_profile_flow",
    "_default_profile_path_for_tool",
    "_ensure_private_directory",
    "_event_message",
    "_event_meta",
    "_list_tool_workspaces",
    "_load_meta",
    "_normalize_setup_tool_name",
    "_policy_defaults_for",
    "_post_with_backoff",
    "_probe_path_writable",
    "_read_profile",
    "_resolve_cli_optional_string",
    "_resolve_existing_file_path",
    "_resolve_optional_path_value",
    "_resolve_optional_string_value",
    "_resolve_path_value",
    "_resolve_profile_events_source",
    "_resolve_profile_path_for_setup",
    "_resolve_profile_path_for_wizard",
    "_resolve_profile_webhook_source",
    "_resolve_setup_events",
    "_resolve_string_value",
    "_resolve_tool_events_path",
    "_resolve_tool_workspace_config_path",
    "_resolve_usr_events_path",
    "_resolve_webhook_config",
    "_resolve_workflow_policy",
    "_split_csv",
    "_status_for_action",
    "_validate_usr_event",
    "_write_profile_file",
)


@lru_cache(maxsize=1)
def _usr_event_version() -> int:
    return helpers.usr_event_version(
        import_module_fn=importlib.import_module,
        notify_config_error_cls=NotifyConfigError,
    )


def _load_meta(meta_path: Path | None) -> dict[str, Any]:
    return helpers.load_meta(
        meta_path,
        notify_error_cls=NotifyError,
    )


def _split_csv(value: str | None) -> list[str]:
    return helpers.split_csv(value)


def _resolve_usr_events_path(events_path: Path, *, require_exists: bool = True) -> Path:
    return helpers.resolve_usr_events_path(
        events_path,
        require_exists=require_exists,
        resolve_usr_events_path_raw_fn=_resolve_usr_events_path_raw,
        validate_usr_event_fn=_validate_usr_event,
    )


def _probe_path_writable(path: Path) -> None:
    helpers.probe_path_writable(path)


def _write_profile_file(profile_path: Path, payload: dict[str, Any], *, force: bool) -> None:
    helpers.write_profile_file(
        profile_path,
        payload,
        force=force,
        notify_config_error_cls=NotifyConfigError,
    )


def _validate_usr_event(event: dict[str, Any], *, allow_unknown_version: bool) -> None:
    helpers.validate_usr_event(
        event,
        allow_unknown_version=allow_unknown_version,
        validate_usr_event_data_fn=_validate_usr_event_data,
        usr_event_version_fn=_usr_event_version,
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
    helpers.post_with_backoff(
        webhook_url,
        formatted_payload,
        tls_ca_bundle=tls_ca_bundle,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retry_max=retry_max,
        retry_base_seconds=retry_base_seconds,
        post_json_fn=post_json,
        notify_delivery_error_cls=NotifyDeliveryError,
        sleep_fn=time.sleep,
        jitter_fn=random.uniform,
    )
