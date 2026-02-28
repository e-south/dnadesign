"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/deps/runtime.py

Runtime-domain dependency exports and helper adapters for notify CLI bindings.

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

from ....delivery.http import post_json
from ....delivery.validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url
from ....errors import NotifyConfigError, NotifyDeliveryError
from ....events.source import normalize_tool_name as _normalize_setup_tool_name
from ....events.source import resolve_tool_events_path as _resolve_tool_events_path
from ....events.transforms import event_message as _event_message
from ....events.transforms import event_meta as _event_meta
from ....events.transforms import status_for_action as _status_for_action
from ....events.transforms import validate_usr_event as _validate_usr_event_data
from ....profiles.policy import default_profile_path_for_tool as _default_profile_path_for_tool
from ....profiles.schema import resolve_profile_events_source as _resolve_profile_events_source
from ....profiles.schema import resolve_profile_webhook_source as _resolve_profile_webhook_source
from ....runtime.runner import run_spool_drain, run_usr_events_watch
from ....runtime.watch import watch_usr_events_loop
from ...commands.delivery.providers import format_for_provider
from ...handlers import run_spool_drain_command, run_usr_events_watch_command
from ...resolve import resolve_cli_optional_string as _resolve_cli_optional_string
from ...resolve import resolve_optional_path_value as _resolve_optional_path_value
from ...resolve import resolve_optional_string_value as _resolve_optional_string_value
from ...resolve import resolve_path_value as _resolve_path_value
from ...resolve import resolve_string_value as _resolve_string_value
from ...resolve import resolve_usr_events_path as _resolve_usr_events_path_raw
from .. import helpers
from .profile import _read_profile
from .setup import _resolve_tool_workspace_config_path


@lru_cache(maxsize=1)
def _usr_event_version() -> int:
    return helpers.usr_event_version(
        import_module_fn=importlib.import_module,
        notify_config_error_cls=NotifyConfigError,
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
