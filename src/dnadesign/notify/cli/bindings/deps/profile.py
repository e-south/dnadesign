"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli/bindings/deps/profile.py

Profile-domain dependency exports and helper adapters for notify CLI bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....delivery.secrets import is_secret_backend_available, resolve_secret_ref, store_secret_ref
from ....delivery.validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url
from ....errors import NotifyConfigError
from ....profiles.flows import create_wizard_profile as _create_wizard_profile_flow
from ....profiles.flows import resolve_profile_path_for_wizard as _resolve_profile_path_for_wizard
from ....profiles.policy import DEFAULT_PROFILE_PATH as _DEFAULT_PROFILE_PATH
from ....profiles.policy import default_profile_path_for_tool as _default_profile_path_for_tool
from ....profiles.policy import policy_defaults as _policy_defaults_for
from ....profiles.policy import resolve_workflow_policy as _resolve_workflow_policy
from ....profiles.schema import read_profile as _read_profile
from ....profiles.schema import resolve_profile_events_source as _resolve_profile_events_source
from ....profiles.schema import resolve_profile_webhook_source as _resolve_profile_webhook_source
from ....runtime.spool import ensure_private_directory as _ensure_private_directory
from ...handlers import (
    run_profile_doctor_command,
    run_profile_init_command,
    run_profile_show_command,
    run_profile_wizard_command,
)
from ...resolve import resolve_existing_file_path as _resolve_existing_file_path
from .. import helpers


def _probe_path_writable(path: Path) -> None:
    helpers.probe_path_writable(path)


def _write_profile_file(profile_path: Path, payload: dict[str, Any], *, force: bool) -> None:
    helpers.write_profile_file(
        profile_path,
        payload,
        force=force,
        notify_config_error_cls=NotifyConfigError,
    )
