"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_resolve_module.py

Module tests for notify CLI option and events-path resolution helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.notify.cli_resolve import resolve_cli_optional_string, resolve_path_value, resolve_usr_events_path
from dnadesign.notify.errors import NotifyConfigError


def test_resolve_cli_optional_string_trims_and_validates() -> None:
    assert resolve_cli_optional_string(field="url_env", cli_value=None) is None
    assert resolve_cli_optional_string(field="url_env", cli_value="  NOTIFY_WEBHOOK  ") == "NOTIFY_WEBHOOK"
    with pytest.raises(NotifyConfigError, match="url_env must be a non-empty string"):
        resolve_cli_optional_string(field="url_env", cli_value="   ")


def test_resolve_path_value_resolves_relative_to_profile() -> None:
    profile_path = Path("/tmp/notify/profile.json")
    resolved = resolve_path_value(
        field="events",
        cli_value=None,
        profile_data={"events": "usr/.events.log"},
        profile_path=profile_path,
    )
    assert resolved == Path("/tmp/notify/usr/.events.log")


def test_resolve_usr_events_path_rejects_yaml_path() -> None:
    with pytest.raises(NotifyConfigError, match="events must point to a USR .events.log JSONL file"):
        resolve_usr_events_path(
            Path("/tmp/config.yaml"),
            validate_usr_event=lambda *_args, **_kwargs: None,
        )
