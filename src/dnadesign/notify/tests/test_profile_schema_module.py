"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_profile_schema_module.py

Module tests for notify profile schema parsing and validation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.profiles.schema import read_profile, resolve_profile_events_source, resolve_profile_webhook_source


def test_read_profile_valid_v2_returns_data(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": "/tmp/usr/.events.log",
                "webhook": {"source": "env", "ref": "NOTIFY_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    data = read_profile(profile_path)

    assert data["provider"] == "slack"
    assert data["events"] == "/tmp/usr/.events.log"


def test_read_profile_rejects_unknown_policy(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": "/tmp/usr/.events.log",
                "policy": "unknown",
                "webhook": {"source": "env", "ref": "NOTIFY_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NotifyConfigError, match="profile field 'policy' must be one of"):
        read_profile(profile_path)


def test_resolve_profile_webhook_source_env_returns_url_env() -> None:
    profile_data = {
        "profile_version": 2,
        "provider": "slack",
        "events": "/tmp/usr/.events.log",
        "webhook": {"source": "env", "ref": "NOTIFY_WEBHOOK"},
    }

    url_env, secret_ref = resolve_profile_webhook_source(profile_data)

    assert url_env == "NOTIFY_WEBHOOK"
    assert secret_ref is None


def test_resolve_profile_events_source_resolves_relative_config_path(tmp_path: Path) -> None:
    profile_path = tmp_path / "notify.profile.json"
    config_path = tmp_path / "configs" / "densegen.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen: {}\n", encoding="utf-8")

    profile_data = {
        "profile_version": 2,
        "provider": "slack",
        "events": "/tmp/usr/.events.log",
        "webhook": {"source": "env", "ref": "NOTIFY_WEBHOOK"},
        "events_source": {"tool": "densegen", "config": "configs/densegen.yaml"},
    }

    resolved = resolve_profile_events_source(profile_data=profile_data, profile_path=profile_path)

    assert resolved == ("densegen", config_path.resolve())
