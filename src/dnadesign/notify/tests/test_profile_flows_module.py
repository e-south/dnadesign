"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_profile_flows_module.py

Tests for notify profile/setup flow orchestration helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.notify.profiles.flows as profile_flows_module
from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.profiles.flows import (
    resolve_profile_path_for_setup,
    resolve_profile_path_for_wizard,
    resolve_setup_events,
    resolve_webhook_config,
)


def test_resolve_profile_path_for_wizard_requires_policy_for_default_profile() -> None:
    with pytest.raises(NotifyConfigError, match="default profile path is ambiguous in wizard mode"):
        resolve_profile_path_for_wizard(profile=Path("outputs/notify/generic/profile.json"), policy=None)


def test_resolve_profile_path_for_setup_prefers_tool_namespace() -> None:
    resolved = resolve_profile_path_for_setup(
        profile=Path("outputs/notify/generic/profile.json"),
        tool_name="densegen",
        policy=None,
        config=None,
    )
    assert str(resolved) == "outputs/notify/densegen/profile.json"


def test_resolve_profile_path_for_setup_anchors_default_to_config_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "workspaces" / "demo" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    resolved = resolve_profile_path_for_setup(
        profile=Path("outputs/notify/generic/profile.json"),
        tool_name="densegen",
        policy=None,
        config=config_path,
    )
    assert resolved == config_path.parent / "outputs" / "notify" / "densegen" / "profile.json"


def test_resolve_setup_events_rejects_mixed_events_and_resolver_modes(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    config = tmp_path / "tool.yaml"
    with pytest.raises(NotifyConfigError, match="pass either --events or --tool with --config/--workspace, not both"):
        resolve_setup_events(
            events=events,
            tool="densegen",
            config=config,
            workspace=None,
            policy=None,
        )


def test_resolve_setup_events_workspace_mode_resolves_config_before_events(tmp_path: Path) -> None:
    resolved_config = tmp_path / "workspaces" / "demo" / "config.yaml"
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    resolved_config.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"

    result = resolve_setup_events(
        events=None,
        tool="densegen",
        config=None,
        workspace="demo",
        policy=None,
        resolve_tool_workspace_config_fn=lambda *, tool, workspace, search_start: resolved_config,
        resolve_tool_events_path_fn=lambda *, tool, config: (resolved_events, "densegen"),
        normalize_tool_name_fn=lambda value: str(value),
    )

    assert result.events_path == resolved_events
    assert result.events_source == {"tool": "densegen", "config": str(resolved_config.resolve())}
    assert result.policy == "densegen"
    assert result.tool_name == "densegen"
    assert result.events_require_exists is False


def test_resolve_webhook_config_file_mode_defaults_to_notify_package_secret_dir() -> None:
    webhook = resolve_webhook_config(
        secret_source="file",  # pragma: allowlist secret
        url_env=None,
        secret_ref=None,
        webhook_url=None,
        store_webhook=False,
        secret_name="densegen-shared",  # pragma: allowlist secret
        secret_backend_available_fn=lambda backend: backend == "file",
    )
    expected = (
        Path(profile_flows_module.__file__).resolve().parents[1] / ".secrets" / "densegen-shared.webhook"
    ).resolve()
    assert webhook == {"source": "secret_ref", "ref": expected.as_uri()}


def test_resolve_webhook_config_auto_file_mode_defaults_to_notify_package_secret_dir() -> None:
    webhook = resolve_webhook_config(
        secret_source="auto",  # pragma: allowlist secret
        url_env=None,
        secret_ref=None,
        webhook_url=None,
        store_webhook=False,  # pragma: allowlist secret
        secret_name="densegen-shared",  # pragma: allowlist secret
        secret_backend_available_fn=lambda backend: backend == "file",
    )
    expected = (
        Path(profile_flows_module.__file__).resolve().parents[1] / ".secrets" / "densegen-shared.webhook"
    ).resolve()
    assert webhook == {"source": "secret_ref", "ref": expected.as_uri()}
