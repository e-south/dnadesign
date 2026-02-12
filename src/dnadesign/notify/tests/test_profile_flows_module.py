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

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.profile_flows import (
    resolve_profile_path_for_setup,
    resolve_profile_path_for_wizard,
    resolve_setup_events,
)


def test_resolve_profile_path_for_wizard_requires_policy_for_default_profile() -> None:
    with pytest.raises(NotifyConfigError, match="default profile path is ambiguous in wizard mode"):
        resolve_profile_path_for_wizard(profile=Path("outputs/notify/generic/profile.json"), policy=None)


def test_resolve_profile_path_for_setup_prefers_tool_namespace() -> None:
    resolved = resolve_profile_path_for_setup(
        profile=Path("outputs/notify/generic/profile.json"),
        tool_name="densegen",
        policy=None,
    )
    assert str(resolved) == "outputs/notify/densegen/profile.json"


def test_resolve_setup_events_rejects_mixed_events_and_resolver_modes(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    config = tmp_path / "tool.yaml"
    with pytest.raises(NotifyConfigError, match="pass either --events or --tool with --config, not both"):
        resolve_setup_events(
            events=events,
            tool="densegen",
            config=config,
            policy=None,
        )
