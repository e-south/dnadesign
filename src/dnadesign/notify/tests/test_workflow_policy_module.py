"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_workflow_policy_module.py

Module tests for notify workflow-policy helpers and default profile namespacing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.workflow_policy import default_profile_path_for_tool, policy_defaults, resolve_workflow_policy


def test_resolve_workflow_policy_accepts_alias_and_known_policy() -> None:
    assert resolve_workflow_policy(policy="infer-evo2") == "infer_evo2"
    assert resolve_workflow_policy(policy="densegen") == "densegen"


def test_resolve_workflow_policy_rejects_unknown_policy() -> None:
    with pytest.raises(NotifyConfigError, match="unsupported policy"):
        resolve_workflow_policy(policy="mystery")


def test_default_profile_path_for_tool_namespaces_by_tool() -> None:
    assert str(default_profile_path_for_tool("densegen")) == "outputs/notify/densegen/profile.json"


def test_policy_defaults_returns_independent_copy() -> None:
    first = policy_defaults("densegen")
    second = policy_defaults("densegen")
    first["only_actions"] = "changed"
    assert second["only_actions"] == "densegen_health,densegen_flush_failed,materialize"
