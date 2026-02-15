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
from dnadesign.notify.workflow_policy import (
    default_profile_path_for_tool,
    policy_defaults,
    register_workflow_policy,
    resolve_workflow_policy,
)


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


def test_register_workflow_policy_supports_custom_policy_with_alias() -> None:
    register_workflow_policy(
        policy="custom_policy",
        defaults={
            "only_actions": "attach",
            "only_tools": "custom_tool",
        },
        aliases=("custom-policy",),
    )

    assert resolve_workflow_policy(policy="custom-policy") == "custom_policy"
    assert policy_defaults("custom_policy")["only_tools"] == "custom_tool"


def test_register_workflow_policy_rejects_duplicate_alias() -> None:
    register_workflow_policy(
        policy="alpha_policy",
        defaults={"only_actions": "materialize"},
        aliases=("alpha-policy",),
    )
    with pytest.raises(NotifyConfigError, match="alias 'alpha-policy' is already registered"):
        register_workflow_policy(
            policy="beta_policy",
            defaults={"only_actions": "attach"},
            aliases=("alpha-policy",),
        )
