"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_policies.py

Policy-family tests for the stress-study response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.policies import (
    audit_policy_specs,
)


def test_multiplicative_screen_has_no_exponent_scale_equivalents() -> None:
    policies = [policy for policy in audit_policy_specs() if policy.kind == "multiplicative"]
    tradeoff_weights = [round(policy.beta / (policy.beta + policy.gamma), 12) for policy in policies]

    assert len(tradeoff_weights) == len(set(tradeoff_weights))


def test_policy_ids_are_unique_and_stable_in_count() -> None:
    policies = audit_policy_specs()

    assert len(policies) == 28
    assert len({policy.id for policy in policies}) == len(policies)
