"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_selection_policy_context.py

Selection policy-context tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    DISTAL_SCAFFOLD_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    selection_policy_context,
)

CLADE9_PROFILE_ID = selection_policy_context.CLADE9_PROFILE_ID
resolve_selection_policy_context = selection_policy_context.resolve_selection_policy_context


def test_policy_context_uses_generation_policy_provenance() -> None:
    context = resolve_selection_policy_context(
        {
            "candidate_id": "candidate_with_policy",
            "primary_policy_id": DISTAL_SCAFFOLD_POLICY_ID,
            "policy_id": DISTAL_SCAFFOLD_POLICY_ID,
        }
    )

    assert context.policy_id == DISTAL_SCAFFOLD_POLICY_ID
    assert context.support_profile_id == CLADE9_PROFILE_ID
    assert context.source_field == "primary_policy_id"


def test_policy_context_rejects_design_class_only_rows() -> None:
    with pytest.raises(ValueError, match="generation policy"):
        resolve_selection_policy_context(
            {
                "candidate_id": "design_class_candidate",
                "design_class_id": "eco1_rt_clade9_plurality25_contact10a_v1",
            }
        )


def test_policy_context_rejects_unknown_generation_policy() -> None:
    with pytest.raises(ValueError, match="Unknown Eco1 generation policy"):
        resolve_selection_policy_context(
            {
                "candidate_id": "unknown_policy_candidate",
                "policy_id": "unknown_policy",
            }
        )
