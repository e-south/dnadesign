"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/selection_policy_context.py

Policy-context helpers for Eco1 RT selection-readiness rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    PRIMARY_POLICY_IDS,
)

CLADE9_PROFILE_ID = "ec86_clade9_conservation_v1"
SUBTYPE_PROFILE_ID = "ec86_iia3_cluster42_1_conservation_v1"

_PROFILE_BY_GENERATION_POLICY_ID = {policy_id: CLADE9_PROFILE_ID for policy_id in PRIMARY_POLICY_IDS}


@dataclass(frozen=True)
class SelectionPolicyContext:
    """Resolved policy context used for MSA-support review."""

    policy_id: str
    support_profile_id: str
    source_field: str


def resolve_selection_policy_context(candidate: Mapping[str, object]) -> SelectionPolicyContext:
    """Resolve the MSA denominator from generation-policy provenance."""

    candidate_id = str(candidate.get("candidate_id") or "unknown")
    primary_policy_id = _clean(candidate.get("primary_policy_id"))
    policy_id = _clean(candidate.get("policy_id"))
    generation_policy_id = primary_policy_id or policy_id
    if not generation_policy_id:
        raise ValueError(
            f"Candidate {candidate_id!r} is missing generation policy provenance; "
            "selection does not accept legacy design-class identifiers."
        )
    if generation_policy_id not in _PROFILE_BY_GENERATION_POLICY_ID:
        raise ValueError(
            f"Unknown Eco1 generation policy id for selection support: {generation_policy_id!r} "
            f"on candidate {candidate_id!r}"
        )
    return SelectionPolicyContext(
        policy_id=generation_policy_id,
        support_profile_id=_PROFILE_BY_GENERATION_POLICY_ID[generation_policy_id],
        source_field="primary_policy_id" if primary_policy_id else "policy_id",
    )


def _clean(value: object) -> str:
    return str(value or "").strip()


__all__ = [
    "CLADE9_PROFILE_ID",
    "SUBTYPE_PROFILE_ID",
    "SelectionPolicyContext",
    "resolve_selection_policy_context",
]
