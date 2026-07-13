"""Authoritative allocation contract for the selected Eco1 RT panel."""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)

SELECTION_POLICY_ID = "eco1_rt_selected_panel_v3"


@dataclass(frozen=True)
class GroupAllocation:
    """Number of sequences selected from one generation-policy group."""

    policy_id: str
    design_group_id: str
    selected_count: int


GROUP_ALLOCATIONS = (
    GroupAllocation(
        policy_id=DISTAL_SCAFFOLD_POLICY_ID,
        design_group_id="distal_scaffold_repack",
        selected_count=2,
    ),
    GroupAllocation(
        policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
        design_group_id="peripheral_shell_repack",
        selected_count=3,
    ),
    GroupAllocation(
        policy_id=COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
        design_group_id="combined_peripheral_and_distal_repack",
        selected_count=3,
    ),
)

_ALLOCATION_BY_POLICY = {allocation.policy_id: allocation for allocation in GROUP_ALLOCATIONS}
PANEL_POLICY_IDS = tuple(allocation.policy_id for allocation in GROUP_ALLOCATIONS)
SELECTED_PANEL_SIZE = sum(allocation.selected_count for allocation in GROUP_ALLOCATIONS)
EXPECTED_SELECTED_POLICY_COUNTS = {allocation.policy_id: allocation.selected_count for allocation in GROUP_ALLOCATIONS}


def allocation_for_policy(policy_id: str) -> GroupAllocation:
    """Return the declared selected-panel allocation for one policy."""

    try:
        return _ALLOCATION_BY_POLICY[policy_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported selected-panel policy id: {policy_id!r}") from exc


__all__ = [
    "EXPECTED_SELECTED_POLICY_COUNTS",
    "GROUP_ALLOCATIONS",
    "PANEL_POLICY_IDS",
    "SELECTED_PANEL_SIZE",
    "SELECTION_POLICY_ID",
    "GroupAllocation",
    "allocation_for_policy",
]
