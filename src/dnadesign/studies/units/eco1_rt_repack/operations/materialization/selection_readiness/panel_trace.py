"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/panel_trace.py

Reviewer-facing candidate flow for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTION_POLICY_ID,
)


def build_selected_panel_trace_rows(
    *,
    triage_rows: Sequence[dict[str, object]],
    panel_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return the row-reducing screen, design groups, and final selection."""

    all_rows = list(triage_rows)
    geometry_rows = [row for row in all_rows if str(row.get("hard_gate_status") or "") == "eligible"]
    contract_rows = [row for row in all_rows if bool(row.get("selection_contract_pass"))]
    if {str(row["candidate_id"]) for row in geometry_rows} != {str(row["candidate_id"]) for row in contract_rows}:
        raise ValueError(
            "The visible Eco1 selection flow assumes generation chemistry and support are invariants among "
            "local-geometry-pass rows."
        )
    policy_counts = Counter(str(row.get("policy_id") or "") for row in contract_rows)
    selected_policy_counts = Counter(str(row.get("policy_id") or "") for row in panel_rows)
    count_fields = {
        "distal_pool_count": policy_counts[DISTAL_SCAFFOLD_POLICY_ID],
        "peripheral_pool_count": policy_counts[NEAR_DNA_RNA_ACID_FREE_POLICY_ID],
        "combined_pool_count": policy_counts[COMBINED_NEAR_PLUS_DISTAL_POLICY_ID],
        "distal_selected_count": selected_policy_counts[DISTAL_SCAFFOLD_POLICY_ID],
        "peripheral_selected_count": selected_policy_counts[NEAR_DNA_RNA_ACID_FREE_POLICY_ID],
        "combined_selected_count": selected_policy_counts[COMBINED_NEAR_PLUS_DISTAL_POLICY_ID],
        "selected_count": len(panel_rows),
    }
    return [
        _trace_row(
            stage_order=1,
            stage_id="candidate_pool",
            stage_label="Complete ProteinMPNN sequences",
            selector_role="input_pool",
            method=("Accepted complete ProteinMPNN sequences generated under one declared policy per sequence."),
            input_count=len(all_rows),
            remaining_count=len(all_rows),
            is_hard_gate=False,
            **count_fields,
        ),
        _trace_row(
            stage_order=2,
            stage_id="local_geometry_screen",
            stage_label="Local geometry retained",
            selector_role="structural_screen",
            method=(
                "Keep sequences that retain fixed-position and generation-chemistry constraints and remain at or "
                "below the declared 2.5 A local C-alpha RMSD review cutoff in every non-distal region."
            ),
            input_count=len(all_rows),
            remaining_count=len(contract_rows),
            is_hard_gate=True,
            **count_fields,
        ),
        _trace_row(
            stage_order=3,
            stage_id="design_groups",
            stage_label="Distal, peripheral, and combined groups",
            selector_role="experimental_design",
            method=(
                "Keep each passing sequence in its ProteinMPNN generation group. The groups define different "
                "interventions, not quality tiers."
            ),
            input_count=len(contract_rows),
            remaining_count=len(contract_rows),
            is_hard_gate=False,
            **count_fields,
        ),
        _trace_row(
            stage_order=4,
            stage_id="selected_panel",
            stage_label="Eight selected sequences",
            selector_role="within_group_mutation_set_selection",
            method=(
                "Select two distal, three peripheral, and three combined sequences. Within each group, maximize "
                "mutated-position distance first and exact-substitution distance second; use chemistry, MSA, "
                "structure, fold metrics, and sequence hash only for later ties."
            ),
            input_count=len(contract_rows),
            remaining_count=len(panel_rows),
            is_hard_gate=False,
            **count_fields,
        ),
    ]


def _trace_row(
    *,
    stage_order: int,
    stage_id: str,
    stage_label: str,
    selector_role: str,
    method: str,
    input_count: int,
    remaining_count: int,
    is_hard_gate: bool,
    distal_pool_count: int = 0,
    peripheral_pool_count: int = 0,
    combined_pool_count: int = 0,
    distal_selected_count: int = 0,
    peripheral_selected_count: int = 0,
    combined_selected_count: int = 0,
    selected_count: int = 0,
) -> dict[str, object]:
    return {
        "selection_policy_id": SELECTION_POLICY_ID,
        "stage_order": stage_order,
        "stage_id": stage_id,
        "stage_label": stage_label,
        "selector_role": selector_role,
        "method": method,
        "input_count": input_count,
        "removed_count": max(input_count - remaining_count, 0),
        "remaining_count": remaining_count,
        "is_hard_gate": is_hard_gate,
        "distal_pool_count": distal_pool_count,
        "peripheral_pool_count": peripheral_pool_count,
        "combined_pool_count": combined_pool_count,
        "distal_selected_count": distal_selected_count,
        "peripheral_selected_count": peripheral_selected_count,
        "combined_selected_count": combined_selected_count,
        "selected_count": selected_count,
    }


__all__ = ["build_selected_panel_trace_rows"]
