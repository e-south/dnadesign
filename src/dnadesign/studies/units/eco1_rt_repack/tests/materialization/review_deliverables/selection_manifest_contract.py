"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/selection_manifest_contract.py

Selection-readiness manifest constants for Eco1 review-deliverable fixtures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

SELECTION_FUNNEL_STAGES = [
    {
        "stage_id": "candidate_pool",
        "stage_label": "Complete ProteinMPNN sequences",
        "selector_role": "input_pool",
        "method": "Accepted complete ProteinMPNN sequences.",
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
    {
        "stage_id": "local_geometry_screen",
        "stage_label": "Local geometry retained",
        "selector_role": "hard_gate",
        "method": "Keep fold models at or below 2.5 A in every non-distal review region.",
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": True,
    },
    {
        "stage_id": "design_groups",
        "stage_label": "Distal, peripheral, and combined groups",
        "selector_role": "experimental_design",
        "method": "Keep each passing sequence in its ProteinMPNN generation group.",
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
    {
        "stage_id": "selected_panel",
        "stage_label": "Eight selected sequences",
        "selector_role": "within_group_mutation_set_selection",
        "method": (
            "Choose mutation-set-diverse rows within each policy by mutated-position Jaccard distance, then "
            "exact-substitution Jaccard distance; use chemistry, MSA, structure, and fold metrics as later ties."
        ),
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
]

PANEL_TIE_BREAK_ORDER = [
    "first pair: largest within-group mutated-position Jaccard distance",
    "first pair: largest within-group exact-substitution Jaccard distance",
    "third row: largest minimum mutated-position Jaccard distance from the within-group pair",
    "third row: largest minimum exact-substitution Jaccard distance from the within-group pair",
    "fewest basic losses near retained DNA/RNA",
    "fewest Pro/Gly gains near retained DNA/RNA",
    "region-wise MSA support",
    "local RMSD values inside declared gates",
    "fold metrics",
    "sequence hash",
]


def selection_trace_rows() -> list[dict[str, object]]:
    return [
        {
            "selection_policy_id": "eco1_rt_selected_panel_v3",
            "stage_order": index,
            "stage_id": stage["stage_id"],
            "stage_label": stage["stage_label"],
            "selector_role": stage["selector_role"],
            "method": stage["method"],
            "input_count": stage["input_count"],
            "removed_count": stage["removed_count"],
            "remaining_count": stage["remaining_count"],
            "is_hard_gate": stage["is_hard_gate"],
        }
        for index, stage in enumerate(SELECTION_FUNNEL_STAGES, start=1)
    ]
