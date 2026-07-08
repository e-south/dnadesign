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
        "stage_label": "Accepted candidate pool",
        "selector_role": "input_pool",
        "filter_rule": "Accepted ProteinMPNN candidate rows before protein-level selection checks.",
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
    {
        "stage_id": "preservation_gate",
        "stage_label": "Preservation gate",
        "selector_role": "hard_gate",
        "filter_rule": "Keep rows passing protein preservation checks.",
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": True,
    },
    {
        "stage_id": "chemistry_support_gate",
        "stage_label": "Chemistry and support gate",
        "selector_role": "hard_gate",
        "filter_rule": (
            "Keep rows with zero acidic gains near retained DNA/RNA and zero unobserved proximal substitutions."
        ),
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
    {
        "stage_id": "global_conservative_diverse_selection",
        "stage_label": "Conservative-diverse six-row selection",
        "selector_role": "global_rank",
        "filter_rule": (
            "Select primary-panel candidates globally by conservative rank fields and mutation-set "
            "dissimilarity; design class is context, not a quota."
        ),
        "input_count": 2,
        "removed_count": 0,
        "remaining_count": 2,
        "is_hard_gate": False,
    },
]

PANEL_TIE_BREAK_ORDER = [
    "fewest proximal unsupported substitutions",
    "fewest acidic gains near retained DNA/RNA or thumb-track",
    "fewest basic losses near retained DNA/RNA or thumb-track",
    "fewest Pro/Gly gains near retained DNA/RNA or thumb-track",
    "largest nearest selected mutation-position Jaccard distance",
    "largest nearest selected exact-substitution Jaccard distance",
    "lowest C-terminal primer-RNA recognition-region C-alpha RMSD",
    "lowest substrate-relevant local C-alpha RMSD",
    "fold metrics",
    "sequence hash",
]


def selection_trace_rows() -> list[dict[str, object]]:
    return [
        {
            "selection_policy_id": "eco1_rt_primary_conservative_panel_v1",
            "stage_order": index,
            "stage_id": stage["stage_id"],
            "stage_label": stage["stage_label"],
            "selector_role": stage["selector_role"],
            "filter_rule": "Fixture primary-panel funnel stage.",
            "input_count": stage["input_count"],
            "removed_count": stage["removed_count"],
            "remaining_count": stage["remaining_count"],
            "is_hard_gate": stage["is_hard_gate"],
        }
        for index, stage in enumerate(SELECTION_FUNNEL_STAGES, start=1)
    ]
