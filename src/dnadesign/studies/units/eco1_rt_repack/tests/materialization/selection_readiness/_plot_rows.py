"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_plot_rows.py

Shared plot-test rows for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    PRIMARY_POLICY_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)


def candidate_sequence_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "sequence": "A" * index + "C" * (8 - index),
            "canonical_mutations": [f"A{index}G", f"L{index + 20}V"],
        }
        for index in range(1, SELECTED_PANEL_SIZE + 1)
    ]


def selected_panel_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "policy_id": policy_id,
            "selection_slot": f"selected_hypothesis_{index:02d}",
            "selection_rank": index,
        }
        for index, policy_id in enumerate(_panel_policy_ids(), start=1)
    ]


def selected_triage_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "policy_id": policy_id,
            "catalytic_or_direct_contact_mutation_count": 0,
            "nucleic_acid_facing_mutation_count": index,
            "thumb_contact_track_mutation_count": 0,
            "distal_scaffold_mutation_count": 8 - index,
            "nucleic_acid_facing_chemistry_warning_count": index % 2,
            "hard_gate_status": "eligible",
            "fold_review_class": "strong_fold_preserved",
            "local_structure_gate_status": "passed",
            "local_structure_max_gated_ca_rmsd_angstrom": 1.25,
            "local_structure_max_all_region_ca_rmsd_angstrom": 1.25,
            "sae_window_status": "wt_like_not_used_for_selection",
            "nucleic_acid_facing_charge_delta": index - 3,
            "nucleic_acid_facing_basic_gain_count": index,
            "nucleic_acid_facing_basic_loss_count": 6 - index,
            "nucleic_acid_facing_acidic_gain_count": index % 2,
            "nucleic_acid_facing_proline_glycine_gain_count": index % 3,
        }
        for index, policy_id in enumerate(_panel_policy_ids(), start=1)
    ]


def _panel_policy_ids() -> list[str]:
    return [PRIMARY_POLICY_IDS[index % len(PRIMARY_POLICY_IDS)] for index in range(SELECTED_PANEL_SIZE)]
