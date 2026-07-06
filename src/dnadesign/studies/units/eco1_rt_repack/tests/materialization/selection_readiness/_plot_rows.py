"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_plot_rows.py

Shared plot-test rows for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)


def candidate_sequence_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "sequence": "A" * index + "C" * (8 - index),
            "canonical_mutations": [f"A{index}G", f"L{index + 20}V"],
        }
        for index in range(1, len(ALL_SPECS) + 1)
    ]


def selected_panel_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": spec.design_class_id,
        }
        for index, spec in enumerate(ALL_SPECS, start=1)
    ]


def selected_triage_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": spec.design_class_id,
            "catalytic_or_direct_contact_mutation_count": 0,
            "nucleic_acid_facing_mutation_count": index,
            "thumb_contact_track_mutation_count": 0,
            "distal_scaffold_mutation_count": 8 - index,
            "nucleic_acid_facing_chemistry_warning_count": index % 2,
            "hard_gate_status": "eligible",
            "fold_review_class": "strong_fold_preserved",
            "sae_window_status": "wt_like_not_used_for_selection",
            "nucleic_acid_facing_charge_delta": index - 3,
            "nucleic_acid_facing_basic_gain_count": index,
            "nucleic_acid_facing_basic_loss_count": 6 - index,
            "nucleic_acid_facing_acidic_gain_count": index % 2,
            "nucleic_acid_facing_proline_glycine_gain_count": index % 3,
        }
        for index, spec in enumerate(ALL_SPECS, start=1)
    ]
