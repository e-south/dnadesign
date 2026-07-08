"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/selection_table_fixtures.py

Panel-selection table row fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def panel_row(
    *,
    slot: str,
    design_class_id: str,
    candidate_id: str,
    mutation_count: int,
    msa_fraction: float,
    na_facing: int,
    chemistry_warnings: int,
) -> dict[str, object]:
    trace_json = (
        f'{{"selection_support_alt_observed_fraction": {msa_fraction}, '
        f'"selection_support_unobserved_mutation_count": 1, '
        f'"mutation_count_total": {mutation_count}, '
        f'"mean_plddt": 92.4, '
        f'"wt_runtime_ca_rmsd": 0.82, '
        f'"cryoem_mapped_ca_rmsd": 1.23, '
        f'"nucleic_acid_facing_mutation_count": {na_facing}, '
        f'"nucleic_acid_facing_charge_delta": 1, '
        f'"nucleic_acid_facing_chemistry_warning_count": {chemistry_warnings}, '
        f'"selection_candidate_tier": "primary_panel_candidate", '
        f'"primary_panel_candidate": true, '
        f'"catalytic_or_direct_contact_mutation_count": 0, '
        f'"thumb_contact_track_mutation_count": 0, '
        f'"c_terminal_primer_rna_recognition_mutation_count": 1, '
        f'"distal_scaffold_mutation_count": {mutation_count}}}'
    )
    return {
        "selection_slot": slot,
        "candidate_id": candidate_id,
        "design_class_id": design_class_id,
        "slot_rank": int(slot.rsplit("_", 1)[-1]),
        "selection_candidate_tier": "primary_panel_candidate",
        "primary_panel_candidate": True,
        "fold_review_class": "strong_fold_preserved",
        "feasibility_status": "pass",
        "nearest_selected_distance_aa": 4,
        "selection_reason": "fixture selected panel row",
        "tie_break_trace_json": trace_json,
    }


def triage_row(*, candidate_id: str, msa_fraction: float, charge_delta: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "selection_support_alt_observed_fraction": msa_fraction,
        "selection_support_unobserved_mutation_count": 1,
        "nucleic_acid_facing_mutation_count": 2,
        "nucleic_acid_facing_charge_delta": charge_delta,
        "nucleic_acid_facing_chemistry_warning_count": 1,
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 1,
        "distal_scaffold_mutation_count": 2,
        "hard_gate_status": "eligible",
        "selection_candidate_tier": "primary_panel_candidate",
        "primary_panel_candidate": True,
        "near_retained_dna_rna_acidic_gain_review_status": "passed",
        "proximal_msa_support_review_status": "passed",
        "sae_window_status": "wt_like_not_used_for_selection",
    }


__all__ = ["panel_row", "triage_row"]
