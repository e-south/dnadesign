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
    policy_id: str,
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
        f'"nucleic_acid_facing_basic_gain_count": 2, '
        f'"nucleic_acid_facing_basic_loss_count": 1, '
        f'"nucleic_acid_facing_acidic_gain_count": 0, '
        f'"nucleic_acid_facing_chemistry_warning_count": {chemistry_warnings}, '
        f'"selection_contract_pass": true, '
        f'"catalytic_or_direct_contact_mutation_count": 0, '
        f'"thumb_contact_track_mutation_count": 0, '
        f'"c_terminal_primer_rna_recognition_mutation_count": 1, '
        f'"distal_scaffold_mutation_count": {mutation_count}}}'
    )
    return {
        "selection_slot": slot,
        "candidate_id": candidate_id,
        "policy_id": policy_id,
        "design_group_id": "peripheral_shell_repack",
        "within_group_rank": int(slot.rsplit("_", 1)[-1]),
        "selection_rank": int(slot.rsplit("_", 1)[-1]),
        "selection_contract_pass": True,
        "fold_review_class": "strong_fold_preserved",
        "mutation_count_total": mutation_count,
        "canonical_mutations": [f"A{position}G" for position in range(1, mutation_count + 1)],
        "mean_plddt": 92.4,
        "local_structure_max_gated_ca_rmsd_angstrom": 1.8,
        "within_group_nearest_mutated_position_jaccard_distance": 0.4,
        "within_group_nearest_exact_substitution_jaccard_distance": 0.7,
        "selection_support_alt_observed_fraction": msa_fraction,
        "nucleic_acid_facing_mutation_count": na_facing,
        "nucleic_acid_facing_charge_delta": 1,
        "nucleic_acid_facing_basic_gain_count": 2,
        "nucleic_acid_facing_basic_loss_count": 1,
        "nucleic_acid_facing_acidic_gain_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 1,
        "wang_alpha1_r13_mutation_count": 0,
        "wang_alpha1_r13_review_status": "retained_wt",
        "wang_alpha1_mutation_count": 2,
        "wang_alpha1_f10_substitution": "WT",
        "wang_alpha1_r13_substitution": "WT",
        "wang_alpha1_cross_protomer_contact_mutation_count": 0,
        "wang_r13a_interface_disruption_evidence_match": False,
        "rt_msdna_oligomeric_state_review_status": "not_established",
        "within_group_nearest_sequence_distance_aa": 4,
        "selection_reason": "fixture selected panel row",
        "tie_break_trace_json": trace_json,
    }


def triage_row(
    *,
    candidate_id: str,
    msa_fraction: float,
    charge_delta: int,
    mutation_count: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "policy_id": "near_dna_rna_acid_free_v1",
        "mutation_count_total": mutation_count,
        "sequence_distance_to_wt": mutation_count,
        "mean_plddt": 92.4,
        "local_structure_max_gated_ca_rmsd_angstrom": 1.8,
        "selection_support_alt_observed_fraction": msa_fraction,
        "selection_support_unobserved_mutation_count": 1,
        "nucleic_acid_facing_mutation_count": 2,
        "nucleic_acid_facing_charge_delta": charge_delta,
        "nucleic_acid_facing_chemistry_warning_count": 1,
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 1,
        "distal_scaffold_mutation_count": 2,
        "local_structure_gate_status": "passed",
        "hard_gate_status": "eligible",
        "wang_alpha1_r13_mutation_count": 0,
        "wang_alpha1_r13_review_status": "retained_wt",
        "wang_alpha1_f10_substitution": "WT",
        "wang_alpha1_r13_substitution": "WT",
        "wang_alpha1_cross_protomer_contact_mutation_count": 0,
        "wang_r13a_interface_disruption_evidence_match": False,
        "rt_msdna_oligomeric_state_review_status": "not_established",
        "selection_contract_pass": True,
        "near_retained_dna_rna_acidic_gain_review_status": "passed",
        "proximal_msa_support_review_status": "passed",
        "sae_window_status": "wt_like_not_used_for_selection",
    }


__all__ = ["panel_row", "triage_row"]
