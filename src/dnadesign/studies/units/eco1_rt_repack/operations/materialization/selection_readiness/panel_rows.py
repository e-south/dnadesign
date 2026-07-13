"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/panel_rows.py

Selected-panel row construction for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTION_POLICY_ID,
)


def build_panel_row(
    row: dict[str, object],
    *,
    within_group_nearest_sequence_distance: int | None,
    input_hashes: dict[str, str | None],
    selection_rank: int,
    design_group_id: str,
    within_group_rank: int,
) -> dict[str, object]:
    """Return one selected-panel row with trace fields kept in one schema owner."""

    reason = (
        "Selected after the declared fixed-position, generation-chemistry, and local-geometry checks. Within "
        "each generation policy, the first pair maximizes mutated-position Jaccard distance and then exact-"
        "substitution distance. A third row, where allocated, maximizes its minimum distance from that pair. "
        "Chemistry counts, regional MSA support, local RMSD, fold metrics, and sequence hash are later tie-breaks. "
        "Policy groups define experimental comparisons, not quality tiers. ESMC and SAE "
        "are not selection inputs. R13 and other alpha-1 substitutions are reported but do not rank rows."
    )
    trace = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "selection_contract_pass": row.get("selection_contract_pass"),
        "selection_contract_failure_reasons_json": row.get("selection_contract_failure_reasons_json"),
        "policy_id": row.get("policy_id"),
        "design_group_id": design_group_id,
        "within_group_rank": within_group_rank,
        "selection_rank": selection_rank,
        "primary_policy_id": row.get("primary_policy_id"),
        "selection_support_policy_id": row.get("selection_support_policy_id"),
        "selection_support_policy_source": row.get("selection_support_policy_source"),
        "source_policy_ids": row.get("source_policy_ids") or [],
        "proximal_review_unobserved_mutation_count": row.get("proximal_review_unobserved_mutation_count"),
        "proximal_review_rare_or_unobserved_mutation_count": row.get(
            "proximal_review_rare_or_unobserved_mutation_count"
        ),
        "wang_alpha1_r13_mutation_count": row.get("wang_alpha1_r13_mutation_count"),
        "wang_alpha1_r13_review_status": row.get("wang_alpha1_r13_review_status"),
        "wang_alpha1_mutation_count": row.get("wang_alpha1_mutation_count"),
        "selection_support_profile_id": row["selection_support_profile_id"],
        "selection_support_alt_observed_fraction": row["selection_support_alt_observed_fraction"],
        "selection_support_alt_frequency_mean": row["selection_support_alt_frequency_mean"],
        "selection_support_unobserved_mutation_count": row["selection_support_unobserved_mutation_count"],
        "mutation_count_total": row["mutation_count_total"],
        "canonical_mutations": row.get("canonical_mutations") or [],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "near_retained_dna_rna_acidic_gain_review_status": row.get("near_retained_dna_rna_acidic_gain_review_status"),
        "proximal_msa_support_review_status": row.get("proximal_msa_support_review_status"),
        "nucleic_acid_facing_charge_delta": row["nucleic_acid_facing_charge_delta"],
        "nucleic_acid_facing_basic_gain_count": row["nucleic_acid_facing_basic_gain_count"],
        "nucleic_acid_facing_acidic_gain_count": row["nucleic_acid_facing_acidic_gain_count"],
        "nucleic_acid_facing_basic_loss_count": row["nucleic_acid_facing_basic_loss_count"],
        "nucleic_acid_facing_proline_glycine_gain_count": row["nucleic_acid_facing_proline_glycine_gain_count"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "c_terminal_primer_rna_recognition_mutation_count": row["c_terminal_primer_rna_recognition_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "within_group_nearest_sequence_distance_aa": within_group_nearest_sequence_distance,
        "within_group_nearest_exact_substitution_jaccard_distance": row.get(
            "within_group_nearest_exact_substitution_jaccard_distance"
        ),
        "within_group_nearest_mutated_position_jaccard_distance": row.get(
            "within_group_nearest_mutated_position_jaccard_distance"
        ),
        "within_group_nearest_exact_substitution_shared_count": row.get(
            "within_group_nearest_exact_substitution_shared_count"
        ),
        "within_group_nearest_mutated_position_shared_count": row.get(
            "within_group_nearest_mutated_position_shared_count"
        ),
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_max_gated_ca_rmsd_angstrom": row["local_structure_max_gated_ca_rmsd_angstrom"],
        "local_structure_max_all_region_ca_rmsd_angstrom": row["local_structure_max_all_region_ca_rmsd_angstrom"],
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": row[
            "local_structure_catalytic_initiation_context_ca_rmsd_angstrom"
        ],
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": row[
            "local_structure_thumb_contact_track_context_ca_rmsd_angstrom"
        ],
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": row[
            "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom"
        ],
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": row[
            "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom"
        ],
        "fold_review_class": row["fold_review_class"],
        "mean_plddt": row["mean_plddt"],
        "wt_runtime_ca_rmsd": row["wt_runtime_ca_rmsd"],
        "cryoem_mapped_ca_rmsd": row["cryoem_mapped_ca_rmsd"],
        "sae_window_status": row["sae_window_status"],
    }
    return {
        "candidate_id": row["candidate_id"],
        "sequence_hash": row["sequence_hash"],
        "policy_id": row.get("policy_id"),
        "primary_policy_id": row.get("primary_policy_id"),
        "selection_support_policy_id": row.get("selection_support_policy_id"),
        "selection_support_policy_source": row.get("selection_support_policy_source"),
        "source_policy_ids": row.get("source_policy_ids") or [],
        "eligible_for_handoff": True,
        "selection_slot": f"selected_{design_group_id}_{within_group_rank:02d}",
        "selection_rank": selection_rank,
        "design_group_id": design_group_id,
        "within_group_rank": within_group_rank,
        "selection_reason": reason,
        "tie_break_trace_json": json.dumps(trace, sort_keys=True),
        "mutation_count_total": row["mutation_count_total"],
        "canonical_mutations": row.get("canonical_mutations") or [],
        "sequence_distance_to_wt": row.get("sequence_distance_to_wt", row["mutation_count_total"]),
        "within_group_nearest_sequence_distance_aa": within_group_nearest_sequence_distance,
        "fold_review_class": row["fold_review_class"],
        "mean_plddt": row["mean_plddt"],
        "wt_runtime_ca_rmsd": row["wt_runtime_ca_rmsd"],
        "cryoem_mapped_ca_rmsd": row["cryoem_mapped_ca_rmsd"],
        "hard_gate_status": row["hard_gate_status"],
        "selection_contract_pass": bool(row.get("selection_contract_pass")),
        "selection_contract_failure_reasons_json": row.get("selection_contract_failure_reasons_json"),
        "wang_alpha1_r13_mutation_count": row.get("wang_alpha1_r13_mutation_count"),
        "wang_alpha1_r13_review_status": row.get("wang_alpha1_r13_review_status"),
        "wang_alpha1_mutation_count": row.get("wang_alpha1_mutation_count"),
        "near_retained_dna_rna_acidic_gain_review_status": row.get("near_retained_dna_rna_acidic_gain_review_status"),
        "proximal_msa_support_review_status": row.get("proximal_msa_support_review_status"),
        "selection_support_alt_observed_fraction": row.get("selection_support_alt_observed_fraction"),
        "selection_support_alt_frequency_mean": row.get("selection_support_alt_frequency_mean"),
        "within_group_nearest_exact_substitution_jaccard_distance": row.get(
            "within_group_nearest_exact_substitution_jaccard_distance"
        ),
        "within_group_nearest_mutated_position_jaccard_distance": row.get(
            "within_group_nearest_mutated_position_jaccard_distance"
        ),
        "within_group_nearest_exact_substitution_shared_count": row.get(
            "within_group_nearest_exact_substitution_shared_count"
        ),
        "within_group_nearest_mutated_position_shared_count": row.get(
            "within_group_nearest_mutated_position_shared_count"
        ),
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_threshold_policy_id": row["local_structure_threshold_policy_id"],
        "local_structure_threshold_failed_region_count": row["local_structure_threshold_failed_region_count"],
        "local_structure_max_gated_ca_rmsd_angstrom": row["local_structure_max_gated_ca_rmsd_angstrom"],
        "local_structure_max_all_region_ca_rmsd_angstrom": row["local_structure_max_all_region_ca_rmsd_angstrom"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "c_terminal_primer_rna_recognition_mutation_count": row["c_terminal_primer_rna_recognition_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_charge_delta": row.get("nucleic_acid_facing_charge_delta"),
        "nucleic_acid_facing_basic_gain_count": row.get("nucleic_acid_facing_basic_gain_count"),
        "nucleic_acid_facing_acidic_gain_count": row.get("nucleic_acid_facing_acidic_gain_count"),
        "nucleic_acid_facing_basic_loss_count": row.get("nucleic_acid_facing_basic_loss_count"),
        "nucleic_acid_facing_proline_glycine_gain_count": row.get("nucleic_acid_facing_proline_glycine_gain_count"),
        "proximal_review_unobserved_mutation_count": row.get("proximal_review_unobserved_mutation_count"),
        "proximal_review_rare_or_unobserved_mutation_count": row.get(
            "proximal_review_rare_or_unobserved_mutation_count"
        ),
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": row[
            "local_structure_thumb_contact_track_context_ca_rmsd_angstrom"
        ],
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": row[
            "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom"
        ],
        "input_candidate_triage_table_hash": input_hashes["candidate_triage_table"],
        "input_foldcheck_review_hash": input_hashes["foldcheck_review"],
        "input_sae_window_summary_hash": input_hashes.get("sae_window_summary"),
    }
