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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    SELECTION_POLICY_ID,
)

NA_FACING_LOW_BURDEN_RATIO = 0.05
NA_FACING_HIGH_BURDEN_RATIO = 0.75


def build_panel_row(
    row: dict[str, object],
    *,
    nearest_distance: int | None,
    input_hashes: dict[str, str | None],
    slot_rank: int,
) -> dict[str, object]:
    """Return one selected-panel row with trace fields kept in one schema owner."""

    reason = (
        "Selected for the primary conservative panel after preservation gates and a chemistry/support gate: no "
        "acidic gains near retained DNA/RNA and no unobserved proximal substitutions. The final panel is selected "
        "globally using simple mutation-set dissimilarity, near retained DNA/RNA basic-loss and Pro/Gly penalties, "
        "regional MSA support, local RMSD values inside the gate, fold metrics, and a deterministic tie-break. "
        "Design classes remain review context rather than quotas. ESMC and SAE rows were retained for review but "
        "not used for selection."
    )
    na_facing_count, na_facing_ratio = _na_facing_count_and_ratio(row)
    trace = {
        "selection_policy_id": SELECTION_POLICY_ID,
        "selection_candidate_tier": row.get("selection_candidate_tier"),
        "primary_panel_candidate": row.get("primary_panel_candidate"),
        "primary_panel_failure_reasons_json": row.get("primary_panel_failure_reasons_json"),
        "design_class_id": row["design_class_id"],
        "proximal_review_unobserved_mutation_count": row.get("proximal_review_unobserved_mutation_count"),
        "proximal_review_rare_or_unobserved_mutation_count": row.get(
            "proximal_review_rare_or_unobserved_mutation_count"
        ),
        "selection_support_profile_id": row["selection_support_profile_id"],
        "selection_support_alt_observed_fraction": row["selection_support_alt_observed_fraction"],
        "selection_support_alt_frequency_mean": row["selection_support_alt_frequency_mean"],
        "selection_support_unobserved_mutation_count": row["selection_support_unobserved_mutation_count"],
        "mutation_count_total": row["mutation_count_total"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "nucleic_acid_facing_burden_ratio": na_facing_ratio,
        "nucleic_acid_facing_burden_band": _na_facing_burden_band(na_facing_count, na_facing_ratio),
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_chemistry_compatible": row.get("nucleic_acid_facing_chemistry_compatible"),
        "nucleic_acid_facing_chemistry_gate_status": row.get("nucleic_acid_facing_chemistry_gate_status"),
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
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_max_ca_rmsd_angstrom": row["local_structure_max_ca_rmsd_angstrom"],
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
        "nearest_selected_distance_aa": nearest_distance,
        "nearest_selected_mutation_token_jaccard_distance": row.get("nearest_selected_mutation_token_jaccard_distance"),
        "nearest_selected_mutation_position_jaccard_distance": row.get(
            "nearest_selected_mutation_position_jaccard_distance"
        ),
        "nearest_selected_mutation_token_shared_count": row.get("nearest_selected_mutation_token_shared_count"),
        "nearest_selected_mutation_position_shared_count": row.get("nearest_selected_mutation_position_shared_count"),
        "fold_review_class": row["fold_review_class"],
        "mean_plddt": row["mean_plddt"],
        "wt_runtime_ca_rmsd": row["wt_runtime_ca_rmsd"],
        "cryoem_mapped_ca_rmsd": row["cryoem_mapped_ca_rmsd"],
        "sae_window_status": row["sae_window_status"],
    }
    return {
        "candidate_id": row["candidate_id"],
        "sequence_hash": row["sequence_hash"],
        "design_class_id": row["design_class_id"],
        "eligible_for_handoff": True,
        "selection_slot": f"primary_panel_{slot_rank:02d}",
        "slot_rank": slot_rank,
        "selected_for_panel": True,
        "selection_reason": reason,
        "tie_break_trace_json": json.dumps(trace, sort_keys=True),
        "mutation_count_total": row["mutation_count_total"],
        "sequence_distance_to_wt": row.get("sequence_distance_to_wt", row["mutation_count_total"]),
        "nearest_selected_distance_aa": nearest_distance,
        "fold_review_class": row["fold_review_class"],
        "feasibility_status": row["feasibility_status"],
        "hard_gate_status": row["hard_gate_status"],
        "primary_panel_candidate": bool(row.get("primary_panel_candidate")),
        "selection_candidate_tier": str(row.get("selection_candidate_tier") or ""),
        "primary_panel_failure_reasons_json": row.get("primary_panel_failure_reasons_json"),
        "near_retained_dna_rna_acidic_gain_review_status": row.get("near_retained_dna_rna_acidic_gain_review_status"),
        "proximal_msa_support_review_status": row.get("proximal_msa_support_review_status"),
        "selection_support_alt_observed_fraction": row.get("selection_support_alt_observed_fraction"),
        "selection_support_alt_frequency_mean": row.get("selection_support_alt_frequency_mean"),
        "nearest_selected_mutation_token_jaccard_distance": row.get("nearest_selected_mutation_token_jaccard_distance"),
        "nearest_selected_mutation_position_jaccard_distance": row.get(
            "nearest_selected_mutation_position_jaccard_distance"
        ),
        "nearest_selected_mutation_token_shared_count": row.get("nearest_selected_mutation_token_shared_count"),
        "nearest_selected_mutation_position_shared_count": row.get("nearest_selected_mutation_position_shared_count"),
        "local_structure_gate_status": row["local_structure_gate_status"],
        "local_structure_threshold_policy_id": row["local_structure_threshold_policy_id"],
        "local_structure_threshold_failed_region_count": row["local_structure_threshold_failed_region_count"],
        "local_structure_max_ca_rmsd_angstrom": row["local_structure_max_ca_rmsd_angstrom"],
        "catalytic_or_direct_contact_mutation_count": row["catalytic_or_direct_contact_mutation_count"],
        "nucleic_acid_facing_mutation_count": row["nucleic_acid_facing_mutation_count"],
        "thumb_contact_track_mutation_count": row["thumb_contact_track_mutation_count"],
        "c_terminal_primer_rna_recognition_mutation_count": row["c_terminal_primer_rna_recognition_mutation_count"],
        "distal_scaffold_mutation_count": row["distal_scaffold_mutation_count"],
        "nucleic_acid_facing_chemistry_warning_count": row["nucleic_acid_facing_chemistry_warning_count"],
        "nucleic_acid_facing_chemistry_compatible": row.get("nucleic_acid_facing_chemistry_compatible"),
        "nucleic_acid_facing_chemistry_gate_status": row.get("nucleic_acid_facing_chemistry_gate_status"),
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
        "input_feasibility_report_hash": input_hashes["feasibility_report"],
        "input_sae_window_summary_hash": input_hashes.get("sae_window_summary"),
    }


def _na_facing_count_and_ratio(row: dict[str, object]) -> tuple[int, float]:
    count = int(row.get("nucleic_acid_facing_mutation_count") or 0)
    total = max(int(row.get("mutation_count_total") or 0), 1)
    return count, count / total


def _na_facing_burden_band(count: int, ratio: float) -> str:
    if count == 0:
        return "none"
    if ratio < NA_FACING_LOW_BURDEN_RATIO:
        return "low"
    if ratio <= NA_FACING_HIGH_BURDEN_RATIO:
        return "moderate"
    return "high"
