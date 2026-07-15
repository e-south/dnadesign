"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_selection_table_assertions.py

Table-level assertions for Eco1 RT selection-readiness materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)


def assert_materialized_selection_tables(result: Any, inputs: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    triage = pq.read_table(result.candidate_triage_table_path).to_pylist()
    low_conf = next(row for row in triage if row["candidate_id"] == "candidate_low_conf")
    assert low_conf["hard_gate_status"] == "eligible"
    assert next(row for row in triage if row["candidate_id"] == "candidate_blocked_by_mask")["hard_gate_status"] == (
        "ineligible"
    )
    assert {row["sae_window_status"] for row in triage} == {"wt_like_not_used_for_selection"}
    _assert_triage_fields(triage)

    panel = pq.read_table(result.candidate_selection_panel_path).to_pylist()
    assert len(panel) == SELECTED_PANEL_SIZE
    assert [row["selection_rank"] for row in panel] == list(range(1, 9))
    assert [row["within_group_rank"] for row in panel] == [1, 2, 1, 2, 3, 1, 2, 3]
    assert [row["selection_slot"] for row in panel] == [
        "selected_distal_scaffold_repack_01",
        "selected_distal_scaffold_repack_02",
        "selected_peripheral_shell_repack_01",
        "selected_peripheral_shell_repack_02",
        "selected_peripheral_shell_repack_03",
        "selected_combined_peripheral_and_distal_repack_01",
        "selected_combined_peripheral_and_distal_repack_02",
        "selected_combined_peripheral_and_distal_repack_03",
    ]
    assert {row["fold_review_class"] for row in panel} == {"strong_fold_preserved"}
    assert all(row["eligible_for_handoff"] for row in panel)
    _assert_panel_fields(panel)
    return triage, panel


def _assert_triage_fields(triage: list[dict]) -> None:
    assert all(row["selection_support_alt_observed_fraction"] is not None for row in triage)
    assert all(row["selection_support_policy_id"] for row in triage)
    assert all(row["selection_support_policy_source"] for row in triage)
    assert all(row["nucleic_acid_facing_mutation_count"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_chemistry_warning_count"] is not None for row in triage)
    assert all(row["proximal_review_unobserved_mutation_count"] is not None for row in triage)
    assert all(row["proximal_review_rare_or_unobserved_mutation_count"] is not None for row in triage)
    assert all(row["local_structure_gate_status"] == "passed" for row in triage)
    assert all(row["local_structure_unavailable_region_count"] == 0 for row in triage)
    assert all(row["local_structure_threshold_failed_region_count"] == 0 for row in triage)
    assert all(row["local_structure_max_gated_ca_rmsd_angstrom"] is not None for row in triage)
    assert all(row["local_structure_max_all_region_ca_rmsd_angstrom"] is not None for row in triage)
    assert all("selection_candidate_tier" not in row for row in triage)
    assert all("proximal_hypothesis_candidate" not in row for row in triage)
    assert all(row["nucleic_acid_facing_acidic_gain_count"] == 0 for row in triage if row["selection_contract_pass"])
    assert all(
        row["proximal_review_unobserved_mutation_count"] == 0 for row in triage if row["selection_contract_pass"]
    )


def _assert_panel_fields(panel: list[dict]) -> None:
    assert {row["local_structure_gate_status"] for row in panel} == {"passed"}
    assert all(row["local_structure_threshold_failed_region_count"] == 0 for row in panel)
    assert all(row["catalytic_or_direct_contact_mutation_count"] == 0 for row in panel)
    assert all("selection_candidate_tier" not in row for row in panel)
    assert all("proximal_hypothesis_candidate" not in row for row in panel)
    assert all(row["selection_contract_pass"] for row in panel)
    assert {row["wang_alpha1_r13_review_status"] for row in panel} <= {"retained_wt", "substituted"}
    assert {row["near_retained_dna_rna_acidic_gain_review_status"] for row in panel} == {"passed"}
    assert {row["proximal_msa_support_review_status"] for row in panel} == {"passed"}
    for field in (
        "mutation_count_total",
        "canonical_mutations",
        "sequence_distance_to_wt",
        "mean_plddt",
        "wt_runtime_ca_rmsd",
        "cryoem_mapped_ca_rmsd",
        "thumb_contact_track_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "nucleic_acid_facing_mutation_count",
        "proximal_review_unobserved_mutation_count",
        "proximal_review_rare_or_unobserved_mutation_count",
        "selection_support_policy_id",
        "selection_support_policy_source",
        "policy_id",
        "selection_rank",
        "design_group_id",
        "within_group_rank",
        "wang_alpha1_r13_mutation_count",
        "wang_alpha1_r13_review_status",
        "wang_alpha1_mutation_count",
        "wang_alpha1_f10_substitution",
        "wang_alpha1_r13_substitution",
        "wang_alpha1_cross_protomer_contact_mutation_count",
        "wang_r13a_interface_disruption_evidence_match",
        "rt_msdna_oligomeric_state_review_status",
        "nucleic_acid_facing_basic_gain_count",
        "nucleic_acid_facing_basic_loss_count",
        "nucleic_acid_facing_proline_glycine_gain_count",
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom",
        "within_group_nearest_sequence_distance_aa",
        "within_group_nearest_exact_substitution_jaccard_distance",
        "within_group_nearest_mutated_position_jaccard_distance",
        "within_group_nearest_exact_substitution_shared_count",
        "within_group_nearest_mutated_position_shared_count",
    ):
        assert all(field in row for row in panel)
    assert all(row["nucleic_acid_facing_acidic_gain_count"] == 0 for row in panel)
    assert all(row["proximal_review_unobserved_mutation_count"] == 0 for row in panel)
    _assert_tie_break_trace(panel[0])


def _assert_tie_break_trace(panel_row: dict) -> None:
    assert "esmc_penalty_rank" not in panel_row
    assert "sae_window_contrast_rank" not in panel_row
    assert "first pair maximizes" in panel_row["selection_reason"]
    assert "experimental comparisons, not quality tiers" in panel_row["selection_reason"].lower()
    assert "not selection inputs" in panel_row["selection_reason"]
    trace_json = panel_row["tie_break_trace_json"]
    for expected in (
        "selection_support_alt_observed_fraction",
        "selection_support_policy_id",
        "selection_support_policy_source",
        "mutation_count_total",
        "distal_scaffold_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "local_structure_gate_status",
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom",
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom",
        "within_group_nearest_sequence_distance_aa",
        "within_group_nearest_exact_substitution_jaccard_distance",
        "within_group_nearest_mutated_position_jaccard_distance",
        "within_group_nearest_exact_substitution_shared_count",
        "within_group_nearest_mutated_position_shared_count",
        "wang_alpha1_f10_substitution",
        "wang_alpha1_r13_substitution",
        "wang_r13a_interface_disruption_evidence_match",
        "rt_msdna_oligomeric_state_review_status",
        "nucleic_acid_facing_basic_gain_count",
        "selection_contract_pass",
    ):
        assert expected in trace_json
    assert "selection_candidate_tier" not in trace_json
    assert "proximal_hypothesis_candidate" not in trace_json
    assert "esmc_6b_additive_llr_total" not in trace_json
    assert "class_local_elimination_policy_id" not in trace_json
    assert "local_structure_substrate_relevant_max" not in trace_json
