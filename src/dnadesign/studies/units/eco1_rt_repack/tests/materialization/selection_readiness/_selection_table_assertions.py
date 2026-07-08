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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)


def assert_materialized_selection_tables(result: Any, inputs: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    feasibility = pq.read_table(result.feasibility_report_path).to_pylist()
    assert {row["candidate_id"] for row in feasibility} == {row["candidate_id"] for row in inputs["candidate_pool"]}
    blocked = next(row for row in feasibility if row["candidate_id"] == "candidate_blocked_by_mask")
    assert blocked["feasibility_status"] == "blocked"
    assert blocked["protected_mutation_violation_count"] == 1

    triage = pq.read_table(result.candidate_triage_table_path).to_pylist()
    low_conf = next(row for row in triage if row["candidate_id"] == "candidate_low_conf")
    assert low_conf["hard_gate_status"] == "ineligible"
    assert next(row for row in triage if row["candidate_id"] == "candidate_blocked_by_mask")["hard_gate_status"] == (
        "ineligible"
    )
    assert {row["sae_window_status"] for row in triage} == {"wt_like_not_used_for_selection"}
    _assert_triage_fields(triage)

    panel = pq.read_table(result.candidate_selection_panel_path).to_pylist()
    assert len(panel) == len(ALL_SPECS)
    assert [row["selection_slot"] for row in panel] == [f"primary_panel_{index:02d}" for index in range(1, 7)]
    assert {row["fold_review_class"] for row in panel} == {"strong_fold_preserved"}
    assert all(row["selected_for_panel"] for row in panel)
    assert all(row["eligible_for_handoff"] for row in panel)
    _assert_panel_fields(panel)
    return triage, panel


def _assert_triage_fields(triage: list[dict]) -> None:
    assert all(row["selection_support_alt_observed_fraction"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_mutation_count"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_chemistry_warning_count"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_chemistry_compatible"] is not None for row in triage)
    assert all(row["proximal_review_unobserved_mutation_count"] is not None for row in triage)
    assert all(row["proximal_review_rare_or_unobserved_mutation_count"] is not None for row in triage)
    assert all(row["local_structure_gate_status"] == "passed" for row in triage)
    assert all(row["local_structure_unavailable_region_count"] == 0 for row in triage)
    assert all(row["local_structure_threshold_failed_region_count"] == 0 for row in triage)
    assert all(row["local_structure_max_ca_rmsd_angstrom"] is not None for row in triage)
    assert {row["selection_candidate_tier"] for row in triage} == {
        "not_panel_candidate",
        "primary_panel_candidate",
    }
    assert all(
        row["primary_panel_candidate"] is True
        for row in triage
        if row["selection_candidate_tier"] == "primary_panel_candidate"
    )
    assert all(
        row["nucleic_acid_facing_acidic_gain_count"] == 0
        for row in triage
        if row["selection_candidate_tier"] == "primary_panel_candidate"
    )
    assert all(
        row["proximal_review_unobserved_mutation_count"] == 0
        for row in triage
        if row["selection_candidate_tier"] == "primary_panel_candidate"
    )


def _assert_panel_fields(panel: list[dict]) -> None:
    assert {row["local_structure_gate_status"] for row in panel} == {"passed"}
    assert all(row["local_structure_threshold_failed_region_count"] == 0 for row in panel)
    assert all(row["catalytic_or_direct_contact_mutation_count"] == 0 for row in panel)
    assert {row["selection_candidate_tier"] for row in panel} == {"primary_panel_candidate"}
    assert {row["near_retained_dna_rna_acidic_gain_review_status"] for row in panel} == {"passed"}
    assert {row["proximal_msa_support_review_status"] for row in panel} == {"passed"}
    for field in (
        "mutation_count_total",
        "sequence_distance_to_wt",
        "thumb_contact_track_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "nucleic_acid_facing_mutation_count",
        "proximal_review_unobserved_mutation_count",
        "proximal_review_rare_or_unobserved_mutation_count",
        "nucleic_acid_facing_basic_gain_count",
        "nucleic_acid_facing_basic_loss_count",
        "nucleic_acid_facing_proline_glycine_gain_count",
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom",
        "nearest_selected_mutation_token_shared_count",
        "nearest_selected_mutation_position_shared_count",
    ):
        assert all(field in row for row in panel)
    assert all(row["nucleic_acid_facing_acidic_gain_count"] == 0 for row in panel)
    assert all(row["proximal_review_unobserved_mutation_count"] == 0 for row in panel)
    _assert_tie_break_trace(panel[0])


def _assert_tie_break_trace(panel_row: dict) -> None:
    assert "esmc_penalty_rank" not in panel_row
    assert "sae_window_contrast_rank" not in panel_row
    assert "primary conservative panel" in panel_row["selection_reason"]
    assert "design classes remain review context rather than quotas" in panel_row["selection_reason"].lower()
    assert "not used for selection" in panel_row["selection_reason"]
    trace_json = panel_row["tie_break_trace_json"]
    for expected in (
        "selection_support_alt_observed_fraction",
        "mutation_count_total",
        "distal_scaffold_mutation_count",
        "c_terminal_primer_rna_recognition_mutation_count",
        "local_structure_gate_status",
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom",
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom",
        "nearest_selected_mutation_token_jaccard_distance",
        "nearest_selected_mutation_position_jaccard_distance",
        "nearest_selected_mutation_token_shared_count",
        "nearest_selected_mutation_position_shared_count",
        "nucleic_acid_facing_basic_gain_count",
        "selection_candidate_tier",
        "nucleic_acid_facing_chemistry_gate_status",
    ):
        assert expected in trace_json
    assert "esmc_6b_additive_llr_total" not in trace_json
    assert "class_local_elimination_policy_id" not in trace_json
    assert "local_structure_substrate_relevant_max" not in trace_json
