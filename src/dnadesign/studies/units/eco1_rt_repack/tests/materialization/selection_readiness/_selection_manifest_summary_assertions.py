"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_selection_manifest_summary_assertions.py

Selection-summary and local-structure assertions for Eco1 RT readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)


def assert_selection_summary_and_trace(
    *, result: Any, manifest: dict[str, Any], triage: list[dict], panel: list[dict]
) -> None:
    summary = manifest["selection_summary"]
    contract_rows = [row for row in triage if bool(row["selection_contract_pass"])]
    assert summary["candidate_counts"] == {
        "candidate_pool": len(triage),
        "selection_contract_pass": len(contract_rows),
        "selection_contract_pass_by_policy": dict(
            sorted(Counter(str(row["policy_id"]) for row in contract_rows).items())
        ),
        "selected_panel": SELECTED_PANEL_SIZE,
    }
    assert summary["gate_counts"]["hard_gate_status"] == manifest["gate_counts"]["hard_gate_status"]
    assert summary["selected_mutation_overlap"]["shared_exact_substitution_count"] == 0
    assert summary["selected_mutation_overlap"]["shared_exact_substitutions"] == []
    assert summary["selected_mutation_overlap"]["unique_mutated_position_count"] == 16
    assert summary["selected_mutation_overlap"]["unique_exact_substitution_count"] == 16
    assert summary["selected_mutation_overlap"]["mean_pairwise_mutated_position_jaccard_distance"] == 1.0
    assert summary["selected_mutation_overlap"]["mean_pairwise_exact_substitution_jaccard_distance"] == 1.0
    assert summary["selected_mutation_overlap"]["mutation_count_range"] == [21, 29]
    overlap_by_policy = summary["selected_mutation_overlap_by_policy"]
    assert sorted(row["selected_candidate_count"] for row in overlap_by_policy.values()) == [2, 3, 3]
    assert all(
        row["minimum_pairwise_mutated_position_jaccard_distance"] is not None for row in overlap_by_policy.values()
    )
    assert all(
        row["minimum_pairwise_exact_substitution_jaccard_distance"] is not None for row in overlap_by_policy.values()
    )
    threshold_counts = summary["local_structure_region_threshold_counts"]
    assert threshold_counts["distal_scaffold_control"] == {"review_only": len(triage)}
    assert all(
        counts == {"passed": len(triage)}
        for region_id, counts in threshold_counts.items()
        if region_id != "distal_scaffold_control"
    )

    funnel_stages = manifest["selection_funnel_stages"]
    assert [row["stage_id"] for row in funnel_stages] == [
        "candidate_pool",
        "local_geometry_screen",
        "design_groups",
        "selected_panel",
    ]
    stage_by_id = {row["stage_id"]: row for row in funnel_stages}
    assert stage_by_id["candidate_pool"]["remaining_count"] == len(triage)
    assert stage_by_id["design_groups"]["remaining_count"] == len(contract_rows)
    assert stage_by_id["design_groups"]["selector_role"] == "experimental_design"
    assert stage_by_id["selected_panel"]["remaining_count"] == len(panel)
    assert stage_by_id["selected_panel"]["selector_role"] == "within_group_mutation_set_selection"
    assert manifest["panel_tie_break_order"] == [
        "first pair: largest within-group mutated-position Jaccard distance",
        "first pair: largest within-group exact-substitution Jaccard distance",
        "third row: largest minimum mutated-position Jaccard distance from the within-group pair",
        "third row: largest minimum exact-substitution Jaccard distance from the within-group pair",
        "fewest basic losses near retained DNA/RNA",
        "fewest Pro/Gly gains near retained DNA/RNA",
        "selection-support MSA observed fraction",
        "selection-support MSA mean alternate-residue frequency",
        "lowest C-terminal primer-RNA recognition-region C-alpha RMSD inside the gate",
        "lowest Wang thumb-contact-track C-alpha RMSD inside the gate",
        "highest mean pLDDT",
        "lowest cryoEM-mapped C-alpha RMSD",
        "sequence hash",
    ]
    trace = pq.read_table(result.hypothesis_panel_selection_trace_path).to_pylist()
    assert {row["stage_id"] for row in trace} == {row["stage_id"] for row in funnel_stages}
    assert len(trace) == len(funnel_stages)


def assert_local_structure_manifest(manifest: dict[str, Any]) -> None:
    assert "hard_gate_allowed_fold_classes" not in manifest
    assert "default_excluded_fold_classes" not in manifest
    assert "local_structure_rmsd_threshold_policy" in manifest
    source_basis_by_id = {row["id"]: row for row in manifest["local_structure_source_basis"]}
    assert source_basis_by_id["tao_et_al_2026_functional_residue_preservation"]["source_ref"] == (
        "doi:10.1038/s41587-026-03149-6"
    )
    assert source_basis_by_id["wang_et_al_2022_ec86_cryoem_structure_priors"]["source_ref"] == (
        "doi:10.1038/s41564-022-01197-7"
    )
    assert [row["region_id"] for row in manifest["local_structure_regions"]] == list(LOCAL_STRUCTURE_REGION_IDS)
    regions = {row["region_id"]: row for row in manifest["local_structure_regions"]}
    assert regions["catalytic_initiation_context"]["region_position_spec"] == "189-204"
    assert "YADD" in regions["catalytic_initiation_context"]["region_position_source"]
    assert regions["catalytic_initiation_context"]["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert regions["thumb_contact_track_context"]["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert regions["near_retained_dna_rna_annulus"]["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert regions["c_terminal_primer_rna_recognition_context"]["region_position_spec"] == "255-311"
    assert regions["c_terminal_primer_rna_recognition_context"]["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert regions["distal_scaffold_control"]["local_ca_rmsd_threshold_angstrom"] is None
