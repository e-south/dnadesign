"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_selection_manifest_artifact_assertions.py

Handoff and artifact assertions for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTED_PANEL_SIZE,
)


def assert_selection_handoff_manifest(manifest: dict[str, Any], *, panel: list[dict]) -> None:
    assert manifest["selected_candidate_ids"] == [row["candidate_id"] for row in panel]
    assert manifest["handoff_readiness"] == {
        "handoff_kind": "rt_only_candidate_handoff",
        "panel_selected": True,
        "candidate_handoff_path": "../../candidate_handoff.yaml",
        "candidate_handoff_sequence_csv_path": "candidate_handoff_sequences.csv",
        "candidate_handoff_sequence_csv_materialized": True,
        "candidate_handoff_file_present": True,
        "candidate_handoff_materialized": True,
        "construct_subject_created": False,
    }
    assert manifest["panel_coverage"]["selected_panel_size"] == SELECTED_PANEL_SIZE
    assert manifest["panel_coverage"]["selected_row_count"] == len(panel)
    assert manifest["panel_coverage"]["policy_allocation_role"] == "experimental_design"
    assert manifest["panel_coverage"]["rt_msdna_oligomeric_state_review_status_counts"] == {
        "not_established": len(panel)
    }
    assert manifest["panel_coverage"]["duplicate_candidate_ids"] == []
    assert manifest["panel_coverage"]["contract_failure_candidate_ids"] == []
    assert manifest["panel_coverage"]["valid"] is True


def assert_selection_artifact_rows(*, result: Any, manifest: dict[str, Any], triage: list[dict]) -> None:
    sensitivity = pq.read_table(result.local_structure_threshold_sensitivity_path).to_pylist()
    support = pq.read_table(result.region_msa_support_path).to_pylist()
    assert manifest["row_counts"]["local_structure_region_metrics"] == len(triage) * len(LOCAL_STRUCTURE_REGION_IDS)
    assert manifest["row_counts"]["local_structure_threshold_sensitivity"] == len(sensitivity)
    assert manifest["row_counts"]["region_msa_support"] == len(support)
    assert manifest["artifacts"]["local_structure_region_metrics"] == "local_structure_region_metrics.parquet"
    assert "local_structure_region_metrics" in manifest["artifact_hashes"]
    assert "near_region_charge_sensitivity" not in manifest["artifacts"]
    assert "near_region_charge_mutation_audit" not in manifest["artifacts"]
    assert "charge_sensitivity_shortlist" not in manifest["artifacts"]
    assert "near_region_charge_sensitivity_policy" not in manifest
    assert {row["scenario_id"] for row in sensitivity} == {
        "tighter_80_percent",
        "declared_threshold",
        "looser_120_percent",
    }
    assert {row["region_id"] for row in sensitivity} == set(LOCAL_STRUCTURE_REGION_IDS) - {"distal_scaffold_control"}
    assert all(row["candidate_count"] == len(triage) for row in sensitivity)
    assert {row["region_id"] for row in support} == {
        "catalytic_or_direct_contact",
        "near_retained_dna_rna_region",
        "thumb_contact_track",
        "c_terminal_primer_rna_recognition_region",
        "distal_scaffold",
    }
    assert len(support) == len(triage) * 5
    assert all(row["region_label"] != "Near retained DNA/RNA annulus" for row in support)
