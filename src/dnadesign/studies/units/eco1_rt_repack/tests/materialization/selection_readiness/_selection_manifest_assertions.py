"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_selection_manifest_assertions.py

Manifest-level assertions for Eco1 RT selection-readiness materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
)

from . import _materialization_assertions as materialization_assertions


def assert_materialized_selection_manifest(
    *,
    result: Any,
    manifest: dict[str, Any],
    triage: list[dict],
    panel: list[dict],
    retired_plot: Path,
) -> None:
    assert manifest["path_policy"] == "paths_relative_to_selection_manifest"
    assert all(not Path(value).is_absolute() for value in manifest["source_tables"].values())
    assert all(not Path(value).is_absolute() for value in manifest["artifacts"].values())
    assert manifest["gate_counts"]["hard_gate_status"] == {"eligible": len(panel), "ineligible": 2}
    assert manifest["gate_counts"]["local_structure_gate_status"] == {"passed": len(triage)}
    assert manifest["gate_counts"]["sae_window_status"] == {"wt_like_not_used_for_selection": len(triage)}
    _assert_summary_and_trace(result=result, manifest=manifest, triage=triage, panel=panel)
    _assert_local_structure_manifest(manifest)
    _assert_handoff_manifest(manifest, panel=panel)
    _assert_artifact_rows(result=result, manifest=manifest, triage=triage)
    materialization_assertions.assert_selection_plot_contract(
        result=result,
        manifest=manifest,
        retired_plot=retired_plot,
    )


def _assert_summary_and_trace(*, result: Any, manifest: dict[str, Any], triage: list[dict], panel: list[dict]) -> None:
    summary = manifest["selection_summary"]
    assert summary["candidate_counts"] == {
        "candidate_pool": len(triage),
        "preservation_pass": len(panel),
        "chemistry_support_pass": len(panel),
        "selected_panel": len(panel),
    }
    assert summary["gate_counts"]["hard_gate_status"] == manifest["gate_counts"]["hard_gate_status"]
    assert summary["selected_mutation_overlap"]["shared_exact_substitution_count"] == 0
    assert summary["selected_mutation_overlap"]["shared_exact_substitutions"] == []
    assert all(
        counts == {"passed": len(triage)} for counts in summary["local_structure_region_threshold_counts"].values()
    )

    funnel_stages = manifest["selection_funnel_stages"]
    assert [row["stage_id"] for row in funnel_stages] == [
        "candidate_pool",
        "preservation_gate",
        "chemistry_support_gate",
        "global_conservative_diverse_selection",
    ]
    stage_by_id = {row["stage_id"]: row for row in funnel_stages}
    assert stage_by_id["candidate_pool"]["remaining_count"] == len(triage)
    assert stage_by_id["global_conservative_diverse_selection"]["remaining_count"] == len(panel)
    assert stage_by_id["global_conservative_diverse_selection"]["selector_role"] == "global_rank"
    trace = pq.read_table(result.primary_panel_selection_trace_path).to_pylist()
    assert {row["stage_id"] for row in trace} == {row["stage_id"] for row in funnel_stages}
    assert len(trace) == len(funnel_stages)


def _assert_local_structure_manifest(manifest: dict[str, Any]) -> None:
    assert manifest["hard_gate_allowed_fold_classes"] == ["strong_fold_preserved"]
    assert "good_fold_preserved" in manifest["default_excluded_fold_classes"]
    assert "local_structure_rmsd_threshold_policy" in manifest
    source_basis_by_id = {row["id"]: row for row in manifest["local_structure_source_basis"]}
    assert source_basis_by_id["tao_et_al_2026_functional_residue_preservation"]["source_ref"] == (
        "doi:10.1038/s41587-026-03149-6"
    )
    assert source_basis_by_id["wang_et_al_2022_ec86_cryoem_structure_priors"]["source_ref"] == (
        "doi:10.1038/s41564-022-01197-7"
    )
    assert [row["region_id"] for row in manifest["local_structure_regions"]] == list(LOCAL_STRUCTURE_REGION_IDS)
    catalytic_region = next(
        row for row in manifest["local_structure_regions"] if row["region_id"] == "catalytic_initiation_context"
    )
    thumb_track_region = next(
        row for row in manifest["local_structure_regions"] if row["region_id"] == "thumb_contact_track_context"
    )
    near_region = next(
        row for row in manifest["local_structure_regions"] if row["region_id"] == "near_retained_dna_rna_annulus"
    )
    c_terminal_region = next(
        row
        for row in manifest["local_structure_regions"]
        if row["region_id"] == "c_terminal_primer_rna_recognition_context"
    )
    assert catalytic_region["region_position_spec"] == "189-204"
    assert "YADD" in catalytic_region["region_position_source"]
    assert catalytic_region["local_ca_rmsd_threshold_angstrom"] == 1.5
    assert thumb_track_region["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert thumb_track_region["local_ca_rmsd_threshold_angstrom"] < near_region["local_ca_rmsd_threshold_angstrom"]
    assert c_terminal_region["region_position_spec"] == "255-311"
    assert c_terminal_region["local_ca_rmsd_threshold_angstrom"] == 2.5


def _assert_handoff_manifest(manifest: dict[str, Any], *, panel: list[dict]) -> None:
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
    assert manifest["panel_coverage"]["required_primary_panel_size"] == len(ALL_SPECS)
    assert manifest["panel_coverage"]["selected_row_count"] == len(panel)
    assert manifest["panel_coverage"]["design_class_quota_enforced"] is False
    assert manifest["panel_coverage"]["duplicate_candidate_ids"] == []
    assert manifest["panel_coverage"]["non_primary_selected_candidate_ids"] == []
    assert manifest["panel_coverage"]["valid"] is True


def _assert_artifact_rows(*, result: Any, manifest: dict[str, Any], triage: list[dict]) -> None:
    sensitivity = pq.read_table(result.local_structure_threshold_sensitivity_path).to_pylist()
    support = pq.read_table(result.region_msa_support_path).to_pylist()
    assert manifest["row_counts"]["local_structure_region_metrics"] == len(triage) * len(LOCAL_STRUCTURE_REGION_IDS)
    assert manifest["row_counts"]["local_structure_threshold_sensitivity"] == len(sensitivity)
    assert manifest["row_counts"]["region_msa_support"] == len(support)
    assert manifest["artifacts"]["local_structure_region_metrics"] == "local_structure_region_metrics.parquet"
    assert "local_structure_region_metrics" in manifest["artifact_hashes"]
    assert {row["scenario_id"] for row in sensitivity} == {
        "tighter_80_percent",
        "declared_threshold",
        "looser_120_percent",
    }
    assert {row["region_id"] for row in sensitivity} == set(LOCAL_STRUCTURE_REGION_IDS)
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
