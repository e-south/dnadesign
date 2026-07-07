"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_materialization.py

Panel-selection materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    cli as selection_readiness_cli,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness import (
    _materialization_assertions as materialization_assertions,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    write_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)


def test_selection_readiness_writes_feasibility_triage_and_primary_panel(tmp_path: Path) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    inputs = write_inputs(class_root, source_root)
    _write_manual_mask_authority_source_basis(repo_root)
    root_handoff_path = source_root / "candidate_handoff.yaml"
    root_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")
    selection_local_handoff_path = selection_root / "candidate_handoff.yaml"
    selection_local_handoff_path.parent.mkdir(parents=True, exist_ok=True)
    selection_local_handoff_path.write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")
    retired_plot = selection_root / "plots" / "selection_panel_review_axes.svg"
    retired_plot.parent.mkdir(parents=True, exist_ok=True)
    retired_plot.write_text("<svg>retired selected-only scatter</svg>\n", encoding="utf-8")

    result = materialize_selection_readiness(
        repo_root=repo_root,
        output_root=class_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    assert result.feasibility_report_path == selection_root / "feasibility_report.parquet"
    assert result.candidate_triage_table_path == selection_root / "candidate_triage_table.parquet"
    assert result.local_structure_region_metrics_path == selection_root / "local_structure_region_metrics.parquet"
    assert result.local_structure_threshold_sensitivity_path == (
        selection_root / "local_structure_threshold_sensitivity.parquet"
    )
    assert result.region_msa_support_path == selection_root / "region_msa_support.parquet"
    assert result.primary_panel_selection_trace_path == selection_root / "primary_panel_selection_trace.parquet"
    assert result.candidate_selection_panel_path == selection_root / "candidate_selection_panel.parquet"
    assert result.candidate_handoff_sequence_csv_path == selection_root / "candidate_handoff_sequences.csv"
    assert result.plots_root == selection_root / "plots"
    assert result.manifest_path == selection_root / "selection_readiness_manifest.yaml"

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
    assert all(row["sae_mechanistic_contrast_window_id"] is None for row in triage)
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
    assert all(row["local_structure_substrate_relevant_max_ca_rmsd_angstrom"] is not None for row in triage)
    assert {row["selection_candidate_tier"] for row in triage} == {"not_panel_candidate", "primary_panel_candidate"}
    assert all(
        row["primary_panel_candidate"] is True
        for row in triage
        if row["selection_candidate_tier"] == "primary_panel_candidate"
    )

    panel = pq.read_table(result.candidate_selection_panel_path).to_pylist()
    assert len(panel) == len(ALL_SPECS)
    assert [row["selection_slot"] for row in panel] == [f"primary_panel_{index:02d}" for index in range(1, 7)]
    assert {row["design_class_id"] for row in panel} == {spec.design_class_id for spec in ALL_SPECS}
    assert {row["fold_review_class"] for row in panel} == {"strong_fold_preserved"}
    assert all(row["selected_for_panel"] for row in panel)
    assert all(row["eligible_for_handoff"] for row in panel)
    assert {row["local_structure_gate_status"] for row in panel} == {"passed"}
    assert all(row["local_structure_threshold_failed_region_count"] == 0 for row in panel)
    assert all(row["local_structure_max_ca_rmsd_angstrom"] is not None for row in panel)
    assert all(row["catalytic_or_direct_contact_mutation_count"] == 0 for row in panel)
    assert all("thumb_contact_track_mutation_count" in row for row in panel)
    assert all("c_terminal_primer_rna_recognition_mutation_count" in row for row in panel)
    assert all("nucleic_acid_facing_mutation_count" in row for row in panel)
    assert {row["nucleic_acid_facing_chemistry_gate_status"] for row in panel} == {"passed"}
    assert all(row["nucleic_acid_facing_chemistry_compatible"] for row in panel)
    assert {row["selection_candidate_tier"] for row in panel} == {"primary_panel_candidate"}
    assert {row["primary_c_terminal_local_rmsd_gate_status"] for row in panel} == {"passed"}
    assert {row["near_retained_dna_rna_acidic_gain_review_status"] for row in panel} == {"passed"}
    assert {row["proximal_msa_support_review_status"] for row in panel} == {"passed"}
    assert {row["local_structure_substrate_relevant_max_gate_status"] for row in panel} == {"passed"}
    assert all(row["local_structure_substrate_relevant_max_ca_rmsd_angstrom"] is not None for row in panel)
    assert all("proximal_review_unobserved_mutation_count" in row for row in panel)
    assert all("proximal_review_rare_or_unobserved_mutation_count" in row for row in panel)
    assert "esmc_penalty_rank" not in panel[0]
    assert "sae_window_contrast_rank" not in panel[0]
    assert "primary conservative panel" in panel[0]["selection_reason"]
    assert "design classes remain review context rather than quotas" in panel[0]["selection_reason"]
    assert "not used for selection" in panel[0]["selection_reason"]
    assert "esmc_6b_additive_llr_total" not in panel[0]["tie_break_trace_json"]
    assert "selection_support_alt_observed_fraction" in panel[0]["tie_break_trace_json"]
    assert "mutation_count_total" in panel[0]["tie_break_trace_json"]
    assert "distal_scaffold_mutation_count" in panel[0]["tie_break_trace_json"]
    assert "c_terminal_primer_rna_recognition_mutation_count" in panel[0]["tie_break_trace_json"]
    assert "local_structure_gate_status" in panel[0]["tie_break_trace_json"]
    assert "local_structure_catalytic_initiation_context_ca_rmsd_angstrom" in panel[0]["tie_break_trace_json"]
    assert (
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom" in panel[0]["tie_break_trace_json"]
    )
    assert "nearest_selected_mutation_token_jaccard_distance" in panel[0]["tie_break_trace_json"]
    assert "nearest_selected_mutation_position_jaccard_distance" in panel[0]["tie_break_trace_json"]
    assert "class_local_elimination_policy_id" not in panel[0]["tie_break_trace_json"]
    assert "selection_candidate_tier" in panel[0]["tie_break_trace_json"]
    assert "nucleic_acid_facing_chemistry_gate_status" in panel[0]["tie_break_trace_json"]

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["path_policy"] == "paths_relative_to_selection_manifest"
    assert all(not Path(value).is_absolute() for value in manifest["source_tables"].values())
    assert all(not Path(value).is_absolute() for value in manifest["artifacts"].values())
    assert manifest["gate_counts"]["hard_gate_status"] == {"eligible": len(panel), "ineligible": 2}
    assert manifest["gate_counts"]["local_structure_gate_status"] == {"passed": len(triage)}
    assert manifest["gate_counts"]["sae_window_status"] == {"wt_like_not_used_for_selection": len(triage)}
    funnel_stages = manifest["selection_funnel_stages"]
    assert [row["stage_id"] for row in funnel_stages] == [
        "candidate_pool",
        "broad_contract_pool",
        "primary_panel_candidate_pool",
        "global_conservative_diverse_selection",
    ]
    stage_by_id = {row["stage_id"]: row for row in funnel_stages}
    assert stage_by_id["candidate_pool"]["remaining_count"] == len(triage)
    assert stage_by_id["broad_contract_pool"]["remaining_count"] == len(panel)
    assert stage_by_id["primary_panel_candidate_pool"]["remaining_count"] == len(panel)
    assert stage_by_id["global_conservative_diverse_selection"]["remaining_count"] == len(panel)
    assert stage_by_id["global_conservative_diverse_selection"]["selector_role"] == "global_rank"
    assert manifest["hard_gate_allowed_fold_classes"] == ["strong_fold_preserved"]
    assert "good_fold_preserved" in manifest["default_excluded_fold_classes"]
    assert "local_structure_rmsd_threshold_policy" in manifest
    assert manifest["artifacts"]["local_structure_threshold_sensitivity"] == (
        "local_structure_threshold_sensitivity.parquet"
    )
    assert manifest["artifacts"]["region_msa_support"] == "region_msa_support.parquet"
    assert manifest["artifacts"]["primary_panel_selection_trace"] == "primary_panel_selection_trace.parquet"
    trace = pq.read_table(result.primary_panel_selection_trace_path).to_pylist()
    trace_stage_ids = {row["stage_id"] for row in trace}
    assert trace_stage_ids == {row["stage_id"] for row in funnel_stages}
    assert len(trace) == len(funnel_stages)
    assert "global_conservative_diverse_selection" in trace_stage_ids
    sensitivity = pq.read_table(result.local_structure_threshold_sensitivity_path).to_pylist()
    assert {row["scenario_id"] for row in sensitivity} == {
        "tighter_80_percent",
        "declared_threshold",
        "looser_120_percent",
    }
    assert {row["region_id"] for row in sensitivity} == set(LOCAL_STRUCTURE_REGION_IDS)
    assert all(row["candidate_count"] == len(triage) for row in sensitivity)
    assert all("selected_failure_count" in row for row in sensitivity)
    support = pq.read_table(result.region_msa_support_path).to_pylist()
    assert {row["region_id"] for row in support} == {
        "catalytic_or_direct_contact",
        "near_retained_dna_rna_region",
        "thumb_contact_track",
        "c_terminal_primer_rna_recognition_region",
        "distal_scaffold",
    }
    assert len(support) == len(triage) * 5
    assert all(row["region_label"] != "Near retained DNA/RNA annulus" for row in support)
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
    assert catalytic_region["region_position_spec"] == "189-204"
    assert "YADD" in catalytic_region["region_position_source"]
    assert "tao_et_al_2026_functional_residue_preservation" in catalytic_region["region_source_basis_ids"]
    assert catalytic_region["coordinate_scope"] == "mapped_rt_chain_ca_after_global_fit"
    assert catalytic_region["local_ca_rmsd_threshold_angstrom"] == 1.5
    thumb_track_region = next(
        row for row in manifest["local_structure_regions"] if row["region_id"] == "thumb_contact_track_context"
    )
    near_region = next(
        row for row in manifest["local_structure_regions"] if row["region_id"] == "near_retained_dna_rna_annulus"
    )
    assert thumb_track_region["local_ca_rmsd_threshold_angstrom"] == 2.5
    assert thumb_track_region["local_ca_rmsd_threshold_angstrom"] < near_region["local_ca_rmsd_threshold_angstrom"]
    assert "Wang/Ec86" in thumb_track_region["region_position_source"]
    c_terminal_region = next(
        row
        for row in manifest["local_structure_regions"]
        if row["region_id"] == "c_terminal_primer_rna_recognition_context"
    )
    assert c_terminal_region["region_position_spec"] == "255-311"
    assert "primer-RNA recognition" in c_terminal_region["region_position_source"]
    assert "inouye_et_al_2004_ec86_thumb_primer_rna_binding" in c_terminal_region["region_source_basis_ids"]
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
    assert manifest["panel_coverage"] == {
        "required_primary_panel_size": len(ALL_SPECS),
        "selected_row_count": len(panel),
        "design_class_quota_enforced": False,
        "selected_design_class_counts": {spec.design_class_id: 1 for spec in ALL_SPECS},
        "duplicate_candidate_ids": [],
        "non_primary_selected_candidate_ids": [],
        "valid": True,
    }
    assert manifest["row_counts"]["local_structure_region_metrics"] == len(triage) * len(LOCAL_STRUCTURE_REGION_IDS)
    assert manifest["row_counts"]["local_structure_threshold_sensitivity"] == len(sensitivity)
    assert manifest["row_counts"]["region_msa_support"] == len(support)
    assert manifest["row_counts"]["primary_panel_selection_trace"] == len(trace)
    assert manifest["row_counts"]["candidate_handoff_sequences"] == len(panel)
    assert manifest["artifacts"]["local_structure_region_metrics"] == "local_structure_region_metrics.parquet"
    assert "local_structure_region_metrics" in manifest["artifact_hashes"]
    assert "local_structure_threshold_sensitivity" in manifest["artifact_hashes"]
    assert "region_msa_support" in manifest["artifact_hashes"]
    assert "primary_panel_selection_trace" in manifest["artifact_hashes"]
    assert "candidate_handoff_sequences" in manifest["artifact_hashes"]
    materialization_assertions.assert_selection_plot_contract(
        result=result,
        manifest=manifest,
        retired_plot=retired_plot,
    )


def test_selection_readiness_cli_reports_handoff_sequence_csv_path(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    write_inputs(class_root, source_root)

    exit_code = selection_readiness_cli.main(
        [
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(class_root),
            "--source-output-root",
            str(source_root),
            "--selection-root",
            str(selection_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["candidate_handoff_sequence_csv_path"] == str(selection_root / "candidate_handoff_sequences.csv")
    assert Path(payload["candidate_handoff_sequence_csv_path"]).exists()


def _write_manual_mask_authority_source_basis(repo_root: Path) -> None:
    path = repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "source_basis": [
                    {
                        "id": "tao_et_al_2026_functional_residue_preservation",
                        "role": "method_prior",
                        "source_ref": "doi:10.1038/s41587-026-03149-6",
                    },
                    {
                        "id": "simon_et_al_2019_retron_rt_motif_grammar",
                        "role": "motif_annotation_prior",
                        "source_ref": "doi:10.1093/nar/gkz865",
                    },
                    {
                        "id": "wang_et_al_2022_ec86_cryoem_structure_priors",
                        "role": "ec86_structure_mask_prior",
                        "source_ref": "doi:10.1038/s41564-022-01197-7",
                    },
                    {
                        "id": "inouye_et_al_1999_ec86_primer_template_recognition",
                        "role": "c_terminal_specificity_review_prior",
                        "source_ref": "doi:10.1074/jbc.274.44.31236",
                    },
                    {
                        "id": "inouye_et_al_2004_ec86_thumb_primer_rna_binding",
                        "role": "c_terminal_specificity_review_prior",
                        "source_ref": "doi:10.1074/jbc.M408462200",
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
