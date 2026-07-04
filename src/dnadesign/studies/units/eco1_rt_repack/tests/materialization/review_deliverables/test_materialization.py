"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_materialization.py

Eco1 review-deliverable materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_FEASIBILITY_AND_HANDOFF,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    visual_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_manifest_visual_contract,
)


def test_review_deliverables_materialize_manifest_figures_and_notebook(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt.review_deliverables"
    assert manifest["status"] == "materialized_complete"
    assert manifest["deliverable_count"] == len(manifest["deliverables"])
    assert manifest["visual_policy"]["requires_alt_text"] is True
    assert manifest["notebook"]["path"] == "notebooks/eco1_review_deliverables.py"
    assert not Path(manifest["notebook"]["path"]).is_absolute()

    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    expected_rendered = {
        "msa_plurality_mask_panel",
        "msa_subtype_plurality_panel",
        "linear_mask_tracks",
        "proteinmpnn_score_mutation_burden",
        "proteinmpnn_mutation_density",
        "proteinmpnn_variant_similarity_heatmap",
        "proteinmpnn_tao_style_fold_validation",
        "mask_structure_context_script",
        "mask_structure_context_orientation_template",
        "mask_structure_browser_manifest",
        "biohub_esmc_sae_structure_browser_manifest",
        "msa_plurality_vs_esmc_entropy",
        "msa_plurality_vs_best_alt_llr",
        "msa_esmc_constraint_tracks",
        "selection_design_class_gate_counts",
        "selection_population_stratification",
        "selection_panel_review_axes",
        "selection_panel_sequence_differences",
        "selection_panel_mutation_geography_chemistry",
        "selection_funnel_summary",
        "selection_panel_table",
        "selection_handoff_sequences",
        "selection_handoff_readiness",
        "selection_handoff_boundary",
        "selected_panel_structure_browser_manifest",
    }
    expected_linked_model_check = {
        "wt_esmc_entropy_by_position",
        "wt_esmc_fraction_negative_alternate_llr",
        "wt_esmc_substitution_llr_heatmap",
    }
    assert expected_rendered.issubset(deliverables)
    assert expected_linked_model_check.issubset(deliverables)
    assert deliverables["mask_structure_context_png"]["status"] == "skipped_optional_render_disabled"
    assert deliverables["foldcheck_review_fold_metric_scatter"]["status"] == "linked_existing"
    assert deliverables["foldcheck_review_structure_overlay_panel"]["status"] == "linked_existing"
    assert deliverables["foldcheck_review_structure_overlay_skipped"]["status"] == "skipped_runtime_unavailable"
    assert deliverables["mask_structure_browser_manifest"]["status"] == "rendered"
    assert "interactively" in deliverables["mask_structure_browser_manifest"]["title"].lower()
    assert deliverables["interactive_structure_browser_manifest"]["status"] == "rendered"
    assert "reference-fitted colabfold" in deliverables["interactive_structure_browser_manifest"]["title"].lower()
    assert deliverables["selected_panel_structure_browser_manifest"]["status"] == "rendered"
    assert "selected eco1 panel" in deliverables["selected_panel_structure_browser_manifest"]["title"].lower()
    visual_ids = {entry["deliverable_id"] for entry in visual_deliverables(manifest["deliverables"])}
    audit_visual_ids = {
        entry["deliverable_id"]
        for entry in visual_deliverables(manifest["deliverables"], selected_lane="audit_supplement")
    }
    assert "mask_structure_browser_manifest" in visual_ids
    assert "interactive_structure_browser_manifest" in visual_ids
    assert "biohub_esmc_sae_structure_browser_manifest" not in visual_ids
    assert "wt_esmc_entropy_by_position" not in visual_ids
    assert "proteinmpnn_score_mutation_burden" not in visual_ids
    assert "proteinmpnn_mutation_density" not in visual_ids
    assert "proteinmpnn_tao_style_fold_validation" not in visual_ids
    assert "proteinmpnn_variant_similarity_heatmap" not in visual_ids
    assert "biohub_esmc_candidate_preference_vs_wt" not in visual_ids
    assert "biohub_esmc_sae_feature_activation_heatmap" not in visual_ids
    assert "biohub_esmc_sae_umap" not in visual_ids
    assert "biohub_esmc_candidate_top_sae_feature_activation_ratio" not in visual_ids
    assert "biohub_esmc_sae_fold_llr_comparison" not in visual_ids
    assert "selection_design_class_gate_counts" in visual_ids
    assert "selection_population_stratification" in visual_ids
    assert "selection_panel_review_axes" in visual_ids
    assert "selection_panel_sequence_differences" in visual_ids
    assert "selection_panel_mutation_geography_chemistry" in visual_ids
    assert "selection_funnel_summary" in visual_ids
    assert "selection_panel_table" in visual_ids
    assert "selection_handoff_sequences" in visual_ids
    assert "selection_handoff_readiness" in visual_ids
    assert "selection_handoff_boundary" in visual_ids
    assert "selected_panel_structure_browser_manifest" in visual_ids
    assert "feasibility_and_handoff_planned" not in visual_ids
    assert "wt_esmc_entropy_by_position" in audit_visual_ids
    assert "proteinmpnn_score_mutation_burden" in audit_visual_ids
    assert "proteinmpnn_mutation_density" in audit_visual_ids
    assert "proteinmpnn_tao_style_fold_validation" in audit_visual_ids
    assert "proteinmpnn_variant_similarity_heatmap" in audit_visual_ids
    assert "biohub_esmc_candidate_preference_vs_wt" in audit_visual_ids
    assert "biohub_esmc_sae_feature_activation_heatmap" in audit_visual_ids
    assert "biohub_esmc_sae_structure_browser_manifest" in audit_visual_ids
    assert "foldcheck_review_fold_metric_scatter" in audit_visual_ids
    assert "mask_structure_context_png" not in visual_ids
    assert "foldcheck_review_structure_overlay_panel" not in visual_ids
    assert "foldcheck_review_structure_overlay_skipped" not in visual_ids
    assert deliverables["wt_esmc_entropy_by_position"]["status"] == "linked_existing"
    assert deliverables["wt_esmc_entropy_by_position"]["section"] == SECTION_CONSTRAINT_EVIDENCE
    assert deliverables["msa_plurality_vs_esmc_entropy"]["status"] == "rendered"
    assert deliverables["proteinmpnn_score_mutation_burden"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["proteinmpnn_score_mutation_burden"]["role"] == "review_only"
    assert deliverables["proteinmpnn_mutation_density"]["role"] == "review_only"
    assert deliverables["proteinmpnn_variant_similarity_heatmap"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["proteinmpnn_variant_similarity_heatmap"]["role"] == "review_only"
    assert "baseline" in deliverables["proteinmpnn_variant_similarity_heatmap"]["title"].lower()
    assert deliverables["proteinmpnn_tao_style_fold_validation"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["proteinmpnn_tao_style_fold_validation"]["role"] == "review_only"
    assert deliverables["interactive_structure_browser_manifest"]["section"] == SECTION_DESIGNS_AND_FOLD_TRIAGE
    assert deliverables["selected_panel_structure_browser_manifest"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_design_class_gate_counts"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_design_class_gate_counts"]["status"] == "linked_existing"
    assert deliverables["selection_population_stratification"]["section"] == SECTION_FEASIBILITY_AND_HANDOFF
    assert deliverables["selection_population_stratification"]["status"] == "linked_existing"
    assert deliverables["selection_funnel_summary"]["artifact_kind"] == "selection_funnel_summary"
    assert deliverables["selection_funnel_summary"]["status"] == "linked_existing"
    assert deliverables["selection_panel_table"]["artifact_kind"] == "selection_panel_table"
    assert deliverables["selection_panel_table"]["status"] == "linked_existing"
    assert deliverables["selection_handoff_sequences"]["artifact_kind"] == "candidate_handoff_sequence_csv"
    assert deliverables["selection_handoff_sequences"]["status"] == "linked_existing"
    assert deliverables["selection_handoff_readiness"]["artifact_kind"] == "handoff_readiness"
    assert deliverables["selection_handoff_readiness"]["status"] == "linked_existing"
    assert deliverables["selection_handoff_boundary"]["artifact_kind"] == "handoff_boundary"
    assert deliverables["selection_handoff_boundary"]["status"] == "linked_existing"

    assert_manifest_visual_contract(
        manifest_path=result.manifest_path,
        manifest=manifest,
        deliverables=deliverables,
        expected_rendered=expected_rendered,
    )


def test_review_deliverables_require_canonical_selection_manifest(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    (tmp_path / "design_classes" / "selection" / "selection_readiness_manifest.yaml").unlink()

    try:
        materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    except FileNotFoundError as exc:
        assert "design_classes/selection/selection_readiness_manifest.yaml" in str(exc)
    else:  # pragma: no cover - pytest assertion path
        raise AssertionError("review deliverables should require the panel-selection manifest")
