"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_visual_routing.py

Notebook visual-routing tests for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.notebook_runtime import (
    visual_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
    RETIRED_SELECTION_PLOT_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_review_notebook_routes_only_visual_artifacts_to_figure_selectors(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path, render_chimerax_png=False)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))

    visual_ids = {entry["deliverable_id"] for entry in visual_deliverables(manifest["deliverables"])}
    audit_visual_ids = {
        entry["deliverable_id"]
        for entry in visual_deliverables(manifest["deliverables"], selected_lane="audit_supplement")
    }
    assert "mask_structure_browser_manifest" in visual_ids
    assert "interactive_structure_browser_manifest" in visual_ids
    assert "selected_panel_structure_browser_manifest" in visual_ids
    assert "design_class_mask_overview" in visual_ids
    assert "expanded_proteinmpnn_fold_validation" in visual_ids
    assert "foldcheck_review_review_class_counts" in visual_ids
    assert "proteinmpnn_residue_frequency_heatmap" in visual_ids
    assert set(CURRENT_SELECTION_PLOT_IDS).issubset(visual_ids)
    assert set(RETIRED_SELECTION_PLOT_IDS).isdisjoint(visual_ids)

    assert "selection_funnel_summary" not in visual_ids
    assert "selection_panel_table" not in visual_ids
    assert "selection_handoff_sequences" not in visual_ids
    assert "selection_handoff_readiness" not in visual_ids
    assert "selection_handoff_boundary" not in visual_ids
    assert "feasibility_and_handoff_planned" not in visual_ids
    assert "linear_mask_tracks" not in visual_ids
    assert "mask_structure_context_png" not in visual_ids
    assert "foldcheck_review_structure_overlay_panel" not in visual_ids
    assert "foldcheck_review_structure_overlay_skipped" not in visual_ids

    assert "biohub_esmc_sae_structure_browser_manifest" not in visual_ids
    assert "wt_esmc_entropy_by_position" not in visual_ids
    assert "proteinmpnn_score_mutation_burden" not in visual_ids
    assert "biohub_esmc_candidate_preference_vs_wt" not in visual_ids
    assert "biohub_esmc_sae_feature_activation_heatmap" not in visual_ids
    assert "biohub_esmc_sae_umap" not in visual_ids
    assert "biohub_esmc_candidate_top_sae_feature_activation_ratio" not in visual_ids
    assert "biohub_esmc_sae_fold_llr_comparison" not in visual_ids

    assert "wt_esmc_entropy_by_position" in audit_visual_ids
    assert "proteinmpnn_score_mutation_burden" in audit_visual_ids
    assert "biohub_esmc_candidate_preference_vs_wt" in audit_visual_ids
    assert "biohub_esmc_sae_feature_activation_heatmap" in audit_visual_ids
    assert "biohub_esmc_sae_delta_umap" not in audit_visual_ids
    assert "biohub_esmc_sae_structure_browser_manifest" in audit_visual_ids
