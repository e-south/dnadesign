"""Contracts for the checked-in promoter-study workspace surface."""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.latentdna.src.notebooks.browser_runtime import _parse_deliverable_markdown
from dnadesign.latentdna.src.services.notebook_controls_service import build_workspace_notebook_controls_payload
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _live_workspace() -> Path:
    return _repo_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "stress_ethanol_cipro_growth"


def _recipe_steps(context, recipe_id: str) -> list[object]:
    return list(context.config.recipes[recipe_id].steps)


def test_live_study_browser_controls_expose_only_canonical_geometry_inventory() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]

    assert controls.schema_version == "latentdna.workspace_notebook_controls.v4"
    assert controls.workspace_id == "stress_ethanol_cipro_growth"
    assert controls.notebook_id == "latent_geometry_browser"
    assert controls.plot_controls.default_surface == "plots"
    assert controls.plot_controls.ordered_plot_ids == [
        "dataset_overview",
        "reference_margin_gallery_wildtype",
        "reference_neighbor_evidence",
        "context_shift_reference_plane",
        "context_delta_distributions",
        "context_geometry_summary",
        "representation_tradeoff_scatter",
        "representation_scree_diagnostic",
        "reference_margin_gallery_synthetic_centroids",
        "appendix_umap_gallery",
    ]
    assert geometry_ids == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]
    assert controls.geometry_controls.default_compare_left == "intermediate_embedding_20b_anchor_60bp"
    assert controls.geometry_controls.default_compare_right == "intermediate_embedding_20b_full_context_1kb"
    assert "design_family" in controls.geometry_controls.preferred_hues
    assert "sig35_variant" in controls.geometry_controls.preferred_hues
    assert "sigma70_variant" not in controls.geometry_controls.preferred_hues
    assert "sigma70_strength_class" not in controls.geometry_controls.preferred_hues
    assert "context_shift_l2" in controls.geometry_controls.preferred_hues
    assert "log_likelihood_per_token_20b_anchor_60bp" not in controls.geometry_controls.preferred_hues
    assert "log_likelihood_per_token_20b_full_context_1kb" not in controls.geometry_controls.preferred_hues
    assert "log_likelihood_per_token_7b_anchor_60bp" not in controls.geometry_controls.preferred_hues
    assert "log_likelihood_per_token_7b_full_context_1kb" not in controls.geometry_controls.preferred_hues
    assert "construct__context_id" not in controls.geometry_controls.preferred_hues
    assert controls.geometry_controls.preferred_hues == list(controls.geometry_controls.hue_kinds)
    assert controls.geometry_controls.hue_kinds["design_family"] == "categorical"
    assert controls.geometry_controls.hue_kinds["is_control"] == "binary"
    assert controls.geometry_controls.hue_kinds["context_shift_l2"] == "continuous"
    assert controls.geometry_controls.reference_labels == ["spyp", "sulap", "j23105"]


def test_live_study_snapshot_and_deliverables_follow_reference_first_contract() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    snapshot = json.loads((workspace / "outputs" / "status" / "workspace_snapshot.json").read_text(encoding="utf-8"))

    assert snapshot["schema_version"] == "latentdna.workspace_snapshot.v1"
    assert snapshot["workspace_id"] == "stress_ethanol_cipro_growth"
    assert snapshot["decision_ladder"] == [
        "dataset_overview",
        "reference_margin_analysis",
        "context_geometry_audit",
        "representation_comparison",
        "representation_health_diagnostic",
    ]
    assert snapshot["browser"]["default_geometry_ids"] == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]
    assert snapshot["browser"]["preferred_hues"] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "source_class",
        "is_control",
        "wildtype_margin_ethanol_vs_control",
        "wildtype_margin_cipro_vs_control",
        "synthetic_margin_ethanol_vs_background",
        "synthetic_margin_cipro_vs_background",
        "context_self_cosine",
        "context_shift_l2",
    ]

    reference_plot = context.config.plots["reference_margin_gallery_wildtype"]
    assert reference_plot.kind == "xy_scatter_grid"
    assert list(reference_plot.scalars) == [
        "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
        "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
        "wildtype_reference_margins_intermediate_embedding_7b_anchor_60bp",
        "wildtype_reference_margins_intermediate_embedding_7b_full_context_1kb",
        "wildtype_reference_margins_pooled_logits_20b_anchor_60bp",
        "wildtype_reference_margins_pooled_logits_20b_full_context_1kb",
        "wildtype_reference_margins_pooled_logits_7b_anchor_60bp",
        "wildtype_reference_margins_pooled_logits_7b_full_context_1kb",
    ]

    context_plane = context.config.plots["context_shift_reference_plane"]
    assert context_plane.kind == "paired_xy_scatter_grid"

    geometry_summary = context.config.plots["context_geometry_summary"]
    assert geometry_summary.kind == "metric_panel_grid"
    assert context.config.plots["reference_neighbor_evidence"].kind == "metric_panel_grid"
    assert "dual_margin_plane" not in context.config.plots
    assert context.config.plots["context_delta_distributions"].kind == "distribution_grid"
    assert list(context.config.plots["context_delta_distributions"].metric_columns or []) == [
        "context_self_cosine",
        "context_shift_l2",
        "context_margin_delta_ethanol",
        "context_margin_delta_cipro",
    ]
    assert context.config.plots["representation_tradeoff_scatter"].kind == "xy_scatter_grid"
    assert context.config.plots["representation_scree_diagnostic"].kind == "curve_grid"
    appendix_gallery = context.config.plots["appendix_umap_gallery"]
    assert appendix_gallery.kind == "projection_grid"
    assert appendix_gallery.visibility_tier == "appendix"
    assert appendix_gallery.shape_column == "sig35_variant"
    assert all(getattr(plot, "semantics_ref", None) for plot in context.config.plots.values())

    reference_requirements = context.config.deliverables["reference_margin_analysis"].requires
    assert list(reference_requirements["views"]) == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]

    geometry_requirements = context.config.deliverables["context_geometry_audit"].requires
    assert list(geometry_requirements["alignments"]) == [
        "intermediate_embedding_20b_anchor_to_full_context",
        "intermediate_embedding_7b_anchor_to_full_context",
        "pooled_logits_20b_anchor_to_full_context",
        "pooled_logits_7b_anchor_to_full_context",
    ]

    assert context.config.deliverables["reference_margin_analysis"].outputs["plots"] == [
        "reference_margin_gallery_wildtype",
        "reference_neighbor_evidence",
    ]
    assert context.config.deliverables["reference_margin_analysis"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/reference_margin_analysis"
    ]
    assert context.config.deliverables["context_geometry_audit"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/context_geometry_audit"
    ]
    assert context.config.deliverables["representation_comparison"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/representation_comparison"
    ]
    assert context.config.deliverables["representation_health_diagnostic"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/representation_health_diagnostic"
    ]
    assert context.config.exports == {}
    assert "benchmark_feature_matrix" not in context.config.deliverables
    assert "benchmark_results_summary" not in context.config.deliverables

    comparison_step = next(
        step
        for step in _recipe_steps(context, "representation_comparison_recipe")
        if step.id == "build_candidate_metrics_long"
    )
    assert comparison_step.params["ethanol_values"] == ["ethanol", "ethanol_ciprofloxacin"]
    assert comparison_step.params["cipro_values"] == ["ciprofloxacin", "ethanol_ciprofloxacin"]
    assert comparison_step.params["dual_values"] == ["ethanol_ciprofloxacin"]


def test_live_study_recipes_rebuild_from_clean_workspace_state() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    reference_steps = {step.id: step for step in _recipe_steps(context, "reference_margin_analysis_recipe")}
    health_step_ids = [step.id for step in _recipe_steps(context, "representation_health_diagnostic_recipe")]
    appendix_steps = {step.id: step for step in _recipe_steps(context, "appendix_umap_gallery_recipe")}

    assert "materialize_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "build_alignment_intermediate_embedding_20b_anchor_to_full_context" in reference_steps
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "fit_scorecard_knn_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "build_context_geometry_metrics_intermediate_embedding_20b" in reference_steps
    assert "compare_context_geometry_agreement_intermediate_embedding_20b" in reference_steps
    assert (
        reference_steps["build_wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb"].params[
            "alignment_id"
        ]
        == "intermediate_embedding_20b_anchor_to_full_context"
    )
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" in health_step_ids
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" in health_step_ids
    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert set(appendix_steps["generate_latent_geometry_browser"].depends_on) >= {
        "render_reference_margin_gallery_wildtype",
        "render_reference_neighbor_evidence",
        "render_context_shift_reference_plane",
        "render_context_delta_distributions",
        "render_context_geometry_summary",
        "render_representation_tradeoff_scatter",
        "render_representation_scree_diagnostic",
        "render_appendix_umap_gallery",
    }


def test_live_study_appendix_deliverable_docs_cover_both_appendix_plots() -> None:
    appendix_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "appendix_umap_gallery.md"
    ).read_text(encoding="utf-8")

    parsed = _parse_deliverable_markdown(appendix_doc)

    assert "reference_margin_gallery_synthetic_centroids" in parsed["plot_sections"]
    assert "appendix_umap_gallery" in parsed["plot_sections"]
