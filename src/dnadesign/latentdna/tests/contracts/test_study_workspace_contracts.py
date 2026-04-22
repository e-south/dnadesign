"""Contracts for the checked-in promoter-study pre-assay workspace surface."""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.latentdna.src.notebooks.browser_runtime import _parse_deliverable_markdown
from dnadesign.latentdna.src.services.catalog_service import workspace_catalog_from_context
from dnadesign.latentdna.src.services.notebook_controls_service import build_workspace_notebook_controls_payload
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config
from dnadesign.latentdna.src.workspaces.plot_semantics import resolve_plot_semantics


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
        "representation_health_summary",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    ]
    assert geometry_ids == [
        "intermediate_embedding_7b_anchor_60bp",
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_7b_full_context_1kb",
        "intermediate_embedding_7b_full_context_anchor_mean",
        "intermediate_embedding_7b_anchor_plus_full_context_concat",
        "intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
    ]
    assert controls.geometry_controls.default_compare_left == "intermediate_embedding_7b_anchor_60bp"
    assert controls.geometry_controls.default_compare_right == "intermediate_embedding_7b_full_context_anchor_mean"
    assert "design_family" in controls.geometry_controls.preferred_hues
    assert "sig35_variant" in controls.geometry_controls.preferred_hues
    assert "spacer_length" in controls.geometry_controls.preferred_hues
    assert "log_likelihood_per_token_7b" in controls.geometry_controls.preferred_hues
    assert "wildtype_margin_ethanol_vs_control" not in controls.geometry_controls.preferred_hues
    assert "wildtype_margin_cipro_vs_control" not in controls.geometry_controls.preferred_hues
    assert "sigma70_variant" not in controls.geometry_controls.preferred_hues
    assert "sigma70_strength_class" not in controls.geometry_controls.preferred_hues
    assert "construct__context_id" not in controls.geometry_controls.preferred_hues
    assert controls.geometry_controls.hue_kinds["design_family"] == "categorical"
    assert controls.geometry_controls.hue_kinds["is_control"] == "binary"
    assert controls.geometry_controls.hue_kinds["spacer_length"] == "ordinal"
    assert controls.geometry_controls.reference_labels == ["spyp", "sulap", "j23105"]


def test_live_study_snapshot_and_deliverables_follow_pre_assay_contract() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    snapshot = json.loads((workspace / "outputs" / "status" / "workspace_snapshot.json").read_text(encoding="utf-8"))
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert snapshot["schema_version"] == "latentdna.workspace_snapshot.v1"
    assert snapshot["workspace_id"] == "stress_ethanol_cipro_growth"
    assert snapshot["decision_ladder"] == [
        "dataset_overview",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
    ]
    assert snapshot["browser"]["default_geometry_ids"] == [
        "intermediate_embedding_7b_anchor_60bp",
        "pooled_logits_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_7b_full_context_1kb",
        "intermediate_embedding_7b_full_context_anchor_mean",
        "intermediate_embedding_7b_anchor_plus_full_context_concat",
        "intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
    ]
    assert controls.geometry_controls.default_model == "7b"
    assert controls.geometry_controls.default_family == "intermediate_embedding"
    assert "spacer_length" in snapshot["browser"]["preferred_hues"]
    assert "log_likelihood_per_token_7b" in snapshot["browser"]["preferred_hues"]
    assert "wildtype_margin_ethanol_vs_control" not in snapshot["browser"]["preferred_hues"]
    assert "wildtype_margin_cipro_vs_control" not in snapshot["browser"]["preferred_hues"]

    assert context.config.plots["representation_health_summary"].kind == "metric_panel_grid"
    assert context.config.plots["design_structure_summary"].kind == "metric_panel_grid"
    assert context.config.plots["sigma35_ordinal_audit"].kind == "metric_panel_grid"
    assert context.config.plots["context_robustness_summary"].kind == "metric_panel_grid"
    assert resolve_plot_semantics(context, plot_id="representation_health_summary").decision_role == "gate"
    dataset_semantics = resolve_plot_semantics(context, plot_id="dataset_overview")
    assert "generation plan" in dataset_semantics.scope
    assert "shared denominator" in dataset_semantics.scope
    assert "anchor_60bp" in " ".join(dataset_semantics.guardrails)
    context_semantics = resolve_plot_semantics(context, plot_id="context_robustness_summary")
    assert "4,096-row design-family-stratified sample" in context_semantics.scope
    umap_semantics = resolve_plot_semantics(context, plot_id="appendix_umap_gallery")
    assert "stored view matrices" in umap_semantics.preprocessing_md
    assert context.config.plots["design_centroid_margin_gallery"].kind == "xy_scatter_grid"
    assert context.config.plots["design_centroid_margin_gallery"].visibility_tier == "appendix"
    assert [option.column for option in context.config.plots["design_centroid_margin_gallery"].hue_options] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "log_likelihood_per_token_7b",
    ]
    assert context.config.plots["design_centroid_margin_gallery"].hue_options[-1].type == "continuous"
    assert context.config.plots["reference_alignment_summary"].kind == "metric_panel_grid"
    assert context.config.plots["reference_alignment_summary"].visibility_tier == "appendix"
    assert context.config.plots["representation_scree_diagnostic"].kind == "curve_grid"
    assert context.config.plots["representation_scree_diagnostic"].visibility_tier == "appendix"
    assert context.config.plots["context_delta_distributions"].kind == "distribution_grid"
    assert context.config.plots["context_delta_distributions"].visibility_tier == "debug"
    assert list(context.config.plots["context_delta_distributions"].metric_columns or []) == [
        "context_self_cosine",
        "context_shift_l2",
    ]
    assert list(context.config.plots["context_delta_distributions"].scalars) == [
        "context_delta_distribution_intermediate_embedding_7b",
        "context_delta_distribution_pooled_logits_7b",
    ]
    appendix_gallery = context.config.plots["appendix_umap_gallery"]
    assert appendix_gallery.kind == "projection_grid"
    assert appendix_gallery.visibility_tier == "appendix"
    assert appendix_gallery.shape_column is None
    assert appendix_gallery.default_hue == "design_family"
    assert [option.column for option in appendix_gallery.hue_options] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "log_likelihood_per_token_7b",
    ]
    assert appendix_gallery.hue_options[-1].type == "continuous"
    assert list(context.config.plots["design_centroid_margin_gallery"].scalars) == [
        "design_centroid_margins_intermediate_embedding_7b_anchor_60bp",
        "design_centroid_margins_pooled_logits_7b_anchor_60bp",
        "design_centroid_margins_intermediate_embedding_7b_full_context_1kb",
        "design_centroid_margins_pooled_logits_7b_full_context_1kb",
        "design_centroid_margins_intermediate_embedding_7b_full_context_anchor_mean",
        "design_centroid_margins_intermediate_embedding_7b_anchor_plus_full_context_concat",
        "design_centroid_margins_intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
    ]
    assert list(context.config.plots["representation_scree_diagnostic"].reducers) == [
        "pca_intermediate_embedding_7b_anchor_60bp",
        "pca_pooled_logits_7b_anchor_60bp",
        "pca_intermediate_embedding_7b_full_context_1kb",
        "pca_pooled_logits_7b_full_context_1kb",
        "pca_intermediate_embedding_7b_full_context_anchor_mean",
        "pca_intermediate_embedding_7b_anchor_plus_full_context_concat",
        "pca_intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
    ]
    assert list(appendix_gallery.projections) == [
        "umap_intermediate_embedding_7b_anchor_60bp",
        "umap_pooled_logits_7b_anchor_60bp",
        "umap_intermediate_embedding_7b_full_context_1kb",
        "umap_pooled_logits_7b_full_context_1kb",
        "umap_intermediate_embedding_7b_full_context_anchor_mean",
        "umap_intermediate_embedding_7b_anchor_plus_full_context_concat",
        "umap_intermediate_embedding_7b_anchor_plus_anchor_mean_concat",
    ]
    assert "reference_margin_gallery_wildtype" not in context.config.plots
    assert "reference_neighbor_evidence" not in context.config.plots
    assert "context_shift_reference_plane" not in context.config.plots
    assert "context_geometry_summary" not in context.config.plots
    assert "representation_tradeoff_scatter" not in context.config.plots
    assert all(getattr(plot, "semantics_ref", None) for plot in context.config.plots.values())

    assert context.config.deliverables["representation_health_summary"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/representation_health_summary"
    ]
    assert context.config.deliverables["design_structure_summary"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/design_structure_summary"
    ]
    assert context.config.deliverables["sigma35_ordinal_audit"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/sigma35_ordinal_audit"
    ]
    assert context.config.deliverables["context_robustness_summary"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/context_robustness_summary"
    ]
    assert context.config.deliverables["appendix_geometry_audit"].outputs["plots"] == [
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
    ]
    assert context.config.deliverables["appendix_umap_gallery"].outputs["plots"] == ["appendix_umap_gallery"]
    assert context.config.deliverables["appendix_umap_gallery"].outputs["notebooks"] == ["latent_geometry_browser"]
    assert context.config.exports == {}


def test_live_study_recipes_rebuild_from_clean_workspace_state() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    pre_assay_steps = {step.id: step for step in _recipe_steps(context, "pre_assay_representation_triage_recipe")}
    appendix_steps = {step.id: step for step in _recipe_steps(context, "appendix_umap_gallery_recipe")}

    assert "materialize_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "build_alignment_intermediate_embedding_20b_anchor_to_full_context" not in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "build_design_centroid_margins_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    design_margin_step = pre_assay_steps["build_design_centroid_margins_intermediate_embedding_7b_anchor_60bp"]
    assert design_margin_step.depends_on == ["materialize_intermediate_embedding_7b_anchor_60bp"]
    assert design_margin_step.params["leave_one_out"] is True
    assert design_margin_step.params["cohort_column"] == "design_family"
    assert "sample_id" not in design_margin_step.params
    assert "build_alignment_intermediate_embedding_7b_anchor_to_anchor_mean" in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_7b_full_context_anchor_mean" in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_7b_anchor_plus_full_context_concat" in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_7b_anchor_plus_anchor_mean_concat" in pre_assay_steps
    assert "build_representation_health_summary_metrics" in pre_assay_steps
    assert "build_design_structure_summary_metrics" in pre_assay_steps
    assert "build_sigma35_ordinal_audit_metrics" in pre_assay_steps
    assert "build_context_robustness_summary_metrics" in pre_assay_steps
    assert "build_reference_alignment_summary_metrics" in pre_assay_steps
    assert all(
        step.params.get("kind")
        not in {
            "similarity_margin",
            "candidate_metrics_long",
            "representation_scorecard",
            "candidate_metric_pairs",
            "candidate_metric_bars",
        }
        for step in pre_assay_steps.values()
        if getattr(step, "op", None) == "scalar.build"
    )
    assert (
        pre_assay_steps["build_sigma35_ordinal_audit_metrics"].params["sig35_order_path"]
        == "study_inputs/sig35_order.yaml"
    )
    assert pre_assay_steps["build_context_robustness_summary_metrics"].params["pairs"] == [
        {
            "pair_id": "intermediate_embedding_7b_anchor_vs_context",
            "label": "7B intermediate: anchor vs 1 kb seq mean",
            "alignment_id": "intermediate_embedding_7b_anchor_to_full_context",
            "anchor_view_id": "intermediate_embedding_7b_anchor_60bp",
            "context_view_id": "intermediate_embedding_7b_full_context_1kb",
        },
        {
            "pair_id": "intermediate_embedding_7b_anchor_vs_anchor_mean",
            "label": "7B intermediate: anchor vs context anchor mean",
            "alignment_id": "intermediate_embedding_7b_anchor_to_anchor_mean",
            "anchor_view_id": "intermediate_embedding_7b_anchor_60bp",
            "context_view_id": "intermediate_embedding_7b_full_context_anchor_mean",
        },
        {
            "pair_id": "pooled_logits_7b_anchor_vs_context",
            "label": "7B pooled logits: anchor vs 1 kb seq mean",
            "alignment_id": "pooled_logits_7b_anchor_to_full_context",
            "anchor_view_id": "pooled_logits_7b_anchor_60bp",
            "context_view_id": "pooled_logits_7b_full_context_1kb",
        },
    ]
    assert "render_design_centroid_margin_gallery" in pre_assay_steps
    assert "render_reference_alignment_summary" in pre_assay_steps
    assert "render_representation_scree_diagnostic" in pre_assay_steps
    assert "render_context_delta_distributions" in pre_assay_steps

    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    assert (
        appendix_steps["build_umap_sample_intermediate_embedding_7b_full_context_anchor_mean"].params["strategy"]
        == "all"
    )
    assert set(appendix_steps["generate_latent_geometry_browser"].depends_on) >= {
        "render_dataset_overview",
        "render_representation_health_summary",
        "render_design_structure_summary",
        "render_sigma35_ordinal_audit",
        "render_context_robustness_summary",
        "render_design_centroid_margin_gallery",
        "render_reference_alignment_summary",
        "render_representation_scree_diagnostic",
        "render_appendix_umap_gallery",
    }


def test_live_study_appendix_deliverable_docs_cover_current_appendix_surfaces() -> None:
    appendix_geometry_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "appendix_geometry_audit.md"
    ).read_text(encoding="utf-8")
    appendix_umap_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "appendix_umap_gallery.md"
    ).read_text(encoding="utf-8")

    parsed_geometry = _parse_deliverable_markdown(appendix_geometry_doc)
    parsed_umap = _parse_deliverable_markdown(appendix_umap_doc)

    assert "design_centroid_margin_gallery" in parsed_geometry["plot_sections"]
    assert "reference_alignment_summary" in parsed_geometry["plot_sections"]
    assert "representation_scree_diagnostic" in parsed_geometry["plot_sections"]
    assert "appendix_umap_gallery" in parsed_umap["plot_sections"]


def test_live_generated_catalog_and_controls_do_not_publish_retired_plot_surfaces() -> None:
    workspace = _live_workspace()
    workspace_catalog_from_context(load_workspace_config(workspace))
    catalog = json.loads((workspace / "outputs" / "catalog.json").read_text(encoding="utf-8"))
    controls = json.loads(
        (workspace / "outputs" / "notebooks" / "latent_geometry_browser" / "controls.json").read_text(encoding="utf-8")
    )

    retired_plot_ids = {
        "context_geometry_summary",
        "context_shift_reference_plane",
        "dual_margin_plane",
        "reference_margin_gallery_synthetic_centroids",
        "reference_margin_gallery_wildtype",
        "reference_neighbor_evidence",
        "representation_tradeoff_scatter",
    }
    catalog_plot_ids = {str(row["plot_id"]) for row in catalog.get("plots", []) if isinstance(row, dict)}
    assert retired_plot_ids.isdisjoint(catalog_plot_ids)
    plot_dirs = {path.name for path in (workspace / "outputs" / "plots").iterdir() if path.is_dir()}
    assert retired_plot_ids.isdisjoint(plot_dirs)

    joinable_tables = controls.get("geometry_controls", {}).get("joinable_tables", [])
    joinable_artifact_ids = {str(row.get("artifact_id")) for row in joinable_tables if isinstance(row, dict)}
    assert any(artifact_id.startswith("design_centroid_margins_") for artifact_id in joinable_artifact_ids)
    assert any(artifact_id.startswith("context_delta_distribution_") for artifact_id in joinable_artifact_ids)
    assert "context_geometry_summary_metrics" not in joinable_artifact_ids
    assert "reference_neighbor_evidence_metrics" not in joinable_artifact_ids
    scalar_dirs = {path.name for path in (workspace / "outputs" / "scalars").iterdir() if path.is_dir()}
    retired_scalar_ids = {
        "context_geometry_summary_metrics",
        "reference_neighbor_evidence_metrics",
        "tradeoff_cipro_context",
        "tradeoff_dual_context",
        "tradeoff_ethanol_context",
        "tradeoff_reference_neighbor_context",
    }
    assert retired_scalar_ids.isdisjoint(scalar_dirs)
    for row in joinable_tables:
        if not isinstance(row, dict):
            continue
        assert "wildtype_margin_ethanol_vs_control" not in row.get("columns", [])
        assert "wildtype_margin_cipro_vs_control" not in row.get("columns", [])
