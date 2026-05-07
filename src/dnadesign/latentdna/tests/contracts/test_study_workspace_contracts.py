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


def _regulondb_workspace() -> Path:
    return _repo_root() / "src" / "dnadesign" / "latentdna" / "workspaces" / "regulondb_native_promoter_panel"


def _recipe_steps(context, recipe_id: str) -> list[object]:
    return list(context.config.recipes[recipe_id].steps)


_EXPANDED_INTERMEDIATE_GEOMETRIES = [
    "intermediate_embedding_7b_anchor_60bp",
    "intermediate_embedding_7b_full_context_1kb",
    "intermediate_embedding_7b_full_context_anchor_mean",
    "intermediate_embedding_7b_reverse_complement_context_1kb",
    "intermediate_embedding_7b_reverse_complement_context_anchor_mean",
    "intermediate_embedding_7b_reference_core60",
    "intermediate_embedding_7b_reference_context_forward_1kb",
    "intermediate_embedding_7b_reference_context_forward_anchor_mean",
    "intermediate_embedding_7b_reference_context_reverse_complement_1kb",
    "intermediate_embedding_7b_reference_context_reverse_complement_anchor_mean",
]


_FIRST_CLASS_CANDIDATE_VIEWS = _EXPANDED_INTERMEDIATE_GEOMETRIES[:5]


def test_live_study_browser_controls_expose_sidecar_geometry_inventory() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")
    snapshot = json.loads((workspace / "outputs" / "status" / "workspace_snapshot.json").read_text(encoding="utf-8"))

    geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]
    # Clean CI checkouts do not include generated joinable scalar tables, so the
    # checked-in snapshot carries the pre-assay browser hue contract there.
    preferred_hues = controls.geometry_controls.preferred_hues or snapshot["browser"]["preferred_hues"]

    assert controls.schema_version == "latentdna.workspace_notebook_controls.v4"
    assert controls.workspace_id == "stress_ethanol_cipro_growth"
    assert controls.notebook_id == "latent_geometry_browser"
    assert controls.plot_controls.default_surface == "plots"
    assert controls.plot_controls.ordered_plot_ids == [
        "dataset_overview",
        "representation_health_summary",
        "design_structure_summary",
        "balanced_design_family_margin_gallery",
        "sigma35_ordinal_audit",
        "sigma35_margin_ladder_gallery",
        "sigma35_stress_margin_gallery",
        "context_robustness_summary",
        "context_pair_summary",
        "candidate_decision_frontier",
        "sigma35_centroid_distance_gallery",
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "reference_core60_strength_umap",
        "reference_core60_pca_scree",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    ]
    assert geometry_ids == _EXPANDED_INTERMEDIATE_GEOMETRIES
    geometry_roles = {row.view_id: row.role for row in controls.geometry_controls.geometries}
    assert geometry_roles["intermediate_embedding_7b_anchor_60bp"] == "primary"
    assert geometry_roles["intermediate_embedding_7b_full_context_anchor_mean"] == "primary"
    assert geometry_roles["intermediate_embedding_7b_full_context_1kb"] == "appendix"
    assert geometry_roles["intermediate_embedding_7b_reference_core60"] == "appendix"
    assert controls.geometry_controls.default_compare_left == "intermediate_embedding_7b_anchor_60bp"
    assert controls.geometry_controls.default_compare_right == "intermediate_embedding_7b_full_context_anchor_mean"
    assert controls.geometry_controls.default_layout == "candidate_grid"
    candidate_grid_layout = next(row for row in controls.geometry_controls.layout_presets if row.id == "candidate_grid")
    assert candidate_grid_layout.view_ids == _FIRST_CLASS_CANDIDATE_VIEWS
    assert [row.candidate_set_id for row in controls.geometry_controls.candidate_sets] == [
        "first_class_intermediate_7b",
        "expanded_reference_intermediate_7b",
        "planned_output_layer_7b",
    ]
    assert controls.geometry_controls.candidate_sets[0].view_ids == _FIRST_CLASS_CANDIDATE_VIEWS
    assert controls.geometry_controls.candidate_sets[0].available_view_ids == _FIRST_CLASS_CANDIDATE_VIEWS
    assert controls.geometry_controls.candidate_sets[2].views[0].status == "planned"
    assert "design_family" in preferred_hues
    assert "sig35_variant" in preferred_hues
    assert "spacer_length" in preferred_hues
    assert "promoter_standard__collection_id" not in preferred_hues
    assert "promoter_standard__strength_value_numeric" not in preferred_hues
    assert "log_likelihood_per_token_7b" not in preferred_hues
    assert "wildtype_margin_ethanol_vs_control" not in preferred_hues
    assert "wildtype_margin_cipro_vs_control" not in preferred_hues
    assert "sigma70_variant" not in preferred_hues
    assert "sigma70_strength_class" not in preferred_hues
    assert "construct__context_id" not in preferred_hues
    if controls.geometry_controls.preferred_hues:
        assert controls.geometry_controls.hue_kinds["design_family"] == "categorical"
        assert controls.geometry_controls.hue_kinds["is_control"] == "binary"
        assert controls.geometry_controls.hue_kinds["spacer_length"] == "ordinal"
    else:
        assert controls.geometry_controls.joinable_tables == []
    assert controls.geometry_controls.reference_labels == ["spyP", "sulAp"]
    assert {row.reference_set_id for row in controls.geometry_controls.reference_sets} >= {
        "reference_spyp_sulap",
        "reference_spyp_sulap_core60",
        "reference_native_mg1655",
        "reference_native_mg1655_core60",
        "reference_anderson_igem",
        "reference_anderson_igem_core60",
        "reference_w_collection",
        "reference_w_collection_core60",
    }


def test_regulondb_deliverable_docs_cover_notebook_visible_plots() -> None:
    workspace = _regulondb_workspace()
    context = load_workspace_config(workspace)
    notebook = context.require_notebook("latent_geometry_browser")
    ordered_plot_ids = set(notebook.ordered_plots)
    covered_plot_ids: set[str] = set()

    assert context.config.study_binding is not None
    docs_root = _repo_root() / context.config.study_binding.deliverable_docs_root
    for deliverable in context.config.deliverables.values():
        if not ordered_plot_ids.intersection(deliverable.outputs.get("plots", [])):
            continue
        for docs_ref in deliverable.docs_refs:
            relative_ref = docs_ref.removeprefix(f"study:{context.config.study_binding.study_id}/")
            markdown_path = docs_root / f"{relative_ref}.md"
            if markdown_path.is_file():
                parsed = _parse_deliverable_markdown(markdown_path.read_text(encoding="utf-8"))
                covered_plot_ids.update(parsed["plot_sections"])

    assert ordered_plot_ids.issubset(covered_plot_ids)


def test_regulondb_umap_plots_expose_metadata_hue_contract() -> None:
    context = load_workspace_config(_regulondb_workspace())
    expected_hues = [
        "regulondb__sigma_factor_set",
        "regulondb__confidence_level_set",
        "regulondb__metadata_completeness_class",
        "regulondb__regulator_composition",
        "regulondb__box_pattern",
        "regulondb__source_strata_set",
        "regulondb__sigma_factor_count",
        "emitted_length_bp",
    ]
    for plot_id in [
        "sigma_umap_intermediate_embedding_7b_native_source_record_seq_mean",
        "sigma_umap_intermediate_embedding_7b_core60_tss_upstream",
    ]:
        plot = context.config.plots[plot_id]
        assert plot.default_hue == "regulondb__sigma_factor_set"
        assert [option.column for option in plot.hue_options] == expected_hues
        assert plot.hue_options[-2].type == "ordinal"
        assert plot.hue_options[-1].type == "ordinal"


def test_notebook_plot_semantics_name_screen_encoding_contracts() -> None:
    requirements = {
        (_live_workspace(), "appendix_umap_gallery"): ["scatter", "fixed coordinates"],
        (_live_workspace(), "design_centroid_margin_gallery"): ["x-axis", "y-axis"],
        (_regulondb_workspace(), "sigma_umap_intermediate_embedding_7b_native_source_record_seq_mean"): [
            "projection",
            "scatter",
        ],
        (_regulondb_workspace(), "sigma_umap_intermediate_embedding_7b_core60_tss_upstream"): [
            "projection",
            "scatter",
        ],
    }

    for (workspace, plot_id), phrases in requirements.items():
        context = load_workspace_config(workspace)
        semantics = resolve_plot_semantics(context, plot_id=plot_id)
        visible_text = "\n".join(
            [
                semantics.encoding,
                semantics.caption,
                semantics.alt_text,
                semantics.preprocessing_md,
                semantics.math_md,
                semantics.limitations_md,
                semantics.failure_modes_md,
            ]
        ).lower()

        for phrase in phrases:
            assert phrase in visible_text, f"{plot_id} semantics should mention {phrase!r}"


def test_live_study_representation_health_compares_first_class_intermediate_and_output_layer_views() -> None:
    context = load_workspace_config(_live_workspace())
    recipe = context.config.recipes["pre_assay_representation_triage_recipe"]
    health_step = next(step for step in recipe.steps if step.id == "build_representation_health_summary_metrics")
    candidates = {str(row["view_id"]) for row in health_step.params["candidates"]}
    omitted = {str(row["view_id"]) for row in health_step.params.get("omitted_candidates", [])}

    expected_first_class_output_views = {
        "output_layer_mean_7b_anchor_60bp",
        "output_layer_mean_7b_full_context_1kb",
        "output_layer_mean_7b_full_context_anchor_mean",
        "output_layer_mean_7b_reverse_complement_context_1kb",
        "output_layer_mean_7b_reverse_complement_context_anchor_mean",
    }
    assert expected_first_class_output_views.issubset(candidates)
    assert not expected_first_class_output_views.intersection(omitted)
    assert all("reference" not in view_id for view_id in candidates)
    assert all("reference" not in view_id for view_id in omitted)


def test_live_study_reference_context_sources_do_not_inherit_promoter_metadata_derivations() -> None:
    context = load_workspace_config(_live_workspace())
    source_ids = [source_id for source_id in context.config.sources if source_id.startswith("reference_context_7b_")]

    assert source_ids
    for source_id in source_ids:
        source = context.config.sources[source_id]
        assert source.metadata_include_mode == "replace"
        assert source.metadata_include == []


def test_live_study_snapshot_and_deliverables_follow_pre_assay_contract() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    snapshot = json.loads((workspace / "outputs" / "status" / "workspace_snapshot.json").read_text(encoding="utf-8"))
    controls = build_workspace_notebook_controls_payload(context, notebook_id="latent_geometry_browser")

    assert snapshot["schema_version"] == "latentdna.workspace_snapshot.v1"
    assert snapshot["workspace_id"] == "stress_ethanol_cipro_growth"
    assert snapshot["sources"]["anchor_7b_seq_mean_log_likelihood_total"]["dataset_id"] == "usr_prom_eth_cip_anchor"
    assert context.config.sources["reference_native"].dataset == "usr_promoter_references"
    assert context.config.sources["reference_core60"].dataset == "construct_prom_eth_cip_reference_core60"
    assert context.config.sources["reference_contexts"].dataset == "construct_prom_eth_cip_reference_contexts"
    assert context.config.views["intermediate_embedding_7b_reference_core60"].role == "appendix"
    expected_appendix_reference_views = [
        "intermediate_embedding_7b_reference_context_forward_1kb",
        "intermediate_embedding_7b_reference_context_forward_anchor_mean",
        "intermediate_embedding_7b_reference_context_reverse_complement_1kb",
        "intermediate_embedding_7b_reference_context_reverse_complement_anchor_mean",
    ]
    assert all(context.config.views[view_id].role == "appendix" for view_id in expected_appendix_reference_views)
    expected_planned_reference_views = [
        "output_layer_mean_7b_reference_core60",
        "output_layer_mean_7b_reference_context_forward_1kb",
        "output_layer_mean_7b_reference_context_forward_anchor_mean",
        "output_layer_mean_7b_reference_context_reverse_complement_1kb",
        "output_layer_mean_7b_reference_context_reverse_complement_anchor_mean",
    ]
    assert all(context.config.views[view_id].role == "planned" for view_id in expected_planned_reference_views)
    initial_control_geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]
    assert all(view_id in initial_control_geometry_ids for view_id in expected_appendix_reference_views)
    assert all(view_id not in initial_control_geometry_ids for view_id in expected_planned_reference_views)
    expected_total_log_likelihood_sources = [
        "anchor_7b_seq_mean_log_likelihood_total",
        "full_context_7b_forward_anchor_mean_log_likelihood_total",
        "full_context_7b_forward_seq_mean_log_likelihood_total",
        "full_context_7b_reverse_complement_anchor_mean_log_likelihood_total",
        "full_context_7b_reverse_complement_seq_mean_log_likelihood_total",
        "reference_core60_7b_core60_mean_log_likelihood_total",
        "reference_context_7b_forward_anchor_mean_log_likelihood_total",
        "reference_context_7b_forward_seq_mean_log_likelihood_total",
        "reference_context_7b_reverse_complement_anchor_mean_log_likelihood_total",
        "reference_context_7b_reverse_complement_seq_mean_log_likelihood_total",
    ]
    assert all(source_id in context.config.sources for source_id in expected_total_log_likelihood_sources)
    expected_decision_prefix = [
        "dataset_overview",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "candidate_decision_frontier",
    ]
    decision_ladder = snapshot["decision_ladder"]
    assert decision_ladder == expected_decision_prefix
    browser_geometry_ids = snapshot["browser"]["default_geometry_ids"]
    control_geometry_ids = [row.view_id for row in controls.geometry_controls.geometries]
    assert browser_geometry_ids == _EXPANDED_INTERMEDIATE_GEOMETRIES
    assert control_geometry_ids == _EXPANDED_INTERMEDIATE_GEOMETRIES
    assert controls.geometry_controls.default_model == "7b"
    assert controls.geometry_controls.default_family == "intermediate_embedding"
    assert controls.geometry_controls.default_layout == "candidate_grid"
    assert "spacer_length" in snapshot["browser"]["preferred_hues"]
    assert "promoter_standard__collection_id" not in snapshot["browser"]["preferred_hues"]
    assert "promoter_standard__strength_value_numeric" not in snapshot["browser"]["preferred_hues"]
    assert "log_likelihood_per_token_7b" not in snapshot["browser"]["preferred_hues"]
    assert "wildtype_margin_ethanol_vs_control" not in snapshot["browser"]["preferred_hues"]
    assert "wildtype_margin_cipro_vs_control" not in snapshot["browser"]["preferred_hues"]

    assert context.config.plots["representation_health_summary"].kind == "metric_panel_grid"
    assert context.config.plots["design_structure_summary"].kind == "metric_panel_grid"
    assert context.config.plots["balanced_design_family_margin_gallery"].kind == "xy_scatter_grid"
    assert context.config.plots["balanced_design_family_margin_gallery"].visibility_tier == "primary"
    assert context.config.plots["balanced_design_family_margin_gallery"].default_hue == "sig35_variant"
    assert context.config.plots["balanced_design_family_margin_gallery"].x_axis_label == (
        r"$m_{\mathrm{eth}}(x)=\cos(z_x,c_{\mathrm{eth}})-\cos(z_x,c_{\mathrm{bg}})$"
    )
    assert context.config.plots["balanced_design_family_margin_gallery"].y_axis_label == (
        r"$m_{\mathrm{cipro}}(x)=\cos(z_x,c_{\mathrm{cipro}})-\cos(z_x,c_{\mathrm{bg}})$"
    )
    assert context.config.plots["sigma35_ordinal_audit"].kind == "metric_panel_grid"
    assert context.config.plots["sigma35_margin_ladder_gallery"].kind == "distribution_grid"
    assert context.config.plots["sigma35_margin_ladder_gallery"].visibility_tier == "primary"
    assert context.config.plots["sigma35_margin_ladder_gallery"].x_axis_label == "Sigma-35 variant"
    assert context.config.plots["sigma35_margin_ladder_gallery"].y_axis_label == (
        r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
    )
    assert context.config.plots["sigma35_stress_margin_gallery"].kind == "xy_scatter_grid"
    assert context.config.plots["sigma35_stress_margin_gallery"].visibility_tier == "primary"
    assert context.config.plots["sigma35_stress_margin_gallery"].x_column == "sig35_margin_f_vs_b"
    assert context.config.plots["sigma35_stress_margin_gallery"].y_column == "synthetic_best_stress_margin"
    assert context.config.plots["sigma35_stress_margin_gallery"].default_hue == "sig35_variant"
    assert context.config.plots["sigma35_stress_margin_gallery"].x_axis_label == (
        r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
    )
    assert context.config.plots["sigma35_stress_margin_gallery"].y_axis_label == (
        r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$"
    )
    assert context.config.plots["context_robustness_summary"].kind == "metric_panel_grid"
    assert context.config.plots["context_pair_summary"].kind == "metric_panel_grid"
    assert context.config.plots["candidate_decision_frontier"].kind == "xy_scatter"
    assert context.config.plots["candidate_decision_frontier"].color_column is None
    assert context.config.plots["candidate_decision_frontier"].size_column == "effective_rank"
    assert context.config.plots["candidate_decision_frontier"].size_range == (140.0, 260.0)
    assert context.config.plots["sigma35_centroid_distance_gallery"].kind == "heatmap_grid"
    assert context.config.plots["sigma35_centroid_distance_gallery"].visibility_tier == "appendix"
    assert context.config.plots["sigma35_centroid_distance_gallery"].x_axis_label == "Sigma-35 variant $h$"
    assert context.config.plots["sigma35_centroid_distance_gallery"].y_axis_label == "Sigma-35 variant $g$"
    assert context.config.plots["sigma35_centroid_distance_gallery"].colorbar_label == (
        r"$d_{\mathrm{emb}}(g,h)=1-\cos(c_g,c_h)$"
    )
    assert resolve_plot_semantics(context, plot_id="representation_health_summary").decision_role == "gate"
    dataset_semantics = resolve_plot_semantics(context, plot_id="dataset_overview")
    assert "generation plan" in dataset_semantics.scope
    assert "shared denominator" in dataset_semantics.scope
    assert "merged anchor-source insert" in " ".join(dataset_semantics.guardrails)
    context_semantics = resolve_plot_semantics(context, plot_id="context_robustness_summary")
    assert "4,096-row design-family-stratified sample" in context_semantics.scope
    umap_semantics = resolve_plot_semantics(context, plot_id="appendix_umap_gallery")
    assert "stored view matrices" in umap_semantics.preprocessing_md
    assert context.config.plots["design_centroid_margin_gallery"].kind == "xy_scatter_grid"
    assert context.config.plots["design_centroid_margin_gallery"].visibility_tier == "appendix"
    expected_reference_hue_options = [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "source_family",
    ]
    design_hue_options = context.config.plots["design_centroid_margin_gallery"].hue_options
    assert [option.column for option in design_hue_options] == expected_reference_hue_options
    assert design_hue_options[3].type == "ordinal"
    assert context.config.plots["reference_alignment_summary"].kind == "metric_panel_grid"
    assert context.config.plots["reference_alignment_summary"].visibility_tier == "appendix"
    assert context.config.plots["representation_scree_diagnostic"].kind == "curve_grid"
    assert context.config.plots["representation_scree_diagnostic"].visibility_tier == "appendix"
    assert context.config.plots["context_pair_summary"].visibility_tier == "primary"
    assert context.config.plots["context_pair_summary"].scalar == "context_pair_summary_metrics"
    assert context.config.plots["context_pair_summary"].label_column == "label"
    assert context.config.plots["context_pair_summary"].color_column == "comparison_role"
    appendix_gallery = context.config.plots["appendix_umap_gallery"]
    assert appendix_gallery.kind == "projection_grid"
    assert appendix_gallery.visibility_tier == "appendix"
    assert appendix_gallery.shape_column is None
    assert appendix_gallery.default_hue == "sig35_variant"
    assert [option.column for option in appendix_gallery.hue_options] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "source_family",
    ]
    assert appendix_gallery.hue_options[3].type == "ordinal"
    assert appendix_gallery.annotation is not None
    assert appendix_gallery.annotation.reference_set == "reference_spyp_sulap"
    assert list(context.config.plots["design_centroid_margin_gallery"].scalars) == [
        f"design_centroid_margins_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(context.config.plots["balanced_design_family_margin_gallery"].scalars) == [
        f"balanced_design_family_margins_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(context.config.plots["sigma35_stress_margin_gallery"].scalars) == [
        f"sigma35_stress_margins_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(context.config.plots["sigma35_margin_ladder_gallery"].scalars) == [
        f"sigma35_stress_margins_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(context.config.plots["sigma35_centroid_distance_gallery"].scalars) == [
        f"sigma35_centroid_distance_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(context.config.plots["representation_scree_diagnostic"].reducers) == [
        f"pca_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS
    ]
    assert list(appendix_gallery.projections) == [f"umap_{view_id}" for view_id in _FIRST_CLASS_CANDIDATE_VIEWS]
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
    assert context.config.deliverables["candidate_decision_frontier"].docs_refs == [
        "study:stress_ethanol_cipro_growth/deliverables/candidate_decision_frontier"
    ]
    assert context.config.deliverables["design_structure_summary"].outputs["plots"] == [
        "design_structure_summary",
        "balanced_design_family_margin_gallery",
    ]
    assert context.config.deliverables["sigma35_ordinal_audit"].outputs["plots"] == [
        "sigma35_ordinal_audit",
        "sigma35_margin_ladder_gallery",
        "sigma35_stress_margin_gallery",
        "sigma35_centroid_distance_gallery",
    ]
    assert context.config.deliverables["context_robustness_summary"].outputs["plots"] == [
        "context_robustness_summary",
        "context_pair_summary",
    ]
    assert context.config.deliverables["candidate_decision_frontier"].outputs["plots"] == [
        "candidate_decision_frontier"
    ]
    assert context.config.deliverables["appendix_geometry_review"].outputs["plots"] == [
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
    ]
    assert context.config.deliverables["appendix_umap_gallery"].outputs["plots"] == [
        "appendix_umap_gallery",
        "reference_core60_strength_umap",
        "reference_core60_pca_scree",
    ]
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
    assert "build_scorecard_sample_intermediate_embedding_7b_anchor_plus_full_context_concat" not in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_7b_anchor_plus_anchor_mean_concat" not in pre_assay_steps
    assert "build_representation_health_summary_metrics" in pre_assay_steps
    representation_health_step = pre_assay_steps["build_representation_health_summary_metrics"]
    assert representation_health_step.params["pairwise_max_rows"] == 4096
    assert representation_health_step.params["pairwise_seed"] == 17
    first_class_output_ids = {
        "output_layer_mean_7b_anchor_60bp",
        "output_layer_mean_7b_full_context_1kb",
        "output_layer_mean_7b_full_context_anchor_mean",
        "output_layer_mean_7b_reverse_complement_context_1kb",
        "output_layer_mean_7b_reverse_complement_context_anchor_mean",
    }
    representation_health_candidates = {row["view_id"] for row in representation_health_step.params["candidates"]}
    omitted_candidate_ids = {row["view_id"] for row in representation_health_step.params.get("omitted_candidates", [])}
    assert first_class_output_ids.issubset(representation_health_candidates)
    assert omitted_candidate_ids == set()
    for view_id in first_class_output_ids:
        assert f"materialize_{view_id}" in pre_assay_steps
        assert f"build_scorecard_sample_{view_id}" in pre_assay_steps
        assert f"reduce_pca_{view_id}" in pre_assay_steps
    assert "build_design_structure_summary_metrics" in pre_assay_steps
    assert "build_sigma35_ordinal_audit_metrics" in pre_assay_steps
    stress_margin_anchor = pre_assay_steps["build_sigma35_stress_margins_intermediate_embedding_7b_anchor_60bp"]
    stress_margin_anchor_mean = pre_assay_steps[
        "build_sigma35_stress_margins_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert "sample_id" not in stress_margin_anchor.params
    assert "sample_id" not in stress_margin_anchor_mean.params
    balanced_margin_anchor = pre_assay_steps[
        "build_balanced_design_family_margins_intermediate_embedding_7b_anchor_60bp"
    ]
    balanced_margin_anchor_mean = pre_assay_steps[
        "build_balanced_design_family_margins_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert balanced_margin_anchor.params["balance_reference_only"] is True
    assert "sample_id" not in balanced_margin_anchor.params
    assert "sample_id" not in balanced_margin_anchor_mean.params
    centroid_distance_anchor = pre_assay_steps["build_sigma35_centroid_distance_intermediate_embedding_7b_anchor_60bp"]
    centroid_distance_anchor_mean = pre_assay_steps[
        "build_sigma35_centroid_distance_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert centroid_distance_anchor.params["kind"] == "axis_centroid_distance"
    assert centroid_distance_anchor.params["axis"]["axis_id"] == "sigma35"
    assert centroid_distance_anchor.params["axis"]["column"] == "sig35_variant"
    assert "sample_id" not in centroid_distance_anchor.params
    assert "sample_id" not in centroid_distance_anchor_mean.params
    assert (
        "build_sigma35_centroid_distance_intermediate_embedding_7b_anchor_plus_anchor_mean_concat"
        not in pre_assay_steps
    )
    assert "build_context_delta_distribution_intermediate_embedding_7b_anchor_mean" in pre_assay_steps
    assert "build_context_delta_distribution_intermediate_embedding_7b_full_context" in pre_assay_steps
    assert "build_context_pair_summary_metrics" in pre_assay_steps
    assert "build_context_robustness_summary_metrics" in pre_assay_steps
    assert "build_reference_alignment_summary_metrics" in pre_assay_steps
    assert "build_candidate_decision_frontier_metrics" in pre_assay_steps
    assert "build_context_delta_distribution_output_layer_mean_7b" not in pre_assay_steps
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
    ordinal_axis_params = pre_assay_steps["build_sigma35_ordinal_audit_metrics"].params
    assert ordinal_axis_params["kind"] == "ordinal_axis_audit"
    assert ordinal_axis_params["axis"]["axis_id"] == "sigma35"
    assert ordinal_axis_params["axis"]["column"] == "sig35_variant"
    assert ordinal_axis_params["axis"]["order_path"] == "study_inputs/sig35_order.yaml"
    assert ordinal_axis_params["axis"]["metric_ids"]["spearman"] == "sig35_ordinal_spearman"
    assert [row["column"] for row in ordinal_axis_params["axis"]["within_groups"]] == [
        "design_family",
        "design_regulator_composition",
    ]
    assert pre_assay_steps["build_candidate_decision_frontier_metrics"].params["ordinal_scalar"] == (
        "sigma35_ordinal_audit_metrics"
    )
    assert pre_assay_steps["build_candidate_decision_frontier_metrics"].params["ordinal_metric_id"] == (
        "sig35_ordinal_spearman"
    )
    context_pair_params = pre_assay_steps["build_context_pair_summary_metrics"].params
    assert [row["comparison_id"] for row in context_pair_params["comparisons"]] == [
        "intermediate_embedding_7b_anchor_vs_anchor_mean",
        "intermediate_embedding_7b_anchor_vs_full_context",
    ]
    reference_params = pre_assay_steps["build_reference_alignment_summary_metrics"].params
    assert "reference_group_columns" not in reference_params
    assert reference_params["reference_sets"] == [
        "reference_spyp_sulap",
        "reference_spyp_sulap_core60",
        "reference_sfxi_archive",
        "reference_native_mg1655",
        "reference_native_mg1655_core60",
        "reference_anderson_igem",
        "reference_anderson_igem_core60",
        "reference_w_collection",
        "reference_w_collection_core60",
    ]
    assert pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b_anchor_mean"].params[
        "where"
    ] == {
        "column": "source_class",
        "equals": "densegen",
    }
    assert (
        pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b_anchor_mean"].params[
            "table_sample_only"
        ]
        is True
    )
    assert pre_assay_steps["build_context_robustness_summary_metrics"].params["pairs"] == [
        {
            "pair_id": "intermediate_embedding_7b_anchor_vs_anchor_mean",
            "label": "7B intermediate: anchor vs context anchor mean",
            "alignment_id": "intermediate_embedding_7b_anchor_to_anchor_mean",
            "anchor_view_id": "intermediate_embedding_7b_anchor_60bp",
            "context_view_id": "intermediate_embedding_7b_full_context_anchor_mean",
        },
    ]
    assert "render_design_centroid_margin_gallery" in pre_assay_steps
    assert "render_reference_alignment_summary" in pre_assay_steps
    assert "render_sigma35_stress_margin_gallery" in pre_assay_steps
    assert "render_sigma35_centroid_distance_gallery" in pre_assay_steps
    assert "render_candidate_decision_frontier" in pre_assay_steps
    assert "render_representation_scree_diagnostic" in pre_assay_steps
    assert "render_context_pair_summary" in pre_assay_steps

    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    for view_id in _FIRST_CLASS_CANDIDATE_VIEWS:
        assert f"build_scorecard_sample_{view_id}" in pre_assay_steps
        assert f"reduce_pca_{view_id}" in pre_assay_steps
        assert f"build_design_centroid_margins_{view_id}" in pre_assay_steps
        assert f"build_balanced_design_family_margins_{view_id}" in pre_assay_steps
        assert f"build_sigma35_stress_margins_{view_id}" in pre_assay_steps
        assert f"build_sigma35_centroid_distance_{view_id}" in pre_assay_steps
        assert f"build_umap_sample_{view_id}" in appendix_steps
        assert f"fit_umap_{view_id}" in appendix_steps
        assert appendix_steps[f"build_umap_sample_{view_id}"].params["strategy"] == "all"
    assert set(appendix_steps["generate_latent_geometry_browser"].depends_on) >= {
        "render_dataset_overview",
        "render_representation_health_summary",
        "render_design_structure_summary",
        "render_sigma35_ordinal_audit",
        "render_sigma35_stress_margin_gallery",
        "render_context_robustness_summary",
        "render_context_pair_summary",
        "render_candidate_decision_frontier",
        "render_sigma35_centroid_distance_gallery",
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
        / "appendix_geometry_review.md"
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


def test_live_study_primary_deliverable_docs_cover_companion_and_frontier_surfaces() -> None:
    sigma_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "sigma35_ordinal_audit.md"
    ).read_text(encoding="utf-8")
    context_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "context_robustness_summary.md"
    ).read_text(encoding="utf-8")
    frontier_doc = (
        _repo_root()
        / "src"
        / "dnadesign"
        / "studies"
        / "stress_ethanol_cipro_growth"
        / "deliverables"
        / "candidate_decision_frontier.md"
    ).read_text(encoding="utf-8")

    parsed_sigma = _parse_deliverable_markdown(sigma_doc)
    parsed_context = _parse_deliverable_markdown(context_doc)
    parsed_frontier = _parse_deliverable_markdown(frontier_doc)

    assert "sigma35_ordinal_audit" in parsed_sigma["plot_sections"]
    assert "sigma35_stress_margin_gallery" in parsed_sigma["plot_sections"]
    assert "sigma35_centroid_distance_gallery" in parsed_sigma["plot_sections"]
    assert "context_robustness_summary" in parsed_context["plot_sections"]
    assert "context_pair_summary" in parsed_context["plot_sections"]
    assert "candidate_decision_frontier" in parsed_frontier["plot_sections"]


def test_live_generated_catalog_and_controls_do_not_publish_retired_plot_surfaces() -> None:
    workspace = _live_workspace()
    context = load_workspace_config(workspace)
    workspace_catalog_from_context(context)
    catalog = json.loads((workspace / "outputs" / "catalog.json").read_text(encoding="utf-8"))
    controls = build_workspace_notebook_controls_payload(
        context,
        notebook_id="latent_geometry_browser",
        catalog_payload=catalog,
    ).model_dump(mode="json")

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
    plots_root = workspace / "outputs" / "plots"
    plot_dirs = {path.name for path in plots_root.iterdir() if path.is_dir()} if plots_root.exists() else set()
    assert retired_plot_ids.isdisjoint(plot_dirs)

    joinable_tables = controls.get("geometry_controls", {}).get("joinable_tables", [])
    joinable_artifact_ids = {str(row.get("artifact_id")) for row in joinable_tables if isinstance(row, dict)}
    if joinable_artifact_ids:
        assert any(artifact_id.startswith("design_centroid_margins_") for artifact_id in joinable_artifact_ids)
        assert any(artifact_id.startswith("context_delta_distribution_") for artifact_id in joinable_artifact_ids)
    assert "context_geometry_summary_metrics" not in joinable_artifact_ids
    assert "reference_neighbor_evidence_metrics" not in joinable_artifact_ids
    scalars_root = workspace / "outputs" / "scalars"
    scalar_dirs = {path.name for path in scalars_root.iterdir() if path.is_dir()} if scalars_root.exists() else set()
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
