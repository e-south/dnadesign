"""Contracts for the promoter-study pre-assay representation-triage template."""

from __future__ import annotations

import yaml

from dnadesign.latentdna.src.workspaces.paths import builtin_templates_dir


def _template_payload() -> dict[str, object]:
    config_path = builtin_templates_dir() / "promoter_reference_margin_benchmark" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _recipe_steps(payload: dict[str, object], recipe_id: str) -> list[dict[str, object]]:
    return list(payload["recipes"][recipe_id]["steps"])


def test_template_tracks_canonical_views_and_pre_assay_deliverables() -> None:
    payload = _template_payload()
    visible_views = [
        view_id for view_id, view in payload["views"].items() if str(view.get("role", "")).strip().lower() != "hidden"
    ]

    assert set(visible_views) == {
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "intermediate_embedding_7b_full_context_anchor_mean",
        "output_layer_mean_7b_anchor_60bp",
        "output_layer_mean_7b_full_context_1kb",
    }
    assert payload["views"]["intermediate_embedding_7b_full_context_anchor_mean"]["role"] == "primary"
    assert payload["candidate_sets"]["five_view_7b_triage"]["views"] == [
        "intermediate_embedding_7b_anchor_60bp",
        "output_layer_mean_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "output_layer_mean_7b_full_context_1kb",
        "intermediate_embedding_7b_full_context_anchor_mean",
    ]
    assert list(payload["deliverables"]) == [
        "dataset_overview",
        "representation_health_summary",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "candidate_decision_frontier",
        "appendix_geometry_review",
        "appendix_umap_gallery",
    ]


def test_template_uses_required_references_and_trimmed_browser_surface() -> None:
    payload = _template_payload()

    assert set(payload["landmarks"]) == {"spyp", "sulap", "j23105"}
    assert payload["reference_sets"]["reference_spyp_sulap"]["display_labels"] == {
        "spyP": "spyP",
        "sulAp": "sulAp",
    }
    assert list(payload["notebooks"]) == ["latent_geometry_browser"]
    assert payload["notebooks"]["latent_geometry_browser"]["default_deliverable"] == "representation_health_summary"
    assert payload["notebooks"]["latent_geometry_browser"]["default_surface"] == "plots"
    assert payload["notebooks"]["latent_geometry_browser"]["candidate_sets"] == ["five_view_7b_triage"]
    assert payload["notebooks"]["latent_geometry_browser"]["default_candidate_set"] == "five_view_7b_triage"
    assert payload["notebooks"]["latent_geometry_browser"]["ordered_plots"] == [
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
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    ]
    assert payload["exports"] == {}
    assert set(payload["alignments"]) == {
        "intermediate_embedding_7b_anchor_to_full_context",
        "intermediate_embedding_7b_anchor_to_anchor_mean",
        "output_layer_mean_7b_anchor_to_full_context",
    }
    assert all("semantics_ref" in plot for plot in payload["plots"].values())

    assert payload["plots"]["representation_health_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["design_structure_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["balanced_design_family_margin_gallery"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["balanced_design_family_margin_gallery"]["visibility_tier"] == "primary"
    assert payload["plots"]["balanced_design_family_margin_gallery"]["default_hue"] == "sig35_variant"
    assert payload["plots"]["sigma35_ordinal_audit"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["sigma35_margin_ladder_gallery"]["kind"] == "distribution_grid"
    assert payload["plots"]["sigma35_margin_ladder_gallery"]["visibility_tier"] == "primary"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["visibility_tier"] == "primary"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["x_column"] == "sig35_margin_f_vs_b"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["y_column"] == "synthetic_best_stress_margin"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["default_hue"] == "sig35_variant"
    assert payload["plots"]["sigma35_centroid_distance_gallery"]["kind"] == "heatmap_grid"
    assert payload["plots"]["sigma35_centroid_distance_gallery"]["visibility_tier"] == "appendix"
    assert payload["plots"]["context_robustness_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["context_pair_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["candidate_decision_frontier"]["kind"] == "xy_scatter"
    assert payload["plots"]["candidate_decision_frontier"]["size_column"] == "effective_rank"
    assert payload["plots"]["candidate_decision_frontier"]["size_range"] == [140, 260]
    assert payload["plots"]["candidate_decision_frontier"]["x_axis_label"] == (
        r"$S_{\mathrm{design}}^{\mathrm{balanced}}="
        r"\operatorname{mean}(d_{\mathrm{between}})/\operatorname{mean}(d_{\mathrm{within}})$"
    )
    assert payload["plots"]["candidate_decision_frontier"]["y_axis_label"] == (
        r"$\rho_{\sigma35}=\operatorname{Spearman}(\Delta_{\mathrm{expected}},\Delta_{\mathrm{observed}})$"
    )
    assert payload["plots"]["balanced_design_family_margin_gallery"]["x_axis_label"] == (
        r"$m_{\mathrm{eth}}(x)=\cos(z_x,c_{\mathrm{eth}})-\cos(z_x,c_{\mathrm{bg}})$"
    )
    assert payload["plots"]["balanced_design_family_margin_gallery"]["y_axis_label"] == (
        r"$m_{\mathrm{cipro}}(x)=\cos(z_x,c_{\mathrm{cipro}})-\cos(z_x,c_{\mathrm{bg}})$"
    )
    assert payload["plots"]["sigma35_margin_ladder_gallery"]["y_axis_label"] == (
        r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
    )
    assert payload["plots"]["sigma35_margin_ladder_gallery"]["x_axis_label"] == "Sigma-35 variant"
    assert payload["plots"]["sigma35_stress_margin_gallery"]["x_axis_label"] == (
        r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
    )
    assert payload["plots"]["sigma35_stress_margin_gallery"]["y_axis_label"] == (
        r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$"
    )
    assert payload["plots"]["sigma35_centroid_distance_gallery"]["colorbar_label"] == (
        r"$d_{\mathrm{emb}}(g,h)=1-\cos(c_g,c_h)$"
    )
    assert payload["plots"]["sigma35_centroid_distance_gallery"]["x_axis_label"] == "Sigma-35 variant $h$"
    assert payload["plots"]["sigma35_centroid_distance_gallery"]["y_axis_label"] == "Sigma-35 variant $g$"
    assert payload["plots"]["design_centroid_margin_gallery"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["design_centroid_margin_gallery"]["visibility_tier"] == "appendix"
    assert payload["plots"]["design_centroid_margin_gallery"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["design_centroid_margin_gallery"]["hue_options"]] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "log_likelihood_per_token_7b",
    ]
    assert payload["plots"]["design_centroid_margin_gallery"]["hue_options"][-1]["type"] == "continuous"
    assert payload["plots"]["design_centroid_margin_gallery"]["scalars"] == [
        "design_centroid_margins_intermediate_embedding_7b_anchor_60bp",
        "design_centroid_margins_intermediate_embedding_7b_full_context_anchor_mean",
        "design_centroid_margins_intermediate_embedding_7b_full_context_1kb",
        "design_centroid_margins_output_layer_mean_7b_anchor_60bp",
        "design_centroid_margins_output_layer_mean_7b_full_context_1kb",
    ]
    assert payload["plots"]["reference_alignment_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["reference_alignment_summary"]["visibility_tier"] == "appendix"
    assert payload["plots"]["representation_scree_diagnostic"]["kind"] == "curve_grid"
    assert payload["plots"]["representation_scree_diagnostic"]["visibility_tier"] == "appendix"
    assert payload["plots"]["representation_scree_diagnostic"]["reducers"] == [
        "pca_intermediate_embedding_7b_anchor_60bp",
        "pca_intermediate_embedding_7b_full_context_anchor_mean",
        "pca_intermediate_embedding_7b_full_context_1kb",
        "pca_output_layer_mean_7b_anchor_60bp",
        "pca_output_layer_mean_7b_full_context_1kb",
    ]
    assert payload["plots"]["context_pair_summary"]["visibility_tier"] == "primary"
    assert payload["plots"]["context_pair_summary"]["scalar"] == "context_pair_summary_metrics"
    assert payload["plots"]["context_pair_summary"]["label_column"] == "label"
    assert payload["plots"]["context_pair_summary"]["color_column"] == "comparison_role"
    assert payload["plots"]["candidate_decision_frontier"].get("color_column") is None
    assert payload["plots"]["appendix_umap_gallery"]["kind"] == "projection_grid"
    assert payload["plots"]["appendix_umap_gallery"]["visibility_tier"] == "appendix"
    assert payload["plots"]["appendix_umap_gallery"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["appendix_umap_gallery"]["hue_options"]] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
        "log_likelihood_per_token_7b",
    ]
    assert payload["plots"]["appendix_umap_gallery"]["hue_options"][-1]["type"] == "continuous"
    assert payload["plots"]["appendix_umap_gallery"]["projections"] == [
        "umap_intermediate_embedding_7b_anchor_60bp",
        "umap_intermediate_embedding_7b_full_context_anchor_mean",
        "umap_intermediate_embedding_7b_full_context_1kb",
        "umap_output_layer_mean_7b_anchor_60bp",
        "umap_output_layer_mean_7b_full_context_1kb",
    ]

    assert payload["deliverables"]["representation_health_summary"]["outputs"]["plots"] == [
        "representation_health_summary"
    ]
    assert payload["deliverables"]["design_structure_summary"]["outputs"]["plots"] == [
        "design_structure_summary",
        "balanced_design_family_margin_gallery",
    ]
    assert payload["deliverables"]["sigma35_ordinal_audit"]["outputs"]["plots"] == [
        "sigma35_ordinal_audit",
        "sigma35_margin_ladder_gallery",
        "sigma35_stress_margin_gallery",
        "sigma35_centroid_distance_gallery",
    ]
    assert payload["deliverables"]["context_robustness_summary"]["outputs"]["plots"] == [
        "context_robustness_summary",
        "context_pair_summary",
    ]
    assert payload["deliverables"]["candidate_decision_frontier"]["outputs"]["plots"] == ["candidate_decision_frontier"]
    assert payload["deliverables"]["appendix_geometry_review"]["outputs"]["plots"] == [
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
    ]
    assert payload["deliverables"]["appendix_umap_gallery"]["outputs"]["plots"] == ["appendix_umap_gallery"]


def test_template_uses_background_relative_internal_margins_and_declared_sig35_order() -> None:
    payload = _template_payload()

    pre_assay_steps = {step["id"]: step for step in _recipe_steps(payload, "pre_assay_representation_triage_recipe")}

    design_margin_step = pre_assay_steps["build_design_centroid_margins_intermediate_embedding_7b_anchor_60bp"]
    assert design_margin_step["params"]["kind"] == "cohort_similarity_margin"
    assert design_margin_step["params"]["leave_one_out"] is True
    assert design_margin_step["params"]["margin_pairs"][0] == {
        "target_values": ["ethanol"],
        "control_values": ["background_only"],
        "output_column": "synthetic_margin_ethanol_vs_background",
    }
    ordinal_axis_params = pre_assay_steps["build_sigma35_ordinal_audit_metrics"]["params"]
    assert ordinal_axis_params["kind"] == "ordinal_axis_audit"
    assert ordinal_axis_params["axis"]["axis_id"] == "sigma35"
    assert ordinal_axis_params["axis"]["column"] == "sig35_variant"
    assert ordinal_axis_params["axis"]["order_path"] == "inputs/sig35_order.yaml"
    assert ordinal_axis_params["axis"]["metric_ids"]["spearman"] == "sig35_ordinal_spearman"
    centroid_distance_params = pre_assay_steps["build_sigma35_centroid_distance_intermediate_embedding_7b_anchor_60bp"][
        "params"
    ]
    assert centroid_distance_params["kind"] == "axis_centroid_distance"
    assert centroid_distance_params["axis"]["axis_id"] == "sigma35"
    assert centroid_distance_params["axis"]["column"] == "sig35_variant"
    assert centroid_distance_params["axis"]["order_path"] == "inputs/sig35_order.yaml"
    design_structure_params = pre_assay_steps["build_design_structure_summary_metrics"]["params"]
    assert design_structure_params["balanced_axis"]["balance_columns"] == [
        "sig35_variant",
        "spacer_length",
    ]
    assert [axis["column"] for axis in design_structure_params["axes"]] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
    ]


def test_template_recipes_are_self_materializing() -> None:
    payload = _template_payload()
    pre_assay_steps = {step["id"]: step for step in _recipe_steps(payload, "pre_assay_representation_triage_recipe")}
    appendix_steps = {step["id"]: step for step in _recipe_steps(payload, "appendix_umap_gallery_recipe")}

    assert "materialize_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "build_alignment_intermediate_embedding_20b_anchor_to_full_context" not in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "build_design_centroid_margins_intermediate_embedding_20b_anchor_60bp" not in pre_assay_steps
    assert "build_alignment_intermediate_embedding_7b_anchor_to_anchor_mean" in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_7b_full_context_anchor_mean" in pre_assay_steps
    assert "build_representation_health_summary_metrics" in pre_assay_steps
    representation_health_params = pre_assay_steps["build_representation_health_summary_metrics"]["params"]
    assert representation_health_params["pairwise_max_rows"] == 4096
    assert representation_health_params["pairwise_seed"] == 17
    assert "build_design_structure_summary_metrics" in pre_assay_steps
    assert "build_sigma35_ordinal_audit_metrics" in pre_assay_steps
    stress_margin_anchor = pre_assay_steps["build_sigma35_stress_margins_intermediate_embedding_7b_anchor_60bp"]
    stress_margin_full = pre_assay_steps["build_sigma35_stress_margins_intermediate_embedding_7b_full_context_1kb"]
    stress_margin_anchor_mean = pre_assay_steps[
        "build_sigma35_stress_margins_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert "sample_id" not in stress_margin_anchor["params"]
    assert "sample_id" not in stress_margin_full["params"]
    assert "sample_id" not in stress_margin_anchor_mean["params"]
    balanced_margin_anchor = pre_assay_steps[
        "build_balanced_design_family_margins_intermediate_embedding_7b_anchor_60bp"
    ]
    balanced_margin_full = pre_assay_steps[
        "build_balanced_design_family_margins_intermediate_embedding_7b_full_context_1kb"
    ]
    balanced_margin_anchor_mean = pre_assay_steps[
        "build_balanced_design_family_margins_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert balanced_margin_anchor["params"]["balance_reference_only"] is True
    assert "sample_id" not in balanced_margin_anchor["params"]
    assert "sample_id" not in balanced_margin_full["params"]
    assert "sample_id" not in balanced_margin_anchor_mean["params"]
    centroid_distance_anchor = pre_assay_steps["build_sigma35_centroid_distance_intermediate_embedding_7b_anchor_60bp"]
    centroid_distance_full = pre_assay_steps[
        "build_sigma35_centroid_distance_intermediate_embedding_7b_full_context_1kb"
    ]
    centroid_distance_anchor_mean = pre_assay_steps[
        "build_sigma35_centroid_distance_intermediate_embedding_7b_full_context_anchor_mean"
    ]
    assert "sample_id" not in centroid_distance_anchor["params"]
    assert "sample_id" not in centroid_distance_full["params"]
    assert "sample_id" not in centroid_distance_anchor_mean["params"]
    assert (
        "build_sigma35_centroid_distance_intermediate_embedding_7b_anchor_plus_anchor_mean_concat"
        not in pre_assay_steps
    )
    assert "build_context_robustness_summary_metrics" in pre_assay_steps
    assert "build_context_delta_distribution_intermediate_embedding_7b_anchor_mean" in pre_assay_steps
    assert "build_context_pair_summary_metrics" in pre_assay_steps
    assert pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b_anchor_mean"]["params"][
        "where"
    ] == {
        "column": "source_class",
        "equals": "densegen",
    }
    assert (
        pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b_anchor_mean"]["params"][
            "table_sample_only"
        ]
        is True
    )
    assert pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b"]["params"]["where"] == {
        "column": "source_class",
        "equals": "densegen",
    }
    assert (
        pre_assay_steps["build_context_delta_distribution_intermediate_embedding_7b"]["params"]["table_sample_only"]
        is True
    )
    assert "build_reference_alignment_summary_metrics" in pre_assay_steps
    assert pre_assay_steps["build_reference_alignment_summary_metrics"]["params"]["reference_sets"] == [
        "reference_spyp_sulap"
    ]
    assert "build_candidate_decision_frontier_metrics" in pre_assay_steps
    assert "render_balanced_design_family_margin_gallery" in pre_assay_steps
    assert "render_sigma35_margin_ladder_gallery" in pre_assay_steps
    assert "render_design_centroid_margin_gallery" in pre_assay_steps
    assert "render_sigma35_stress_margin_gallery" in pre_assay_steps
    assert "render_sigma35_centroid_distance_gallery" in pre_assay_steps
    assert "render_representation_scree_diagnostic" in pre_assay_steps
    assert "render_context_pair_summary" in pre_assay_steps
    assert "render_candidate_decision_frontier" in pre_assay_steps

    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" not in appendix_steps
    assert "materialize_intermediate_embedding_7b_anchor_plus_full_context_concat" not in appendix_steps
    assert "materialize_intermediate_embedding_7b_anchor_plus_anchor_mean_concat" not in appendix_steps
    assert (
        appendix_steps["build_umap_sample_intermediate_embedding_7b_full_context_anchor_mean"]["params"]["strategy"]
        == "all"
    )
    assert set(appendix_steps["generate_latent_geometry_browser"]["depends_on"]) >= {
        "render_dataset_overview",
        "render_representation_health_summary",
        "render_design_structure_summary",
        "render_balanced_design_family_margin_gallery",
        "render_sigma35_ordinal_audit",
        "render_sigma35_margin_ladder_gallery",
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
