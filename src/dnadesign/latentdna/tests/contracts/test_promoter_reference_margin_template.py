"""Contracts for the promoter-study reference-margin template."""

from __future__ import annotations

import yaml

from dnadesign.latentdna.src.workspaces.paths import builtin_templates_dir


def _template_payload() -> dict[str, object]:
    config_path = builtin_templates_dir() / "promoter_reference_margin_benchmark" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _recipe_steps(payload: dict[str, object], recipe_id: str) -> list[dict[str, object]]:
    return list(payload["recipes"][recipe_id]["steps"])


def test_reference_margin_template_tracks_canonical_views_and_deliverables() -> None:
    payload = _template_payload()

    assert set(payload["views"]) == {
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    }
    assert list(payload["deliverables"]) == [
        "dataset_overview",
        "reference_margin_analysis",
        "context_geometry_audit",
        "representation_comparison",
        "representation_health_diagnostic",
        "appendix_umap_gallery",
    ]


def test_reference_margin_template_uses_required_landmarks_and_browser_surface() -> None:
    payload = _template_payload()

    assert set(payload["landmarks"]) == {"spyp", "sulap", "j23105"}
    assert payload["reference_sets"]["promoter_wildtype_primary"]["display_labels"] == {
        "spyP": "spyp",
        "sulAp": "sulap",
        "J23105": "j23105",
    }
    assert list(payload["notebooks"]) == ["latent_geometry_browser"]
    assert payload["notebooks"]["latent_geometry_browser"]["default_deliverable"] == "reference_margin_analysis"
    assert payload["notebooks"]["latent_geometry_browser"]["default_surface"] == "plots"
    assert payload["notebooks"]["latent_geometry_browser"]["ordered_plots"] == [
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
    assert payload["exports"] == {}
    assert set(payload["alignments"]) == {
        "intermediate_embedding_20b_anchor_to_full_context",
        "intermediate_embedding_7b_anchor_to_full_context",
        "pooled_logits_20b_anchor_to_full_context",
        "pooled_logits_7b_anchor_to_full_context",
    }
    assert all("semantics_ref" in plot for plot in payload["plots"].values())
    assert payload["plots"]["reference_margin_gallery_wildtype"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["reference_margin_gallery_wildtype"]["scalars"] == [
        "wildtype_reference_margins_intermediate_embedding_7b_anchor_60bp",
        "wildtype_reference_margins_pooled_logits_7b_anchor_60bp",
        "wildtype_reference_margins_intermediate_embedding_7b_full_context_1kb",
        "wildtype_reference_margins_pooled_logits_7b_full_context_1kb",
        "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
        "wildtype_reference_margins_pooled_logits_20b_anchor_60bp",
        "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
        "wildtype_reference_margins_pooled_logits_20b_full_context_1kb",
    ]
    assert payload["plots"]["reference_margin_gallery_wildtype"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["reference_margin_gallery_wildtype"]["hue_options"]] == [
        "design_family",
        "sig35_variant",
    ]
    assert payload["plots"]["reference_neighbor_evidence"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["reference_neighbor_evidence"]["visibility_tier"] == "primary"
    assert "dual_margin_plane" not in payload["plots"]
    assert payload["plots"]["reference_margin_gallery_synthetic_centroids"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["reference_margin_gallery_synthetic_centroids"]["visibility_tier"] == "appendix"
    assert payload["plots"]["reference_margin_gallery_synthetic_centroids"]["default_hue"] == "design_family"
    assert payload["plots"]["context_shift_reference_plane"]["kind"] == "paired_xy_scatter_grid"
    assert payload["plots"]["context_shift_reference_plane"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["context_shift_reference_plane"]["hue_options"]] == [
        "design_family",
        "sig35_variant",
        "context_self_cosine",
        "context_shift_l2",
    ]
    assert payload["plots"]["context_delta_distributions"]["kind"] == "distribution_grid"
    assert payload["plots"]["context_delta_distributions"]["metric_columns"] == [
        "context_self_cosine",
        "context_shift_l2",
        "context_margin_delta_ethanol",
        "context_margin_delta_cipro",
    ]
    assert payload["plots"]["context_geometry_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["representation_tradeoff_scatter"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["representation_scree_diagnostic"]["kind"] == "curve_grid"
    assert payload["plots"]["appendix_umap_gallery"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["appendix_umap_gallery"]["hue_options"]] == [
        "design_family",
        "sig35_variant",
        "context_self_cosine",
        "context_shift_l2",
    ]
    assert payload["plots"]["appendix_umap_gallery"]["visibility_tier"] == "appendix"
    assert payload["deliverables"]["reference_margin_analysis"]["requires"]["views"] == [
        "intermediate_embedding_20b_anchor_60bp",
        "intermediate_embedding_20b_full_context_1kb",
        "intermediate_embedding_7b_anchor_60bp",
        "intermediate_embedding_7b_full_context_1kb",
        "pooled_logits_20b_anchor_60bp",
        "pooled_logits_20b_full_context_1kb",
        "pooled_logits_7b_anchor_60bp",
        "pooled_logits_7b_full_context_1kb",
    ]
    assert payload["deliverables"]["context_geometry_audit"]["requires"]["alignments"] == [
        "intermediate_embedding_20b_anchor_to_full_context",
        "intermediate_embedding_7b_anchor_to_full_context",
        "pooled_logits_20b_anchor_to_full_context",
        "pooled_logits_7b_anchor_to_full_context",
    ]
    assert payload["deliverables"]["reference_margin_analysis"]["outputs"]["plots"] == [
        "reference_margin_gallery_wildtype",
        "reference_neighbor_evidence",
    ]
    assert payload["deliverables"]["representation_comparison"]["outputs"]["plots"] == [
        "representation_tradeoff_scatter"
    ]
    assert payload["deliverables"]["representation_health_diagnostic"]["outputs"]["plots"] == [
        "representation_scree_diagnostic"
    ]
    assert payload["deliverables"]["appendix_umap_gallery"]["outputs"]["plots"] == [
        "reference_margin_gallery_synthetic_centroids",
        "appendix_umap_gallery",
    ]


def test_reference_margin_template_treats_dual_promoters_as_present_for_single_regulator_tasks() -> None:
    payload = _template_payload()

    comparison_step = next(
        step
        for step in _recipe_steps(payload, "representation_comparison_recipe")
        if step["id"] == "build_candidate_metrics_long"
    )
    params = comparison_step["params"]

    assert params["label_column"] == "design_family"
    assert params["ethanol_values"] == ["ethanol", "ethanol_ciprofloxacin"]
    assert params["cipro_values"] == ["ciprofloxacin", "ethanol_ciprofloxacin"]
    assert params["dual_values"] == ["ethanol_ciprofloxacin"]


def test_reference_margin_template_recipes_are_self_materializing() -> None:
    payload = _template_payload()
    reference_steps = {step["id"]: step for step in _recipe_steps(payload, "reference_margin_analysis_recipe")}
    health_step_ids = [step["id"] for step in _recipe_steps(payload, "representation_health_diagnostic_recipe")]
    appendix_steps = {step["id"]: step for step in _recipe_steps(payload, "appendix_umap_gallery_recipe")}

    assert "materialize_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "build_alignment_intermediate_embedding_20b_anchor_to_full_context" in reference_steps
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "fit_scorecard_knn_intermediate_embedding_20b_anchor_60bp" in reference_steps
    assert "build_context_geometry_metrics_intermediate_embedding_20b" in reference_steps
    assert "compare_context_geometry_agreement_intermediate_embedding_20b" in reference_steps
    assert (
        reference_steps["build_wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb"]["params"][
            "alignment_id"
        ]
        == "intermediate_embedding_20b_anchor_to_full_context"
    )
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" in health_step_ids
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" in health_step_ids
    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert appendix_steps["build_umap_sample_intermediate_embedding_20b_anchor_60bp"]["params"]["strategy"] == "all"
    assert set(appendix_steps["generate_latent_geometry_browser"]["depends_on"]) >= {
        "render_reference_margin_gallery_wildtype",
        "render_reference_neighbor_evidence",
        "render_context_shift_reference_plane",
        "render_context_delta_distributions",
        "render_context_geometry_summary",
        "render_representation_tradeoff_scatter",
        "render_representation_scree_diagnostic",
        "render_appendix_umap_gallery",
    }
