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
        "representation_health_summary",
        "design_structure_summary",
        "sigma35_ordinal_audit",
        "context_robustness_summary",
        "appendix_geometry_audit",
        "appendix_umap_gallery",
    ]


def test_template_uses_required_references_and_trimmed_browser_surface() -> None:
    payload = _template_payload()

    assert set(payload["landmarks"]) == {"spyp", "sulap", "j23105"}
    assert payload["reference_sets"]["promoter_wildtype_primary"]["display_labels"] == {
        "spyP": "spyp",
        "sulAp": "sulap",
        "J23105": "j23105",
    }
    assert list(payload["notebooks"]) == ["latent_geometry_browser"]
    assert payload["notebooks"]["latent_geometry_browser"]["default_deliverable"] == "representation_health_summary"
    assert payload["notebooks"]["latent_geometry_browser"]["default_surface"] == "plots"
    assert payload["notebooks"]["latent_geometry_browser"]["ordered_plots"] == [
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
    assert payload["exports"] == {}
    assert set(payload["alignments"]) == {
        "intermediate_embedding_20b_anchor_to_full_context",
        "intermediate_embedding_7b_anchor_to_full_context",
        "pooled_logits_20b_anchor_to_full_context",
        "pooled_logits_7b_anchor_to_full_context",
    }
    assert all("semantics_ref" in plot for plot in payload["plots"].values())

    assert payload["plots"]["representation_health_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["design_structure_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["sigma35_ordinal_audit"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["context_robustness_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["design_centroid_margin_gallery"]["kind"] == "xy_scatter_grid"
    assert payload["plots"]["design_centroid_margin_gallery"]["visibility_tier"] == "appendix"
    assert payload["plots"]["design_centroid_margin_gallery"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["design_centroid_margin_gallery"]["hue_options"]] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
    ]
    assert payload["plots"]["reference_alignment_summary"]["kind"] == "metric_panel_grid"
    assert payload["plots"]["reference_alignment_summary"]["visibility_tier"] == "appendix"
    assert payload["plots"]["representation_scree_diagnostic"]["kind"] == "curve_grid"
    assert payload["plots"]["representation_scree_diagnostic"]["visibility_tier"] == "appendix"
    assert payload["plots"]["context_delta_distributions"]["kind"] == "distribution_grid"
    assert payload["plots"]["context_delta_distributions"]["visibility_tier"] == "debug"
    assert payload["plots"]["context_delta_distributions"]["metric_columns"] == [
        "context_self_cosine",
        "context_shift_l2",
    ]
    assert payload["plots"]["appendix_umap_gallery"]["kind"] == "projection_grid"
    assert payload["plots"]["appendix_umap_gallery"]["visibility_tier"] == "appendix"
    assert payload["plots"]["appendix_umap_gallery"]["default_hue"] == "design_family"
    assert [option["column"] for option in payload["plots"]["appendix_umap_gallery"]["hue_options"]] == [
        "design_family",
        "design_regulator_composition",
        "sig35_variant",
        "spacer_length",
    ]

    assert payload["deliverables"]["representation_health_summary"]["outputs"]["plots"] == [
        "representation_health_summary"
    ]
    assert payload["deliverables"]["design_structure_summary"]["outputs"]["plots"] == ["design_structure_summary"]
    assert payload["deliverables"]["sigma35_ordinal_audit"]["outputs"]["plots"] == ["sigma35_ordinal_audit"]
    assert payload["deliverables"]["context_robustness_summary"]["outputs"]["plots"] == ["context_robustness_summary"]
    assert payload["deliverables"]["appendix_geometry_audit"]["outputs"]["plots"] == [
        "design_centroid_margin_gallery",
        "reference_alignment_summary",
        "representation_scree_diagnostic",
    ]
    assert payload["deliverables"]["appendix_umap_gallery"]["outputs"]["plots"] == ["appendix_umap_gallery"]


def test_template_uses_background_relative_internal_margins_and_declared_sig35_order() -> None:
    payload = _template_payload()

    pre_assay_steps = {step["id"]: step for step in _recipe_steps(payload, "pre_assay_representation_triage_recipe")}

    design_margin_step = pre_assay_steps["build_design_centroid_margins_intermediate_embedding_20b_anchor_60bp"]
    assert design_margin_step["params"]["kind"] == "cohort_similarity_margin"
    assert design_margin_step["params"]["leave_one_out"] is True
    assert design_margin_step["params"]["margin_pairs"][0] == {
        "target_values": ["ethanol"],
        "control_values": ["background_only"],
        "output_column": "synthetic_margin_ethanol_vs_background",
    }
    assert (
        pre_assay_steps["build_sigma35_ordinal_audit_metrics"]["params"]["sig35_order_path"]
        == "inputs/sig35_order.yaml"
    )
    assert pre_assay_steps["build_design_structure_summary_metrics"]["params"]["balance_columns"] == [
        "sig35_variant",
        "spacer_length",
    ]


def test_template_recipes_are_self_materializing() -> None:
    payload = _template_payload()
    pre_assay_steps = {step["id"]: step for step in _recipe_steps(payload, "pre_assay_representation_triage_recipe")}
    appendix_steps = {step["id"]: step for step in _recipe_steps(payload, "appendix_umap_gallery_recipe")}

    assert "materialize_intermediate_embedding_20b_anchor_60bp" in pre_assay_steps
    assert "build_alignment_intermediate_embedding_20b_anchor_to_full_context" in pre_assay_steps
    assert "build_scorecard_sample_intermediate_embedding_20b_anchor_60bp" in pre_assay_steps
    assert "reduce_pca_intermediate_embedding_20b_anchor_60bp" in pre_assay_steps
    assert "build_design_centroid_margins_intermediate_embedding_20b_anchor_60bp" in pre_assay_steps
    assert "build_representation_health_summary_metrics" in pre_assay_steps
    assert "build_design_structure_summary_metrics" in pre_assay_steps
    assert "build_sigma35_ordinal_audit_metrics" in pre_assay_steps
    assert "build_context_robustness_summary_metrics" in pre_assay_steps
    assert "build_reference_alignment_summary_metrics" in pre_assay_steps
    assert "render_design_centroid_margin_gallery" in pre_assay_steps
    assert "render_representation_scree_diagnostic" in pre_assay_steps
    assert "render_context_delta_distributions" in pre_assay_steps

    assert "build_umap_sample_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert "fit_umap_intermediate_embedding_20b_anchor_60bp" in appendix_steps
    assert appendix_steps["build_umap_sample_intermediate_embedding_20b_anchor_60bp"]["params"]["strategy"] == "all"
    assert set(appendix_steps["generate_latent_geometry_browser"]["depends_on"]) >= {
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
