"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/contracts/test_plot_gallery_contracts.py

Contracts for multi-panel scalar and agreement plot galleries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dnadesign.latentdna.src.contracts.plot import PlotConfig
from dnadesign.latentdna.src.plots.recipes import resolve_plot_spec

_PLOT_CONFIG_ADAPTER = TypeAdapter(PlotConfig)


def test_xy_scatter_grid_accepts_scalar_panels() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter_grid",
            "scalars": [
                "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                "design_centroid_margins_intermediate_embedding_20b_full_context_1kb",
            ],
            "panel_titles": [
                "intermediate_embedding 20b anchor_60bp",
                "intermediate_embedding 20b full_context_1kb",
            ],
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "color_column": "design_family",
        }
    )

    spec = resolve_plot_spec(
        plots={"design_centroid_margin_gallery": config},
        plot_id="design_centroid_margin_gallery",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "xy_scatter_grid"
    assert spec.scalar_ids == [
        "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
        "design_centroid_margins_intermediate_embedding_20b_full_context_1kb",
    ]
    assert spec.panel_titles == [
        "intermediate_embedding 20b anchor_60bp",
        "intermediate_embedding 20b full_context_1kb",
    ]


def test_xy_scatter_grid_rejects_misaligned_panel_titles() -> None:
    with pytest.raises(ValidationError, match="xy_scatter_grid panel_titles must match scalars length"):
        _PLOT_CONFIG_ADAPTER.validate_python(
            {
                "kind": "xy_scatter_grid",
                "scalars": [
                    "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                    "design_centroid_margins_intermediate_embedding_20b_full_context_1kb",
                ],
                "panel_titles": ["only one title"],
                "x_column": "synthetic_margin_ethanol_vs_background",
                "y_column": "synthetic_margin_cipro_vs_background",
            }
        )


def test_agreement_summary_grid_preserves_panel_inventory() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "agreement_summary_grid",
            "agreements": [
                "context_geometry_agreement_intermediate_embedding_20b",
                "context_geometry_agreement_output_layer_mean_20b",
            ],
            "panel_titles": [
                "intermediate_embedding 20b",
                "output_layer_mean 20b",
            ],
        }
    )

    spec = resolve_plot_spec(
        plots={"agreement_demo": config},
        plot_id="agreement_demo",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "agreement_summary_grid"
    assert spec.agreement_ids == [
        "context_geometry_agreement_intermediate_embedding_20b",
        "context_geometry_agreement_output_layer_mean_20b",
    ]
    assert spec.panel_titles == ["intermediate_embedding 20b", "output_layer_mean 20b"]


def test_categorical_count_accepts_panelled_count_plots() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "categorical_count",
            "scalar": "dataset_overview_counts",
            "category_column": "dimension",
            "label_column": "category_label",
            "value_column": "fraction",
            "panel_column": "dimension_label",
        }
    )

    spec = resolve_plot_spec(
        plots={"dataset_overview": config},
        plot_id="dataset_overview",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "categorical_count"
    assert spec.scalar_id == "dataset_overview_counts"
    assert spec.row_column == "dimension"
    assert spec.column_column == "category_label"
    assert spec.value_column == "fraction"


def test_metric_panel_grid_accepts_candidate_metric_summary_plots() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "metric_panel_grid",
            "scalar": "representation_health_summary_metrics",
            "facet_column": "category",
            "panel_title_column": "display_name",
            "category_column": "label",
            "label_column": "candidate_label",
            "value_column": "metric_value",
            "ci_lower_column": "ci_lower",
            "ci_upper_column": "ci_upper",
            "color_column": "candidate_family",
            "direction_column": "direction",
            "unit_column": "unit",
            "sort_rule": "panel_direction",
            "measure_kind": "metric",
            "value_kind": "score",
            "value_label": "Metric value",
        }
    )

    spec = resolve_plot_spec(
        plots={"representation_health_summary": config},
        plot_id="representation_health_summary",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "metric_panel_grid"
    assert spec.scalar_id == "representation_health_summary_metrics"
    assert spec.row_column == "category"
    assert spec.panel_column == "display_name"
    assert spec.column_column == "label"
    assert spec.label_column == "candidate_label"
    assert spec.value_column == "metric_value"
    assert spec.ci_lower_column == "ci_lower"
    assert spec.ci_upper_column == "ci_upper"
    assert spec.measure_kind == "metric"
    assert spec.value_kind == "score"
    assert spec.value_label == "Metric value"
    assert spec.sort_rule == "panel_direction"


def test_distribution_grid_accepts_explicit_metric_inventory() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "distribution_grid",
            "scalars": [
                "context_geometry_metrics_intermediate_embedding_20b",
                "context_geometry_metrics_output_layer_mean_20b",
            ],
            "metric_columns": [
                "context_self_cosine",
                "context_shift_l2",
                "context_margin_delta_ethanol",
                "context_margin_delta_cipro",
            ],
            "color_column": "design_family",
            "render_mode": "violin_box",
        }
    )

    spec = resolve_plot_spec(
        plots={"distribution_demo": config},
        plot_id="distribution_demo",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "distribution_grid"
    assert spec.scalar_ids == [
        "context_geometry_metrics_intermediate_embedding_20b",
        "context_geometry_metrics_output_layer_mean_20b",
    ]
    assert spec.metric_columns == [
        "context_self_cosine",
        "context_shift_l2",
        "context_margin_delta_ethanol",
        "context_margin_delta_cipro",
    ]


def test_heatmap_grid_accepts_panelled_sigma35_distance_gallery() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "heatmap_grid",
            "scalars": [
                "sigma35_centroid_distance_intermediate_embedding_7b_anchor_60bp",
                "sigma35_centroid_distance_intermediate_embedding_7b_full_context_anchor_mean",
            ],
            "panel_titles": [
                "Anchor",
                "Anchor mean",
            ],
            "row_column": "row_variant",
            "column_column": "column_variant",
            "value_column": "metric_value",
            "colorbar_label": r"$d_{\mathrm{emb}}(g,h)=1-\cos(c_g,c_h)$",
            "row_order": ["TTGACA (f)", "TAGACA (e)"],
            "column_order": ["TTGACA (f)", "TAGACA (e)"],
            "color_scale": "sequential",
        }
    )

    spec = resolve_plot_spec(
        plots={"sigma35_centroid_distance_gallery": config},
        plot_id="sigma35_centroid_distance_gallery",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "heatmap_grid"
    assert spec.scalar_ids == [
        "sigma35_centroid_distance_intermediate_embedding_7b_anchor_60bp",
        "sigma35_centroid_distance_intermediate_embedding_7b_full_context_anchor_mean",
    ]
    assert spec.row_order == ["TTGACA (f)", "TAGACA (e)"]
    assert spec.column_order == ["TTGACA (f)", "TAGACA (e)"]
    assert spec.color_scale == "sequential"
    assert spec.colorbar_label == r"$d_{\mathrm{emb}}(g,h)=1-\cos(c_g,c_h)$"


def test_xy_scatter_accepts_labelled_decision_frontier_config() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "candidate_decision_frontier_metrics",
            "x_column": "design_family_balanced_separation_ratio",
            "y_column": "sig35_ordinal_spearman",
            "size_column": "effective_rank",
            "size_range": [140, 260],
            "label_column": "frontier_label",
            "label_values": ["60 bp anchor", "1 kb seq mean", "1 kb anchor mean"],
        }
    )

    spec = resolve_plot_spec(
        plots={"candidate_decision_frontier": config},
        plot_id="candidate_decision_frontier",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "xy_scatter"
    assert spec.scalar_id == "candidate_decision_frontier_metrics"
    assert spec.label_column == "frontier_label"
    assert spec.label_values == ["60 bp anchor", "1 kb seq mean", "1 kb anchor mean"]
    assert spec.size_column == "effective_rank"
    assert spec.size_range == (140.0, 260.0)


def test_distribution_grid_accepts_explicit_math_axis_label() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "distribution_grid",
            "scalars": ["sigma35_stress_margins_intermediate_embedding_7b_anchor_60bp"],
            "metric_columns": ["sig35_margin_f_vs_b"],
            "y_axis_label": r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$",
        }
    )

    spec = resolve_plot_spec(
        plots={"sigma35_margin_ladder_gallery": config},
        plot_id="sigma35_margin_ladder_gallery",
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )

    assert spec.kind == "distribution_grid"
    assert spec.y_axis_label == r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
