"""Contracts for multi-panel scalar and agreement plot galleries."""

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
                "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
                "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
            ],
            "panel_titles": [
                "intermediate_embedding 20b anchor_60bp",
                "intermediate_embedding 20b full_context_1kb",
            ],
            "x_column": "wildtype_margin_ethanol_vs_control",
            "y_column": "wildtype_margin_cipro_vs_control",
            "color_column": "design_family",
        }
    )

    spec = resolve_plot_spec(
        plots={"reference_margin_gallery_wildtype": config},
        plot_id="reference_margin_gallery_wildtype",
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
        "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
        "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
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
                    "wildtype_reference_margins_intermediate_embedding_20b_anchor_60bp",
                    "wildtype_reference_margins_intermediate_embedding_20b_full_context_1kb",
                ],
                "panel_titles": ["only one title"],
                "x_column": "wildtype_margin_ethanol_vs_control",
                "y_column": "wildtype_margin_cipro_vs_control",
            }
        )


def test_agreement_summary_grid_preserves_panel_inventory() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "agreement_summary_grid",
            "agreements": [
                "context_geometry_agreement_intermediate_embedding_20b",
                "context_geometry_agreement_pooled_logits_20b",
            ],
            "panel_titles": [
                "intermediate_embedding 20b",
                "pooled_logits 20b",
            ],
        }
    )

    spec = resolve_plot_spec(
        plots={"context_geometry_summary": config},
        plot_id="context_geometry_summary",
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
        "context_geometry_agreement_pooled_logits_20b",
    ]
    assert spec.panel_titles == ["intermediate_embedding 20b", "pooled_logits 20b"]


def test_categorical_count_accepts_panelled_count_plots() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "categorical_count",
            "scalar": "dataset_overview_counts",
            "category_column": "category",
            "label_column": "label",
            "value_column": "row_count",
            "panel_column": "source_id",
            "color_column": "category",
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
    assert spec.row_column == "category"
    assert spec.column_column == "label"
    assert spec.value_column == "row_count"


def test_metric_panel_grid_accepts_candidate_metric_summary_plots() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "metric_panel_grid",
            "scalar": "reference_neighbor_evidence_metrics",
            "facet_column": "category",
            "panel_title_column": "display_name",
            "category_column": "label",
            "label_column": "candidate_label",
            "value_column": "metric_value",
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
        plots={"reference_neighbor_evidence": config},
        plot_id="reference_neighbor_evidence",
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
    assert spec.scalar_id == "reference_neighbor_evidence_metrics"
    assert spec.row_column == "category"
    assert spec.panel_column == "display_name"
    assert spec.column_column == "label"
    assert spec.label_column == "candidate_label"
    assert spec.value_column == "metric_value"
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
                "context_geometry_metrics_pooled_logits_20b",
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
        plots={"context_delta_distributions": config},
        plot_id="context_delta_distributions",
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
        "context_geometry_metrics_pooled_logits_20b",
    ]
    assert spec.metric_columns == [
        "context_self_cosine",
        "context_shift_l2",
        "context_margin_delta_ethanol",
        "context_margin_delta_cipro",
    ]
