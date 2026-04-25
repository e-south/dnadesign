"""Plot-contract tests for hue configuration on live scatter surfaces."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.plot import PlotConfig
from dnadesign.latentdna.src.plots.recipes import resolve_plot_spec
from dnadesign.latentdna.src.plots.render import _effective_shape_column

_PLOT_CONFIG_ADAPTER = TypeAdapter(PlotConfig)


def test_xy_scatter_plot_config_accepts_default_hue_and_hue_options() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "design_centroid_margins",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "default_hue": "design_family",
            "hue_options": [
                {"column": "design_family", "label": "Design family", "type": "categorical"},
                {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
            ],
        }
    )

    assert config.default_hue == "design_family"
    assert [option.column for option in config.hue_options] == ["design_family", "sig35_variant"]


def test_xy_scatter_plot_config_accepts_ordinal_hue_options() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "design_centroid_margins",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "default_hue": "spacer_length",
            "hue_options": [
                {"column": "spacer_length", "label": "Spacer length", "type": "ordinal"},
            ],
        }
    )

    assert config.default_hue == "spacer_length"
    assert config.hue_options[0].type == "ordinal"


def test_plot_config_rejects_default_hue_not_declared_in_hue_options() -> None:
    with pytest.raises(ValueError, match="default_hue"):
        _PLOT_CONFIG_ADAPTER.validate_python(
            {
                "kind": "xy_scatter",
                "scalar": "design_centroid_margins",
                "x_column": "synthetic_margin_ethanol_vs_background",
                "y_column": "synthetic_margin_cipro_vs_background",
                "default_hue": "design_family",
                "hue_options": [
                    {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
                ],
            }
        )


def test_resolve_plot_spec_preserves_hue_configuration() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "design_centroid_margins",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "default_hue": "design_family",
            "hue_options": [
                {"column": "design_family", "label": "Design family", "type": "categorical"},
                {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
            ],
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

    assert spec.color_column == "design_family"
    assert spec.default_hue == "design_family"
    assert [option.column for option in spec.hue_options] == ["design_family", "sig35_variant"]


def test_hue_switchable_scatter_surfaces_ignore_shape_channel_at_render_time() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "design_centroid_margins",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "shape_column": "sig35_variant",
            "default_hue": "design_family",
            "hue_options": [
                {"column": "design_family", "label": "Design family", "type": "categorical"},
                {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
            ],
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

    assert spec.shape_column == "sig35_variant"
    assert _effective_shape_column(spec) is None


def test_metric_panel_grid_inline_spec_rejects_shape_column() -> None:
    with pytest.raises(ContractViolationError, match="metric_panel_grid does not support shape_column"):
        resolve_plot_spec(
            plots={},
            plot_id="representation_health_summary",
            kind="metric_panel_grid",
            projection_ids=[],
            panel_titles=[],
            enrichment_id=None,
            distance_id=None,
            scalar_id="representation_health_summary_metrics",
            scalar_ids=[],
            agreement_id=None,
            agreement_ids=[],
            reducer_id=None,
            left_cluster_id=None,
            right_cluster_id=None,
            value_column="value",
            x_column="facet",
            y_column="panel_title",
            color_column="category",
            render_mode=None,
            label_column="metric_label",
            label_values=[],
            shape_column="sig35_variant",
        )
