"""
Plot-contract tests for optional shape-channel support.
"""

from __future__ import annotations

from pydantic import TypeAdapter

from dnadesign.latentdna.src.contracts.plot import PlotConfig
from dnadesign.latentdna.src.plots.recipes import resolve_plot_spec

_PLOT_CONFIG_ADAPTER = TypeAdapter(PlotConfig)


def test_xy_scatter_plot_config_accepts_shape_column() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "wildtype_reference_margins",
            "x_column": "wildtype_margin_ethanol_vs_control",
            "y_column": "wildtype_margin_cipro_vs_control",
            "color_column": "design_family",
            "shape_column": "sig35_variant",
        }
    )

    assert config.shape_column == "sig35_variant"


def test_resolve_plot_spec_preserves_shape_column() -> None:
    config = _PLOT_CONFIG_ADAPTER.validate_python(
        {
            "kind": "xy_scatter",
            "scalar": "wildtype_reference_margins",
            "x_column": "wildtype_margin_ethanol_vs_control",
            "y_column": "wildtype_margin_cipro_vs_control",
            "color_column": "design_family",
            "shape_column": "sig35_variant",
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

    assert spec.shape_column == "sig35_variant"
