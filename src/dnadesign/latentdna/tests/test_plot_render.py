from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from matplotlib.collections import PathCollection

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.plot import ResolvedPlotSpec, metric_panel_uses_square_axes
from dnadesign.latentdna.src.contracts.plot_semantics import PlotSemantics
from dnadesign.latentdna.src.metadata_axes import axis_style_map_from_payload
from dnadesign.latentdna.src.plots.annotation_rendering import (
    annotation_continuous_color_encoding as _annotation_continuous_color_encoding,
)
from dnadesign.latentdna.src.plots.annotation_rendering import draw_annotation_callouts as _draw_annotation_callouts
from dnadesign.latentdna.src.plots.annotation_rendering import draw_resolved_annotations as _draw_resolved_annotations
from dnadesign.latentdna.src.plots.annotations import resolve_annotation_rows
from dnadesign.latentdna.src.plots.layout import (
    _grid_figure_size,
    _panel_grid_dimensions,
    metric_panel_grid_layout,
    plot_tight_layout_kwargs,
)
from dnadesign.latentdna.src.plots.legends import (
    add_figure_legends as _add_figure_legends,
)
from dnadesign.latentdna.src.plots.legends import (
    add_side_figure_legends as _add_side_figure_legends,
)
from dnadesign.latentdna.src.plots.render import render_plot_artifact
from dnadesign.latentdna.src.plots.render_state import LayoutReservation
from dnadesign.latentdna.src.plots.renderers.agreement import render_correspondence_heatmap_plot
from dnadesign.latentdna.src.plots.renderers.distribution import (
    derived_panel_label,
    render_distribution_panel,
    render_distribution_plot,
)
from dnadesign.latentdna.src.plots.renderers.heatmap import (
    heatmap_grid_from_rows as _heatmap_grid_from_rows,
)
from dnadesign.latentdna.src.plots.renderers.heatmap import (
    render_heatmap_panel as _render_heatmap_panel,
)
from dnadesign.latentdna.src.plots.renderers.metric import (
    load_metric_panel_grid_input,
    metric_panel_groups,
    metric_panel_needs_candidate_label_ticks,
    metric_panel_uses_grouped_family_bars,
    render_metric_panel,
)
from dnadesign.latentdna.src.plots.renderers.projection import render_projection_plot
from dnadesign.latentdna.src.plots.renderers.scatter import axis_category_value as _axis_category_value
from dnadesign.latentdna.src.plots.renderers.scatter import category_color_map as _category_color_map
from dnadesign.latentdna.src.plots.renderers.scatter import category_key as _category_key
from dnadesign.latentdna.src.plots.renderers.scatter import continuous_color_encoding as _continuous_color_encoding
from dnadesign.latentdna.src.plots.renderers.xy import render_xy_plot
from dnadesign.latentdna.src.plots.tables import read_table_rows
from dnadesign.latentdna.src.services._plot_payloads import plot_input_payload
from dnadesign.latentdna.src.visual_style import compact_candidate_title, wrap_plot_title

SIGMA35_NONCANONICAL_BUCKET = "__latentdna_reference_or_other__"
SIGMA35_AXIS_STYLES = axis_style_map_from_payload(
    {
        "sig35_variant": {
            "axis_id": "sigma35",
            "column": "sig35_variant",
            "label": "Sigma-35 variant",
            "kind": "categorical",
            "category_order": ["f", "e", "d", "c", "b", "control"],
            "ordinal_subset": ["f", "e", "d", "c", "b"],
            "display_labels": {
                "f": "TTGACA (f)",
                "e": "TAGACA (e)",
                "d": "TTTACA (d)",
                "c": "TTGTGA (c)",
                "b": "CTGACA (b)",
                "control": "Control",
            },
            "compact_display_labels": {
                "f": "f\nTTGACA",
                "e": "e\nTAGACA",
                "d": "d\nTTTACA",
                "c": "c\nTTGTGA",
                "b": "b\nCTGACA",
                "control": "Control",
            },
            "category_colors": {
                "f": "#B2182B",
                "e": "#D6604D",
                "d": "#F4A582",
                "c": "#92C5DE",
                "b": "#2166AC",
                "control": "#7F8894",
            },
            "noncanonical_bucket": SIGMA35_NONCANONICAL_BUCKET,
            "noncanonical_label": "Reference/other",
            "include_noncanonical_in_legend": False,
            "canonical_row_match": "any",
            "canonical_row_selectors": [
                {"column": "source_class", "in_values": ["densegen"]},
                {"column": "source_family", "in_values": ["densegen_generated"]},
            ],
        }
    }
)


def _row_xlabel_title_overlaps(fig, *, columns: int) -> list[bool]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [axis for axis in fig.axes if axis.axison]
    overlaps: list[bool] = []
    for top_axis, bottom_axis in zip(axes[:columns], axes[columns : columns * 2], strict=False):
        xlabel_bbox = top_axis.xaxis.label.get_window_extent(renderer=renderer)
        title_bbox = bottom_axis.title.get_window_extent(renderer=renderer)
        overlaps.append(bool(xlabel_bbox.overlaps(title_bbox)))
    return overlaps


SIGMA35_HEATMAP_AXIS_STYLES = axis_style_map_from_payload(
    {
        "row_variant": {
            "axis_id": "row_variant",
            "column": "row_variant",
            "compact_display_labels": {"TTGACA (f)": "f\nTTGACA", "CTGACA (b)": "b\nCTGACA"},
        },
        "column_variant": {
            "axis_id": "column_variant",
            "column": "column_variant",
            "compact_display_labels": {"TTGACA (f)": "f", "CTGACA (b)": "b"},
        },
    }
)


def _metric_spec(*, plot_id: str, color_column: str | None) -> ResolvedPlotSpec:
    payload = {
        "plot_id": plot_id,
        "kind": "metric_panel_grid",
        "scalar_id": "fixture_metrics",
        "row_column": "category",
        "panel_column": "display_name",
        "column_column": "label",
        "label_column": "candidate_label",
        "value_column": "metric_value",
        "ci_lower_column": "ci_lower",
        "ci_upper_column": "ci_upper",
        "direction_column": "direction",
        "unit_column": "unit",
        "sort_rule": "panel_direction",
        "measure_kind": "metric",
        "value_kind": "score",
        "value_label": "Metric value",
    }
    if color_column is not None:
        payload["color_column"] = color_column
    return ResolvedPlotSpec.model_validate(payload)


def test_category_key_compacts_list_values_for_categorical_legends() -> None:
    rows = [
        {"sigma_set": ["SIGMA24", "SIGMA38", "SIGMA70"]},
        {"sigma_set": ["SIGMA70"]},
    ]

    color_map, categories = _category_color_map([rows], "sigma_set")

    assert _category_key(["SIGMA24", "SIGMA38", "SIGMA70"]) == "SIGMA24+38+70"
    assert categories == ["SIGMA24+38+70", "SIGMA70"]
    assert set(color_map) == set(categories)


def test_axis_style_hue_keeps_context_derived_densegen_rows_categorical() -> None:
    assert (
        _axis_category_value(
            {
                "sig35_variant": "f",
                "source_family": "construct_derived",
                "source_class": "densegen",
            },
            "sig35_variant",
            axis_styles=SIGMA35_AXIS_STYLES,
        )
        == "f"
    )
    assert (
        _axis_category_value(
            {
                "sig35_variant": "f",
                "source_family": "construct_derived",
                "source_class": "construct_derived",
            },
            "sig35_variant",
            axis_styles=SIGMA35_AXIS_STYLES,
        )
        == SIGMA35_NONCANONICAL_BUCKET
    )


def test_continuous_color_encoding_honors_explicit_hue_option() -> None:
    rows = [
        {"x": 0.0, "y": 0.0, "strength": 0.01},
        {"x": 1.0, "y": 1.0, "strength": 0.9},
        {"x": 2.0, "y": 2.0, "strength": None},
    ]
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "reference_strength_umap",
            "kind": "projection_scatter",
            "projection_ids": ["umap_reference_strength"],
            "color_column": "strength",
            "hue_options": [{"column": "strength", "label": "Reference strength", "type": "continuous"}],
        }
    )

    encoding = _continuous_color_encoding(rows, spec)

    assert encoding is not None
    assert list(encoding["values"][:2]) == [0.01, 0.9]
    assert math.isnan(float(encoding["values"][2]))


def test_projection_renderer_requires_xy_columns(tmp_path) -> None:
    projection_dir = tmp_path / "projections" / "fixture_umap"
    projection_dir.mkdir(parents=True)
    pq.write_table(pa.table({"x": [0.0, 1.0]}), projection_dir / "coords.parquet")
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_projection",
            "kind": "projection_scatter",
            "projection_ids": ["fixture_umap"],
        }
    )

    with pytest.raises(ContractViolationError, match="projection artifact fixture_umap.*'y'"):
        render_projection_plot(
            SimpleNamespace(output_root=tmp_path),
            spec,
            pyplot=plt,
            axis_styles=None,
        )


def test_projection_grid_labels_every_panel_x_axes(tmp_path: Path) -> None:
    projection_ids = [f"fixture_umap_{index}" for index in range(12)]
    for projection_id in projection_ids:
        projection_dir = tmp_path / "projections" / projection_id
        projection_dir.mkdir(parents=True)
        pq.write_table(pa.table({"x": [0.0, 1.0], "y": [1.0, 0.0]}), projection_dir / "coords.parquet")
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "appendix_umap_gallery",
            "kind": "projection_grid",
            "projection_ids": projection_ids,
        }
    )

    result = render_projection_plot(
        SimpleNamespace(output_root=tmp_path),
        spec,
        pyplot=plt,
        axis_styles=None,
    )

    try:
        panel_axes = result.figure.axes[:12]
        assert {axis.get_xlabel() for axis in panel_axes} == {"Projection 1"}
        result.figure.tight_layout(
            **plot_tight_layout_kwargs(spec.plot_id, legend_bottom=result.layout_reservation.legend_bottom)
        )
        assert not any(_row_xlabel_title_overlaps(result.figure, columns=6))
    finally:
        plt.close(result.figure)


def test_xy_scatter_grid_preserves_full_axis_labels_per_panel(tmp_path: Path) -> None:
    for scalar_id in ["left", "right"]:
        scalar_dir = tmp_path / "scalars" / scalar_id
        scalar_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "x": [0.0, 1.0],
                    "y": [1.0, 0.0],
                    "sig35_variant": ["f", "b"],
                }
            ),
            scalar_dir / "table.parquet",
        )
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "sigma35_stress_margin_gallery",
            "kind": "xy_scatter_grid",
            "scalar_ids": ["left", "right"],
            "x_column": "x",
            "y_column": "y",
            "x_axis_label": r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$",
            "y_axis_label": r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$",
            "color_column": "sig35_variant",
        }
    )

    result = render_xy_plot(
        SimpleNamespace(output_root=tmp_path, config=SimpleNamespace(reference_sets={})),
        spec,
        pyplot=plt,
        axis_styles=None,
    )

    try:
        panel_axes = result.figure.axes[:2]
        expected_x_label = r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
        expected_y_label = r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$"
        assert [axis.get_xlabel() for axis in panel_axes] == [expected_x_label, expected_x_label]
        assert [axis.get_ylabel() for axis in panel_axes] == [expected_y_label, expected_y_label]
        figure_texts = {text.get_text() for text in result.figure.texts}
        assert expected_x_label not in figure_texts
        assert result.layout_reservation.legend_bottom >= 0.14
    finally:
        plt.close(result.figure)


def test_xy_scatter_with_no_finite_rows_records_explicit_annotation_state(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "fixture_scalar"
    scalar_dir.mkdir(parents=True)
    pq.write_table(pa.table({"x": [math.nan], "y": [math.nan]}), scalar_dir / "table.parquet")
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_xy",
            "kind": "xy_scatter",
            "scalar_id": "fixture_scalar",
            "x_column": "x",
            "y_column": "y",
        }
    )
    semantics = PlotSemantics(
        plot_id="fixture_xy",
        question="Does empty xy rendering fail visibly?",
        decision_role="debug",
        encoding="xy scatter",
        scope="unit fixture",
        guardrails=["No finite rows should not crash metadata generation."],
        caption="Fixture xy scatter.",
        alt_text="Fixture xy scatter.",
        preprocessing_md="Fixture.",
        math_md="Fixture.",
        rationale_md="Fixture.",
        limitations_md="Fixture.",
        failure_modes_md="Fixture.",
    )

    _, outputs, metadata = render_plot_artifact(
        SimpleNamespace(
            output_root=tmp_path,
            config=SimpleNamespace(defaults=SimpleNamespace(plot_formats=["png"]), reference_sets={}),
        ),
        spec=spec,
        output_dir=tmp_path / "plots" / "fixture_xy",
        semantics=semantics,
    )

    assert [Path(output).name for output in outputs] == ["plot.png"]
    assert metadata["reference_panels"]["fixture_scalar"]["complete"] is True


def test_correspondence_heatmap_rejects_duplicate_assignment_keys(tmp_path: Path) -> None:
    left_dir = tmp_path / "clusters" / "left"
    right_dir = tmp_path / "clusters" / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["row0", "row0"],
                "cluster_label": [0, 1],
            }
        ),
        left_dir / "assignments.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["row0"],
                "cluster_label": [0],
            }
        ),
        right_dir / "assignments.parquet",
    )
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_correspondence",
            "kind": "correspondence_heatmap",
            "left_cluster_id": "left",
            "right_cluster_id": "right",
        }
    )

    with pytest.raises(ContractViolationError, match="duplicate row key"):
        render_correspondence_heatmap_plot(
            SimpleNamespace(output_root=tmp_path),
            spec,
            pyplot=plt,
        )


def test_side_figure_legend_expands_single_panel_canvas() -> None:
    fig, _ = plt.subplots(figsize=(5.15, 5.0))
    categories = [f"SIGMA{index}" for index in range(15)]
    color_map = {category: "#0072B2" for category in categories}
    try:
        right_margin = _add_side_figure_legends(
            fig,
            plt,
            color_categories=categories,
            color_map=color_map,
            color_title="regulondb__sigma_factor_set",
            shape_categories=[],
            shape_map={},
            shape_title=None,
        )

        assert right_margin > 0.0
        assert fig.get_size_inches()[0] >= 7.35
        assert fig.legends
        assert fig.legends[0].get_texts()[0].get_text() == "SIGMA0"
    finally:
        plt.close(fig)


def test_figure_legend_refuses_single_row_when_many_categories() -> None:
    fig, _ = plt.subplots(figsize=(8.5, 5.0))
    categories = [f"variant_{index}" for index in range(18)]
    color_map = {category: "#0072B2" for category in categories}
    try:
        bottom_margin = _add_figure_legends(
            fig,
            plt,
            plot_id="appendix_umap_gallery",
            color_categories=categories,
            color_map=color_map,
            color_title="sig35_variant",
            shape_categories=[],
            shape_map={},
            shape_title=None,
            single_row=True,
        )

        assert bottom_margin > 0.1
        assert fig.legends
        assert getattr(fig.legends[0], "_ncols") < len(categories)
    finally:
        plt.close(fig)


def test_appendix_umap_legend_uses_compact_bottom_band_for_strength_ladder() -> None:
    fig, _ = plt.subplots(figsize=(24.9, 12.0))
    categories = ["f", "e", "d", "c", "b"]
    color_map = {category: "#0072B2" for category in categories}
    try:
        bottom_margin = _add_figure_legends(
            fig,
            plt,
            plot_id="appendix_umap_gallery",
            color_categories=categories,
            color_map=color_map,
            color_title="sig35_variant",
            shape_categories=[],
            shape_map={},
            shape_title=None,
            single_row=True,
        )

        assert 0.08 <= bottom_margin <= 0.095
        assert fig.legends
        anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        assert 0.03 <= anchor.y0 <= 0.045
    finally:
        plt.close(fig)


def test_margin_gallery_legend_uses_tight_reserved_bottom_band() -> None:
    fig, _ = plt.subplots(figsize=(21.2, 4.3))
    categories = ["background_only", "ethanol", "ciprofloxacin", "ethanol_ciprofloxacin", "control"]
    color_map = {category: "#0072B2" for category in categories}
    try:
        bottom_margin = _add_figure_legends(
            fig,
            plt,
            plot_id="design_centroid_margin_gallery",
            color_categories=categories,
            color_map=color_map,
            color_title="design_family",
            shape_categories=[],
            shape_map={},
            shape_title=None,
            single_row=True,
        )

        assert 0.105 <= bottom_margin <= 0.125
        assert fig.legends
        anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        assert 0.04 <= anchor.y0 <= 0.06
    finally:
        plt.close(fig)


def test_twelve_panel_full_population_gallery_uses_intermediate_output_rows() -> None:
    assert _panel_grid_dimensions(12) == (2, 6)


def test_metric_panel_groups_preserve_distinct_display_metrics_within_category() -> None:
    rows = [
        {
            "category": "design_family",
            "display_name": "Design-family separation ratio",
            "label": "design_family_separation_ratio",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.2,
        },
        {
            "category": "design_family",
            "display_name": "Balanced design-family separation ratio",
            "label": "design_family_balanced_separation_ratio",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.1,
        },
    ]
    spec = _metric_spec(plot_id="design_structure_summary", color_column="candidate_family")

    groups = metric_panel_groups(rows, spec)

    assert [group.title for group in groups] == [
        "Design-family separation ratio",
        "Balanced design-family separation ratio",
    ]
    assert [len(group.rows) for group in groups] == [1, 1]


def test_metric_panel_disables_grouped_family_bars_when_keys_would_overwrite_rows() -> None:
    rows = [
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "context_self_cosine_median",
            "candidate_label": "anchor vs context anchor mean",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.12,
        },
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "context_self_cosine_median",
            "candidate_label": "anchor vs full 1 kb context",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.07,
        },
    ]
    spec = _metric_spec(plot_id="context_robustness_summary", color_column="candidate_family")

    assert not metric_panel_uses_grouped_family_bars(rows, spec)
    assert metric_panel_needs_candidate_label_ticks(rows, spec)

    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Context self cosine",
            color_map=color_map,
            square=False,
        )

        tick_labels = [label.get_text().replace("\n", " ") for label in ax.get_xticklabels()]
        assert any("Anchor Vs Context" in label for label in tick_labels)
        assert any("Anchor Vs Full" in label for label in tick_labels)
    finally:
        plt.close(fig)


def test_metric_panel_input_requires_configured_value_column(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "fixture_metrics"
    scalar_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "category": ["effective_rank"],
                "display_name": ["Effective rank"],
                "label": ["candidate_a"],
                "candidate_label": ["candidate_a"],
                "metric_value": [6.4],
                "row_count": [1],
                "direction": ["higher_is_better"],
                "unit": ["dims"],
            }
        ),
        scalar_dir / "table.parquet",
    )
    spec = _metric_spec(plot_id="representation_health_summary", color_column=None).model_copy(
        update={"value_column": "missing_score"}
    )

    with pytest.raises(ContractViolationError, match="missing_score"):
        load_metric_panel_grid_input(SimpleNamespace(output_root=tmp_path), spec)


def testrender_metric_panel_ignores_nan_values_when_setting_limits_and_annotations() -> None:
    rows = [
        {
            "category": "design_family_separation_ratio",
            "display_name": "Design-family separation ratio",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.52,
            "ci_lower": 1.42,
            "ci_upper": 1.61,
        },
        {
            "category": "design_family_separation_ratio",
            "display_name": "Design-family separation ratio",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_7b",
            "candidate_scope": "full_context_1kb",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
        },
        {
            "category": "design_family_separation_ratio",
            "display_name": "Design-family separation ratio",
            "label": "candidate_c",
            "candidate_label": "candidate_c",
            "candidate_model": "evo2_20b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 0.67,
            "ci_lower": 0.61,
            "ci_upper": 0.72,
        },
    ]
    spec = _metric_spec(plot_id="design_structure_summary", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Design-family separation ratio",
            color_map=color_map,
            square=False,
        )

        lower, upper = ax.get_xlim()
        assert upper > 1.0
        assert lower < 0.0
        assert any(text.get_text() == "NA" for text in ax.texts)
        finite_text_positions = [text.get_position()[0] for text in ax.texts if text.get_text() != "NA"]
        assert finite_text_positions
        assert max(finite_text_positions) <= upper
    finally:
        plt.close(fig)


def testrender_metric_panel_keeps_bootstrap_replicates_off_bar_plot() -> None:
    rows = [
        {
            "category": "sig35_ordinal_spearman",
            "display_name": "Sigma-35 Spearman",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "rho",
            "metric_value": 0.42,
            "ci_lower": 0.31,
            "ci_upper": 0.52,
            "bootstrap_replicates": [0.35, 0.40, 0.48],
        }
    ]
    spec = _metric_spec(plot_id="sigma35_ordinal_audit", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Sigma-35 Spearman",
            color_map=color_map,
            square=True,
        )

        assert not any(isinstance(collection, PathCollection) for collection in ax.collections)
        assert "white dots" not in " ".join(text.get_text() for text in ax.texts)
        assert "Whiskers: 95% bootstrap CI" in " ".join(text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def testrender_metric_panel_uses_compact_candidate_tick_labels() -> None:
    rows = [
        {
            "category": "effective_rank",
            "display_name": "Effective rank",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "health_status": "pass",
            "direction": "higher_is_better",
            "unit": "dims",
            "metric_value": 8.4,
        },
        {
            "category": "effective_rank",
            "display_name": "Effective rank",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_20b",
            "candidate_scope": "full_context_1kb",
            "candidate_family": "output_layer_mean",
            "health_status": "warn",
            "direction": "higher_is_better",
            "unit": "dims",
            "metric_value": 1.2,
        },
    ]
    spec = _metric_spec(plot_id="representation_health_summary", color_column=None)
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Effective rank",
            color_map=color_map,
            square=True,
        )

        tick_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert tick_labels == ["7B anchor insert Block", "20B 1kb ctx Output"]
        assert float(ax.get_box_aspect()) == 1.0
    finally:
        plt.close(fig)


def test_render_metric_panel_rotates_compact_regulondb_scope_labels() -> None:
    rows = [
        {
            "category": "sigma_factor_separation",
            "display_name": "Sigma-factor separation ratio",
            "label": "native_block",
            "candidate_label": "native block",
            "candidate_model": "evo2_7b",
            "candidate_scope": "native_source_record",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.14,
        },
        {
            "category": "sigma_factor_separation",
            "display_name": "Sigma-factor separation ratio",
            "label": "core60_block",
            "candidate_label": "core60 block",
            "candidate_model": "evo2_7b",
            "candidate_scope": "core60_tss_upstream",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.12,
        },
        {
            "category": "sigma_factor_separation",
            "display_name": "Sigma-factor separation ratio",
            "label": "native_output",
            "candidate_label": "native output",
            "candidate_model": "evo2_7b",
            "candidate_scope": "native_source_record",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.03,
        },
        {
            "category": "sigma_factor_separation",
            "display_name": "Sigma-factor separation ratio",
            "label": "core60_output",
            "candidate_label": "core60 output",
            "candidate_model": "evo2_7b",
            "candidate_scope": "core60_tss_upstream",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": 1.02,
        },
    ]
    spec = _metric_spec(plot_id="sigma_factor_structure_summary", color_column=None)
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Sigma-factor separation ratio",
            color_map=color_map,
            square=False,
        )

        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert all("Source Record" not in label for label in tick_labels)
        assert any("native 81 bp" in label for label in tick_labels)
        assert any("core60 TSS" in label for label in tick_labels)
        assert all(label.get_rotation() == 32.0 for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def testrender_metric_panel_uses_placeholder_when_all_values_are_missing() -> None:
    rows = [
        {
            "category": "balanced_design_family_separation_ratio",
            "display_name": "Balanced design-family separation ratio",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": math.nan,
        },
        {
            "category": "balanced_design_family_separation_ratio",
            "display_name": "Balanced design-family separation ratio",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_20b",
            "candidate_scope": "full_context_1kb",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": math.nan,
        },
    ]
    spec = _metric_spec(plot_id="design_structure_summary", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Balanced design-family separation ratio",
            color_map=color_map,
            square=True,
        )

        text_values = [text.get_text() for text in ax.texts]
        assert "Metric unavailable" in text_values
        assert "No finite values in this snapshot" in text_values
        assert "NA" not in text_values
        assert all(not label.get_text() for label in ax.get_xticklabels())
        assert all(not label.get_text() for label in ax.get_yticklabels())
    finally:
        plt.close(fig)


def testrender_metric_panel_suppresses_redundant_scope_in_grouped_ticks() -> None:
    rows = [
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.19,
        },
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.8,
        },
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "candidate_c",
            "candidate_label": "candidate_c",
            "candidate_model": "evo2_20b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.0,
        },
        {
            "category": "context_self_cosine_median",
            "display_name": "Context self cosine",
            "label": "candidate_d",
            "candidate_label": "candidate_d",
            "candidate_model": "evo2_20b",
            "candidate_scope": "anchor_vs_context",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.04,
        },
    ]
    spec = _metric_spec(plot_id="context_robustness_summary", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Context self cosine",
            color_map=color_map,
            square=False,
        )

        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert tick_labels == ["7B", "20B"]
        assert all(label.get_rotation() == 32.0 for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_render_distribution_panel_orders_sig35_categories_by_strength_and_uses_display_labels() -> None:
    rows = [
        {"sig35_variant": "b", "sig35_margin_f_vs_b": -0.4},
        {"sig35_variant": "f", "sig35_margin_f_vs_b": 0.7},
        {"sig35_variant": "d", "sig35_margin_f_vs_b": 0.1},
        {"sig35_variant": "e", "sig35_margin_f_vs_b": 0.3},
        {"sig35_variant": "c", "sig35_margin_f_vs_b": -0.1},
        {"sig35_variant": "control", "sig35_margin_f_vs_b": 0.0},
        {"sig35_variant": "TTGACA", "sig35_margin_f_vs_b": 0.5},
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        render_distribution_panel(
            ax,
            rows=rows,
            metric_column="sig35_margin_f_vs_b",
            color_column="sig35_variant",
            render_mode="violin_box",
            panel_title="Sigma-35 ladder",
            square=False,
            axis_styles=SIGMA35_AXIS_STYLES,
        )

        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert tick_labels == [
            "f\nTTGACA",
            "e\nTAGACA",
            "d\nTTTACA",
            "c\nTTGTGA",
            "b\nCTGACA",
        ]
        assert all(label.get_rotation() == 0.0 for label in ax.get_xticklabels())
        assert all(label.get_fontsize() <= 9.5 for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_render_distribution_panel_ordinal_swarm_draws_sampled_points_and_rank_annotation() -> None:
    rows = [
        {
            "ordinal_label": "W1",
            "ordinal_plot_order": 1,
            "ordinal_margin": -0.3,
        },
        {
            "ordinal_label": "W3",
            "ordinal_plot_order": 3,
            "ordinal_margin": 0.1,
        },
        {
            "ordinal_label": "W2",
            "ordinal_plot_order": 2,
            "ordinal_margin": -0.1,
        },
        {
            "ordinal_label": "W3",
            "ordinal_plot_order": 3,
            "ordinal_margin": 0.25,
        },
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        render_distribution_panel(
            ax,
            rows=rows,
            metric_column="ordinal_margin",
            color_column="ordinal_label",
            render_mode="ordinal_swarm",
            panel_title="W strength ladder",
            square=False,
        )

        assert [label.get_text() for label in ax.get_xticklabels()] == ["W1", "W2", "W3"]
        collections = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
        assert collections, "ordinal swarm should render row-level points"
        line_labels = {line.get_label() for line in ax.lines}
        assert "_ordinal_linear_fit" in line_labels
        assert "_ordinal_class_median_connector" in line_labels
        assert any("Ordinal-order rho" in text.get_text() and "R^2" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)


def test_render_distribution_panel_ordinal_swarm_does_not_draw_singleton_whiskers() -> None:
    rows = [
        {
            "ordinal_label": f"J231{index:02d}",
            "ordinal_plot_order": index + 1,
            "ordinal_margin": float(index) / 10.0,
        }
        for index in range(19)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        render_distribution_panel(
            ax,
            rows=rows,
            metric_column="ordinal_margin",
            color_column="ordinal_label",
            render_mode="ordinal_swarm",
            panel_title="Anderson strength ladder",
            square=False,
        )

        collections = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
        assert collections, "ordinal swarm should render visible singleton points"
        assert all(float(size) >= 30.0 for collection in collections for size in collection.get_sizes())
        assert all((collection.get_alpha() or 0.0) >= 0.78 for collection in collections)
        line_labels = {line.get_label() for line in ax.lines}
        assert "_ordinal_linear_fit" in line_labels
        assert "_ordinal_class_median_connector" not in line_labels
        assert "_ordinal_class_median_tick" not in line_labels
        assert "_ordinal_class_iqr" not in line_labels
        assert all(label.get_rotation() >= 45.0 for label in ax.get_xticklabels())
        assert all(label.get_fontsize() <= 8.0 for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_render_distribution_panel_ordinal_swarm_skips_trend_for_degenerate_rank() -> None:
    rows = [
        {
            "ordinal_label": "W1",
            "ordinal_plot_order": 1,
            "ordinal_margin": margin,
        }
        for margin in [-0.2, 0.0, 0.2]
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        render_distribution_panel(
            ax,
            rows=rows,
            metric_column="ordinal_margin",
            color_column="ordinal_label",
            render_mode="ordinal_swarm",
            panel_title="Degenerate ladder",
            square=False,
        )

        line_labels = {line.get_label() for line in ax.lines}
        assert "_ordinal_linear_fit" not in line_labels
        assert "_ordinal_class_median_connector" not in line_labels
        assert "_ordinal_class_median_tick" in line_labels
    finally:
        plt.close(fig)


def test_render_distribution_panel_preserves_explicit_math_axis_label() -> None:
    rows = [
        {"sig35_variant": "f", "sig35_margin_f_vs_b": 0.7},
        {"sig35_variant": "b", "sig35_margin_f_vs_b": -0.4},
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        label = r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
        render_distribution_panel(
            ax,
            rows=rows,
            metric_column="sig35_margin_f_vs_b",
            color_column="sig35_variant",
            render_mode="violin_box",
            panel_title="Sigma-35 ladder",
            square=False,
            y_axis_label=label,
            axis_styles=SIGMA35_AXIS_STYLES,
        )

        assert ax.get_ylabel() == label
    finally:
        plt.close(fig)


def test_render_distribution_grid_ordinal_swarm_keeps_math_axis_labels_on_every_panel(tmp_path: Path) -> None:
    for scalar_id, offset in [("left", 0.0), ("right", 0.1)]:
        scalar_dir = tmp_path / "scalars" / scalar_id
        scalar_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "ordinal_label": ["weak", "middle", "strong"],
                    "ordinal_plot_order": [1.0, 2.0, 3.0],
                    "ordinal_margin": [-0.3 + offset, 0.0 + offset, 0.35 + offset],
                }
            ),
            scalar_dir / "table.parquet",
        )
    x_label = r"$r_{\mathrm{ord}}(x)$ weak$\to$strong class rank"
    y_label = r"$m_{\mathrm{ord}}(x)=\cos(z_x,c_{\mathrm{strong}})-\cos(z_x,c_{\mathrm{weak}})$"
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "ordinal_ladder",
            "kind": "distribution_grid",
            "scalar_ids": ["left", "right"],
            "metric_columns": ["ordinal_margin"],
            "color_column": "ordinal_label",
            "render_mode": "ordinal_swarm",
            "single_row_panels": True,
            "hide_repeated_y_axis": True,
            "x_axis_label": x_label,
            "y_axis_label": y_label,
        }
    )

    result = render_distribution_plot(SimpleNamespace(output_root=tmp_path), spec, pyplot=plt, axis_styles=None)

    try:
        panel_axes = result.figure.axes[:2]
        assert [axis.get_xlabel() for axis in panel_axes] == [x_label, x_label]
        assert [axis.get_ylabel() for axis in panel_axes] == [y_label, y_label]
    finally:
        plt.close(result.figure)


def test_plot_input_payload_preserves_filter_options_for_notebook_dropdowns() -> None:
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "ordinal_ladder",
            "kind": "distribution_grid",
            "scalar_ids": ["ordinal_ladder_rows_anchor"],
            "metric_columns": ["ordinal_margin"],
            "color_column": "ordinal_label",
            "render_mode": "ordinal_swarm",
            "filter_options": [
                {
                    "column": "ordinal_group_id",
                    "label": "Ordinal group",
                    "include_all": False,
                    "values": [
                        {"value": "sigma35", "label": "Sigma-35"},
                        {"value": "t7_w_collection_core60", "label": "W collection core60"},
                    ],
                }
            ],
        }
    )

    payload = plot_input_payload(spec)

    assert payload["filter_options"] == [
        {
            "column": "ordinal_group_id",
            "label": "Ordinal group",
            "type": "categorical",
            "include_all": False,
            "values": [
                {"value": "sigma35", "label": "Sigma-35"},
                {"value": "t7_w_collection_core60", "label": "W collection core60"},
            ],
        }
    ]


def test_plot_input_payload_preserves_annotation_hue_for_notebook_reference_controls() -> None:
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "sfxi_reference_umap",
            "kind": "projection_grid",
            "projection_ids": ["umap_reference_core60"],
            "annotation": {
                "reference_set": "reference_sfxi_archive",
                "hue_column": "sfxi_ref__metric_value",
                "colorbar_label": "SFXI metric",
            },
        }
    )

    payload = plot_input_payload(spec)

    assert payload["annotation"]["reference_set"] == "reference_sfxi_archive"
    assert payload["annotation"]["hue_column"] == "sfxi_ref__metric_value"
    assert payload["annotation"]["colorbar_label"] == "SFXI metric"


def test_heatmap_grid_respects_configured_sig35_order_without_reference_pollution() -> None:
    rows = [
        {"row_variant": "TTGACA (f)", "column_variant": "TTGACA (f)", "metric_value": 0.0},
        {"row_variant": "TTGACA (f)", "column_variant": "TAGACA (e)", "metric_value": 0.4},
        {"row_variant": "TAGACA (e)", "column_variant": "TTGACA (f)", "metric_value": 0.4},
        {"row_variant": "TAGACA (e)", "column_variant": "TAGACA (e)", "metric_value": 0.0},
        {
            "row_variant": "TTGACA (annotated, unranked)",
            "column_variant": "TTGACA (f)",
            "metric_value": 0.8,
        },
        {
            "row_variant": "TTGACA (f)",
            "column_variant": "TTGACA (annotated, unranked)",
            "metric_value": 0.8,
        },
    ]

    grid, row_values, column_values = _heatmap_grid_from_rows(
        rows,
        row_column="row_variant",
        column_column="column_variant",
        value_column="metric_value",
        row_order=["TTGACA (f)", "TAGACA (e)"],
        column_order=["TTGACA (f)", "TAGACA (e)"],
    )

    assert row_values == ["TTGACA (f)", "TAGACA (e)"]
    assert column_values == ["TTGACA (f)", "TAGACA (e)"]
    assert [[round(float(value), 3) for value in row] for row in grid.tolist()] == [[0.0, 0.4], [0.4, 0.0]]


def test_heatmap_grid_fails_fast_on_duplicate_semantic_cells() -> None:
    rows = [
        {"row_variant": "TTGACA (f)", "column_variant": "TAGACA (e)", "metric_value": 0.4},
        {"row_variant": "TTGACA (f)", "column_variant": "TAGACA (e)", "metric_value": 0.5},
    ]

    with pytest.raises(ContractViolationError, match="duplicate heatmap cell"):
        _heatmap_grid_from_rows(
            rows,
            row_column="row_variant",
            column_column="column_variant",
            value_column="metric_value",
            row_order=[],
            column_order=[],
        )


def test_heatmap_grid_fails_fast_when_later_rows_miss_required_columns() -> None:
    rows = [
        {"row_variant": "TTGACA (f)", "column_variant": "TAGACA (e)", "metric_value": 0.4},
        {"row_variant": "TAGACA (e)", "metric_value": 0.5},
    ]

    with pytest.raises(ContractViolationError, match="column_variant"):
        _heatmap_grid_from_rows(
            rows,
            row_column="row_variant",
            column_column="column_variant",
            value_column="metric_value",
            row_order=[],
            column_order=[],
        )


def test_read_table_rows_validates_required_schema_columns(tmp_path) -> None:
    table_path = tmp_path / "table.parquet"
    pq.write_table(pa.table({"x": [1.0], "metric_value": [0.4]}), table_path)

    with pytest.raises(ContractViolationError, match="missing required column"):
        read_table_rows(table_path, required_columns=["x", "y"], artifact_label="fixture table")


def test_layout_reservation_tracks_explicit_legend_space() -> None:
    reservation = LayoutReservation()

    reservation.reserve_bottom(0.08)
    reservation.reserve_bottom(0.04)
    reservation.reserve_right(0.20)

    assert reservation.legend_bottom == 0.08
    assert reservation.legend_right == 0.20
    assert reservation.has_reservation


def test_resolve_annotation_rows_reports_later_row_missing_label_column() -> None:
    reference_set = SimpleNamespace(
        ids=["target"],
        match_column="id",
        label_column="label",
        label_mode="label_and_highlight",
        display_labels={},
        where=[],
        where_all=[],
        require_non_empty=True,
    )
    context = SimpleNamespace(config=SimpleNamespace(reference_sets={"fixture_reference": reference_set}))
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_projection",
            "kind": "projection_scatter",
            "projection_ids": ["fixture_projection"],
            "annotation": {"reference_set": "fixture_reference", "missing_policy": "allow"},
        }
    )

    resolved = resolve_annotation_rows(
        context,
        [
            {"id": "target", "label": "Target", "x": 0.0, "y": 0.0},
            {"id": "other", "x": 1.0, "y": 1.0},
        ],
        spec=spec,
    )

    assert resolved.selected_rows == []
    assert resolved.label_column is None
    assert resolved.state["complete"] is False
    assert resolved.state["error"] == "missing_reference_columns"
    assert resolved.state["missing_columns"] == ["label"]


def test_render_heatmap_panel_can_hide_redundant_y_tick_labels() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        image = _render_heatmap_panel(
            ax,
            grid=[[0.0, 0.5], [0.5, 0.0]],
            row_values=["TTGACA (f)", "CTGACA (b)"],
            column_values=["TTGACA (f)", "CTGACA (b)"],
            row_column="row_variant",
            column_column="column_variant",
            title="Sigma-35 distance",
            cmap="cividis",
            norm=None,
            show_y_tick_labels=False,
            show_y_axis_label=False,
        )

        assert image is not None
        assert ax.get_ylabel() == ""
        assert all(not tick.get_text() for tick in ax.get_yticklabels())
    finally:
        plt.close(fig)


def test_render_heatmap_panel_can_hide_redundant_x_tick_labels() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        image = _render_heatmap_panel(
            ax,
            grid=[[0.0, 0.5], [0.5, 0.0]],
            row_values=["TTGACA (f)", "CTGACA (b)"],
            column_values=["TTGACA (f)", "CTGACA (b)"],
            row_column="row_variant",
            column_column="column_variant",
            title="Sigma-35 distance",
            cmap="cividis",
            norm=None,
            square_cells=True,
            show_x_tick_labels=False,
            show_x_axis_label=False,
        )

        assert image is not None
        assert ax.get_xlabel() == ""
        assert all(not tick.get_text() for tick in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_render_heatmap_panel_compacts_sigma35_tick_labels_for_dense_grids() -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    try:
        image = _render_heatmap_panel(
            ax,
            grid=[[0.0, 0.2], [0.2, 0.0]],
            row_values=["TTGACA (f)", "CTGACA (b)"],
            column_values=["TTGACA (f)", "CTGACA (b)"],
            row_column="row_variant",
            column_column="column_variant",
            title="Sigma-35 centroid distance",
            cmap="viridis",
            norm=None,
            square_cells=True,
            axis_styles=SIGMA35_HEATMAP_AXIS_STYLES,
        )

        assert image is not None
        assert [label.get_text() for label in ax.get_xticklabels()] == ["F", "B"]
        assert [label.get_text() for label in ax.get_yticklabels()] == ["f\nTTGACA", "b\nCTGACA"]
        assert all(label.get_rotation() == 0.0 for label in ax.get_xticklabels())
        assert all(label.get_fontsize() <= 9.5 for label in ax.get_xticklabels())
    finally:
        plt.close(fig)


def test_draw_annotation_callouts_can_skip_marker_overlay() -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        _draw_annotation_callouts(
            ax,
            rows=[{"x": 0.3, "y": 0.7}],
            resolved_x="x",
            resolved_y="y",
            label_texts=["anchor"],
            marker_colors=["#111111"],
            marker=None,
            marker_size=0.0,
        )

        assert len(ax.collections) == 0
        assert [text.get_text() for text in ax.texts] == ["anchor"]
    finally:
        plt.close(fig)


def test_draw_resolved_annotations_highlights_all_reference_rows_when_labels_are_suppressed() -> None:
    rows = [{"x": float(index), "y": float(index), "usr_label__primary": f"J231{index:02d}"} for index in range(6)]
    context = SimpleNamespace(
        config=SimpleNamespace(
            reference_sets={
                "reference_anderson_igem": SimpleNamespace(label_mode="label_and_highlight"),
            }
        )
    )
    spec = SimpleNamespace(annotation=SimpleNamespace(reference_set="reference_anderson_igem"), color_column=None)
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        _draw_resolved_annotations(
            ax,
            context=context,
            spec=spec,
            rows=rows,
            resolved_x="x",
            resolved_y="y",
            resolved_label_column="usr_label__primary",
            color_map={},
        )

        assert len(ax.collections) == 1
        assert len(ax.collections[0].get_offsets()) == 6
        assert len(ax.texts) == 0
    finally:
        plt.close(fig)


def test_annotation_hue_requires_finite_numeric_values_by_default() -> None:
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_projection",
            "kind": "projection_scatter",
            "projection_ids": ["fixture_projection"],
            "annotation": {
                "reference_set": "reference_sfxi",
                "hue_column": "sfxi_ref__metric_value",
            },
        }
    )

    with pytest.raises(ContractViolationError, match="non-finite annotation hue"):
        _annotation_continuous_color_encoding(
            [
                {"x": 0.0, "y": 0.0, "sfxi_ref__metric_value": 0.15},
                {"x": 1.0, "y": 1.0, "sfxi_ref__metric_value": math.nan},
            ],
            spec,
        )


def test_projection_scatter_reference_stars_use_continuous_annotation_hue(tmp_path: Path) -> None:
    projection_dir = tmp_path / "projections" / "fixture_umap"
    projection_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["ref-low", "ref-high", "background"],
                "x": [0.0, 1.0, 0.5],
                "y": [0.0, 1.0, 0.5],
                "is_reference": [True, True, False],
                "usr_label__primary": ["ES low", "ES high", "background"],
                "sfxi_ref__metric_value": [0.15, 0.85, 0.4],
            }
        ),
        projection_dir / "coords.parquet",
    )
    reference_set = SimpleNamespace(
        ids=[],
        match_column="id",
        label_column="usr_label__primary",
        label_mode="highlight_only",
        display_labels={},
        where=[
            SimpleNamespace(
                column="is_reference",
                equals=True,
                in_values=[],
                regex=None,
                not_regex=None,
                non_null=True,
            )
        ],
        where_all=[],
        require_non_empty=True,
    )
    spec = ResolvedPlotSpec.model_validate(
        {
            "plot_id": "fixture_sfxi_reference_umap",
            "kind": "projection_scatter",
            "projection_ids": ["fixture_umap"],
            "annotation": {
                "reference_set": "reference_sfxi",
                "hue_column": "sfxi_ref__metric_value",
                "colorbar_label": "SFXI",
            },
        }
    )

    result = render_projection_plot(
        SimpleNamespace(
            output_root=tmp_path,
            config=SimpleNamespace(reference_sets={"reference_sfxi": reference_set}),
        ),
        spec,
        pyplot=plt,
        axis_styles=None,
    )

    try:
        arrays = [
            collection.get_array()
            for collection in result.figure.axes[0].collections
            if collection.get_array() is not None
        ]
        assert len(arrays) == 1
        assert arrays[0].tolist() == [0.15, 0.85]
        assert len(result.figure.axes) == 2
        assert result.figure.axes[1].get_ylabel() == "SFXI"
    finally:
        plt.close(result.figure)


def test_derived_panel_label_humanizes_context_delta_distribution_ids() -> None:
    assert derived_panel_label("context_delta_distribution_output_layer_mean_7b") == "Output Layer Mean Evo 2 7B"


def test_wrap_plot_title_respects_explicit_max_lines() -> None:
    wrapped = wrap_plot_title(
        "intermediate embedding 20b full context 1 kb candidate comparison surface",
        width=14,
        max_lines=2,
    )

    assert wrapped.count("\n") == 1
    assert wrapped.endswith("...")


def test_wrap_plot_title_preserves_explicit_line_breaks() -> None:
    wrapped = wrap_plot_title("Panel title\nN = 157,164", width=18)

    assert wrapped.splitlines() == ["Panel Title", "N = 157,164"]


def test_compact_candidate_title_shortens_candidate_surface_titles() -> None:
    compact = compact_candidate_title("Evo 2 7B · 1 Kb Construct Context · Intermediate Block Mean")

    assert compact == "7B · 1 kb ctx · Block"


def test_single_row_candidate_grid_layout_is_opt_in_for_four_panel_galleries() -> None:
    assert _panel_grid_dimensions(4) == (2, 2)
    assert _panel_grid_dimensions(4, prefer_single_row=True) == (1, 4)
    assert (
        _grid_figure_size(4, square_panels=True, prefer_single_row=True)[0]
        > _grid_figure_size(
            4,
            square_panels=True,
        )[0]
    )


def test_panel_grid_dimensions_cap_seven_and_eight_panel_galleries_at_two_rows() -> None:
    assert _panel_grid_dimensions(7) == (2, 4)
    assert _panel_grid_dimensions(8) == (2, 4)


def test_single_row_candidate_layout_can_show_six_configured_panels() -> None:
    assert _panel_grid_dimensions(6, prefer_single_row=True) == (1, 6)


def testrender_metric_panel_wraps_representation_health_axis_label_to_at_most_two_lines() -> None:
    rows = [
        {
            "category": "pairwise_cosine_distance_iqr",
            "display_name": "Pairwise cosine distance interquartile range",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "anchor_60bp",
            "candidate_family": "intermediate_embedding",
            "direction": "higher_is_better",
            "unit": "cosine distance",
            "metric_value": 0.18,
        },
        {
            "category": "pairwise_cosine_distance_iqr",
            "display_name": "Pairwise cosine distance interquartile range",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_20b",
            "candidate_scope": "full_context_1kb",
            "candidate_family": "output_layer_mean",
            "direction": "higher_is_better",
            "unit": "cosine distance",
            "metric_value": 0.05,
        },
    ]
    spec = _metric_spec(plot_id="representation_health_summary", color_column=None)
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Pairwise cosine distance interquartile range",
            color_map=color_map,
            square=True,
        )

        assert ax.get_xlabel().count("\n") <= 1
    finally:
        plt.close(fig)


def test_render_metric_panel_keeps_reference_set_qualifier_out_of_axis_label() -> None:
    rows = [
        {
            "category": "reference_set: reference_spyp_sulap",
            "display_name": "Reference group size\nReference set: spyP / sulAp",
            "label": "candidate_a",
            "candidate_label": "candidate_a",
            "candidate_model": "evo2_7b",
            "candidate_scope": "merged_anchor_insert_seq_mean",
            "candidate_family": "intermediate_embedding",
            "direction": "descriptive",
            "unit": "rows",
            "metric_value": 2.0,
        },
        {
            "category": "reference_set: reference_spyp_sulap",
            "display_name": "Reference group size\nReference set: spyP / sulAp",
            "label": "candidate_b",
            "candidate_label": "candidate_b",
            "candidate_model": "evo2_7b",
            "candidate_scope": "full_context_anchor_mean",
            "candidate_family": "intermediate_embedding",
            "direction": "descriptive",
            "unit": "rows",
            "metric_value": 2.0,
        },
    ]
    spec = _metric_spec(plot_id="reference_alignment_summary", color_column=None)
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Reference group size\nReference set: spyP / sulAp",
            color_map=color_map,
            square=True,
        )

        assert "Reference Set" in ax.get_title()
        assert "Reference Set" not in ax.get_ylabel()
        assert "..." not in ax.get_ylabel()
        assert "Reference Group Size" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_representation_health_summary_declares_square_metric_panels() -> None:
    assert metric_panel_uses_square_axes("representation_health_summary")


def test_representation_health_summary_uses_horizontal_metric_panel_layout() -> None:
    rows, columns, figsize = metric_panel_grid_layout("representation_health_summary", 3)

    assert (rows, columns) == (1, 3)
    assert figsize[0] > figsize[1]


def test_reference_alignment_summary_uses_landscape_appendix_layout() -> None:
    rows, columns, figsize = metric_panel_grid_layout("reference_alignment_summary", 31)

    assert (rows, columns) == (4, 8)
    assert figsize[0] > figsize[1]
