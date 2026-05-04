from __future__ import annotations

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt

from dnadesign.latentdna.src.contracts.plot import ResolvedPlotSpec
from dnadesign.latentdna.src.plots.render import (
    _add_figure_legends,
    _add_side_figure_legends,
    _category_color_map,
    _category_key,
    _continuous_color_encoding,
    _derived_panel_label,
    _draw_annotation_callouts,
    _draw_resolved_annotations,
    _grid_figure_size,
    _heatmap_grid_from_rows,
    _panel_grid_dimensions,
    _render_distribution_panel,
    _render_heatmap_panel,
    _render_metric_panel,
    _row_sig35_plot_category,
)
from dnadesign.latentdna.src.visual_style import compact_candidate_title, wrap_plot_title


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


def test_static_sig35_hue_keeps_context_derived_densegen_rows_categorical() -> None:
    assert (
        _row_sig35_plot_category(
            {
                "sig35_variant": "f",
                "source_family": "construct_derived",
                "source_class": "densegen",
            },
            "sig35_variant",
        )
        == "f"
    )
    assert (
        _row_sig35_plot_category(
            {
                "sig35_variant": "f",
                "source_family": "construct_derived",
                "source_class": "construct_derived",
            },
            "sig35_variant",
        )
        == "__latentdna_noncanonical_sig35__"
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


def test_render_metric_panel_ignores_nan_values_when_setting_limits_and_annotations() -> None:
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
            "candidate_family": "pooled_logits",
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
        _render_metric_panel(
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


def test_render_metric_panel_uses_compact_candidate_tick_labels() -> None:
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
            "candidate_family": "pooled_logits",
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
        _render_metric_panel(
            ax,
            rows=rows,
            spec=spec,
            panel_title="Effective rank",
            color_map=color_map,
            square=True,
        )

        tick_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert tick_labels == ["7B anchor insert Block", "20B 1kb ctx Logits"]
        assert float(ax.get_box_aspect()) == 1.0
    finally:
        plt.close(fig)


def test_render_metric_panel_uses_placeholder_when_all_values_are_missing() -> None:
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
            "candidate_family": "pooled_logits",
            "direction": "higher_is_better",
            "unit": "ratio",
            "metric_value": math.nan,
        },
    ]
    spec = _metric_spec(plot_id="design_structure_summary", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        _render_metric_panel(
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


def test_render_metric_panel_suppresses_redundant_scope_in_grouped_ticks() -> None:
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
            "candidate_family": "pooled_logits",
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
            "candidate_family": "pooled_logits",
            "direction": "higher_is_better",
            "unit": "cosine",
            "metric_value": 0.04,
        },
    ]
    spec = _metric_spec(plot_id="context_robustness_summary", color_column="candidate_family")
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        _render_metric_panel(
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
        _render_distribution_panel(
            ax,
            rows=rows,
            metric_column="sig35_margin_f_vs_b",
            color_column="sig35_variant",
            render_mode="violin_box",
            panel_title="Sigma-35 ladder",
            square=False,
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


def test_render_distribution_panel_preserves_explicit_math_axis_label() -> None:
    rows = [
        {"sig35_variant": "f", "sig35_margin_f_vs_b": 0.7},
        {"sig35_variant": "b", "sig35_margin_f_vs_b": -0.4},
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        label = r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
        _render_distribution_panel(
            ax,
            rows=rows,
            metric_column="sig35_margin_f_vs_b",
            color_column="sig35_variant",
            render_mode="violin_box",
            panel_title="Sigma-35 ladder",
            square=False,
            y_axis_label=label,
        )

        assert ax.get_ylabel() == label
    finally:
        plt.close(fig)


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
        )

        assert image is not None
        assert [label.get_text() for label in ax.get_xticklabels()] == ["f", "b"]
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


def test_derived_panel_label_humanizes_context_delta_distribution_ids() -> None:
    assert _derived_panel_label("context_delta_distribution_pooled_logits_7b") == "Pooled Logits Evo 2 7B"


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


def test_render_metric_panel_wraps_representation_health_axis_label_to_at_most_two_lines() -> None:
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
            "candidate_family": "pooled_logits",
            "direction": "higher_is_better",
            "unit": "cosine distance",
            "metric_value": 0.05,
        },
    ]
    spec = _metric_spec(plot_id="representation_health_summary", color_column=None)
    color_map, _ = _category_color_map([rows], spec.color_column)
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        _render_metric_panel(
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
