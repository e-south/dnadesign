from __future__ import annotations

import math

import matplotlib.pyplot as plt

from dnadesign.latentdna.src.contracts.plot import ResolvedPlotSpec
from dnadesign.latentdna.src.plots.render import (
    _category_color_map,
    _derived_panel_label,
    _grid_figure_size,
    _panel_grid_dimensions,
    _render_metric_panel,
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

        lower, upper = ax.get_ylim()
        assert upper > 1.0
        assert lower < 0.0
        assert any(text.get_text() == "NA" for text in ax.texts)
        finite_text_positions = [text.get_position()[1] for text in ax.texts if text.get_text() != "NA"]
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
        assert tick_labels == ["7B 60bp Block", "20B 1kb ctx Logits"]
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
