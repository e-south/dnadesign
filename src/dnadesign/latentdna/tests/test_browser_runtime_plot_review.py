import json
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import pandas as pd

from dnadesign.latentdna.src.notebooks import browser_runtime_support as runtime_support
from dnadesign.latentdna.src.notebooks.browser_runtime_plot_review import render_plot_review_surface


def test_render_plot_review_surface_supports_semantic_xy_columns(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "synthetic_margin_ethanol_vs_background": [0.4, -0.2],
                "synthetic_margin_cipro_vs_background": [0.25, -0.1],
                "design_family": ["ethanol", "background_only"],
                "sig35_variant": ["b", "f"],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "panel_titles": ["Design-centroid margin gallery"],
            "hue_options": [
                {"column": "design_family", "label": "Design family", "type": "categorical"},
            ],
        },
        frames=frames,
        hue_column="design_family",
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "design_centroid_margin_gallery" in rendered.text


def test_render_plot_review_surface_supports_metric_panel_grid_from_current_scalar_rows(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "category": ["effective_rank", "effective_rank"],
                "label": ["candidate_a", "candidate_b"],
                "panel_id": ["Effective rank", "Effective rank"],
                "metric_value": [6.4, 2.1],
                "display_name": ["Effective rank", "Effective rank"],
                "direction": ["higher_is_better", "higher_is_better"],
                "unit": ["dims", "dims"],
                "candidate_family": ["intermediate_embedding", "pooled_logits"],
                "candidate_model": ["20b", "20b"],
                "candidate_scope": ["anchor_60bp", "full_context_1kb"],
                "candidate_label": ["candidate_a", "candidate_b"],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "representation_health_summary",
            "kind": "metric_panel_grid",
            "scalar_id": "representation_health_summary_metrics",
            "row_column": "category",
            "panel_column": "display_name",
            "column_column": "label",
            "label_column": "candidate_label",
            "value_column": "metric_value",
            "color_column": "candidate_family",
            "direction_column": "direction",
            "unit_column": "unit",
            "value_label": "Metric value",
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "representation_health_summary" in rendered.text


def test_render_plot_review_surface_supports_categorical_count_grid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "dimension": ["provenance", "provenance", "generation_plan", "generation_plan"],
                "dimension_label": ["Provenance", "Provenance", "Generation plan", "Generation plan"],
                "category_label": ["DenseGen", "Manual/wildtype control", "Background only", "Control"],
                "fraction": [0.99997, 0.00003, 0.75, 0.25],
                "count": [157160, 4, 3, 1],
                "percent": [99.997, 0.003, 75.0, 25.0],
                "denominator": [157164, 157164, 4, 4],
                "order": [1, 2, 1, 2],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "dataset_overview",
            "kind": "categorical_count",
            "scalar_id": "dataset_overview_counts",
            "row_column": "dimension",
            "column_column": "category_label",
            "value_column": "fraction",
            "panel_column": "dimension_label",
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "dataset_overview" in rendered.text


def test_render_plot_review_surface_supports_curve_grid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    reducer_dir = tmp_path / "reducers" / "demo_curve"
    reducer_dir.mkdir(parents=True)
    (reducer_dir / "summary.json").write_text(
        json.dumps({"explained_variance_ratio": [0.5, 0.3, 0.2]}),
        encoding="utf-8",
    )

    rendered = render_plot_review_surface(
        {
            "plot_id": "representation_scree_diagnostic",
            "kind": "curve_grid",
            "reducer_ids": ["demo_curve"],
            "panel_titles": ["Demo curve"],
        },
        frames=[],
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "representation_scree_diagnostic" in rendered.text
