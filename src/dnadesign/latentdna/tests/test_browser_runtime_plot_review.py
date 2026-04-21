import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.latentdna.src.notebooks import browser_runtime_plot_review as plot_review_runtime
from dnadesign.latentdna.src.notebooks import browser_runtime_support as runtime_support
from dnadesign.latentdna.src.notebooks.browser_runtime_plot_review import render_plot_review_surface


def _decode_svg_markup(rendered: mo.Html) -> str:
    match = re.search(r"data:image/svg\+xml;base64,([^']+)", rendered.text)
    assert match is not None
    return base64.b64decode(match.group(1)).decode("utf-8")


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


def test_render_plot_review_surface_uses_placeholder_for_scatter_panels_without_finite_values(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "synthetic_margin_ethanol_vs_background": [float("nan")],
                "synthetic_margin_cipro_vs_background": [float("nan")],
                "design_family": ["background_only"],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "synthetic_margin_ethanol_vs_background",
            "y_column": "synthetic_margin_cipro_vs_background",
            "panel_titles": ["20B · 60 bp · Block"],
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
    svg_markup = _decode_svg_markup(rendered)
    assert "Margins unavailable" in svg_markup
    assert "No finite values in this snapshot" in svg_markup


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


def test_render_plot_review_surface_compacts_regulator_composition_legend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "y": [0.0, 1.0, 2.0, 3.0, 4.0],
                "design_regulator_composition": [
                    "cpxR_MANWWHTTTAM",
                    "lexA_CTGTATAWAWWHACA",
                    "baeR_TTTCTSCVHNA+lexA_CTGTATAWAWWHACA",
                    "sig35=b",
                    "control",
                ],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "appendix_geometry_audit",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["Geometry audit"],
            "hue_options": [
                {"column": "design_regulator_composition", "label": "Reg. comp.", "type": "categorical"},
            ],
        },
        frames=frames,
        hue_column="design_regulator_composition",
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    svg_markup = _decode_svg_markup(rendered)
    for label in ("BaeR+LexA", "Bg", "CpxR", "Ctrl", "LexA"):
        assert label in svg_markup
    assert svg_markup.count("<!-- Bg -->") == 1


def test_render_plot_review_surface_orders_sig35_legend_by_strength(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 1.0, 2.0, 3.0],
                "sig35_variant": ["b", "f", "d", "control"],
            }
        )
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "appendix_geometry_audit",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["Geometry audit"],
            "hue_options": [
                {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
            ],
        },
        frames=frames,
        hue_column="sig35_variant",
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        labels = [text.get_text() for text in legend.get_texts()]
        colors = [handle.get_color() for handle in legend.legend_handles]
        assert labels == ["TTGACA (f)", "TTTACA (d)", "CTGACA (b)", "Control"]
        assert colors == ["#B2182B", "#F4A582", "#2166AC", "#7F8894"]
    finally:
        plt.close(fig)


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


def test_render_plot_review_surface_prefers_single_row_for_four_panel_candidate_galleries(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [1.0, 0.0],
                "design_family": ["ethanol", "background_only"],
            }
        )
        for _ in range(4)
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["A", "B", "C", "D"],
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

    try:
        y_positions = {round(axis.get_position().y0, 3) for axis in fig.axes[:4]}
        assert len(y_positions) == 1
    finally:
        plt.close(fig)


def test_render_plot_review_surface_keeps_seven_panel_candidate_gallery_within_two_rows(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [1.0, 0.0],
                "design_family": ["ethanol", "background_only"],
            }
        )
        for _ in range(7)
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": [f"Panel {index}" for index in range(7)],
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

    try:
        panel_axes = fig.axes[:7]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 2
        assert fig.axes[7].axison is False
    finally:
        plt.close(fig)


def test_render_plot_review_surface_places_continuous_colorbar_below_design_centroid_gallery(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [2.0, 1.0, 0.0],
                "synthetic_margin_ethanol_vs_background": [-0.35, 0.0, 0.28],
            }
        )
        for _ in range(2)
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["A", "B"],
            "hue_options": [
                {
                    "column": "synthetic_margin_ethanol_vs_background",
                    "label": "Ethanol margin",
                    "type": "continuous",
                },
            ],
        },
        frames=frames,
        hue_column="synthetic_margin_ethanol_vs_background",
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:2]
        colorbar_ax = fig.axes[-1]
        assert len(fig.axes) == 3
        assert panel_axes[0].collections[0].cmap.name == "coolwarm"
        assert colorbar_ax.get_position().width > colorbar_ax.get_position().height
        assert colorbar_ax.get_position().y1 < min(axis.get_position().y0 for axis in panel_axes)
    finally:
        plt.close(fig)


def test_render_plot_review_surface_uses_single_row_for_dataset_overview_three_panels(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "dimension": [
                    "provenance",
                    "provenance",
                    "generation_plan",
                    "generation_plan",
                    "sig35_variant",
                    "sig35_variant",
                ],
                "dimension_label": [
                    "Provenance",
                    "Provenance",
                    "Generation plan",
                    "Generation plan",
                    "Sigma-35 variant",
                    "Sigma-35 variant",
                ],
                "category_label": ["DenseGen", "Control", "Background only", "Control", "f", "control"],
                "fraction": [0.99997, 0.00003, 0.75, 0.25, 0.85, 0.15],
                "count": [157160, 4, 3, 1, 5, 1],
                "percent": [99.997, 0.003, 75.0, 25.0, 85.0, 15.0],
                "denominator": [157164, 157164, 4, 4, 6, 6],
                "order": [1, 2, 1, 2, 1, 2],
            }
        )
    ]

    fig = render_plot_review_surface(
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

    try:
        panel_axes = fig.axes[:3]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 1
    finally:
        plt.close(fig)


def test_render_plot_review_surface_keeps_scree_gallery_within_two_rows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    for index in range(7):
        reducer_dir = tmp_path / "reducers" / f"demo_curve_{index}"
        reducer_dir.mkdir(parents=True)
        (reducer_dir / "summary.json").write_text(
            json.dumps({"explained_variance_ratio": [0.5, 0.3, 0.2]}),
            encoding="utf-8",
        )

    fig = render_plot_review_surface(
        {
            "plot_id": "representation_scree_diagnostic",
            "kind": "curve_grid",
            "reducer_ids": [f"demo_curve_{index}" for index in range(7)],
            "panel_titles": [f"Panel {index}" for index in range(7)],
        },
        frames=[],
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:7]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 2
    finally:
        plt.close(fig)


def test_load_plot_review_frames_enriches_projection_frames_from_projection_view_rows(tmp_path: Path) -> None:
    projection_dir = tmp_path / "projections" / "umap_anchor"
    projection_dir.mkdir(parents=True)
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )
    (projection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": [
                    {
                        "kind": "view_matrix",
                        "id": "intermediate_embedding_7b_anchor_60bp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    view_dir = tmp_path / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "log_likelihood_per_token_7b": [-1.2, -0.9],
            "design_family": ["ethanol", "ciprofloxacin"],
        }
    ).to_parquet(view_dir / "rows.parquet", index=False)

    frames = plot_review_runtime.load_plot_review_frames(
        {
            "plot_id": "appendix_umap_gallery",
            "kind": "projection_grid",
            "projection_ids": ["umap_anchor"],
            "hue_options": [
                {"column": "log_likelihood_per_token_7b", "label": "7B log likelihood / token", "type": "continuous"},
            ],
        },
        joinable_tables=[],
        output_root=tmp_path,
    )

    assert len(frames) == 1
    assert frames[0]["log_likelihood_per_token_7b"].tolist() == [-1.2, -0.9]
    assert frames[0].attrs["view_id"] == "intermediate_embedding_7b_anchor_60bp"
