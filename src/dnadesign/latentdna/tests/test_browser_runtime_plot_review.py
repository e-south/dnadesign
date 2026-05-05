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

SIGMA35_NONCANONICAL_BUCKET = "__latentdna_reference_or_other__"
SIGMA35_AXIS_STYLES = {
    "sig35_variant": {
        "axis_id": "sigma35",
        "column": "sig35_variant",
        "label": "Sigma-35 variant",
        "kind": "categorical",
        "category_order": ["f", "e", "d", "c", "b", "control"],
        "display_labels": {
            "f": "TTGACA (f)",
            "e": "TAGACA (e)",
            "d": "TTTACA (d)",
            "c": "TTGTGA (c)",
            "b": "CTGACA (b)",
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
REGULATOR_AXIS_STYLES = {
    "design_regulator_composition": {
        "axis_id": "regulator_composition",
        "column": "design_regulator_composition",
        "label": "Reg. comp.",
        "kind": "categorical",
        "category_order": [
            "baeR_TTTCTSCVHNA+lexA_CTGTATAWAWWHACA",
            "sig35=b",
            "cpxR_MANWWHTTTAM",
            "control",
            "lexA_CTGTATAWAWWHACA",
        ],
        "display_labels": {
            "baeR_TTTCTSCVHNA+lexA_CTGTATAWAWWHACA": "BaeR+LexA",
            "cpxR_MANWWHTTTAM": "CpxR",
            "lexA_CTGTATAWAWWHACA": "LexA",
            "sig35=b": "Bg",
            "control": "Ctrl",
        },
    }
}


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
                "candidate_family": ["intermediate_embedding", "output_layer_mean"],
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
            "plot_id": "appendix_geometry_review",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["Geometry browser"],
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
        axis_styles=REGULATOR_AXIS_STYLES,
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
            "plot_id": "appendix_geometry_review",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "panel_titles": ["Geometry browser"],
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
        axis_styles=SIGMA35_AXIS_STYLES,
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


def test_plot_review_sig35_hue_keeps_context_derived_densegen_rows_categorical() -> None:
    frame = pd.DataFrame(
        {
            "sig35_variant": ["f", "b", "f", "ACCGCG"],
            "source_family": [
                "construct_derived",
                "construct_derived",
                "construct_derived",
                "reference_source",
            ],
            "source_class": ["densegen", "densegen", "construct_derived", "reference_control"],
        }
    )

    series = plot_review_runtime._categorical_hue_series(frame, "sig35_variant", axis_styles=SIGMA35_AXIS_STYLES)

    assert series.tolist() == [
        "f",
        "b",
        SIGMA35_NONCANONICAL_BUCKET,
        SIGMA35_NONCANONICAL_BUCKET,
    ]


def test_render_plot_review_surface_preserves_explicit_math_axis_labels(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "sig35_margin_f_vs_b": [0.8, -0.1],
                "synthetic_best_stress_margin": [0.6, 0.2],
                "sig35_variant": ["f", "b"],
            }
        )
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "sigma35_stress_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "sig35_margin_f_vs_b",
            "y_column": "synthetic_best_stress_margin",
            "x_axis_label": r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$",
            "y_axis_label": r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$",
            "panel_titles": ["Stress margin"],
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
        axis = fig.axes[0]
        assert axis.get_xlabel() == r"$m_{\sigma35}(x)=\cos(z_x,c_f)-\cos(z_x,c_b)$"
        assert axis.get_ylabel() == r"$m_{\mathrm{stress}}(x)=\max\{m_{\mathrm{eth}}(x),m_{\mathrm{cipro}}(x)\}$"
    finally:
        plt.close(fig)


def test_render_plot_review_surface_renders_sigma35_margin_ladder_as_single_row_square_panels(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame(
            {
                "sig35_margin_f_vs_b": [0.8, 0.55, 0.22, -0.15],
                "sig35_variant": ["f", "e", "d", "b"],
            }
        ),
        pd.DataFrame(
            {
                "sig35_margin_f_vs_b": [0.62, 0.41, 0.08, -0.24],
                "sig35_variant": ["f", "e", "d", "b"],
            }
        ),
        pd.DataFrame(
            {
                "sig35_margin_f_vs_b": [0.93, 0.68, 0.33, -0.09],
                "sig35_variant": ["f", "e", "d", "b"],
            }
        ),
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "sigma35_margin_ladder_gallery",
            "kind": "distribution_grid",
            "scalar_ids": ["anchor", "full_context", "anchor_mean"],
            "panel_titles": ["Anchor", "1 kb seq", "Anchor mean"],
            "hue_options": [
                {"column": "sig35_variant", "label": "Sigma-35 variant", "type": "categorical"},
            ],
            "metric_columns": ["sig35_margin_f_vs_b"],
            "color_column": "sig35_variant",
            "render_mode": "violin_box",
            "single_row_panels": True,
            "square_panels": True,
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        axis_styles=SIGMA35_AXIS_STYLES,
    )

    try:
        fig.canvas.draw()
        assert len(fig.axes) == 3
        axis_positions = [axis.get_position() for axis in fig.axes]
        assert max(position.y0 for position in axis_positions) - min(position.y0 for position in axis_positions) < 0.02
        for axis in fig.axes:
            assert abs(float(axis.get_box_aspect()) - 1.0) < 0.05
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


def test_render_plot_review_surface_does_not_leak_debug_distribution_scalar_ids(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))
    frames = [
        pd.DataFrame(
            {
                "score": [0.25, 0.5, 0.75],
            }
        )
    ]

    rendered = render_plot_review_surface(
        {
            "plot_id": "distribution_demo",
            "kind": "distribution_grid",
            "scalar_ids": ["debug_distribution_demo"],
            "value_column": "score",
            "metric_columns": ["score"],
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    svg_markup = _decode_svg_markup(rendered)
    assert "debug_distribution_demo" not in svg_markup
    assert "debug distribution demo" not in svg_markup.lower()
    assert "Score" in svg_markup


def test_render_plot_review_surface_preserves_context_distribution_family_titles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame({"score": [0.25, 0.5, 0.75]}),
        pd.DataFrame({"score": [0.35, 0.55, 0.85]}),
    ]

    fig = render_plot_review_surface(
        {
            "plot_id": "distribution_demo",
            "kind": "distribution_grid",
            "scalar_ids": [
                "context_delta_distribution_intermediate_embedding_7b",
                "context_delta_distribution_output_layer_mean_7b",
            ],
            "value_column": "score",
            "metric_columns": ["score"],
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        titles = [" ".join(axis.get_title().split()) for axis in fig.axes]
        assert any("Intermediate Block Mean Evo 2 7B" in title for title in titles)
        assert any("Output Layer Mean Evo 2 7B" in title for title in titles)
    finally:
        plt.close(fig)


def test_load_plot_review_frames_marks_stale_scalar_manifest_as_unavailable(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "context_delta_distribution_intermediate_embedding_7b"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame({"score": [0.2, 0.4, 0.6]}).to_parquet(scalar_dir / "table.parquet", index=False)
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "scalar_table",
                "artifact_id": "context_delta_distribution_intermediate_embedding_7b",
                "status": "attention",
            }
        ),
        encoding="utf-8",
    )

    frames = plot_review_runtime.load_plot_review_frames(
        {
            "plot_id": "distribution_demo",
            "kind": "distribution_grid",
            "scalar_ids": ["context_delta_distribution_intermediate_embedding_7b"],
            "value_column": "score",
            "metric_columns": ["score"],
        },
        joinable_tables=[],
        output_root=tmp_path,
    )

    assert len(frames) == 1
    assert frames[0].empty
    assert "scalar_table artifact is not fresh" in str(frames[0].attrs.get("load_error"))


def test_load_plot_review_frames_rejects_scalar_manifest_without_identity_fields(tmp_path: Path) -> None:
    scalar_dir = tmp_path / "scalars" / "context_delta_distribution_intermediate_embedding_7b"
    scalar_dir.mkdir(parents=True)
    pd.DataFrame({"score": [0.2, 0.4, 0.6]}).to_parquet(scalar_dir / "table.parquet", index=False)
    (scalar_dir / "manifest.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    frames = plot_review_runtime.load_plot_review_frames(
        {
            "plot_id": "distribution_demo",
            "kind": "distribution_grid",
            "scalar_ids": ["context_delta_distribution_intermediate_embedding_7b"],
            "value_column": "score",
            "metric_columns": ["score"],
        },
        joinable_tables=[],
        output_root=tmp_path,
    )

    assert len(frames) == 1
    assert frames[0].empty
    assert "artifact_id=missing" in str(frames[0].attrs.get("load_error"))


def test_load_plot_review_frames_allows_attention_projection_manifest_and_recovers_view_id(tmp_path: Path) -> None:
    view_dir = tmp_path / "views" / "intermediate_embedding_7b_anchor_plus_full_context_concat"
    view_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["row0", "row1"],
            "design_family": ["ethanol", "cipro"],
        }
    ).to_parquet(view_dir / "rows.parquet", index=False)
    (view_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "intermediate_embedding_7b_anchor_plus_full_context_concat",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )

    projection_dir = tmp_path / "projections" / "umap_intermediate_embedding_7b_anchor_plus_full_context_concat"
    projection_dir.mkdir(parents=True)
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )
    (projection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "projection",
                "artifact_id": "umap_intermediate_embedding_7b_anchor_plus_full_context_concat",
                "status": "attention",
                "inputs": [
                    {
                        "kind": "view_matrix",
                        "id": "intermediate_embedding_7b_anchor_plus_full_context_concat",
                    }
                ],
                "warnings": ["projection fit estimated peak 8.45 GiB exceeds warn threshold"],
            }
        ),
        encoding="utf-8",
    )

    frames = plot_review_runtime.load_plot_review_frames(
        {
            "plot_id": "appendix_umap_gallery",
            "kind": "projection_grid",
            "projection_ids": ["umap_intermediate_embedding_7b_anchor_plus_full_context_concat"],
            "hue_options": [{"column": "design_family"}],
        },
        joinable_tables=[],
        output_root=tmp_path,
    )

    assert len(frames) == 1
    assert not frames[0].empty
    assert frames[0]["design_family"].tolist() == ["ethanol", "cipro"]
    assert frames[0].attrs["artifact_status"] == "attention"
    assert "attention-state artifact" in str(frames[0].attrs["artifact_warning"])


def test_render_plot_review_surface_allows_projection_grids_with_partial_panel_errors(
    monkeypatch, tmp_path: Path
) -> None:
    sentinel = object()
    monkeypatch.setattr(plot_review_runtime, "render_projection_grid", lambda *args, **kwargs: sentinel)
    frames = [pd.DataFrame({"x": [0.0], "y": [1.0]}), pd.DataFrame()]
    frames[1].attrs["load_error"] = "projection artifact is not fresh for `proj_b`: status=attention"

    rendered = render_plot_review_surface(
        {
            "plot_id": "appendix_umap_gallery",
            "kind": "projection_grid",
            "projection_ids": ["proj_a", "proj_b"],
            "panel_titles": ["Panel A", "Panel B"],
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert rendered is sentinel


def test_render_plot_review_surface_allows_scatter_grids_with_partial_panel_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]}),
        pd.DataFrame(),
    ]
    frames[1].attrs["load_error"] = "scalar_table artifact is not fresh for `stale_scalar`: status=attention"

    fig = render_plot_review_surface(
        {
            "plot_id": "design_centroid_margin_gallery",
            "kind": "xy_scatter_grid",
            "x_column": "x",
            "y_column": "y",
            "scalar_ids": ["healthy_scalar", "stale_scalar"],
            "panel_titles": ["Healthy panel", "Stale panel"],
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert fig.axes[0].get_title() == "Healthy Panel"
    assert fig.axes[1].get_title() == "Stale Panel"
    stale_panel_text = " ".join(text.get_text().replace("\n", " ") for text in fig.axes[1].texts).lower()
    assert "artifact is not fresh" in stale_panel_text
    assert "status=attention" in stale_panel_text


def test_render_plot_review_surface_allows_distribution_grids_with_partial_panel_errors(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot_review_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frames = [
        pd.DataFrame({"score": [0.2, 0.4, 0.6]}),
        pd.DataFrame(),
    ]
    frames[1].attrs["load_error"] = "scalar_table artifact is not fresh for `stale_scalar`: status=attention"

    fig = render_plot_review_surface(
        {
            "plot_id": "distribution_demo",
            "kind": "distribution_grid",
            "scalar_ids": ["healthy_scalar", "stale_scalar"],
            "value_column": "score",
            "panel_titles": ["Healthy panel", "Stale panel"],
        },
        frames=frames,
        hue_column=None,
        reference_labels=[],
        joinable_tables=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert fig.axes[0].get_title() == "Healthy Panel"
    assert fig.axes[1].get_title() == "Stale Panel"
    stale_panel_text = " ".join(text.get_text().replace("\n", " ") for text in fig.axes[1].texts).lower()
    assert "artifact is not fresh" in stale_panel_text
    assert "status=attention" in stale_panel_text


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
                "artifact_kind": "projection",
                "artifact_id": "umap_anchor",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "view_matrix",
                        "id": "intermediate_embedding_7b_anchor_60bp",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    view_dir = tmp_path / "views" / "intermediate_embedding_7b_anchor_60bp"
    view_dir.mkdir(parents=True)
    (view_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "intermediate_embedding_7b_anchor_60bp",
                "status": "ok",
            }
        ),
        encoding="utf-8",
    )
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
