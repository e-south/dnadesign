"""Projection rendering regression tests for notebook geometry browser surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from dnadesign.latentdna.src.notebooks import browser_runtime_projection as projection_runtime

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


def _panel_offsets(fig) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for collection in fig.axes[0].collections:
        collection_offsets = np.asarray(collection.get_offsets())
        if collection_offsets.size == 0:
            continue
        offsets.extend((float(x), float(y)) for x, y in collection_offsets.tolist())
    return sorted(offsets)


def _legend_clears_panel_axes(fig) -> bool:
    fig.canvas.draw()
    if not fig.legends:
        return True
    renderer = fig.canvas.get_renderer()
    legend_bbox = fig.legends[0].get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    panel_y0 = min(axis.get_position().y0 for axis in fig.axes if axis.axison)
    return bool(legend_bbox.y1 < panel_y0)


def _legend_panel_gap(fig) -> float:
    fig.canvas.draw()
    if not fig.legends:
        return 0.0
    renderer = fig.canvas.get_renderer()
    legend_bbox = fig.legends[0].get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    panel_y0 = min(axis.get_position().y0 for axis in fig.axes if axis.axison)
    return float(panel_y0 - legend_bbox.y1)


def _write_manifest(
    artifact_dir: Path,
    *,
    artifact_kind: str,
    artifact_id: str,
    status: str = "ok",
    inputs: list[dict[str, object]] | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "status": status,
    }
    if inputs is not None:
        payload["inputs"] = inputs
    (artifact_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_view_rows(output_root: Path, view_id: str, rows: pd.DataFrame) -> None:
    view_dir = output_root / "views" / view_id
    _write_manifest(view_dir, artifact_kind="view", artifact_id=view_id)
    rows.to_parquet(view_dir / "rows.parquet")


def _write_scalar_table(output_root: Path, scalar_id: str, table: pd.DataFrame) -> None:
    scalar_dir = output_root / "scalars" / scalar_id
    _write_manifest(scalar_dir, artifact_kind="scalar_table", artifact_id=scalar_id)
    table.to_parquet(scalar_dir / "table.parquet")


def test_render_projection_grid_keeps_point_coordinates_fixed_across_hues(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0, 0.0],
            "design_family": ["ethanol", "ethanol", "cipro", "cipro"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig_without_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column=None,
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )
    fig_with_hue = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _panel_offsets(fig_without_hue) == _panel_offsets(fig_with_hue)
    finally:
        plt.close(fig_without_hue)
        plt.close(fig_with_hue)


def test_render_projection_grid_default_alt_names_layout_hue_and_reference_overlay(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_render(fig, *, alt=None):
        plt.close(fig)
        captured["alt"] = str(alt or "")
        return captured["alt"]

    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", fake_render)
    monkeypatch.setattr(
        projection_runtime,
        "resolve_reference_annotation",
        lambda *args, **kwargs: {
            "labels": ["spyP"],
            "match_column": "usr_label__primary",
            "display_labels": {},
            "label_limit": 5,
            "warnings": [],
        },
    )
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "sig35_variant": ["b", "c"],
        }
    )

    alt = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[frame],
        hue_column="sig35_variant",
        hue_kinds={"sig35_variant": "categorical"},
        axis_styles=SIGMA35_AXIS_STYLES,
        joinable_tables=[],
        reference_labels=["spyP"],
        reference_set_id="reference_spyp_sulap",
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert alt == captured["alt"]
    assert "Single persisted projection" in alt
    assert "colored by Sigma-35 variant" in alt
    assert "reference labels overlay enabled" in alt
    assert "1 matched row" in alt
    assert "grid" not in alt.lower()


def test_render_projection_grid_prepends_dynamic_contract_to_semantic_alt_text(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}

    def fake_render(fig, *, alt=None):
        plt.close(fig)
        captured["alt"] = str(alt or "")
        return captured["alt"]

    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", fake_render)
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0], "gc_fraction": [0.42, 0.51]})

    alt = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[frame],
        hue_column="gc_fraction",
        hue_kinds={"gc_fraction": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        reference_set_id="",
        output_root=tmp_path,
        workspace_dir=tmp_path,
        alt_text="Configured UMAP gallery semantics.",
    )

    assert alt == captured["alt"]
    assert alt.startswith("Single persisted projection colored by GC fraction; reference labels overlay off")
    assert alt.endswith("Configured UMAP gallery semantics.")


def test_render_projection_grid_surfaces_reference_overlay_warnings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: "rendered plot")
    monkeypatch.setattr(
        projection_runtime,
        "resolve_reference_annotation",
        lambda *args, **kwargs: {
            "labels": [],
            "match_column": "usr_label__primary",
            "display_labels": {},
            "label_limit": 0,
            "warnings": ["reference set `demo` has 2 unmatched reference rows"],
        },
    )

    rendered = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[pd.DataFrame({"x": [0.0], "y": [1.0], "usr_label__primary": ["row0"]})],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        reference_set_id="demo",
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "Reference overlay warning" in rendered.text
    assert "unmatched reference rows" in rendered.text


def test_render_projection_grid_surfaces_missing_reference_hue_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: "rendered plot")
    monkeypatch.setattr(
        projection_runtime,
        "resolve_reference_annotation",
        lambda *args, **kwargs: {
            "labels": ["spyP"],
            "match_column": "usr_label__primary",
            "display_labels": {},
            "label_limit": 5,
            "warnings": [],
        },
    )

    rendered = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[pd.DataFrame({"x": [0.0], "y": [1.0], "usr_label__primary": ["spyP"]})],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        reference_set_id="reference_spyp_sulap",
        reference_hue_column="promoter_standard__strength_value_numeric",
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "Reference overlay warning" in rendered.text
    assert "reference hue `promoter_standard__strength_value_numeric` has no finite values" in rendered.text


def test_render_projection_grid_warns_when_reference_hue_is_missing_in_one_panel(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: "rendered plot")
    monkeypatch.setattr(
        projection_runtime,
        "resolve_reference_annotation",
        lambda *args, **kwargs: {
            "labels": ["W1"],
            "match_column": "usr_label__primary",
            "display_labels": {},
            "label_limit": 5,
            "warnings": [],
        },
    )
    frames = [
        pd.DataFrame(
            {
                "x": [0.0],
                "y": [1.0],
                "usr_label__primary": ["W1"],
                "promoter_standard__strength_value_numeric": [1.0],
            }
        ),
        pd.DataFrame({"x": [1.0], "y": [0.0], "usr_label__primary": ["W1"]}),
    ]
    frames[0].attrs["view_id"] = "strength_panel"
    frames[1].attrs["view_id"] = "missing_hue_panel"

    rendered = projection_runtime.render_projection_grid(
        [
            {"view_id": "strength_panel", "projection_id": "proj_a", "title": "Strength panel"},
            {"view_id": "missing_hue_panel", "projection_id": "proj_b", "title": "Missing hue panel"},
        ],
        frames=frames,
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        reference_set_id="reference_w_collection",
        reference_hue_column="promoter_standard__strength_value_numeric",
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "Reference overlay warning" in rendered.text
    assert "reference hue `promoter_standard__strength_value_numeric` is missing" in rendered.text
    assert "missing_hue_panel" in rendered.text


def test_render_projection_grid_suppresses_degenerate_continuous_colorbar(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "wildtype_margin_ethanol_vs_control": [0.25, 0.25, 0.25],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="wildtype_margin_ethanol_vs_control",
        hue_kinds={"wildtype_margin_ethanol_vs_control": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_render_projection_grid_renders_full_population_for_large_browser_frames(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    row_count = 30_001
    frame = pd.DataFrame(
        {
            "x": np.arange(row_count, dtype=float),
            "y": np.arange(row_count, dtype=float),
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "demo", "projection_id": "umap_demo", "title": "Demo"}],
        frames=[frame],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        offsets = _panel_offsets(fig)
        assert len(offsets) == row_count
        assert offsets[0] == (0.0, 0.0)
        assert offsets[-1] == (float(row_count - 1), float(row_count - 1))
    finally:
        plt.close(fig)


def test_render_projection_grid_rasterizes_very_large_frames_without_sampling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    monkeypatch.setattr(projection_runtime, "should_rasterize_scatter", lambda row_count: row_count > 3)
    captured: dict[str, int] = {}

    def fake_raster(ax, *, x_values, y_values, color, **kwargs):
        del y_values, color, kwargs
        captured["row_count"] = len(x_values)
        ax.imshow(np.zeros((2, 2, 4)), extent=(-0.1, 1.1, -0.1, 1.1), origin="lower")

    monkeypatch.setattr(projection_runtime, "draw_single_color_raster_scatter", fake_raster)
    row_count = 5
    frame = pd.DataFrame(
        {
            "x": np.arange(row_count, dtype=float),
            "y": np.arange(row_count, dtype=float),
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "demo", "projection_id": "umap_demo", "title": "Demo"}],
        frames=[frame],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert captured["row_count"] == row_count
        assert len(fig.axes[0].images) == 1
    finally:
        plt.close(fig)


def test_render_projection_grid_keeps_missing_continuous_hue_rows_visible(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0, 0.0],
            "context_shift_l2": [0.1, np.nan, 0.8, np.nan],
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "demo", "projection_id": "umap_demo", "title": "Demo"}],
        frames=[frame],
        hue_column="context_shift_l2",
        hue_kinds={"context_shift_l2": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _panel_offsets(fig) == [(0.0, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]
    finally:
        plt.close(fig)


def test_render_projection_grid_rasterizes_missing_continuous_hue_background(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    monkeypatch.setattr(projection_runtime, "should_rasterize_scatter", lambda row_count: row_count > 3)
    captured: dict[str, int] = {}

    def fake_background(ax, *, x_values, y_values, color, **kwargs):
        del y_values, color, kwargs
        captured["background_count"] = len(x_values)
        ax.imshow(np.zeros((2, 2, 4)), extent=(-0.1, 4.1, -0.1, 4.1), origin="lower")

    def fake_continuous(ax, *, x_values, y_values, hue_values, **kwargs):
        del y_values, kwargs
        captured["colored_count"] = len(x_values)
        captured["colored_hue_count"] = len(hue_values)
        ax.imshow(np.zeros((2, 2, 4)), extent=(-0.1, 4.1, -0.1, 4.1), origin="lower")
        return ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis")

    monkeypatch.setattr(projection_runtime, "draw_single_color_raster_scatter", fake_background)
    monkeypatch.setattr(projection_runtime, "draw_continuous_raster_scatter", fake_continuous)
    frame = pd.DataFrame(
        {
            "x": np.arange(5, dtype=float),
            "y": np.arange(5, dtype=float),
            "context_shift_l2": [0.1, np.nan, 0.8, np.nan, 0.4],
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "demo", "projection_id": "umap_demo", "title": "Demo"}],
        frames=[frame],
        hue_column="context_shift_l2",
        hue_kinds={"context_shift_l2": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert captured == {"background_count": 2, "colored_count": 3, "colored_hue_count": 3}
    finally:
        plt.close(fig)


def test_render_projection_grid_rasterized_continuous_hue_keeps_full_extent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    monkeypatch.setattr(projection_runtime, "should_rasterize_scatter", lambda row_count: row_count > 3)
    frame = pd.DataFrame(
        {
            "x": np.arange(5, dtype=float),
            "y": np.arange(5, dtype=float),
            "context_shift_l2": [np.nan, 0.1, 0.8, 0.4, np.nan],
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "demo", "projection_id": "umap_demo", "title": "Demo"}],
        frames=[frame],
        hue_column="context_shift_l2",
        hue_kinds={"context_shift_l2": "continuous"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        x_min, x_max = fig.axes[0].get_xlim()
        y_min, y_max = fig.axes[0].get_ylim()
        assert x_min < 0.0
        assert x_max > 4.0
        assert y_min < 0.0
        assert y_max > 4.0
    finally:
        plt.close(fig)


def test_render_projection_grid_compacts_regulator_composition_legend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
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
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="design_regulator_composition",
        hue_kinds={"design_regulator_composition": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        axis_styles=REGULATOR_AXIS_STYLES,
    )

    try:
        assert len(fig.legends) == 1
        legend = fig.legends[0]
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["BaeR+LexA", "Bg", "CpxR", "Ctrl", "LexA"]
        assert legend._ncols == 3
    finally:
        plt.close(fig)


def test_render_projection_grid_orders_sig35_legend_by_strength(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "sig35_variant": ["b", "f", "d", "control"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="sig35_variant",
        hue_kinds={"sig35_variant": "categorical"},
        joinable_tables=[],
        reference_labels=[],
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
        assert colors == ["#B2182B", "#E69B7B", "#2166AC", "#7F8894"]
    finally:
        plt.close(fig)


def test_render_projection_grid_keeps_reference_sig35_rows_neutral_and_out_of_legend(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "sig35_variant": ["b", "f", "f", "ACCGCG"],
            "source_family": ["densegen_generated", "densegen_generated", "reference_source", "sfxi_archive"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="sig35_variant",
        hue_kinds={"sig35_variant": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        axis_styles=SIGMA35_AXIS_STYLES,
    )

    try:
        assert _panel_offsets(fig) == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert labels == ["TTGACA (f)", "CTGACA (b)"]
    finally:
        plt.close(fig)


def test_render_projection_grid_uses_vectorized_reference_matching(monkeypatch, tmp_path: Path) -> None:
    from dnadesign.latentdna.src.notebooks import browser_runtime_support as support

    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    calls = 0
    original = support.normalize_label

    def counting_normalize(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(support, "normalize_label", counting_normalize)
    row_count = 2048
    frame = pd.DataFrame(
        {
            "x": np.arange(row_count, dtype=float),
            "y": np.arange(row_count, dtype=float),
            "usr_label__primary": ["dense"] * (row_count - 2) + ["J23105_core60", "W1_core60"],
        }
    )

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[frame],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=["J23105_core60", "W1_core60"],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        highlight_offsets = np.asarray(fig.axes[0].collections[-1].get_offsets(), dtype=float)
        assert sorted(map(tuple, highlight_offsets.tolist())) == [
            (float(row_count - 2), float(row_count - 2)),
            (float(row_count - 1), float(row_count - 1)),
        ]
        assert calls == 2
    finally:
        plt.close(fig)


def test_projection_sig35_hue_keeps_context_derived_densegen_rows_categorical() -> None:
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

    series = projection_runtime._categorical_hue_series(frame, "sig35_variant", axis_styles=SIGMA35_AXIS_STYLES)

    assert series.tolist() == [
        "f",
        "b",
        SIGMA35_NONCANONICAL_BUCKET,
        SIGMA35_NONCANONICAL_BUCKET,
    ]


def test_projection_required_value_check_accepts_array_backed_categorical_columns() -> None:
    frame = pd.DataFrame({"regulondb__sigma_factor_set": [np.array(["sigma38", "sigma70"], dtype=object)]})

    assert projection_runtime._column_has_required_values(frame, "regulondb__sigma_factor_set")


def test_render_projection_grid_draws_reference_stars_as_hue_independent_overlay(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "sig35_variant": ["f", "b", "f", "b"],
            "source_class": ["densegen", "densegen", "reference_control", "reference_control"],
            "usr_label__primary": ["dense_a", "dense_b", "J23105_core60", "W1_core60"],
        }
    )
    panel_specs = [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame],
        hue_column="sig35_variant",
        hue_kinds={"sig35_variant": "categorical"},
        joinable_tables=[],
        reference_labels=["J23105_core60", "W1_core60"],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        axis_styles=SIGMA35_AXIS_STYLES,
    )

    try:
        highlight_offsets = np.asarray(fig.axes[0].collections[-1].get_offsets(), dtype=float)
        assert sorted(map(tuple, highlight_offsets.tolist())) == [(2.0, 2.0), (3.0, 3.0)]
        assert [text.get_text() for text in fig.legends[0].get_texts()] == ["TTGACA (f)", "CTGACA (b)"]
    finally:
        plt.close(fig)


def test_render_projection_grid_prefers_single_row_when_requested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(4)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame, frame, frame, frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
        prefer_single_row=True,
    )

    try:
        y_positions = {round(axis.get_position().y0, 3) for axis in fig.axes[:4]}
        assert len(y_positions) == 1
    finally:
        plt.close(fig)


def test_render_projection_grid_handles_seven_panel_gallery_without_axis_zip_failures(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(7)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame for _ in range(7)],
        plot_id="appendix_umap_gallery",
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:7]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 2
        assert fig.axes[7].axison is False
    finally:
        plt.close(fig)


def test_render_projection_grid_uses_two_rows_for_intermediate_and_output_gallery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(12)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame for _ in range(12)],
        plot_id="appendix_umap_gallery",
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        panel_axes = fig.axes[:12]
        assert len({round(axis.get_position().y0, 3) for axis in panel_axes}) == 2
        assert len({round(axis.get_position().x0, 3) for axis in panel_axes[:6]}) == 6
        assert 0.0 <= _legend_panel_gap(fig) <= 0.12
    finally:
        plt.close(fig)


def test_render_projection_grid_keeps_legend_clear_of_multi_panel_gallery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    frame = pd.DataFrame(
        {
            "x": np.arange(30, dtype=float),
            "y": np.arange(30, dtype=float),
            "design_family": ["background_only", "ethanol", "cipro", "dual", "control"] * 6,
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(7)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame for _ in range(7)],
        plot_id="appendix_umap_gallery",
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _legend_clears_panel_axes(fig)
    finally:
        plt.close(fig)


def test_render_projection_grid_keeps_long_two_panel_legend_clear(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)
    sigma_labels = [
        "Sigma19",
        "Sigma24",
        "Sigma24 + Sigma38 + Sigma70",
        "Sigma24 + Sigma70",
        "Sigma28",
        "Sigma28 + Sigma32",
        "Sigma28 + Sigma70",
        "Sigma32",
        "Sigma32 + Sigma38",
        "Sigma32 + Sigma54",
        "Sigma32 + Sigma70",
        "Sigma38",
        "Sigma38 + Sigma54",
        "Sigma38 + Sigma70",
        "Sigma54",
        "Sigma54 + Sigma70",
        "Sigma70",
    ]
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, len(sigma_labels)),
            "y": np.linspace(1.0, 0.0, len(sigma_labels)),
            "regulondb__sigma_factor_set": sigma_labels,
        }
    )
    panel_specs = [
        {"view_id": "native", "projection_id": "proj_native", "title": "Native source-record intermediate"},
        {"view_id": "core60", "projection_id": "proj_core60", "title": "Core60 TSS-upstream intermediate"},
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame, frame],
        hue_column="regulondb__sigma_factor_set",
        hue_kinds={"regulondb__sigma_factor_set": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        assert _legend_clears_panel_axes(fig)
    finally:
        plt.close(fig)


def test_render_projection_grid_surfaces_preloaded_frame_errors(tmp_path: Path) -> None:
    errored_frame = pd.DataFrame()
    errored_frame.attrs["load_error"] = "projection artifact is not fresh for `proj_a`: stale"

    rendered = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor view"}],
        frames=[errored_frame],
        hue_column=None,
        hue_kinds={},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    assert isinstance(rendered, mo.Html)
    assert "projection artifact is not fresh" in rendered.text


def test_render_projection_grid_preserves_healthy_panels_when_some_preloaded_frames_error(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    healthy_frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    errored_frame = pd.DataFrame()
    errored_frame.attrs["load_error"] = "projection artifact is not fresh for `proj_a`: status=attention"

    fig = projection_runtime.render_projection_grid(
        [
            {"view_id": "view_a", "projection_id": "proj_a", "title": "Anchor concat"},
            {"view_id": "view_b", "projection_id": "proj_b", "title": "Anchor base"},
        ],
        frames=[errored_frame, healthy_frame],
        plot_id="appendix_umap_gallery",
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        first_axis, second_axis = fig.axes[:2]
        first_axis_text = " ".join(text.get_text() for text in first_axis.texts)
        assert "Projection unavailable" in first_axis_text
        assert "status=attention" in first_axis_text.lower()
        assert len(second_axis.collections) == 2
    finally:
        plt.close(fig)


def test_load_projection_frame_allows_attention_manifest_and_preserves_warning_attrs(tmp_path: Path) -> None:
    output_root = tmp_path
    _write_view_rows(
        output_root,
        "view_a",
        pd.DataFrame(
            {
                "id": ["row0", "row1"],
                "design_family": ["ethanol", "cipro"],
            }
        ),
    )
    projection_dir = output_root / "projections" / "proj_attention"
    _write_manifest(
        projection_dir,
        artifact_kind="projection",
        artifact_id="proj_attention",
        status="attention",
        inputs=[{"kind": "view_matrix", "id": "view_a"}],
    )
    (projection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "projection",
                "artifact_id": "proj_attention",
                "status": "attention",
                "inputs": [{"kind": "view_matrix", "id": "view_a"}],
                "warnings": ["projection fit estimated peak 8.45 GiB exceeds warn threshold"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )

    frame = projection_runtime.load_projection_frame(
        "view_a",
        "proj_attention",
        [],
        output_root=output_root,
        required_columns=["design_family"],
    )

    assert not frame.empty
    assert frame["design_family"].tolist() == ["ethanol", "cipro"]
    assert frame.attrs["artifact_status"] == "attention"
    assert "attention-state artifact" in str(frame.attrs["artifact_warning"])


def test_load_projection_frame_reads_only_join_and_required_view_row_columns(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path
    _write_view_rows(
        output_root,
        "view_a",
        pd.DataFrame(
            {
                "id": ["row0", "row1"],
                "design_family": ["ethanol", "cipro"],
                "unused_large_metadata": ["x" * 1024, "y" * 1024],
            }
        ),
    )
    projection_dir = output_root / "projections" / "proj_a"
    _write_manifest(
        projection_dir,
        artifact_kind="projection",
        artifact_id="proj_a",
        status="ok",
        inputs=[{"kind": "view_matrix", "id": "view_a"}],
    )
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )
    observed_columns_by_name: dict[str, list[str] | None] = {}
    original_read_parquet = projection_runtime.load_table.__globals__["pd"].read_parquet

    def recording_read_parquet(path, *args, **kwargs):
        observed_columns_by_name[Path(path).name] = kwargs.get("columns")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(projection_runtime.load_table.__globals__["pd"], "read_parquet", recording_read_parquet)

    frame = projection_runtime.load_projection_frame(
        "view_a",
        "proj_a",
        [],
        output_root=output_root,
        required_columns=["design_family"],
    )

    assert not frame.empty
    assert frame["design_family"].tolist() == ["ethanol", "cipro"]
    assert "unused_large_metadata" not in frame.columns
    assert observed_columns_by_name["rows.parquet"] == ["design_family", "id"]


def test_load_projection_frame_does_not_badge_warning_only_ok_manifest(tmp_path: Path) -> None:
    output_root = tmp_path
    _write_view_rows(
        output_root,
        "view_a",
        pd.DataFrame(
            {
                "id": ["row0", "row1"],
                "design_family": ["ethanol", "cipro"],
            }
        ),
    )
    projection_dir = output_root / "projections" / "proj_ok_warning"
    _write_manifest(
        projection_dir,
        artifact_kind="projection",
        artifact_id="proj_ok_warning",
        status="ok",
        inputs=[{"kind": "view_matrix", "id": "view_a"}],
    )
    (projection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "projection",
                "artifact_id": "proj_ok_warning",
                "status": "ok",
                "inputs": [{"kind": "view_matrix", "id": "view_a"}],
                "warnings": ["projection fit estimated peak 8.45 GiB exceeds warn threshold"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"id": ["row0", "row1"], "x": [0.0, 1.0], "y": [1.0, 0.0]}).to_parquet(
        projection_dir / "coords.parquet",
        index=False,
    )

    frame = projection_runtime.load_projection_frame(
        "view_a",
        "proj_ok_warning",
        [],
        output_root=output_root,
        required_columns=["design_family"],
    )

    assert not frame.empty
    assert frame["design_family"].tolist() == ["ethanol", "cipro"]
    assert "artifact_status" not in frame.attrs
    assert "artifact_warning" not in frame.attrs


def test_render_projection_grid_marks_attention_artifacts_without_hiding_panel(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_family": ["ethanol", "cipro"],
        }
    )
    frame.attrs["artifact_warning"] = "projection `proj_attention` is rendered from an attention-state artifact."

    fig = projection_runtime.render_projection_grid(
        [{"view_id": "view_a", "projection_id": "proj_attention", "title": "Attention panel"}],
        frames=[frame],
        hue_column="design_family",
        hue_kinds={"design_family": "categorical"},
        joinable_tables=[],
        reference_labels=[],
        output_root=tmp_path,
        workspace_dir=tmp_path,
    )

    try:
        axis_text = " ".join(text.get_text() for text in fig.axes[0].texts)
        assert "Attention" in axis_text
        assert len(fig.axes[0].collections) == 2
    finally:
        plt.close(fig)


def test_render_projection_grid_places_continuous_colorbar_below_panels_and_uses_margin_palette(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(projection_runtime, "render_matplotlib_figure", lambda fig, alt=None: fig)

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [2.0, 1.0, 0.0],
            "synthetic_margin_ethanol_vs_background": [-0.35, 0.0, 0.28],
        }
    )
    panel_specs = [
        {"view_id": f"view_{index}", "projection_id": f"proj_{index}", "title": f"Panel {index}"} for index in range(2)
    ]

    fig = projection_runtime.render_projection_grid(
        panel_specs,
        frames=[frame, frame],
        hue_column="synthetic_margin_ethanol_vs_background",
        hue_kinds={"synthetic_margin_ethanol_vs_background": "continuous"},
        joinable_tables=[],
        reference_labels=[],
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


def test_enrich_projection_frame_joins_requested_columns_by_same_named_key(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_demo",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
                "synthetic_margin_cipro_vs_background": [0.05, 0.15],
                "unused_metric": [5.0, 6.0],
            }
        ),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_demo",
                "relative_path": "scalars/design_centroid_margins_demo/table.parquet",
                "columns": [
                    "id",
                    "synthetic_margin_ethanol_vs_background",
                    "synthetic_margin_cipro_vs_background",
                    "unused_metric",
                ],
            }
        ],
        output_root=tmp_path,
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.25, -0.10]
    assert "synthetic_margin_cipro_vs_background" not in enriched.columns
    assert "unused_metric" not in enriched.columns


def test_enrich_projection_frame_joins_context_metrics_from_anchor_id(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "context_delta_distribution_demo",
        pd.DataFrame(
            {
                "id": ["context_row_1", "context_row_2"],
                "construct__anchor_id": ["anchor_1", "anchor_2"],
                "context_self_cosine": [0.91, 0.84],
                "context_shift_l2": [0.02, 0.09],
            }
        ),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "context_delta_distribution_demo",
                "relative_path": "scalars/context_delta_distribution_demo/table.parquet",
                "columns": ["id", "construct__anchor_id", "context_self_cosine", "context_shift_l2"],
            }
        ],
        output_root=tmp_path,
        required_columns=["context_self_cosine"],
    )

    assert enriched["context_self_cosine"].tolist() == [0.91, 0.84]
    assert "construct__anchor_id" not in enriched.columns
    assert "context_shift_l2" not in enriched.columns


def test_enrich_projection_frame_respects_explicit_table_view_targets(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "context_delta_distribution_intermediate_embedding_7b",
        pd.DataFrame(
            {
                "id": ["context_row_1", "context_row_2"],
                "construct__anchor_id": ["anchor_1", "anchor_2"],
                "context_self_cosine": [0.91, 0.84],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "intermediate_embedding_7b_anchor_plus_full_context_concat",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "context_delta_distribution_intermediate_embedding_7b",
                "relative_path": "scalars/context_delta_distribution_intermediate_embedding_7b/table.parquet",
                "columns": ["id", "construct__anchor_id", "context_self_cosine"],
                "view_ids": ["intermediate_embedding_7b_anchor_60bp"],
            }
        ],
        output_root=tmp_path,
        view_id="intermediate_embedding_7b_anchor_plus_full_context_concat",
        required_columns=["context_self_cosine"],
        strict_required_columns=False,
    )

    assert "context_self_cosine" not in enriched.columns


def test_enrich_projection_frame_limits_candidate_metric_tables_to_active_view(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [np.nan, np.nan],
            }
        ),
    )
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.35, -0.22],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "output_layer_mean_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
            {
                "artifact_id": "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_output_layer_mean_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
        ],
        output_root=tmp_path,
        view_id="output_layer_mean_20b_anchor_60bp",
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.35, -0.22]


def test_enrich_projection_frame_prefers_authoritative_view_row_metadata(tmp_path: Path) -> None:
    _write_view_rows(
        tmp_path,
        "intermediate_embedding_7b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "design_regulator_composition": ["baeR", "background"],
                "spacer_length": [18, 17],
            }
        ),
    )

    frame = pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "design_regulator_composition": ["baeR_TTTCTSCVHNA", "sig35=c"],
            "spacer_length": [None, None],
        }
    )
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [],
        output_root=tmp_path,
        view_id="intermediate_embedding_7b_anchor_60bp",
        required_columns=["design_regulator_composition", "spacer_length"],
    )

    assert enriched["design_regulator_composition"].tolist() == ["baeR", "background"]
    assert enriched["spacer_length"].tolist() == [18, 17]


def test_enrich_projection_frame_refreshes_required_column_from_joinable_source(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame(
        {
            "id": ["anchor_1", "anchor_2"],
            "x": [0.0, 1.0],
            "y": [1.0, 0.0],
            "synthetic_margin_ethanol_vs_background": [999.0, -999.0],
        }
    )
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            }
        ],
        output_root=tmp_path,
        view_id="intermediate_embedding_20b_anchor_60bp",
        required_columns=["synthetic_margin_ethanol_vs_background"],
    )

    assert enriched["synthetic_margin_ethanol_vs_background"].tolist() == [0.25, -0.10]


def test_enrich_projection_frame_rejects_required_column_when_joined_values_are_empty(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [None, None],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="required metadata columns are empty"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                    "relative_path": (
                        "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet"
                    ),
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                }
            ],
            output_root=tmp_path,
            view_id="intermediate_embedding_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )


def test_enrich_projection_frame_rejects_ambiguous_required_column_sources(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.35, -0.22],
            }
        ),
    )
    _write_scalar_table(
        tmp_path,
        "context_delta_distribution_output_layer_mean_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.10, 0.20],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "output_layer_mean_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="ambiguous metadata source"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
                    "relative_path": "scalars/design_centroid_margins_output_layer_mean_20b_anchor_60bp/table.parquet",
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                },
                {
                    "artifact_id": "context_delta_distribution_output_layer_mean_20b_anchor_60bp",
                    "relative_path": (
                        "scalars/context_delta_distribution_output_layer_mean_20b_anchor_60bp/table.parquet"
                    ),
                    "columns": ["id", "synthetic_margin_ethanol_vs_background"],
                },
            ],
            output_root=tmp_path,
            view_id="output_layer_mean_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )


def test_enrich_projection_frame_skips_ambiguous_optional_columns_in_non_strict_mode(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.35, -0.22],
            }
        ),
    )
    _write_scalar_table(
        tmp_path,
        "sigma35_stress_margins_output_layer_mean_20b_anchor_60bp",
        pd.DataFrame(
            {
                "id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.10, 0.20],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "output_layer_mean_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})
    enriched = projection_runtime.enrich_projection_frame(
        frame,
        [
            {
                "artifact_id": "design_centroid_margins_output_layer_mean_20b_anchor_60bp",
                "relative_path": "scalars/design_centroid_margins_output_layer_mean_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
            {
                "artifact_id": "sigma35_stress_margins_output_layer_mean_20b_anchor_60bp",
                "relative_path": "scalars/sigma35_stress_margins_output_layer_mean_20b_anchor_60bp/table.parquet",
                "columns": ["id", "synthetic_margin_ethanol_vs_background"],
            },
        ],
        output_root=tmp_path,
        view_id="output_layer_mean_20b_anchor_60bp",
        required_columns=["synthetic_margin_ethanol_vs_background"],
        strict_required_columns=False,
    )

    assert "synthetic_margin_ethanol_vs_background" not in enriched.columns
    assert enriched[["x", "y"]].to_dict(orient="records") == [{"x": 0.0, "y": 1.0}, {"x": 1.0, "y": 0.0}]


def test_enrich_projection_frame_rejects_required_column_when_join_keys_do_not_resolve(tmp_path: Path) -> None:
    _write_scalar_table(
        tmp_path,
        "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame(
            {
                "record_id": ["anchor_1", "anchor_2"],
                "synthetic_margin_ethanol_vs_background": [0.25, -0.10],
            }
        ),
    )
    _write_view_rows(
        tmp_path,
        "intermediate_embedding_20b_anchor_60bp",
        pd.DataFrame({"id": ["anchor_1", "anchor_2"]}),
    )

    frame = pd.DataFrame({"id": ["anchor_1", "anchor_2"], "x": [0.0, 1.0], "y": [1.0, 0.0]})

    with pytest.raises(ValueError, match="cannot join"):
        projection_runtime.enrich_projection_frame(
            frame,
            [
                {
                    "artifact_id": "design_centroid_margins_intermediate_embedding_20b_anchor_60bp",
                    "relative_path": (
                        "scalars/design_centroid_margins_intermediate_embedding_20b_anchor_60bp/table.parquet"
                    ),
                    "columns": ["record_id", "synthetic_margin_ethanol_vs_background"],
                }
            ],
            output_root=tmp_path,
            view_id="intermediate_embedding_20b_anchor_60bp",
            required_columns=["synthetic_margin_ethanol_vs_background"],
        )
