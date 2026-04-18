from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.latentdna.src.notebooks import browser_runtime_support as runtime_support
from dnadesign.latentdna.src.notebooks.browser_runtime_support import (
    MAX_INLINE_SVG_BYTES,
    available_hues_for_frames,
    candidate_hue_columns,
    category_color_map,
    classify_hue_series,
    draw_reference_labels,
    key_value_table,
    normalize_hue_kind,
    notebook_theme,
    render_matplotlib_figure,
    render_plot_asset,
    resolve_plot_render_asset,
    select_plot_render_path,
    table_from_records,
)
from dnadesign.latentdna.src.visual_style import wrap_plot_title


def test_select_plot_render_path_prefers_svg_assets(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    png_path = tmp_path / "plot.png"
    pdf_path = tmp_path / "plot.pdf"
    svg_path.write_text("<svg />", encoding="utf-8")
    png_path.write_bytes(b"png")
    pdf_path.write_bytes(b"%PDF-1.4")

    assert select_plot_render_path([png_path, pdf_path, svg_path]) == svg_path


def test_resolve_plot_render_asset_uses_raster_fallback_for_large_svg(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    png_path = tmp_path / "plot.png"
    svg_path.write_bytes(b"x" * (MAX_INLINE_SVG_BYTES + 1))
    png_path.write_bytes(b"png")

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path == png_path
    assert notice is not None
    assert "plot.svg" in notice
    assert "plot.png" in notice
    assert "inline notebook limit" in notice


def test_resolve_plot_render_asset_reports_missing_raster_fallback(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    svg_path.write_bytes(b"x" * (MAX_INLINE_SVG_BYTES + 1))

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path is None
    assert notice is not None
    assert "no raster fallback" in notice.lower()


def test_category_color_map_prefers_stable_semantic_colors() -> None:
    color_map = category_color_map(["ethanol", "ciprofloxacin", "background_only", "ethanol_ciprofloxacin", "control"])

    assert color_map["background_only"] == "#56B4E9"
    assert color_map["ethanol"] == "#E69F00"
    assert color_map["ciprofloxacin"] == "#009E73"
    assert color_map["ethanol_ciprofloxacin"] == "#CC79A7"
    assert color_map["control"] == "#111111"


def test_notebook_theme_only_styles_notebook_owned_classes() -> None:
    html = notebook_theme().text

    assert "Avenir Next" in html
    assert ".latentdna-plot-asset" in html
    assert ".latentdna-badge" in html
    assert "font-family" in html
    assert "latentdna-media-max-width" in html
    assert ".latentdna-browser" not in html
    assert ".mo-callout" not in html
    assert ".mo-ui-table" not in html
    assert " button" not in html
    assert " input" not in html
    assert " select" not in html
    assert " label" not in html
    assert "<style>body" not in html
    assert "<style>html" not in html
    assert "#App.bg-background" not in html
    assert "body.dark.dark-theme" not in html
    assert "color-scheme: light" not in html
    assert "MutationObserver" not in html


def test_candidate_hue_columns_restricts_to_preferred_surface() -> None:
    frame = pd.DataFrame(
        {
            "design_family": ["control", "ethanol"],
            "sig35_variant": ["control", "b"],
            "usr_label__primary": ["j23105", "spyp"],
            "construct__anchor_id": ["a1", "a2"],
        }
    )

    assert candidate_hue_columns(
        frame,
        ["design_family", "sig35_variant", "wildtype_margin_ethanol_vs_control"],
    ) == ["design_family", "sig35_variant"]


def test_table_from_records_uses_marimo_native_table_widget() -> None:
    table = table_from_records([{"Artifact": "table_a", "Columns": ["x", "y"]}], columns=["Artifact", "Columns"])

    assert isinstance(table, mo.ui.table)


def test_render_matplotlib_figure_prefers_inline_svg_in_app_run_mode(monkeypatch) -> None:
    monkeypatch.setattr(runtime_support.mo, "app_meta", lambda: SimpleNamespace(mode="run"))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    rendered = render_matplotlib_figure(fig, alt="run mode figure")

    assert isinstance(rendered, mo.Html)
    assert "run mode figure" in rendered.text
    assert "data:image/svg+xml;base64," in rendered.text


def test_render_plot_asset_wraps_svg_as_data_uri_image(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    svg_path.write_text(
        "<?xml version='1.0' encoding='utf-8'?><ns0:svg xmlns:ns0='http://www.w3.org/2000/svg'></ns0:svg>",
        encoding="utf-8",
    )

    rendered = render_plot_asset(svg_path, workspace_dir=tmp_path)

    assert isinstance(rendered, mo.Html)
    assert "data:image/svg+xml;base64," in rendered.text
    assert "<img" in rendered.text
    assert "latentdna-plot-asset" in rendered.text


def test_draw_reference_labels_uses_requested_coordinate_columns() -> None:
    frame = pd.DataFrame(
        {
            "usr_label__primary": ["spyp", "background_only"],
            "wildtype_margin_ethanol_vs_control": [0.4, -0.2],
            "wildtype_margin_cipro_vs_control": [0.25, -0.1],
        }
    )
    fig, ax = plt.subplots()

    draw_reference_labels(
        ax,
        frame,
        reference_labels=["spyp"],
        x_column="wildtype_margin_ethanol_vs_control",
        y_column="wildtype_margin_cipro_vs_control",
    )

    assert len(ax.collections) == 1
    assert any(text.get_text() == "spyP" for text in ax.texts)


def test_draw_reference_labels_skips_frames_without_requested_coordinates() -> None:
    frame = pd.DataFrame({"usr_label__primary": ["spyp"], "other_metric": [0.4]})
    fig, ax = plt.subplots()

    draw_reference_labels(
        ax,
        frame,
        reference_labels=["spyp"],
        x_column="wildtype_margin_ethanol_vs_control",
        y_column="wildtype_margin_cipro_vs_control",
    )

    assert len(ax.collections) == 0
    assert len(ax.texts) == 0


def test_key_value_table_formats_summary_values() -> None:
    table = key_value_table([("Deliverables", 7), ("Families", ["intermediate_embedding", "pooled_logits"])])

    assert isinstance(table, mo.ui.table)
    frame = table.data.drop(columns=["_marimo_row_id"])
    assert frame.to_dict(orient="records") == [
        {"Field": "Deliverables", "Value": 7},
        {"Field": "Families", "Value": ["intermediate_embedding", "pooled_logits"]},
    ]


def test_table_from_records_preserves_dict_values_in_native_widget() -> None:
    table = table_from_records(
        [{"Metric": "summary", "Payload": {"rows": 128, "status": "ok"}}],
        columns=["Metric", "Payload"],
    )

    assert isinstance(table, mo.ui.table)
    frame = table.data.drop(columns=["_marimo_row_id"])
    assert frame.to_dict(orient="records") == [{"Metric": "summary", "Payload": {"rows": 128, "status": "ok"}}]


def test_table_from_records_preserves_long_lists_in_native_widget() -> None:
    table = table_from_records(
        [{"Paths": ["a", "b", "c", "d", "e"]}],
        columns=["Paths"],
    )

    assert isinstance(table, mo.ui.table)
    frame = table.data.drop(columns=["_marimo_row_id"])
    assert frame.to_dict(orient="records") == [{"Paths": ["a", "b", "c", "d", "e"]}]


def test_wrap_plot_title_breaks_long_titles_without_splitting_words() -> None:
    wrapped = wrap_plot_title("intermediate embedding 20b full context 1 kb", width=18)

    assert "\n" in wrapped
    assert "Intermediate" in wrapped
    assert "Construct Context" in wrapped
    assert "..." not in wrapped


def test_classify_hue_series_treats_boolean_values_as_categorical() -> None:
    series = pd.Series([True, False, True])

    assert classify_hue_series(series) == "categorical"


def test_classify_hue_series_respects_configured_kind_over_dtype() -> None:
    series = pd.Series([0.0, 1.0, 0.5])

    assert classify_hue_series(series, configured_kind="categorical") == "categorical"


def test_normalize_hue_kind_accepts_binary_as_categorical_legend_kind() -> None:
    assert normalize_hue_kind("binary") == "binary"


def test_available_hues_for_frames_intersects_support_across_visible_panels() -> None:
    frames = [
        pd.DataFrame({"design_family": ["ethanol", "cipro"], "context_shift_l2": [1.0, 2.0]}),
        pd.DataFrame({"design_family": ["control", "ethanol"], "context_shift_l2": [3.0, 4.0]}),
    ]

    assert available_hues_for_frames(
        frames,
        preferred_hues=["design_family", "context_shift_l2"],
        hue_kinds={"design_family": "categorical", "context_shift_l2": "continuous"},
    ) == ["design_family", "context_shift_l2"]


def test_available_hues_for_frames_excludes_null_only_and_missing_panel_support() -> None:
    frames = [
        pd.DataFrame({"design_family": ["ethanol"], "context_shift_l2": [1.0]}),
        pd.DataFrame({"design_family": [None], "context_shift_l2": [None]}),
    ]

    assert (
        available_hues_for_frames(
            frames,
            preferred_hues=["design_family", "context_shift_l2"],
            hue_kinds={"design_family": "categorical", "context_shift_l2": "continuous"},
        )
        == []
    )


def test_available_hues_for_frames_excludes_degenerate_continuous_hues() -> None:
    frames = [
        pd.DataFrame({"context_shift_l2": [1.0, 1.0]}),
        pd.DataFrame({"context_shift_l2": [1.0, 1.0]}),
    ]

    assert (
        available_hues_for_frames(
            frames,
            preferred_hues=["context_shift_l2"],
            hue_kinds={"context_shift_l2": "continuous"},
        )
        == []
    )
