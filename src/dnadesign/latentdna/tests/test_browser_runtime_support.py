from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dnadesign.latentdna.src.notebooks import rendering as notebook_rendering
from dnadesign.latentdna.src.notebooks.browser_runtime_support import (
    available_hues_for_frames,
    candidate_hue_columns,
    category_color_map,
    classify_hue_series,
    continuous_hue_render_params,
    display_hue_label,
    display_hue_value,
    display_reference_label,
    draw_reference_labels,
    key_value_table,
    labeled_options,
    normalize_categorical_hue_value,
    normalize_hue_kind,
    notebook_theme,
    resolve_join_keys,
    table_from_records,
)
from dnadesign.latentdna.src.notebooks.rendering import (
    MAX_INLINE_NOTEBOOK_ASSET_BYTES,
    MAX_INLINE_SVG_BYTES,
    render_math_markdown,
    render_matplotlib_figure,
    render_plot_asset,
    resolve_plot_render_asset,
    select_plot_render_path,
)
from dnadesign.latentdna.src.visual_style import NONCANONICAL_SIG35_CATEGORY, wrap_plot_title


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
    assert "inline notebook rendering" in notice


def test_resolve_plot_render_asset_reports_missing_alternate(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    svg_path.write_bytes(b"x" * (MAX_INLINE_SVG_BYTES + 1))

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path is None
    assert notice is not None
    assert "no raster or pdf alternate" in notice.lower()


def test_resolve_plot_render_asset_uses_small_raster_for_large_inline_svg(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    png_path = tmp_path / "plot.png"
    svg_path.write_bytes(b"x" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))
    png_path.write_bytes(b"png")

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path == png_path
    assert notice is not None
    assert "plot.png" in notice
    assert "large for inline notebook rendering" in notice


def test_resolve_plot_render_asset_uses_pdf_alternate_when_raster_is_missing(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    pdf_path = tmp_path / "plot.pdf"
    svg_path.write_bytes(b"x" * (MAX_INLINE_SVG_BYTES + 1))
    pdf_path.write_bytes(b"%PDF-1.4")

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path == pdf_path
    assert notice is not None
    assert "plot.pdf" in notice


def test_resolve_plot_render_asset_skips_oversize_raster_alternate(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    png_path = tmp_path / "plot.png"
    pdf_path = tmp_path / "plot.pdf"
    svg_path.write_bytes(b"x" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))
    png_path.write_bytes(b"p" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))
    pdf_path.write_bytes(b"%PDF-1.4")

    render_path, notice = resolve_plot_render_asset(svg_path)

    assert render_path == pdf_path
    assert notice is not None
    assert "plot.pdf" in notice


def test_resolve_plot_render_asset_uses_pdf_for_oversize_raster(tmp_path: Path) -> None:
    png_path = tmp_path / "plot.png"
    pdf_path = tmp_path / "plot.pdf"
    png_path.write_bytes(b"p" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))
    pdf_path.write_bytes(b"%PDF-1.4")

    render_path, notice = resolve_plot_render_asset(png_path)

    assert render_path == pdf_path
    assert notice is not None
    assert "plot.png" in notice
    assert "plot.pdf" in notice
    assert "inline notebook asset limit" in notice


def test_resolve_plot_render_asset_refuses_oversize_raster_without_pdf(tmp_path: Path) -> None:
    png_path = tmp_path / "plot.png"
    png_path.write_bytes(b"p" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))

    render_path, notice = resolve_plot_render_asset(png_path)

    assert render_path is None
    assert notice is not None
    assert "no pdf alternate" in notice.lower()


def test_render_plot_asset_does_not_read_oversize_raster_without_pdf(tmp_path: Path, monkeypatch) -> None:
    png_path = tmp_path / "plot.png"
    png_path.write_bytes(b"p" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))

    def _fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("oversize raster should not be read into notebook memory")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)

    rendered = render_plot_asset(png_path, workspace_dir=tmp_path, alt_text="large raster")

    assert "plot.png" in rendered.text
    assert "data:image/png;base64" not in rendered.text


def test_category_color_map_prefers_stable_semantic_colors() -> None:
    color_map = category_color_map(["ethanol", "ciprofloxacin", "background_only", "ethanol_ciprofloxacin", "control"])

    assert color_map["background_only"] == "#56B4E9"
    assert color_map["ethanol"] == "#E69F00"
    assert color_map["ciprofloxacin"] == "#009E73"
    assert color_map["ethanol_ciprofloxacin"] == "#CC79A7"
    assert color_map["control"] == "#111111"


def test_category_color_map_orders_sig35_variant_by_reverse_alphabetical_strength() -> None:
    color_map = category_color_map(["b", "f", "d", "control"], column="sig35_variant")

    assert list(color_map) == ["f", "d", "b", "control"]
    assert color_map["f"] == "#B2182B"
    assert color_map["d"] == "#F4A582"
    assert color_map["b"] == "#2166AC"
    assert color_map["control"] == "#7F8894"


def test_category_color_map_orders_spacer_length_from_cool_to_warm() -> None:
    color_map = category_color_map(["18", "16", "17"], column="spacer_length")

    assert list(color_map) == ["16", "17", "18"]
    assert color_map["16"] == "#2C7BB6"
    assert color_map["17"] == "#FEE090"
    assert color_map["18"] == "#D73027"


def test_notebook_theme_only_styles_notebook_owned_classes() -> None:
    html = notebook_theme().text

    assert "Avenir Next" in html
    assert ".latentdna-badge" in html
    assert "font-family" in html
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


def test_resolve_join_keys_supports_construct_anchor_id_to_id_orientation() -> None:
    left = pd.DataFrame({"construct__anchor_id": ["a1", "a2"], "x": [0.0, 1.0]})
    right = pd.DataFrame({"id": ["a1", "a2"], "metric": [0.25, -0.1]})

    assert resolve_join_keys(left, right) == ("construct__anchor_id", "id")


def test_display_hue_label_and_value_compact_design_regulator_composition() -> None:
    assert display_hue_label("design_regulator_composition") == "Reg. comp."
    assert display_hue_value("design_regulator_composition", "cpxR_MANWWHTTTAM+lexA_CTGTATAWAWWHACA") == "CpxR+LexA"
    assert display_hue_value("design_regulator_composition", "sig35=b") == "Bg"
    assert display_hue_value("design_regulator_composition", "control") == "Ctrl"
    assert normalize_categorical_hue_value("design_regulator_composition", float("nan")) == "NA"


def test_display_hue_value_formats_sig35_variant_for_legends() -> None:
    assert display_hue_value("sig35_variant", "f") == "TTGACA (f)"
    assert display_hue_value("sig35_variant", "control") == "Control"


def test_sig35_hue_normalization_keeps_reference_variants_out_of_densegen_legend() -> None:
    assert normalize_categorical_hue_value("sig35_variant", "TTTACA") == NONCANONICAL_SIG35_CATEGORY
    assert normalize_categorical_hue_value("sig35_variant", "ACCGCG") == NONCANONICAL_SIG35_CATEGORY
    assert normalize_categorical_hue_value("sig35_variant", "f") == "f"


def test_reference_display_label_strips_core60_context_suffixes() -> None:
    assert display_reference_label("J23118_core60") == "J23118"
    assert display_reference_label("W2_core60_context1kb_rc") == "W2"
    assert display_reference_label("spyp") == "spyP"
    assert display_reference_label("sulAp") == "sulAp"


def test_labeled_options_disambiguates_duplicate_labels_without_dropping_values() -> None:
    options = labeled_options(
        [
            ("Control", "view_a"),
            ("Control", "view_b"),
            ("Treatment", "view_c"),
        ]
    )

    assert list(options.values()) == ["view_a", "view_b", "view_c"]
    assert "Control [view_a]" in options
    assert "Control [view_b]" in options
    assert options["Treatment"] == "view_c"


def test_table_from_records_uses_marimo_native_table_widget() -> None:
    table = table_from_records([{"Artifact": "table_a", "Columns": ["x", "y"]}], columns=["Artifact", "Columns"])

    assert isinstance(table, mo.ui.table)


def test_render_matplotlib_figure_prefers_inline_svg_in_app_run_mode(monkeypatch) -> None:
    monkeypatch.setattr(notebook_rendering.mo, "app_meta", lambda: SimpleNamespace(mode="run"))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    rendered = render_matplotlib_figure(fig, alt="run mode figure")

    assert isinstance(rendered, mo.Html)
    assert "run mode figure" in rendered.text
    assert "data:image/svg+xml;base64," in rendered.text
    assert "overflow-x: auto" in rendered.text
    assert "max-width: 100%" in rendered.text


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
    assert "overflow-x: auto" in rendered.text
    assert "max-width: 100%" in rendered.text


def test_render_plot_asset_uses_plot_alt_text_for_raster_assets(tmp_path: Path) -> None:
    png_path = tmp_path / "plot.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    rendered = render_plot_asset(png_path, workspace_dir=tmp_path, alt_text="Context robustness summary")

    assert isinstance(rendered, mo.Html)
    assert "Context robustness summary" in rendered.text
    assert "data:image/png;base64," in rendered.text
    assert "overflow-x: auto" in rendered.text
    assert "max-width: 100%" in rendered.text


def test_render_plot_asset_prefers_png_for_large_svg_with_raster_fallback(tmp_path: Path) -> None:
    svg_path = tmp_path / "plot.svg"
    png_path = tmp_path / "plot.png"
    svg_path.write_bytes(b"x" * (MAX_INLINE_NOTEBOOK_ASSET_BYTES + 1))
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    rendered = render_plot_asset(svg_path, workspace_dir=tmp_path, alt_text="UMAP gallery")

    assert isinstance(rendered, mo.Html)
    assert "data:image/png;base64," in rendered.text
    assert "Displaying `plot.png` because `plot.svg` is large for inline notebook rendering" in rendered.text


def test_render_math_markdown_emits_equation_images_for_display_math() -> None:
    rendered = render_math_markdown(
        """
        Effective rank uses
        $$
        r_{\\mathrm{eff}} = \\exp\\left(-\\sum_i p_i \\log p_i\\right).
        $$
        """
    )

    assert isinstance(rendered, mo.Html)
    assert "data:image/" in rendered.text
    assert "Math expression" in rendered.text
    assert "Effective rank uses" in rendered.text


def test_render_math_markdown_normalizes_common_latex_inequalities() -> None:
    rendered = render_math_markdown(
        """
        Cumulative variance uses
        $$
        \\sum_{k \\le i} p_k.
        $$
        """
    )

    assert isinstance(rendered, mo.Html)
    assert "Math expression" in rendered.text
    assert "fell back to plain text" not in rendered.text


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


def test_draw_reference_labels_uses_reference_set_display_labels() -> None:
    frame = pd.DataFrame(
        {
            "usr_label__primary": ["J23105", "background_only"],
            "x": [0.4, -0.2],
            "y": [0.25, -0.1],
        }
    )
    fig, ax = plt.subplots()

    draw_reference_labels(
        ax,
        frame,
        reference_labels=["J23105"],
        reference_display_labels={"J23105": "Anderson J23105"},
    )

    assert len(ax.collections) == 1
    assert any(text.get_text() == "Anderson J23105" for text in ax.texts)


def test_draw_reference_labels_separates_close_reference_annotations() -> None:
    frame = pd.DataFrame(
        {
            "usr_label__primary": ["spyp", "sulAp", "J23105"],
            "x": [0.02, 0.03, 0.04],
            "y": [0.02, 0.025, 0.03],
        }
    )
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.1, 0.2)

    draw_reference_labels(ax, frame, reference_labels=["spyp", "sulAp", "J23105"])
    fig.canvas.draw()
    text_positions = [tuple(round(value, 3) for value in text.get_position()) for text in ax.texts]

    assert len(text_positions) == 3
    assert len(set(text_positions)) == 3


def test_draw_reference_labels_uses_translucent_label_boxes() -> None:
    frame = pd.DataFrame(
        {
            "usr_label__primary": ["spyp"],
            "x": [0.02],
            "y": [0.02],
        }
    )
    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    draw_reference_labels(ax, frame, reference_labels=["spyp"])

    assert len(ax.texts) == 1
    bbox_patch = ax.texts[0].get_bbox_patch()
    assert bbox_patch is not None
    assert bbox_patch.get_alpha() == 0.82


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
    table = key_value_table([("Deliverables", 7), ("Families", ["intermediate_embedding", "output_layer_mean"])])

    assert isinstance(table, mo.ui.table)
    frame = table.data.drop(columns=["_marimo_row_id"])
    assert frame.to_dict(orient="records") == [
        {"Field": "Deliverables", "Value": 7},
        {"Field": "Families", "Value": ["intermediate_embedding", "output_layer_mean"]},
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
    assert classify_hue_series(series, configured_kind="ordinal") == "ordinal"


def test_normalize_hue_kind_accepts_binary_as_categorical_legend_kind() -> None:
    assert normalize_hue_kind("binary") == "binary"
    assert normalize_hue_kind("ordinal") == "ordinal"


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


def test_available_hues_for_frames_accepts_array_backed_categorical_hues() -> None:
    frames = [
        pd.DataFrame({"regulondb__sigma_factor_set": [np.array(["sigma70"], dtype=object)]}),
        pd.DataFrame({"regulondb__sigma_factor_set": [np.array(["sigma38", "sigma70"], dtype=object)]}),
    ]

    assert available_hues_for_frames(
        frames,
        preferred_hues=["regulondb__sigma_factor_set"],
        hue_kinds={"regulondb__sigma_factor_set": "categorical"},
    ) == ["regulondb__sigma_factor_set"]


def test_normalize_categorical_hue_value_formats_array_backed_sets() -> None:
    assert (
        normalize_categorical_hue_value(
            "regulondb__sigma_factor_set",
            np.array(["sigma38", "sigma70"], dtype=object),
        )
        == "Sigma38 + Sigma70"
    )


def test_available_hues_for_frames_ignores_empty_panels_when_resolving_hues() -> None:
    frames = [
        pd.DataFrame({"design_family": ["ethanol"], "context_shift_l2": [1.0]}),
        pd.DataFrame(),
    ]

    assert available_hues_for_frames(
        frames,
        preferred_hues=["design_family", "context_shift_l2"],
        hue_kinds={"design_family": "categorical", "context_shift_l2": "continuous"},
    ) == ["design_family"]


def test_available_hues_for_frames_excludes_null_only_visible_panel_support() -> None:
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


def test_continuous_hue_render_params_uses_diverging_norm_for_margin_columns() -> None:
    params = continuous_hue_render_params(
        "synthetic_margin_ethanol_vs_background",
        pd.Series([-0.4, -0.1, 0.0, 0.2, 0.35]),
    )

    assert params["cmap"] == "coolwarm"
    assert type(params["norm"]).__name__ == "TwoSlopeNorm"
    assert params["norm"].vcenter == 0.0
    assert params["vmin"] is None
    assert params["vmax"] is None


def test_continuous_hue_render_params_uses_sequential_scale_for_log_likelihood() -> None:
    params = continuous_hue_render_params(
        "log_likelihood_per_token_7b",
        pd.Series([-3.2, -2.9, -2.1, -1.8]),
    )

    assert params["cmap"] == "viridis"
    assert params["norm"] is None
    assert params["vmin"] < params["vmax"]
