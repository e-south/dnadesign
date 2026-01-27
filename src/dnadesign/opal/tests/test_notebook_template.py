# ABOUTME: Tests the OPAL marimo notebook template rendering.
# ABOUTME: Ensures generated notebooks include expected scaffolding choices.
from pathlib import Path

from dnadesign.opal.src.analysis.notebook_template import render_campaign_notebook


def test_notebook_template_defines_filtered_df() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "filtered_df = pred_df" in text
    assert "return filtered_df" in text


def test_notebook_template_score_cast_is_strict_false() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "scores = filtered_df.get_column(score_field).cast(pl.Float64, strict=False)" in text


def test_notebook_template_data_source_options() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "predictions (selected run)" in text
    assert "labels (all rounds)" in text


def test_notebook_template_uses_medium_width_and_altair() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'marimo.App(width="medium")' in text
    assert "import altair as alt" in text


def test_notebook_template_has_color_by_and_no_max_rank() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Color by" in text
    assert "Max rank" not in text


def test_notebook_template_hstack_altair_plots() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "alt.hconcat" in text


def test_notebook_template_has_plot_row_controls() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Plot rows" in text
    assert "Use all rows for plots" not in text


def test_notebook_template_sets_altair_theme() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "setup_altair_theme()" in text


def test_notebook_template_dedupes_color_columns() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "select_cols = list(dict.fromkeys(select_cols))" in text
    assert "sfxi_cols = list(dict.fromkeys(sfxi_cols))" in text


def test_notebook_template_removes_extra_tables() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "mo.ui.dataframe(summary_df)" not in text
    assert "mo.ui.dataframe(labels_df)" not in text
    assert "mo.ui.data_explorer(filtered_df)" not in text


def test_notebook_template_has_plot_gallery() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Plot gallery" in text
    assert "outputs/plots" in text
