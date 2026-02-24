import ast
from pathlib import Path

from dnadesign.opal.src.analysis.notebook_template import render_campaign_notebook


def test_notebook_template_data_source_options() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "predictions (selected run)" in text
    assert "labels (all rounds)" in text


def test_notebook_template_uses_medium_width() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'marimo.App(width="medium")' in text


def test_notebook_template_removes_extra_tables() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "mo.ui.dataframe(summary_df)" not in text
    assert "mo.ui.dataframe(labels_df)" not in text
    assert "mo.ui.data_explorer(filtered_df)" not in text


def test_notebook_template_has_plot_gallery() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Plot gallery" in text
    assert "outputs/plots" in text
    assert "load_plot_config" in text


def test_notebook_template_omits_altair_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "import altair as alt" not in text


def test_notebook_template_is_valid_python() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    ast.parse(text)
