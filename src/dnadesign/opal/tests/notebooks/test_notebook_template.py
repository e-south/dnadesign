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


def test_notebook_template_is_campaign_specific_accordion_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "# OPAL Campaign Notebook" in text
    assert "Campaign-specific artifact viewer" in text
    assert "mo.accordion(" in text
    for section in [
        "Campaign contract",
        "Round and run",
        "Ledger readiness",
        "Records and active record",
        "Labels and predictions",
        "Plot deliverables",
        "Optional context boundaries",
    ]:
        assert section in text


def test_notebook_template_uses_shared_campaign_progress_helpers() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "from dnadesign.opal.src.analysis.campaign_progress import" in text
    assert "assess_records_contract_for_values" in text
    assert "build_ledger_status_table" in text
    assert "build_records_preview" in text
    assert "cli_handoff_lines" in text
    assert "read_optional_table" in text
    assert "records_status_lines" in text


def test_notebook_template_degrades_without_runs() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "No runs available yet" in text
    assert "mo.stop(len(rounds) == 0" not in text
    assert 'default_source = "records"' in text


def test_notebook_template_keeps_lateral_tools_out() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "dnadesign.baserender" not in text
    assert "densegen__visual" not in text
    assert "cluster__ldn_v1__umap_x" not in text
    assert "cluster__ldn_v1__umap_y" not in text


def test_notebook_template_omits_altair_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "import altair as alt" not in text


def test_notebook_template_is_valid_python() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    ast.parse(text)
