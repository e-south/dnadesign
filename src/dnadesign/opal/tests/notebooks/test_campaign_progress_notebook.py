"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_campaign_progress_notebook.py

Tests OPAL campaign progress notebook structure.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

NOTEBOOK_PATH = Path("src/dnadesign/opal/notebooks/campaign_progress.py")


def test_campaign_progress_has_no_load_button() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "Load records.parquet" not in text
    assert "Click **Load**" not in text


def test_campaign_progress_uses_semantic_dashboard_api_imports() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "from dnadesign.opal.notebooks.api.generated import" in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "from dnadesign.opal import" not in text
    assert "from dnadesign.opal.notebooks.api import" not in text
    assert "dnadesign.opal.src" not in text
    assert "dnadesign.opal.src.analysis.dashboard.api" not in text
    assert "dnadesign.opal.dashboard" not in text


def test_campaign_progress_is_not_atlas() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "# Campaign Review" in text
    assert "mo.accordion(" in text
    assert "mo.ui.table" in text
    assert "Campaigns at a glance" in text
    assert "Selected campaign" in text
    assert "Validity" in text
    assert "Visual surface" in text


def test_campaign_progress_uses_tables_for_contract_and_record_status() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "build_notebook_at_a_glance_rows" in text
    assert "build_notebook_validity_rows" in text
    assert "build_notebook_evidence_rows" in text
    assert "campaign_contract_rows(" not in text
    assert "active_record_rows(" not in text
    assert 'f"- Campaign:' not in text
    assert 'f"- id:' not in text
    assert 'f"- X column:' not in text


def test_campaign_progress_uses_canonical_campaign_set_view_model() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "build_campaign_set_round_options" not in text
    assert 'label="Round"' not in text
    assert 'selected_round_selector = "all"' in text
    assert 'label="Campaign"' in text
    assert 'label="Campaign set"' in text
    assert 'label="Visual surface"' in text
    assert 'label="Review surface"' in text
    assert "view_mode_ui = mo.ui.radio(" in text
    assert 'default_view_mode = "Campaign set" if collection_set_choices else "Campaign"' in text
    assert "value=default_view_mode" in text
    assert "visual_label_memory, set_visual_label_memory = mo.state(None)" in text
    assert "on_change=set_visual_label_memory" in text
    assert "build_notebook_collection_set_choices" in text
    assert "build_notebook_collection_visual_choices" in text
    assert "build_notebook_campaign_set_visual_choices" not in text


def test_campaign_progress_keeps_lateral_tools_out_of_opal_surface() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "dnadesign.baserender" not in text
    assert "densegen__visual" not in text
    assert "cluster__ldn_v1__umap_x" not in text
    assert "cluster__ldn_v1__umap_y" not in text


def test_campaign_progress_has_no_diagnostics_sampling_controls() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "Diagnostics sample" not in text
    assert "diagnostics_sample_slider" not in text


def test_legacy_prom60_archive_removed() -> None:
    assert not Path("src/dnadesign/opal/archived/notebooks/prom60_eda_legacy.py").exists()
