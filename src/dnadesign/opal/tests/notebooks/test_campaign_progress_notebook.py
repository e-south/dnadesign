"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/notebooks/test_campaign_progress_notebook.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

NOTEBOOK_PATH = Path("src/dnadesign/opal/notebooks/campaign_progress.py")


def test_campaign_progress_has_no_load_button() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "Load records.parquet" not in text
    assert "Click **Load**" not in text


def test_campaign_progress_uses_public_opal_imports() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "from dnadesign.opal import" in text
    assert "dnadesign.opal.src" not in text


def test_campaign_progress_is_not_atlas() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "# OPAL Campaign Progress" in text
    assert "mo.accordion(" in text
    assert "Campaign contract" in text
    assert "Records and active record" in text
    assert "Ledger and CLI handoff" in text
    assert "X provenance and limitations" in text


def test_campaign_progress_uses_canonical_cli_handoff() -> None:
    text = NOTEBOOK_PATH.read_text()
    assert "cli_handoff_lines" in text
    assert "cli_handoff_lines(config_text)" in text


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
