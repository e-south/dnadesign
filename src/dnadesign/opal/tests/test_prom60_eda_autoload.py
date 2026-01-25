# ABOUTME: Ensures prom60 EDA notebook auto-loads data without a manual button.
# ABOUTME: Guards against reintroducing load-button gating that hides data by default.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_prom60_eda_autoload.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path


def test_prom60_eda_has_no_load_button() -> None:
    nb_path = Path("src/dnadesign/opal/notebooks/prom60_eda.py")
    text = nb_path.read_text()
    assert "Load records.parquet" not in text
    assert "Click **Load**" not in text


def test_prom60_eda_explorer_defaults_use_opal_view_score() -> None:
    nb_path = Path("src/dnadesign/opal/notebooks/prom60_eda.py")
    text = nb_path.read_text()
    assert '_preferred_x = "opal__view__score"' in text
    assert '_preferred_y = "opal__view__effect_scaled"' in text


def test_prom60_eda_includes_sfxi_brush_pool_option() -> None:
    nb_path = Path("src/dnadesign/opal/notebooks/prom60_eda.py")
    text = nb_path.read_text()
    assert "SFXI brush selection" in text
    assert "sfxi_brush" in text
