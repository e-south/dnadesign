"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_public_api_boundary.py

Regression tests for public API boundary OPAL API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
import sys

import dnadesign.opal as opal


def test_package_root_defers_heavy_public_imports() -> None:
    code = """
import json
import sys
import dnadesign.opal as opal

from dnadesign.opal import load_config

print(json.dumps({
    "has_load_config": callable(load_config),
    "load_config_exported": "load_config" in opal.__all__,
    "gaussian_process_loaded": "dnadesign.opal.src.models.gaussian_process" in sys.modules,
    "sklearn_loaded": any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["has_load_config"] is True
    assert payload["load_config_exported"] is True
    assert payload["gaussian_process_loaded"] is False
    assert payload["sklearn_loaded"] is False


def test_package_root_does_not_export_dashboard_helpers() -> None:
    prohibited = {
        "campaign_label_from_path",
        "diagnostics_to_lines",
        "find_repo_root",
        "list_campaign_paths",
        "load_campaign_selection",
        "load_parquet_cached",
    }

    assert prohibited.isdisjoint(set(opal.__all__))
    for name in prohibited:
        assert not hasattr(opal, name)


def test_package_root_does_not_export_generated_notebook_components() -> None:
    prohibited = {
        "build_notebook_artifact_garden_rows",
        "build_notebook_at_a_glance_rows",
        "build_notebook_baserender_contract",
        "build_notebook_plot_card_rows",
        "build_notebook_plot_method_rows",
        "build_notebook_visual_surface_model",
        "render_notebook_baserender_record",
        "resolve_notebook_round_default",
    }

    assert prohibited.isdisjoint(set(opal.__all__))
    for name in prohibited:
        assert not hasattr(opal, name)
