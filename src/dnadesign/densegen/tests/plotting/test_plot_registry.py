"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_registry.py

Plot registry coverage for the canonical DenseGen plot surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.viz.plot_registry import PLOT_SPECS


def test_plot_registry_has_descriptions() -> None:
    for name, meta in PLOT_SPECS.items():
        assert "description" in meta, f"Missing description for plot '{name}'"
        assert str(meta["description"]).strip(), f"Empty description for plot '{name}'"


def test_plot_registry_is_canonical_set() -> None:
    assert set(PLOT_SPECS.keys()) == {
        "placement_map",
        "tfbs_usage",
        "run_health",
        "stage_a_summary",
    }
