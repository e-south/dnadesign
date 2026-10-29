"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_spop_unavailable.py

Negative coverage for the retired historical SPOP surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.opal.src.analysis.dashboard.datasets import list_campaign_paths
from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.registries.objectives import list_objectives


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_spop_v1_is_unavailable_from_objective_registry_and_checked_in_campaign_configs() -> None:
    assert "spop_v1" not in list_objectives()

    configured_objectives = {
        view.objective.name
        for config_path in list_campaign_paths(_repo_root())
        for view in load_config(config_path).selection_views
    }
    assert "spop_v1" not in configured_objectives
