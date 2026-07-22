"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_campaign_navigation.py

Tests for study-owned OPAL campaign navigation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.campaign_navigation import (
    discover_current_campaign_navigation,
    load_current_campaign_navigation,
)


def test_current_campaign_navigation_resolves_study_record() -> None:
    repo_root = Path(__file__).resolve().parents[8]

    navigation = load_current_campaign_navigation(repo_root)

    assert navigation.campaign_slug == "secg_msrb_greedy"
    assert navigation.selection_view_ids == ("ethanol", "ciprofloxacin", "and")
    assert navigation.objective_names == ("multistate_response_behavior_v1",)
    assert navigation.config_path.as_posix().endswith("secg_msrb_greedy/configs/campaign.yaml")
    assert navigation.notebook_path.as_posix().endswith("notebooks/opal_secg_msrb_greedy_analysis.py")
    assert navigation.notebook_materialized is (repo_root / navigation.notebook_path).is_file()
    assert navigation.run_command == f"uv run opal notebook run -c {navigation.config_path}"


def test_campaign_navigation_rejects_repo_escape(tmp_path: Path) -> None:
    record = tmp_path / "campaign.yaml"
    record.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "path_base": "repo",
                "campaign_id": "stress_ethanol_cipro_growth",
                "steps": [{"inputs": {"opal_config": "repo:../outside/campaign.yaml"}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes repository root"):
        load_current_campaign_navigation(tmp_path, record_path=record)


def test_campaign_navigation_is_optional_without_a_source_checkout(tmp_path: Path) -> None:
    assert discover_current_campaign_navigation(tmp_path) is None
