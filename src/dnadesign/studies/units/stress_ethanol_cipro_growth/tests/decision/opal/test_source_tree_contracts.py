"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/test_source_tree_contracts.py

Source-tree boundaries for the stress-study OPAL integration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    SCHEMA_ID,
    STUDY_ID,
)

STUDY_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth")
OPAL_CAMPAIGNS_ROOT = Path("src/dnadesign/opal/campaigns")


def test_candidate_identity_contract_is_owned_at_study_scope() -> None:
    assert SCHEMA_ID == "dnadesign.study.promoter_candidate_bindings.v1"
    assert STUDY_ID == "stress_ethanol_cipro_growth"
    assert (STUDY_ROOT / "promoter_candidate_bindings" / "README.md").is_file()


def test_sfxi_source_campaigns_have_no_executable_configs() -> None:
    source_slugs = {
        "secg_and_rf_sfxi_topn",
        "secg_cipro_rf_sfxi_topn",
        "secg_ethanol_rf_sfxi_topn",
    }

    for slug in source_slugs:
        assert not (OPAL_CAMPAIGNS_ROOT / slug / "configs" / "campaign.yaml").exists()
