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

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.source_evidence import (
    SFXI_ROUND0_SOURCE_EVIDENCE_ROOT,
)
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


def test_opal_campaign_root_contains_only_executable_campaigns() -> None:
    assert {path.name for path in OPAL_CAMPAIGNS_ROOT.iterdir() if path.is_dir()} == {
        "demo_gp_ei",
        "demo_gp_topn",
        "demo_rf_sfxi_topn",
        "secg_rmf_greedy",
    }


def test_sfxi_source_evidence_root_is_study_owned() -> None:
    assert SFXI_ROUND0_SOURCE_EVIDENCE_ROOT == (STUDY_ROOT / "workbench" / "source_evidence" / "opal_sfxi_round0")
