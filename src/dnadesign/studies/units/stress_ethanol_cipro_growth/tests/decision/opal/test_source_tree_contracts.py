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
STUDY_OPAL_ROOT = STUDY_ROOT / "decision/opal"
OPAL_CAMPAIGNS_ROOT = Path("src/dnadesign/opal/campaigns")


def test_candidate_identity_contract_is_owned_at_study_scope() -> None:
    assert SCHEMA_ID == "dnadesign.study.promoter_candidate_bindings.v1"
    assert STUDY_ID == "stress_ethanol_cipro_growth"
    assert (STUDY_ROOT / "promoter_candidate_bindings" / "README.md").is_file()
    assert not (STUDY_OPAL_ROOT / "reader_candidate_bindings").exists()


def test_measured_sfxi_wrapper_is_absent() -> None:
    wrapper = STUDY_OPAL_ROOT / "measured_reader_vec8"

    assert not wrapper.exists() or not any(
        path.is_file() and (path.suffix == ".py" or path.name == "README.md") for path in wrapper.rglob("*")
    )


def test_metric_specific_stress_promoter_strategy_config_is_absent() -> None:
    config = STUDY_OPAL_ROOT / "synthesis_handoff/configs/sfxi_promoter_insert_v1.yaml"

    assert not config.exists()


def test_sfxi_source_campaigns_have_no_executable_configs() -> None:
    source_slugs = {
        "secg_and_rf_sfxi_topn",
        "secg_cipro_rf_sfxi_topn",
        "secg_ethanol_rf_sfxi_topn",
    }

    for slug in source_slugs:
        assert not (OPAL_CAMPAIGNS_ROOT / slug / "configs" / "campaign.yaml").exists()
