"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_rt_lnrna_metadata.py

Tests for generic metadata derivations used by RT-lnRNA LatentDNA rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.contracts.workspace import (
    MetadataDelimitedNumericMeanDerivationConfig,
    MetadataNumericQuantileBinDerivationConfig,
    MetadataSingleCategoricalTokenDerivationConfig,
    MetadataTokenPresenceDerivationConfig,
)
from dnadesign.latentdna.src.metadata.derivations import derive_metadata_value

RT_WORKSPACE_DIR = Path("src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage")
CRAWFORD_ABUNDANCE_QUANTILE_EDGES_V1: tuple[float, float, float, float] = (
    8.90033258134951,
    95.97675900112976,
    212.83370392895753,
    380.6567556708186,
)


def test_rt_lnrna_metadata_derives_source_scoped_abundance_values() -> None:
    row = {
        "construct_subject__khan_abundance_observation_ids": "khan_obs_1",
        "construct_subject__khan_abundance_raw_values": "0.66128418",
        "construct_subject__khan_abundance_normalized_values": "0.66128418",
        "construct_subject__khan_abundance_ordinal_bins": "high",
        "construct_subject__crawford_abundance_observation_ids": "crawford_obs_1;crawford_obs_2",
        "construct_subject__crawford_abundance_raw_values": "524.213430898279;570.9851335062278",
        "construct_subject__crawford_reference_record_ids": "crawford_ref_1",
    }

    khan_status = MetadataTokenPresenceDerivationConfig(
        kind="token_presence",
        source="construct_subject__khan_abundance_observation_ids",
        present_value="abundance_affiliated",
        absent_value="not_abundance_affiliated",
    )
    khan_raw = MetadataDelimitedNumericMeanDerivationConfig(
        kind="delimited_numeric_mean",
        source="construct_subject__khan_abundance_raw_values",
    )
    khan_ordinal = MetadataSingleCategoricalTokenDerivationConfig(
        kind="single_categorical_token",
        source="construct_subject__khan_abundance_ordinal_bins",
    )
    crawford_status = MetadataTokenPresenceDerivationConfig(
        kind="token_presence",
        source="construct_subject__crawford_abundance_observation_ids",
        present_value="abundance_affiliated",
        absent_value="not_abundance_affiliated",
    )
    crawford_raw = MetadataDelimitedNumericMeanDerivationConfig(
        kind="delimited_numeric_mean",
        source="construct_subject__crawford_abundance_raw_values",
    )
    crawford_ordinal = MetadataNumericQuantileBinDerivationConfig(
        kind="numeric_quantile_bin",
        source="construct_subject__crawford_abundance_raw_values",
        edges=list(CRAWFORD_ABUNDANCE_QUANTILE_EDGES_V1),
        labels=["very_low", "low", "medium", "high", "very_high"],
    )
    crawford_reference_status = MetadataTokenPresenceDerivationConfig(
        kind="token_presence",
        source="construct_subject__crawford_reference_record_ids",
        present_value="design_reference_affiliated",
        absent_value="not_design_reference_affiliated",
    )

    assert derive_metadata_value(row, khan_status) == "abundance_affiliated"
    assert derive_metadata_value(row, khan_raw) == pytest.approx(0.66128418)
    assert derive_metadata_value(row, khan_ordinal) == "high"
    assert derive_metadata_value(row, crawford_status) == "abundance_affiliated"
    assert derive_metadata_value(row, crawford_raw) == pytest.approx(547.5992822022534)
    assert derive_metadata_value(row, crawford_ordinal) == "very_high"
    assert derive_metadata_value(row, crawford_reference_status) == "design_reference_affiliated"


def test_rt_lnrna_metadata_fails_on_conflicting_ordinal_values() -> None:
    row = {"construct_subject__khan_abundance_ordinal_bins": "low;high"}
    derivation = MetadataSingleCategoricalTokenDerivationConfig(
        kind="single_categorical_token",
        source="construct_subject__khan_abundance_ordinal_bins",
    )

    with pytest.raises(ContractViolationError, match="conflicting categorical values"):
        derive_metadata_value(row, derivation)


def test_crawford_abundance_edges_match_workspace_order_contract() -> None:
    payload = yaml.safe_load((RT_WORKSPACE_DIR / "study_inputs/crawford_abundance_order.yaml").read_text())

    assert tuple(payload["binning"]["edges"]) == CRAWFORD_ABUNDANCE_QUANTILE_EDGES_V1


def test_retired_reader_spop_snapshots_are_absent_and_not_active_sources() -> None:
    payload = yaml.safe_load((RT_WORKSPACE_DIR / "config.yaml").read_text(encoding="utf-8"))

    assert tuple((RT_WORKSPACE_DIR / "study_inputs").glob("reader_spop_*.parquet")) == ()
    assert "rt_lnrna_reader_spop_candidate_summary" not in payload["sources"]
    assert not {
        "reader_spop_overlay_status",
        "reader_spop_metric_id",
        "reader_spop_experiment_ids",
        "reader_spop_normalized_value",
        "reader_spop_score_median",
        "reader_spop_observation_count",
        "reader_spop_qc_flags",
    }.intersection(payload["metadata"]["include"])
    assert payload["metadata"]["derivations"]["label_readiness_status"] == {
        "kind": "constant",
        "value": "reporter_response_profiles_absent_pending_meta_study",
        "value_type": "string",
    }
