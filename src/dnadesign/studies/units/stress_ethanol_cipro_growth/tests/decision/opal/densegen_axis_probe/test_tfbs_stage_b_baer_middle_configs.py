"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_baer_middle_configs.py

Regression tests for TFBS stage b baer middle configs studies units.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_stage_b_source_fixture

_profiles = probe_module("tfbs.profiles")
SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID
SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_LABELS = _profiles.slot_position_count_fixed_baer_middle_label_names()

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

_stage_b_configs = probe_module("tfbs.stage_b.configs")
TfbsStageBConfig = _stage_b_configs.TfbsStageBConfig
materialize_tfbs_stage_b_sentinel_configs = _stage_b_configs.materialize_tfbs_stage_b_sentinel_configs


def test_tfbs_stage_b_baer_middle_profile_writes_count_fixed_scope(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-baer-middle-count-fixed"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
            target_profile_id=SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
        )
    )

    result = materialize_tfbs_stage_b_sentinel_configs(
        TfbsStageBConfig(
            stage_a_run_root=stage_a_root,
            out_dir=stage_a_root / "stage_b_count_fixed_baer_middle_configs",
            label_names=SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_LABELS,
            target_profile_id=SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
            validate_configs=False,
        )
    )

    manifest = _read_json(result.config_manifest_path)
    assert result.campaign_count == 2
    assert manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID
    assert manifest["target_profile"]["profile_role"] == "boundary_stage_b_count_fixed_minimal_placement_probe"
    assert manifest["candidate_scope_mode"] == "label_specific_count_fixed"
    assert [row["label_name"] for row in manifest["candidate_scopes"]] == ["baeR_in_slot1"]
    assert manifest["candidate_scopes"][0]["target_family_count_column"] == "baeR_count"
    assert manifest["candidate_scopes"][0]["positive_label_marginal"] == {"0": 10, "1": 5}

    by_role = {row["oracle_role"]: row for row in manifest["campaigns"]}
    positive = by_role["positive"]
    control = by_role["matched_null"]
    assert positive["candidate_scope_path"] == control["candidate_scope_path"]
    assert positive["candidate_scope_hash"] == control["candidate_scope_hash"]
    assert positive["initial_label_ids_hash"] == control["initial_label_ids_hash"]
    assert positive["target_family_count_column"] == "baeR_count"
    positive_labels = pd.read_parquet(positive["label_table_path"])
    control_labels = pd.read_parquet(control["label_table_path"])
    assert len(positive_labels) == 15
    assert (positive_labels["baeR_count"] == 1).all()
    assert (control_labels["baeR_count"] == 1).all()
    assert positive_labels["baeR_in_slot1"].value_counts().sort_index().to_dict() == {0: 10, 1: 5}
    assert control_labels["baeR_in_slot1"].value_counts().sort_index().to_dict() == {0: 10, 1: 5}
    cfg = yaml.safe_load(Path(positive["config_path"]).read_text(encoding="utf-8"))
    assert cfg["campaign"]["metadata"]["target_label"] == "BaeR in middle slot"
    assert cfg["transforms_y"]["params"]["value_columns"] == ["baeR_in_slot1"]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
