"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/manual_mask_authority/test_materialization.py

Manual motif mask-authority tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority import (
    materialize_manual_mask_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_manual_mask_authority_materializer_writes_protected_motifs_and_rt_review_labels(
    tmp_path: Path,
) -> None:
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_manual_mask_authority(repo_root=repo_root(), output_root=tmp_path)

    authority = _load_yaml(result.manual_mask_authority_path)
    assert authority["schema_id"] == "eco1_rt_repack.manual_mask_authority"
    assert authority["artifact_id"] == "eco1_rt_conservative_v1.manual_mask_authority"
    assert authority["status"] == "materialized"
    assert authority["coordinate_space"] == "canonical_position"
    assert authority["mask_policy_id"] == "eco1_rt_manual_motif_wang_direct_contact_v1"
    assert authority["summary"]["protected_feature_count"] == 3
    assert authority["summary"]["rt_interval_feature_count"] == 7
    assert authority["summary"]["manual_mask_position_count"] == 12
    assert authority["summary"]["candidate_prior_position_count"] == 8
    assert authority["summary"]["deferred_authority_count"] == 0

    source_basis_ids = {source["id"] for source in authority["source_basis"]}
    assert "wang_et_al_2022_ec86_cryoem_structure_priors" in source_basis_ids

    feature_positions = {feature["feature_id"]: feature["canonical_positions"] for feature in authority["features"]}
    assert feature_positions["retron_x_naxxh"] == [105, 106, 107, 108, 109]
    assert feature_positions["catalytic_yadd"] == [195, 196, 197, 198]
    assert feature_positions["retron_y_vtg"] == [243, 244, 245]
    assert feature_positions["rt1_interval"] == list(range(33, 65))
    assert feature_positions["rt2_interval"] == list(range(65, 100))
    assert feature_positions["rt3_interval"] == list(range(111, 152))
    assert feature_positions["rt4_interval"] == list(range(159, 191))
    assert feature_positions["rt5_interval"] == list(range(192, 212))
    assert feature_positions["rt6_interval"] == list(range(212, 231))
    assert feature_positions["rt7_interval"] == list(range(231, 246))
    assert authority["features"][0]["structure_residue_ids"] == [105, 106, 107, 108, 109]
    assert all(feature["source_locator"] for feature in authority["features"])

    position_reasons = {row["canonical_position"]: row["manual_mask_reason"] for row in authority["residues"]}
    assert position_reasons[195] == "catalytic_yadd"
    assert position_reasons[105] == "retron_x_naxxh"
    assert position_reasons[243] == "retron_y_vtg"
    assert 33 not in position_reasons
    assert 230 not in position_reasons

    candidate_positions = {row["canonical_position"] for row in authority["candidate_prior_residues"]}
    assert candidate_positions == {49, 51, 55, 56, 73, 231, 257, 264}
    active_manual_positions = {row["canonical_position"] for row in authority["residues"]}
    assert candidate_positions & active_manual_positions == set()
    assert all(
        row["policy"] == "candidate_prior_not_mask_authoritative" for row in authority["candidate_prior_residues"]
    )
    rt_interval_features = [
        feature for feature in authority["features"] if feature["authority_type"] == "rt_core_interval"
    ]
    assert len(rt_interval_features) == 7
    assert {feature["policy"] for feature in rt_interval_features} == {"review_label"}

    assert authority["deferred_authority"] == []


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
