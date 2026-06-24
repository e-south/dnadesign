"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/mask_set/test_materialization.py

Mask-set materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_mask_set_materializer_writes_plurality25_direct_contact_mask(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)

    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)

    mask_set = _load_yaml(result.mask_set_path)
    assert mask_set["schema_id"] == "thread.mask_set"
    assert mask_set["status"] == "materialized"
    assert mask_set["mask_policy_id"] == "eco1_rt_clade9_plurality25_direct_contact5a_v1"
    assert "selected_tier_id" not in mask_set
    assert "relaxed_tier_projections" not in mask_set
    assert mask_set["sampling_status"] == "pending_sampling_plan"
    assert mask_set["sampling_allowed"] is True

    summary = mask_set["summary"]
    assert summary["total_positions"] == 320
    assert summary["mapped_position_count"] == 309
    assert summary["missing_backbone_position_count"] == 11
    assert summary["non_fixed_missing_backbone_position_count"] == 11
    assert summary["non_fixed_mapped_position_count"] == 1
    assert summary["protected_position_count"] == 308
    assert summary["source_protected_counts"]["motif_anchor"] == 12
    assert summary["source_protected_counts"]["wang_ec86_direct_contact_prior"] == 8
    assert summary["source_protected_counts"]["evolutionarily_conserved_clade9_25pct_plurality"] > 0
    assert summary["source_protected_counts"]["direct_retained_dna_rna_contact_5a"] > 0
    assert summary["non_fixed_mapped_positions"] == [4]

    residues = mask_set["residues"]
    assert len(residues) == 320
    assert all("final_fixed" not in row for row in residues)
    assert all("proteinmpnn_designable" not in row for row in residues)
    residues_by_position = {row["canonical_position"]: row for row in residues}
    assert residues_by_position[1]["protected"] is False
    assert residues_by_position[1]["non_fixed_missing_backbone"] is True
    assert residues_by_position[4]["non_fixed"] is True
    assert residues_by_position[4]["protected"] is False
    assert residues_by_position[195]["motif_protected"] is True
    assert residues_by_position[195]["manual_mask_reason"] == "catalytic_yadd"
    assert "motif_anchor" in residues_by_position[195]["protection_reasons"]
    assert residues_by_position[105]["manual_mask_reason"] == "retron_x_naxxh"
    assert residues_by_position[243]["manual_mask_reason"] == "retron_y_vtg"
    assert residues_by_position[33]["rt_interval_review_label"] == "RT1"
    assert "rt1_core_interval" not in residues_by_position[33]["protection_reasons"]
    assert residues_by_position[230]["rt_interval_review_label"] == "RT6"
    assert "rt6_core_interval" not in residues_by_position[230]["protection_reasons"]
    assert mask_set["manual_mask_authority_status"] == "materialized_eco1_rt_manual_motif_wang_direct_contact_v1"


def test_phase1_with_mask_set_passes_thread_contract(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is True
    assert "eco1_rt.mask.mask_set_not_materialized" not in {issue.check_id for issue in report.issues}


def test_phase1_rejects_mask_set_protected_mismatch(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    mask_set = _load_yaml(result.mask_set_path)
    non_fixed_row = next(row for row in mask_set["residues"] if row["non_fixed"] is True)
    non_fixed_row["protected"] = True
    result.mask_set_path.write_text(yaml.safe_dump(mask_set, sort_keys=False), encoding="utf-8")

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.mask.mask_set_value_mismatch" in {issue.check_id for issue in report.issues}


def test_phase1_rejects_protected_residue_without_source_reason(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    mask_set = _load_yaml(result.mask_set_path)
    protected_row = next(row for row in mask_set["residues"] if row["protected"] is True)
    protected_row["protection_reasons"] = []
    result.mask_set_path.write_text(yaml.safe_dump(mask_set, sort_keys=False), encoding="utf-8")

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.mask.mask_set_missing_protection_reason" in {issue.check_id for issue in report.issues}


def test_phase1_rejects_motif_protection_mismatch(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    mask_set = _load_yaml(result.mask_set_path)
    yadd_row = next(row for row in mask_set["residues"] if row["canonical_position"] == 195)
    yadd_row["motif_protected"] = False
    yadd_row["manual_mask_reason"] = ""
    yadd_row["protection_reasons"] = [source for source in yadd_row["protection_reasons"] if source != "motif_anchor"]
    result.mask_set_path.write_text(yaml.safe_dump(mask_set, sort_keys=False), encoding="utf-8")

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.mask.mask_set_value_mismatch" in {issue.check_id for issue in report.issues}


def test_phase1_rejects_missing_wang_candidate_priors(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    manual_authority = _load_yaml(result.manual_mask_authority_path)
    manual_authority["candidate_prior_residues"] = []
    result.manual_mask_authority_path.write_text(yaml.safe_dump(manual_authority, sort_keys=False), encoding="utf-8")

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.mask.manual_mask_authority_missing_candidate_priors" in {issue.check_id for issue in report.issues}


def test_phase1_rejects_missing_audited_rt_interval_authority(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    result = materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    manual_authority = _load_yaml(result.manual_mask_authority_path)
    manual_authority["features"] = [
        feature for feature in manual_authority["features"] if feature["feature_id"] != "rt5_interval"
    ]
    result.manual_mask_authority_path.write_text(yaml.safe_dump(manual_authority, sort_keys=False), encoding="utf-8")

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    assert "eco1_rt.mask.manual_mask_authority_missing_rt_intervals" in {issue.check_id for issue in report.issues}


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
