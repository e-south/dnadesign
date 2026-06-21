"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/test_phase_contracts.py

Phase-gate tests for Eco1 RT repack contract validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root, require_ec86kit_source_artifacts


def test_phase0_checked_in_contracts_pass_as_scaffold() -> None:
    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase0_scaffold")

    assert report.passed is True
    assert report.issue_count == 0


def test_phase1_contracts_fail_on_missing_materialized_structure_artifacts(tmp_path: Path) -> None:
    report = validate_checked_in_contracts(
        repo_root=repo_root(),
        phase="phase1_thread_contract",
        output_root=tmp_path / "missing-thread-output",
    )

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.profile.pending_reference_authority" not in check_ids
    assert "eco1_rt.structure.pending_source_selection" not in check_ids
    assert "eco1_rt.structure.residue_numbering_not_started" not in check_ids
    assert "eco1_rt.structure.backbone_bundle_not_materialized" in check_ids
    assert "eco1_rt.structure.residue_map_not_materialized" in check_ids


def test_phase1_with_materialized_structure_artifacts_reaches_evidence_gate(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.structure.backbone_bundle_not_materialized" not in check_ids
    assert "eco1_rt.structure.residue_map_not_materialized" not in check_ids
    assert "eco1_rt.evidence.conservation_profile_not_materialized" in check_ids
    assert "eco1_rt.evidence.contact_profile_not_materialized" in check_ids
    assert "eco1_rt.mask.mask_set_not_materialized" in check_ids


def test_phase1_with_contact_profile_reaches_conservation_and_mask_gate(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    materialize_structure_authority(repo_root=repo_root(), output_root=tmp_path)
    materialize_contact_profile(repo_root=repo_root(), output_root=tmp_path)

    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase1_thread_contract", output_root=tmp_path)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.evidence.contact_profile_not_materialized" not in check_ids
    assert "eco1_rt.evidence.contact_profile_source_hash_mismatch" not in check_ids
    assert "eco1_rt.evidence.conservation_profile_not_materialized" in check_ids
    assert "eco1_rt.mask.mask_set_not_materialized" in check_ids
