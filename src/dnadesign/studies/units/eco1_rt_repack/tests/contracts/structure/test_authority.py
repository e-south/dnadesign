"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/structure/test_authority.py

Structure-authority contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure import (
    validate_authority_consistency_payload,
    validate_residue_numbering_policy_payload,
    validate_structure_authority_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import load_yaml


def test_structure_authority_rejects_missing_chain_and_invalid_context_policy() -> None:
    payload = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/structure-sources.yaml")
    selected = payload["selected_source"]
    selected["rt_chain_id"] = "pending"
    selected["retained_context_policy"] = "rt_with_everything_guess"

    report = validate_structure_authority_payload(payload, phase="phase1_thread_contract")

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.structure.pending_selection_field" in check_ids
    assert "eco1_rt.structure.invalid_retained_context_policy" in check_ids


def test_residue_numbering_policy_rejects_source_hash_mismatch() -> None:
    structure_sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/structure-sources.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    numbering["reference_sequence_hash"] = "sha256:not-the-selected-sequence"

    report = validate_residue_numbering_policy_payload(
        numbering,
        structure_sources=structure_sources,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert {issue.check_id for issue in report.issues} == {"eco1_rt.structure.numbering_sequence_hash_mismatch"}


def test_authority_consistency_rejects_profile_source_mismatch() -> None:
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    structure_sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/structure-sources.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    profile["reference"]["structure_authority"] = "wrong_structure"

    report = validate_authority_consistency_payload(
        profile=profile,
        structure_sources=structure_sources,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert {issue.check_id for issue in report.issues} == {"eco1_rt.profile.structure_authority_mismatch"}
