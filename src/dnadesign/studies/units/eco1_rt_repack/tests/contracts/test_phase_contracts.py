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

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import validate_sampling_artifacts
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.suite import validate_checked_in_contracts
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact import materialize_contact_profile
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.structure import (
    materialize_structure_authority,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root, require_ec86kit_source_artifacts

_CANDIDATE_HANDOFF_SCHEMA_PATH = (
    "docs/studies/eco1_rt_repack/operations/contract/schemas/thread-candidate-handoff.schema.yaml"
)
_ARTIFACT_CHAIN_SCHEMA_PATH = (
    "docs/studies/eco1_rt_repack/operations/contract/schemas/thread-artifact-chain.schema.yaml"
)
_CANDIDATE_HANDOFF_READINESS_PATH = (
    "docs/studies/eco1_rt_repack/operations/contract/readiness/checks/candidate_handoff.yaml"
)


def test_phase0_checked_in_contracts_pass_as_scaffold() -> None:
    report = validate_checked_in_contracts(repo_root=repo_root(), phase="phase0_scaffold")

    assert report.passed is True
    assert report.issue_count == 0


def test_phase4_sampling_gate_requires_materialized_rt_only_candidate_handoff(tmp_path: Path) -> None:
    issues = validate_sampling_artifacts(
        repo_root=repo_root(),
        structure_root=tmp_path,
        phase="phase4_downstream_promotion",
    )

    check_ids = {issue.check_id for issue in issues}
    assert "eco1_rt.handoff.candidate_handoff_not_materialized" in check_ids


def test_candidate_handoff_schema_requires_flat_sequence_csv_authority() -> None:
    schema = yaml.safe_load((repo_root() / _CANDIDATE_HANDOFF_SCHEMA_PATH).read_text(encoding="utf-8"))

    source_artifact_fields = set(schema["field_contract"]["source_artifacts"]["required"])
    assert "candidate_handoff_sequences" in source_artifact_fields

    candidate_fields = set(schema["field_contract"]["candidates"]["required_fields"])
    assert "candidate_handoff_sequence_csv_hash" in candidate_fields
    assert "protein_sequence" not in candidate_fields
    assert "dna_design_status" not in candidate_fields

    sequence_csv_columns = set(schema["field_contract"]["candidate_handoff_sequences"]["required_columns"])
    assert {
        "candidate_id",
        "selection_slot",
        "design_class_id",
        "protein_sequence",
        "sequence_hash",
        "amino_acid_length",
        "codon_policy_id",
        "dna_design_status",
        "restriction_site_screen_status",
    }.issubset(sequence_csv_columns)
    assert "candidate_handoff_sequences_must_match_candidate_sequence_hashes" in schema["invariants"]


def test_candidate_handoff_readiness_requires_flat_sequence_csv_hash_closure() -> None:
    readiness = yaml.safe_load((repo_root() / _CANDIDATE_HANDOFF_READINESS_PATH).read_text(encoding="utf-8"))
    checks = readiness["checks"]["phase2_real_backend_ingest"]
    handoff_check = next(
        check for check in checks if check["check_id"] == "eco1_rt.thread.materialized_candidate_handoff"
    )
    intent = handoff_check["validator_intent"]

    assert "candidate_handoff_sequences" in intent["required_source_artifacts"]
    assert "candidate_handoff_sequences" in intent["required_upstream_hashes"]


def test_artifact_chain_places_sequence_csv_before_candidate_handoff() -> None:
    artifact_chain = yaml.safe_load((repo_root() / _ARTIFACT_CHAIN_SCHEMA_PATH).read_text(encoding="utf-8"))
    artifact_order = list(artifact_chain["artifact_order"])

    assert artifact_order.index("candidate_selection_panel") < artifact_order.index("candidate_handoff_sequences")
    assert artifact_order.index("candidate_handoff_sequences") < artifact_order.index("candidate_handoff")
    assert "protein_sequence" in artifact_chain["artifacts"]["candidate_handoff_sequences"]["required_columns"]


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
