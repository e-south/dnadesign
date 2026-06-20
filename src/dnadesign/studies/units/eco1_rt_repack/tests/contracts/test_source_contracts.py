"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/test_source_contracts.py

Profile, artifact-chain, and MSA source-contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.artifact_chain import (
    validate_artifact_chain_schema_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.conservation_sources import (
    validate_conservation_sources_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.mask_cases import (
    validate_conservative_mask_cases_payload,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.profile import validate_profile_payload
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import load_yaml


def test_conservation_sources_contract_accepts_selected_mestre_sources() -> None:
    sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml")
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")

    report = validate_conservation_sources_payload(
        sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is True


def test_conservation_sources_reject_target_sequence_hash_mismatch() -> None:
    sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml")
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    changed_sources = deepcopy(sources)
    changed_sources["target_sequence"]["reference_sequence_hash"] = "sha256:not-the-ec86kit-reference"

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert "eco1_rt.conservation.target_sequence_hash_mismatch" in {issue.check_id for issue in report.issues}


def test_conservation_sources_reject_missing_provider_policy() -> None:
    sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml")
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    changed_sources = deepcopy(sources)
    changed_sources["sequence_providers"] = [
        provider
        for provider in changed_sources["sequence_providers"]
        if provider["id"] != "bv_brc_feature_protein_fasta"
    ]

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert "eco1_rt.conservation.missing_required_provider" in {issue.check_id for issue in report.issues}


def test_conservation_sources_reject_missing_tao_rule_fields() -> None:
    sources = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml")
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    numbering = load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml")
    changed_sources = deepcopy(sources)
    changed_sources["source_method"]["plurality_rule"] = "frequency_only"
    changed_sources["source_method"]["gap_denominator_policy"] = "all_alignment_rows"

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.conservation.invalid_gap_denominator_policy" in check_ids
    assert "eco1_rt.conservation.invalid_plurality_rule" in check_ids


def test_profile_validator_rejects_forbidden_cross_tool_identity_fields() -> None:
    profile = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml")
    schema = load_yaml("docs/studies/eco1_rt_repack/operations/contract/schemas/eco1-rt-profile.schema.yaml")
    profile["permuter__var_id"] = "should-not-be-here"
    profile["downstream"]["rt_lnrna_sponging_construct_triage"]["construct_subject_id"] = "preclaimed"

    report = validate_profile_payload(profile=profile, schema=schema, phase="phase0_scaffold")

    assert report.passed is False
    messages = "\n".join(issue.message for issue in report.issues)
    assert "permuter__var_id" in messages
    assert "construct_subject_id" in messages


def test_artifact_chain_schema_requires_no_fallback_and_fixture_boundaries() -> None:
    schema = load_yaml("docs/studies/eco1_rt_repack/operations/contract/schemas/thread-artifact-chain.schema.yaml")
    schema["invariants"] = [
        invariant
        for invariant in schema["invariants"]
        if invariant
        not in {
            "fallback_policy_must_be_explicit_no_fallback_for_sampling",
            "fixture_artifacts_cannot_satisfy_materialized_handoff",
        }
    ]

    report = validate_artifact_chain_schema_payload(schema)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "thread.artifact_chain.missing_no_fallback_invariant" in check_ids
    assert "thread.artifact_chain.missing_fixture_boundary_invariant" in check_ids


def test_conservative_mask_cases_keep_required_fail_fast_gates() -> None:
    cases = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/conservative_mask_cases.yaml")
    cases["cases"] = [case for case in cases["cases"] if case["id"] != "reject_missing_contact_threshold"]

    report = validate_conservative_mask_cases_payload(cases)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.mask_cases.missing_required_case" in check_ids
