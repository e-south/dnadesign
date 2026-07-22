"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/conservation/test_sources.py

Conservation-source contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.conservation import (
    validate_conservation_sources_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import load_yaml


def _validation_inputs() -> tuple[dict, dict, dict]:
    return (
        load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/conservation-sources.yaml"),
        load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/eco1_rt_v1.profile.yaml"),
        load_yaml("docs/studies/eco1_rt_repack/workbench/provenance/residue-numbering-policy.yaml"),
    )


def test_conservation_sources_contract_accepts_selected_mestre_sources() -> None:
    sources, profile, numbering = _validation_inputs()

    report = validate_conservation_sources_payload(
        sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is True


def test_conservation_sources_reject_target_sequence_hash_mismatch() -> None:
    sources, profile, numbering = _validation_inputs()
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
    sources, profile, numbering = _validation_inputs()
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
    sources, profile, numbering = _validation_inputs()
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


def test_conservation_sources_reject_legacy_broad_profile_acceptance() -> None:
    sources, profile, numbering = _validation_inputs()
    changed_sources = deepcopy(sources)
    changed_sources["phase1_acceptance"]["required_profile_ids"] = [
        "broad_retron_rt",
        "ec86_iia3_cluster42_1_conservation_v1",
    ]

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.conservation.phase1_unapproved_profile" in check_ids
    assert "eco1_rt.conservation.phase1_missing_required_profile" in check_ids


def test_conservation_sources_reject_full_mestre_roster_as_scoring_denominator() -> None:
    sources, profile, numbering = _validation_inputs()
    changed_sources = deepcopy(sources)
    broad_group = next(
        group for group in changed_sources["source_groups"] if group["profile_id"] == "ec86_clade9_conservation_v1"
    )
    broad_group["selection_rule"]["included_records"] = "mestre_s1_all_retron_rt_records_context"

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.conservation.forbidden_full_roster_denominator" in check_ids
    assert "eco1_rt.conservation.ec86_clade9_roster_scope_mismatch" in check_ids


def test_conservation_sources_reject_non_clade9_broad_scope() -> None:
    sources, profile, numbering = _validation_inputs()
    changed_sources = deepcopy(sources)
    broad_group = next(
        group for group in changed_sources["source_groups"] if group["profile_id"] == "ec86_clade9_conservation_v1"
    )
    broad_group["selection_rule"]["parent_rt_clade"] = 8

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert "eco1_rt.conservation.ec86_clade9_roster_scope_mismatch" in {issue.check_id for issue in report.issues}


def test_conservation_sources_reject_missing_motif_qc_policy() -> None:
    sources, profile, numbering = _validation_inputs()
    changed_sources = deepcopy(sources)
    broad_group = next(
        group for group in changed_sources["source_groups"] if group["profile_id"] == "ec86_clade9_conservation_v1"
    )
    broad_group["selection_rule"].pop("motif_qc_markers")
    broad_group["selection_rule"]["hard_reject_filters"] = ["missing_catalytic_rt_core"]

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.conservation.missing_motif_qc_markers" in check_ids
    assert "eco1_rt.conservation.missing_hard_reject_filter" in check_ids


def test_conservation_sources_reject_silent_alignment_backend_fallback() -> None:
    sources, profile, numbering = _validation_inputs()
    changed_sources = deepcopy(sources)
    changed_sources["alignment_policy"]["alternative_backend_policy"]["fallback_policy"] = "fallback_to_available"

    report = validate_conservation_sources_payload(
        changed_sources,
        profile=profile,
        numbering_policy=numbering,
        phase="phase1_thread_contract",
    )

    assert report.passed is False
    assert "eco1_rt.conservation.invalid_alternative_backend_fallback_policy" in {
        issue.check_id for issue in report.issues
    }
