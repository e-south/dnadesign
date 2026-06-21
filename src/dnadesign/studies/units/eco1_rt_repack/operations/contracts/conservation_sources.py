"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation_sources.py

Eco1 RT repack contract validation primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _as_string_list,
    _is_pending_value,
    _is_positive_int,
    _is_sha256_text,
    _nested_get,
    _phase_rank,
    _require_known_phase,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.conservation_source_selection import (
    validate_conservation_selection_rule,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _CONSERVATION_GAP_DENOMINATOR_POLICY,
    _CONSERVATION_PLURALITY_RULE,
    _CONSERVATION_TARGET_POLICY,
    _PROVIDER_FAILURE_POLICY,
    _REQUIRED_CONSERVATION_PROFILE_IDS,
    _REQUIRED_CONSERVATION_PROVIDER_IDS,
    _STUDY_ID,
    _TARGET_MISMATCH_POLICY,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_conservation_sources_payload(
    payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
    phase: str,
) -> ContractReport:
    """Validate declared MSA/conservation source authority without accepting generated evidence."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []
    if _phase_rank(phase) < _phase_rank("phase1_thread_contract"):
        return ContractReport(phase=phase)

    if payload.get("study_id") != _STUDY_ID:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.study_id_mismatch",
                message=f"conservation source study_id must be {_STUDY_ID!r}",
                path="workbench/provenance/conservation-sources.yaml:study_id",
            )
        )

    if payload.get("status") != "selected":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.source_contract_not_selected",
                message="Phase 1 requires selected conservation source authority before conservation profiles",
                path="workbench/provenance/conservation-sources.yaml:status",
            )
        )

    source_method = payload.get("source_method")
    if not isinstance(source_method, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_source_method",
                message="conservation-sources.yaml must declare source_method",
                path="workbench/provenance/conservation-sources.yaml:source_method",
            )
        )
    else:
        _validate_conservation_source_method(issues, source_method=source_method, profile=profile)

    alignment_policy = payload.get("alignment_policy")
    if not isinstance(alignment_policy, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_alignment_policy",
                message="conservation-sources.yaml must declare alignment_policy",
                path="workbench/provenance/conservation-sources.yaml:alignment_policy",
            )
        )
    else:
        _validate_conservation_alignment_policy(issues, alignment_policy=alignment_policy)

    target_sequence = payload.get("target_sequence")
    if not isinstance(target_sequence, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_target_sequence",
                message="conservation-sources.yaml must declare target_sequence",
                path="workbench/provenance/conservation-sources.yaml:target_sequence",
            )
        )
    else:
        _validate_conservation_target_sequence(
            issues,
            target_sequence=target_sequence,
            numbering_policy=numbering_policy,
        )

    providers = payload.get("sequence_providers")
    if not isinstance(providers, list):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_sequence_providers",
                message="conservation-sources.yaml must declare sequence_providers",
                path="workbench/provenance/conservation-sources.yaml:sequence_providers",
            )
        )
        provider_ids: set[str] = set()
    else:
        provider_ids = _validate_conservation_providers(issues, providers=providers)

    source_groups = payload.get("source_groups")
    if not isinstance(source_groups, list):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_source_groups",
                message="conservation-sources.yaml must declare source_groups",
                path="workbench/provenance/conservation-sources.yaml:source_groups",
            )
        )
    else:
        _validate_conservation_source_groups(
            issues,
            source_groups=source_groups,
            provider_ids=provider_ids,
            profile=profile,
        )

    _validate_phase1_acceptance(issues, payload=payload)

    return ContractReport(phase=phase, issues=tuple(issues))


def _validate_conservation_source_method(
    issues: list[ContractIssue],
    *,
    source_method: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    policy = profile.get("conservative_policy")
    profile_threshold = policy.get("conservation_threshold") if isinstance(policy, Mapping) else None
    profile_rule = policy.get("conservation_rule") if isinstance(policy, Mapping) else None

    if source_method.get("conservation_rule") != profile_rule:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.conservation_rule_mismatch",
                message="source_method conservation_rule must match the Eco1 profile policy",
                path="workbench/provenance/conservation-sources.yaml:source_method.conservation_rule",
            )
        )
    if source_method.get("threshold") != profile_threshold:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.threshold_mismatch",
                message="source_method threshold must match the Eco1 profile conservation_threshold",
                path="workbench/provenance/conservation-sources.yaml:source_method.threshold",
            )
        )
    if source_method.get("gap_denominator_policy") != _CONSERVATION_GAP_DENOMINATOR_POLICY:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_gap_denominator_policy",
                message="source_method must use non-gap rows as the conservation denominator",
                path="workbench/provenance/conservation-sources.yaml:source_method.gap_denominator_policy",
            )
        )
    if source_method.get("plurality_rule") != _CONSERVATION_PLURALITY_RULE:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_plurality_rule",
                message="source_method plurality_rule must require WT amino acid to equal the plurality amino acid",
                path="workbench/provenance/conservation-sources.yaml:source_method.plurality_rule",
            )
        )
    if source_method.get("missing_evidence_policy") != "fail_closed_no_designability":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_missing_evidence_policy",
                message="missing conservation evidence must fail closed and cannot imply designability",
                path="workbench/provenance/conservation-sources.yaml:source_method.missing_evidence_policy",
            )
        )


def _validate_conservation_alignment_policy(
    issues: list[ContractIssue],
    *,
    alignment_policy: Mapping[str, Any],
) -> None:
    if alignment_policy.get("target_sequence_policy") != _CONSERVATION_TARGET_POLICY:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_target_sequence_policy",
                message="alignment policy must pin the ec86kit reference sequence as the target row",
                path="workbench/provenance/conservation-sources.yaml:alignment_policy.target_sequence_policy",
            )
        )
    if _is_pending_value(alignment_policy.get("alignment_command")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_alignment_command",
                message="alignment policy must declare the exact alignment command shape before materialization",
                path="workbench/provenance/conservation-sources.yaml:alignment_policy.alignment_command",
            )
        )
    if alignment_policy.get("alignment_scope") != "protein":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_alignment_scope",
                message="conservation alignment scope must be protein",
                path="workbench/provenance/conservation-sources.yaml:alignment_policy.alignment_scope",
            )
        )
    alternative_backend_policy = alignment_policy.get("alternative_backend_policy")
    if isinstance(alternative_backend_policy, Mapping) and (
        alternative_backend_policy.get("fallback_policy") != "no_silent_backend_fallback"
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_alternative_backend_fallback_policy",
                message="alternative alignment backend policy must reject silent fallback",
                path=(
                    "workbench/provenance/conservation-sources.yaml:"
                    "alignment_policy.alternative_backend_policy.fallback_policy"
                ),
            )
        )


def _validate_conservation_target_sequence(
    issues: list[ContractIssue],
    *,
    target_sequence: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
) -> None:
    if target_sequence.get("reference_sequence_hash") != numbering_policy.get("reference_sequence_hash"):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.target_sequence_hash_mismatch",
                message="target_sequence reference_sequence_hash must match the selected residue-numbering policy",
                path="workbench/provenance/conservation-sources.yaml:target_sequence.reference_sequence_hash",
            )
        )
    coverage = numbering_policy.get("coverage")
    reference_length = coverage.get("reference_sequence_length") if isinstance(coverage, Mapping) else None
    if target_sequence.get("reference_sequence_length") != reference_length:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.target_sequence_length_mismatch",
                message="target_sequence reference_sequence_length must match residue-numbering coverage",
                path="workbench/provenance/conservation-sources.yaml:target_sequence.reference_sequence_length",
            )
        )
    if not _is_sha256_text(target_sequence.get("reference_sequence_hash")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_target_sequence_hash",
                message="target_sequence reference_sequence_hash must be a SHA-256 digest",
                path="workbench/provenance/conservation-sources.yaml:target_sequence.reference_sequence_hash",
            )
        )

    mismatch = _nested_get(target_sequence, ("known_public_accession", "mismatch"))
    if not isinstance(mismatch, Mapping) or mismatch.get("policy") != _TARGET_MISMATCH_POLICY:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_target_mismatch_policy",
                message="known public Eco1 accession mismatch must be rejected or explicitly adjudicated",
                path="workbench/provenance/conservation-sources.yaml:target_sequence.known_public_accession.mismatch",
            )
        )


def _validate_conservation_providers(
    issues: list[ContractIssue],
    *,
    providers: list[Any],
) -> set[str]:
    provider_ids: set[str] = set()
    for index, provider in enumerate(providers):
        if not isinstance(provider, Mapping):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.invalid_provider",
                    message="each sequence provider must be a mapping",
                    path=f"workbench/provenance/conservation-sources.yaml:sequence_providers[{index}]",
                )
            )
            continue
        provider_id = str(provider.get("id", "")).strip()
        if provider_id:
            provider_ids.add(provider_id)
        if provider.get("status") != "selected" or provider.get("failure_policy") != _PROVIDER_FAILURE_POLICY:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.invalid_provider_policy",
                    message="sequence providers must be selected and fail by explicit exclude-or-fail policy",
                    path=f"workbench/provenance/conservation-sources.yaml:sequence_providers[{index}]",
                )
            )
    missing_providers = sorted(_REQUIRED_CONSERVATION_PROVIDER_IDS - provider_ids)
    for provider_id in missing_providers:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_required_provider",
                message=f"conservation source contract must declare sequence provider {provider_id!r}",
                path="workbench/provenance/conservation-sources.yaml:sequence_providers",
            )
        )
    return provider_ids


def _validate_conservation_source_groups(
    issues: list[ContractIssue],
    *,
    source_groups: list[Any],
    provider_ids: set[str],
    profile: Mapping[str, Any],
) -> None:
    observed_profile_ids: set[str] = set()
    for index, source_group in enumerate(source_groups):
        if not isinstance(source_group, Mapping):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.invalid_source_group",
                    message="each conservation source group must be a mapping",
                    path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}]",
                )
            )
            continue
        profile_id = str(source_group.get("profile_id", "")).strip()
        observed_profile_ids.add(profile_id)
        if source_group.get("status") != "selected":
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.source_group_not_selected",
                    message=f"source group {profile_id!r} must be selected",
                    path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].status",
                )
            )
        _validate_conservation_roster_source(issues, source_group=source_group, index=index)
        validate_conservation_selection_rule(issues, source_group=source_group, index=index)
        group_provider_ids = set(_as_string_list(source_group.get("provider_ids")))
        if not _REQUIRED_CONSERVATION_PROVIDER_IDS <= group_provider_ids <= provider_ids:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.source_group_provider_mismatch",
                    message=f"source group {profile_id!r} must use the declared required sequence providers",
                    path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].provider_ids",
                )
            )
        if not _is_positive_int(source_group.get("min_non_gap_count")):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.conservation.invalid_min_non_gap_count",
                    message=f"source group {profile_id!r} must declare positive min_non_gap_count",
                    path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].min_non_gap_count",
                )
            )

    profile_policy = profile.get("conservative_policy")
    profile_declared_ids = (
        set(_as_string_list(profile_policy.get("conservation_profiles")))
        if isinstance(profile_policy, Mapping)
        else set()
    )
    required_ids = _REQUIRED_CONSERVATION_PROFILE_IDS | profile_declared_ids
    for profile_id in sorted(required_ids - observed_profile_ids):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_required_source_group",
                message=f"conservation source contract must declare source group {profile_id!r}",
                path="workbench/provenance/conservation-sources.yaml:source_groups",
            )
        )


def _validate_phase1_acceptance(
    issues: list[ContractIssue],
    *,
    payload: Mapping[str, Any],
) -> None:
    acceptance = payload.get("phase1_acceptance")
    if not isinstance(acceptance, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_phase1_acceptance",
                message="conservation-sources.yaml must declare phase1_acceptance",
                path="workbench/provenance/conservation-sources.yaml:phase1_acceptance",
            )
        )
        return
    required_profile_ids = set(_as_string_list(acceptance.get("required_profile_ids")))
    missing_profile_ids = _REQUIRED_CONSERVATION_PROFILE_IDS - required_profile_ids
    forbidden_profile_ids = required_profile_ids - _REQUIRED_CONSERVATION_PROFILE_IDS
    for profile_id in sorted(missing_profile_ids):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.phase1_missing_required_profile",
                message=f"phase1_acceptance.required_profile_ids must include {profile_id!r}",
                path="workbench/provenance/conservation-sources.yaml:phase1_acceptance.required_profile_ids",
            )
        )
    for profile_id in sorted(forbidden_profile_ids):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.phase1_unapproved_profile",
                message=f"phase1_acceptance.required_profile_ids includes unapproved profile {profile_id!r}",
                path="workbench/provenance/conservation-sources.yaml:phase1_acceptance.required_profile_ids",
            )
        )


def _validate_conservation_roster_source(
    issues: list[ContractIssue],
    *,
    source_group: Mapping[str, Any],
    index: int,
) -> None:
    roster_source = source_group.get("roster_source")
    profile_id = str(source_group.get("profile_id", "")).strip()
    if not isinstance(roster_source, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.missing_roster_source",
                message=f"source group {profile_id!r} must declare roster_source",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].roster_source",
            )
        )
        return
    if not _is_sha256_text(roster_source.get("source_sha256")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.invalid_roster_source_hash",
                message=f"source group {profile_id!r} must declare roster source SHA-256",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].roster_source.source_sha256",
            )
        )
    if _is_pending_value(roster_source.get("source_ref")) or _is_pending_value(roster_source.get("accession_field")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.conservation.incomplete_roster_source",
                message=f"source group {profile_id!r} must declare source_ref and accession_field",
                path=f"workbench/provenance/conservation-sources.yaml:source_groups[{index}].roster_source",
            )
        )
