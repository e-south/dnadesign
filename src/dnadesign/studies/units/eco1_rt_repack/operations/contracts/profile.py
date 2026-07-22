"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/profile.py

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
    _is_positive_number,
    _iter_forbidden_field_paths,
    _phase_rank,
    _require_known_phase,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import _STUDY_ID
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_profile_payload(*, profile: Mapping[str, Any], schema: Mapping[str, Any], phase: str) -> ContractReport:
    """Validate the Eco1-specific fixed-backbone profile fixture."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []

    if profile.get("study_id") != _STUDY_ID:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.study_id_mismatch",
                message=f"profile study_id must be {_STUDY_ID!r}",
                path="study_id",
            )
        )

    for field in _as_string_list(schema.get("required_fields")):
        if field not in profile:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.profile.missing_required_field",
                    message=f"profile is missing required field {field!r}",
                    path=field,
                )
            )

    field_contract = schema.get("field_contract")
    if isinstance(field_contract, Mapping):
        for section, contract in field_contract.items():
            _validate_profile_section_required_fields(
                issues=issues,
                profile=profile,
                section=str(section),
                contract=contract,
            )
        _validate_profile_pending_reference_fields(
            issues=issues,
            profile=profile,
            reference_contract=field_contract.get("reference"),
            phase=phase,
        )

    _validate_profile_forbidden_fields(issues=issues, profile=profile, forbidden=schema.get("forbidden_fields"))
    _validate_profile_conservative_policy(issues=issues, profile=profile)
    _validate_profile_sampling_policy(issues=issues, profile=profile)

    return ContractReport(phase=phase, issues=tuple(issues))


def _validate_profile_section_required_fields(
    *,
    issues: list[ContractIssue],
    profile: Mapping[str, Any],
    section: str,
    contract: Any,
) -> None:
    if not isinstance(contract, Mapping):
        return
    section_payload = profile.get(section)
    if not isinstance(section_payload, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.missing_policy_section",
                message=f"profile section {section!r} must be a mapping",
                path=section,
            )
        )
        return
    for field in _as_string_list(contract.get("required")):
        if field not in section_payload:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.profile.missing_section_field",
                    message=f"profile section {section!r} is missing required field {field!r}",
                    path=f"{section}.{field}",
                )
            )


def _validate_profile_pending_reference_fields(
    *,
    issues: list[ContractIssue],
    profile: Mapping[str, Any],
    reference_contract: Any,
    phase: str,
) -> None:
    if not isinstance(reference_contract, Mapping):
        return
    pending_forbidden_from = str(reference_contract.get("pending_forbidden_from", "")).strip()
    if not pending_forbidden_from or _phase_rank(phase) < _phase_rank(pending_forbidden_from):
        return
    reference = profile.get("reference")
    if not isinstance(reference, Mapping):
        return
    pending_fields = [
        field
        for field in _as_string_list(reference_contract.get("required"))
        if _is_pending_value(reference.get(field))
    ]
    if pending_fields:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.pending_reference_authority",
                message=f"Phase {phase} forbids pending reference authority fields: {pending_fields}",
                path="reference",
            )
        )


def _validate_profile_forbidden_fields(
    *,
    issues: list[ContractIssue],
    profile: Mapping[str, Any],
    forbidden: Any,
) -> None:
    forbidden_fields = set(_as_string_list(forbidden))
    if not forbidden_fields:
        return
    for path in _iter_forbidden_field_paths(profile, forbidden_fields):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.forbidden_field",
                message=f"profile contains forbidden field {path}",
                path=path,
            )
        )


def _validate_profile_conservative_policy(*, issues: list[ContractIssue], profile: Mapping[str, Any]) -> None:
    policy = profile.get("conservative_policy")
    if not isinstance(policy, Mapping):
        return

    if policy.get("mask_policy_id") != "eco1_rt_clade9_plurality25_direct_contact5a_v1":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_mask_policy_id",
                message="mask_policy_id must be eco1_rt_clade9_plurality25_direct_contact5a_v1",
                path="conservative_policy.mask_policy_id",
            )
        )

    direct_threshold = policy.get("direct_contact_threshold_angstrom")
    if not _is_positive_number(direct_threshold) or float(direct_threshold) != 5.0:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_contact_threshold",
                message="direct_contact_threshold_angstrom must equal 5",
                path="conservative_policy.direct_contact_threshold_angstrom",
            )
        )
    for legacy_field in ("substrate_contact_threshold_angstrom", "relaxed_contact_thresholds_angstrom"):
        if legacy_field in policy:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.profile.legacy_mask_policy_field",
                    message=f"conservative_policy must not retain legacy field {legacy_field!r}",
                    path=f"conservative_policy.{legacy_field}",
                )
            )

    conservation_threshold = policy.get("conservation_threshold")
    if not isinstance(conservation_threshold, int | float) or not 0 < float(conservation_threshold) <= 1:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_conservation_threshold",
                message="conservation_threshold must be a number in (0, 1]",
                path="conservative_policy.conservation_threshold",
            )
        )

    mask_groups = policy.get("manual_mask_groups")
    if not isinstance(mask_groups, list) or not mask_groups:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_manual_mask_groups",
                message="manual_mask_groups must be a non-empty list",
                path="conservative_policy.manual_mask_groups",
            )
        )
    else:
        for index, group in enumerate(mask_groups):
            if not isinstance(group, Mapping) or not {"id", "status", "policy"} <= set(group):
                issues.append(
                    ContractIssue(
                        check_id="eco1_rt.profile.invalid_manual_mask_group",
                        message="each manual mask group must include id, status, and policy",
                        path=f"conservative_policy.manual_mask_groups[{index}]",
                    )
                )

    designable_rule = str(policy.get("designable_rule", "")).strip()
    if not designable_rule or ("missing" in designable_rule and "designable" in designable_rule):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_designable_rule",
                message="designable_rule must not treat missing evidence as designable",
                path="conservative_policy.designable_rule",
            )
        )


def _validate_profile_sampling_policy(*, issues: list[ContractIssue], profile: Mapping[str, Any]) -> None:
    policy = profile.get("sampling_policy")
    if not isinstance(policy, Mapping):
        return

    if policy.get("backend_selection_policy") != "explicit_no_fallback":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.implicit_backend_fallback",
                message="backend_selection_policy must be explicit_no_fallback",
                path="sampling_policy.backend_selection_policy",
            )
        )

    selected_backend = policy.get("selected_backend")
    allowed_backends = policy.get("backends_allowed")
    if (
        not isinstance(selected_backend, str)
        or not selected_backend.strip()
        or not isinstance(allowed_backends, list)
        or selected_backend not in allowed_backends
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_selected_backend",
                message="selected_backend must be declared and listed in backends_allowed",
                path="sampling_policy.selected_backend",
            )
        )

    temperatures = policy.get("temperatures")
    if (
        not isinstance(temperatures, list)
        or not temperatures
        or not all(_is_positive_number(item) for item in temperatures)
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_temperatures",
                message="temperatures must be a non-empty list of positive numbers",
                path="sampling_policy.temperatures",
            )
        )

    seed_set = policy.get("seed_set")
    if (
        policy.get("seed_policy") != "explicit_seed_list_required"
        or not isinstance(seed_set, list)
        or not seed_set
        or not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in seed_set)
        or len(set(seed_set)) != len(seed_set)
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_seed_set",
                message="seed_policy requires a non-empty explicit unique positive integer seed_set",
                path="sampling_policy.seed_set",
            )
        )

    batch_id = policy.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_batch_id",
                message="sampling_policy.batch_id must be explicit for scaled backend ingest",
                path="sampling_policy.batch_id",
            )
        )
    num_seq_per_target = policy.get("num_seq_per_target")
    batch_size = policy.get("batch_size")
    if not isinstance(num_seq_per_target, int) or isinstance(num_seq_per_target, bool) or num_seq_per_target <= 0:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_num_seq_per_target",
                message="sampling_policy.num_seq_per_target must be a positive integer",
                path="sampling_policy.num_seq_per_target",
            )
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_batch_size",
                message="sampling_policy.batch_size must be a positive integer",
                path="sampling_policy.batch_size",
            )
        )
    if (
        isinstance(num_seq_per_target, int)
        and not isinstance(num_seq_per_target, bool)
        and isinstance(batch_size, int)
        and not isinstance(batch_size, bool)
        and batch_size > 0
        and num_seq_per_target % batch_size != 0
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.profile.invalid_sampling_batch_divisibility",
                message="sampling_policy.num_seq_per_target must be divisible by batch_size",
                path="sampling_policy",
            )
        )
