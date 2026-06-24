"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/authority.py

Eco1 RT repack contract validation primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _append_mismatch_issue,
    _as_string_list,
    _is_pending_value,
    _is_positive_int,
    _nested_get,
    _phase_rank,
    _require_known_phase,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _REQUIRED_NUMBERING_POLICY_FIELDS,
    _REQUIRED_RESIDUE_MAP_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_structure_authority_payload(payload: Mapping[str, Any], *, phase: str) -> ContractReport:
    """Validate selected Eco1/Ec86 structure authority fields."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []
    if _phase_rank(phase) < _phase_rank("phase1_thread_contract"):
        return ContractReport(phase=phase)

    status = str(payload.get("status", "")).strip()
    if status != "selected":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.pending_source_selection",
                message="Phase 1 requires a selected structure authority before residue maps or sampling plans",
                path="workbench/provenance/structure-sources.yaml:status",
            )
        )
        return ContractReport(phase=phase, issues=tuple(issues))

    selected_source = payload.get("selected_source")
    if not isinstance(selected_source, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.missing_selected_source",
                message="selected structure authority must be recorded in selected_source",
                path="workbench/provenance/structure-sources.yaml:selected_source",
            )
        )
        return ContractReport(phase=phase, issues=tuple(issues))

    required_fields = sorted(
        set(_as_string_list(_nested_get(payload, ("selection_contract", "required_fields"))))
        | set(
            _as_string_list(
                _nested_get(payload, ("selection_contract", "phase1_acceptance", "required_non_pending_fields"))
            )
        )
    )
    pending_fields = [field for field in required_fields if _is_pending_value(selected_source.get(field))]
    if pending_fields:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.pending_selection_field",
                message=f"selected structure authority has pending fields: {pending_fields}",
                path="workbench/provenance/structure-sources.yaml:selected_source",
            )
        )

    allowed_policies = set(
        _as_string_list(_nested_get(payload, ("selection_contract", "retained_context_policy", "allowed_values")))
    )
    retained_context_policy = str(selected_source.get("retained_context_policy", "")).strip()
    if retained_context_policy not in allowed_policies:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.invalid_retained_context_policy",
                message="selected structure authority retained_context_policy is not allowed",
                path="workbench/provenance/structure-sources.yaml:selected_source.retained_context_policy",
            )
        )
    return ContractReport(phase=phase, issues=tuple(issues))


def validate_residue_numbering_policy_payload(
    payload: Mapping[str, Any],
    *,
    structure_sources: Mapping[str, Any],
    phase: str,
) -> ContractReport:
    """Validate the selected numbering policy without accepting runtime residue maps."""

    _require_known_phase(phase)
    issues: list[ContractIssue] = []
    if _phase_rank(phase) < _phase_rank("phase1_thread_contract"):
        return ContractReport(phase=phase)

    status = str(payload.get("status", "")).strip()
    if status != "selected":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_numbering_not_started",
                message="Phase 1 requires a selected residue-numbering policy before mutable residues are valid",
                path="workbench/provenance/residue-numbering-policy.yaml:status",
            )
        )
        return ContractReport(phase=phase, issues=tuple(issues))

    pending_fields = sorted(
        field for field in _REQUIRED_NUMBERING_POLICY_FIELDS if _is_pending_value(payload.get(field))
    )
    if pending_fields:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.pending_numbering_policy_field",
                message=f"selected residue-numbering policy has pending fields: {pending_fields}",
                path="workbench/provenance/residue-numbering-policy.yaml",
            )
        )

    selected_source = structure_sources.get("selected_source")
    if isinstance(selected_source, Mapping):
        selected_source_id = str(selected_source.get("source_id", "")).strip()
        if payload.get("selected_structure_source_id") != selected_source_id:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.numbering_source_mismatch",
                    message=(
                        "residue-numbering policy selected_structure_source_id must match selected structure source_id"
                    ),
                    path="workbench/provenance/residue-numbering-policy.yaml:selected_structure_source_id",
                )
            )
        selected_sequence_hash = str(selected_source.get("reference_sequence_hash", "")).strip()
        if payload.get("reference_sequence_hash") != selected_sequence_hash:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.numbering_sequence_hash_mismatch",
                    message="residue-numbering policy reference_sequence_hash must match selected structure source",
                    path="workbench/provenance/residue-numbering-policy.yaml:reference_sequence_hash",
                )
            )
        selected_origin = str(selected_source.get("residue_numbering_origin", "")).strip()
        if payload.get("residue_numbering_origin") != selected_origin:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.numbering_origin_mismatch",
                    message="residue-numbering policy origin must match selected structure source",
                    path="workbench/provenance/residue-numbering-policy.yaml:residue_numbering_origin",
                )
            )

    declared_columns = set(_as_string_list(payload.get("required_mapping_columns")))
    missing_columns = sorted(_REQUIRED_RESIDUE_MAP_COLUMNS - declared_columns)
    if missing_columns:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.numbering_missing_required_columns",
                message=f"residue-numbering policy is missing required residue-map columns: {missing_columns}",
                path="workbench/provenance/residue-numbering-policy.yaml:required_mapping_columns",
            )
        )

    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or not _is_positive_int(coverage.get("mapped_residue_count")):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.invalid_numbering_coverage",
                message="residue-numbering policy must declare positive mapped_residue_count coverage",
                path="workbench/provenance/residue-numbering-policy.yaml:coverage",
            )
        )

    return ContractReport(phase=phase, issues=tuple(issues))


def validate_authority_consistency_payload(
    *,
    profile: Mapping[str, Any],
    structure_sources: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
    phase: str,
) -> ContractReport:
    """Validate consistency among profile, structure authority, and numbering policy."""

    _require_known_phase(phase)
    if _phase_rank(phase) < _phase_rank("phase1_thread_contract"):
        return ContractReport(phase=phase)

    reference = profile.get("reference")
    selected_source = structure_sources.get("selected_source")
    if not isinstance(reference, Mapping) or not isinstance(selected_source, Mapping):
        return ContractReport(phase=phase)

    issues: list[ContractIssue] = []
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.structure_authority_mismatch",
        message="profile structure_authority must match selected structure source_id",
        profile_value=reference.get("structure_authority"),
        authority_value=selected_source.get("source_id"),
        path="reference.structure_authority",
    )
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.sequence_authority_mismatch",
        message="profile sequence_authority must match selected reference_sequence_authority",
        profile_value=reference.get("sequence_authority"),
        authority_value=selected_source.get("reference_sequence_authority"),
        path="reference.sequence_authority",
    )
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.structure_chain_mismatch",
        message="profile structure_chain_id must match selected rt_chain_id",
        profile_value=reference.get("structure_chain_id"),
        authority_value=selected_source.get("rt_chain_id"),
        path="reference.structure_chain_id",
    )
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.retained_context_policy_mismatch",
        message="profile retained_context_policy must match selected retained_context_policy",
        profile_value=reference.get("retained_context_policy"),
        authority_value=selected_source.get("retained_context_policy"),
        path="reference.retained_context_policy",
    )
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.residue_numbering_origin_mismatch",
        message="profile residue_numbering_origin must match selected residue_numbering_origin",
        profile_value=reference.get("residue_numbering_origin"),
        authority_value=numbering_policy.get("residue_numbering_origin"),
        path="reference.residue_numbering_origin",
    )
    _append_mismatch_issue(
        issues,
        check_id="eco1_rt.profile.reference_sequence_hash_mismatch",
        message="profile reference_sequence_hash must match selected numbering policy hash",
        profile_value=reference.get("reference_sequence_hash"),
        authority_value=numbering_policy.get("reference_sequence_hash"),
        path="reference.reference_sequence_hash",
    )
    return ContractReport(phase=phase, issues=tuple(issues))
