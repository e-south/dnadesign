"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/sampling/candidate_handoff.py

RT-only candidate-handoff contract validation for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue

_REQUIRED_ROOT_FIELDS = {
    "schema_id",
    "schema_version",
    "handoff_id",
    "handoff_kind",
    "study_id",
    "subject_kind",
    "construct_subject_created",
    "downstream_acceptance_required",
    "source_artifacts",
    "selection_policy",
    "candidates",
}
_REQUIRED_VALUES = {
    "handoff_kind": "rt_only_candidate_handoff",
    "study_id": "eco1_rt_repack",
    "subject_kind": "reverse_transcriptase_protein_only",
    "construct_subject_created": False,
    "downstream_acceptance_required": True,
}
_REQUIRED_SOURCE_ARTIFACTS = {
    "candidate_table",
    "foldcheck_report",
    "foldcheck_review",
    "feasibility_report",
    "candidate_triage_table",
    "candidate_selection_panel",
    "candidate_handoff_sequences",
    "upstream_artifact_hashes",
}
_REQUIRED_CANDIDATE_FIELDS = {
    "candidate_id",
    "sequence_hash",
    "candidate_handoff_sequence_csv_hash",
    "eligible_for_handoff",
    "foldcheck_status",
    "feasibility_status",
    "selection_slot",
}
_FORBIDDEN_FIELDS = {"permuter__var_id", "construct_subject_id", "downstream_target", "acceptance_summary"}


def validate_candidate_handoff_content(path: Path) -> list[ContractIssue]:
    """Validate the study-local RT-only candidate handoff record."""

    issues: list[ContractIssue] = []
    payload = _load_yaml_mapping(path, issues)
    if payload is None:
        return issues
    _validate_root_fields(issues, payload=payload, path=path)
    _validate_required_values(issues, payload=payload, path=path)
    _validate_forbidden_fields(issues, payload=payload, path=path)
    _validate_source_artifacts(issues, payload=payload, path=path)
    _validate_selection_policy(issues, payload=payload, path=path)
    _validate_candidates(issues, payload=payload, path=path)
    return issues


def _load_yaml_mapping(path: Path, issues: list[ContractIssue]) -> Mapping[str, Any] | None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.invalid_yaml",
                message=f"candidate_handoff.yaml is not valid YAML: {exc}",
                path=str(path),
            )
        )
        return None
    if not isinstance(loaded, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.invalid_yaml_mapping",
                message="candidate_handoff.yaml must be a YAML mapping",
                path=str(path),
            )
        )
        return None
    return loaded


def _validate_root_fields(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    for field in sorted(_REQUIRED_ROOT_FIELDS - set(payload)):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.missing_required_field",
                message=f"candidate_handoff.yaml must declare {field!r}",
                path=str(path),
            )
        )


def _validate_required_values(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    for field, expected in _REQUIRED_VALUES.items():
        if field in payload and payload.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.handoff.required_value_mismatch",
                    message=f"candidate_handoff.yaml field {field!r} must equal {expected!r}",
                    path=str(path),
                )
            )


def _validate_forbidden_fields(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    for field in sorted(_FORBIDDEN_FIELDS & set(payload)):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.forbidden_field",
                message=f"candidate_handoff.yaml must not declare {field!r}",
                path=str(path),
            )
        )


def _validate_source_artifacts(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    source_artifacts = payload.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.invalid_source_artifacts",
                message="candidate_handoff.yaml source_artifacts must be a mapping",
                path=str(path),
            )
        )
        return
    for field in sorted(_REQUIRED_SOURCE_ARTIFACTS - set(source_artifacts)):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.missing_source_artifact",
                message=f"candidate_handoff.yaml source_artifacts must declare {field!r}",
                path=str(path),
            )
        )
    upstream_hashes = source_artifacts.get("upstream_artifact_hashes")
    if not isinstance(upstream_hashes, Mapping) or not upstream_hashes:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.missing_upstream_hash_closure",
                message="candidate_handoff.yaml must include non-empty upstream_artifact_hashes",
                path=str(path),
            )
        )


def _validate_selection_policy(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    policy = payload.get("selection_policy")
    if not isinstance(policy, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.invalid_selection_policy",
                message="candidate_handoff.yaml selection_policy must be a mapping",
                path=str(path),
            )
        )
        return
    for field in ("eligibility_rule", "sae_acceptance_gate"):
        if field not in policy:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.handoff.missing_selection_policy_field",
                    message=f"candidate_handoff.yaml selection_policy must declare {field!r}",
                    path=str(path),
                )
            )
    if policy.get("sae_acceptance_gate") is not False:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.required_value_mismatch",
                message="candidate_handoff.yaml selection_policy.sae_acceptance_gate must be false",
                path=str(path),
            )
        )


def _validate_candidates(issues: list[ContractIssue], *, payload: Mapping[str, Any], path: Path) -> None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.handoff.invalid_candidates",
                message="candidate_handoff.yaml candidates must be a non-empty list",
                path=str(path),
            )
        )
        return
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.handoff.invalid_candidate_row",
                    message=f"candidate_handoff.yaml candidates[{index}] must be a mapping",
                    path=str(path),
                )
            )
            continue
        for field in sorted(_REQUIRED_CANDIDATE_FIELDS - set(candidate)):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.handoff.missing_candidate_field",
                    message=f"candidate_handoff.yaml candidates[{index}] must declare {field!r}",
                    path=str(path),
                )
            )
