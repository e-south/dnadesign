"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/contact_risk/artifacts.py

Contact-risk profile validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue

_REQUIRED_EVIDENCE_KEYS = {
    "nearest_context_atom_distance",
    "sidechain_context_distance",
    "backbone_context_distance",
    "contact_atom_density",
    "retained_context_chain_count",
}
_REQUIRED_ROW_FIELDS = {
    "canonical_position",
    "wt_aa",
    "mapping_status",
    "contact_risk_class",
    "nearest_context_atom_distance_angstrom",
    "sidechain_context_distance_angstrom",
    "backbone_context_distance_angstrom",
    "sidechain_atom_status",
    "contact_atom_count_within_20a",
    "retained_context_chain_count_within_20a",
    "manual_mask",
    "conservation_mask",
    "wang_candidate_prior",
}


def validate_contact_risk_profile_content(path: Path) -> list[ContractIssue]:
    """Validate a materialized contact-risk profile as an audit artifact."""

    issues: list[ContractIssue] = []
    profile = _load_yaml(path)
    _validate_top_level(issues, profile=profile, path=path)
    _validate_evidence_availability(issues, profile=profile, path=path)
    _validate_rows(issues, profile=profile, path=path)
    return issues


def _validate_top_level(issues: list[ContractIssue], *, profile: Mapping[str, Any], path: Path) -> None:
    expected = {
        "schema_id": "eco1_rt_repack.contact_risk_profile",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.contact_risk_profile",
        "status": "materialized",
        "contact_risk_policy_id": "eco1_rt_contact_risk_audit_v1",
    }
    for key, value in expected.items():
        if profile.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.contact_risk.metadata_mismatch",
                    message=f"contact_risk_profile.yaml field {key!r} must equal {value!r}",
                    path=str(path),
                )
            )
    decision = profile.get("sampling_decision")
    if not isinstance(decision, Mapping) or decision.get("status") != "not_sampling_authoritative":
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.sampling_decision_mismatch",
                message="contact_risk_profile.yaml must be explicit that it is not sampling-authoritative",
                path=str(path),
            )
        )


def _validate_evidence_availability(issues: list[ContractIssue], *, profile: Mapping[str, Any], path: Path) -> None:
    evidence = profile.get("evidence_availability")
    if not isinstance(evidence, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.evidence_availability_missing",
                message="contact_risk_profile.yaml must declare evidence_availability",
                path=str(path),
            )
        )
        return
    missing = sorted(_REQUIRED_EVIDENCE_KEYS - set(evidence))
    if missing:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.evidence_availability_missing_keys",
                message=f"contact_risk_profile.yaml evidence_availability is missing keys: {missing}",
                path=str(path),
            )
        )
    missing_status = [
        key
        for key, value in evidence.items()
        if key in _REQUIRED_EVIDENCE_KEYS and not _mapping_has_text(value, "status")
    ]
    if missing_status:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.evidence_availability_missing_status",
                message=f"contact-risk evidence entries must declare status: {missing_status}",
                path=str(path),
            )
        )


def _validate_rows(issues: list[ContractIssue], *, profile: Mapping[str, Any], path: Path) -> None:
    rows = profile.get("residues")
    if not isinstance(rows, list):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.rows_missing",
                message="contact_risk_profile.yaml must contain a residues list",
                path=str(path),
            )
        )
        return
    missing_fields = sorted(
        {
            field
            for field in _REQUIRED_ROW_FIELDS
            if any(not isinstance(row, Mapping) or field not in row for row in rows)
        }
    )
    if missing_fields:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.row_missing_required_field",
                message=f"contact-risk rows are missing required fields: {missing_fields}",
                path=str(path),
            )
        )
        return
    observed_positions = [row.get("canonical_position") for row in rows if isinstance(row, Mapping)]
    if observed_positions != sorted(observed_positions):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.position_order_mismatch",
                message="contact-risk rows must be sorted by canonical position",
                path=str(path),
            )
        )
    summary = profile.get("summary")
    if isinstance(summary, Mapping) and summary.get("total_positions") != len(rows):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.contact_risk.summary_mismatch",
                message="contact-risk summary total_positions must match residue row count",
                path=str(path),
            )
        )


def _mapping_has_text(value: Any, field: str) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get(field), str) and bool(value[field].strip())


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded
