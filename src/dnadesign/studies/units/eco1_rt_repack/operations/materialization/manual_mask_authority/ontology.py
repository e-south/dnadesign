"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/manual_mask_authority/ontology.py

Ontology helpers for Eco1 RT manual mask authority.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WANG_DIRECT_CONTACT_MASK_PRIOR_POLICY = "active_direct_contact_mask_prior"


def structure_residue_ids_for_positions(
    canonical_positions: list[int],
    *,
    residue_by_position: Mapping[int, Mapping[str, Any]],
) -> list[int]:
    """Return mapped structure residue ids for canonical Eco1 positions."""

    structure_ids: list[int] = []
    for position in canonical_positions:
        residue = residue_by_position[position]
        if residue.get("mapping_status") != "mapped":
            raise ValueError(f"manual mask feature references unmapped position {position}")
        structure_ids.append(_require_int(residue, "structure_residue_id"))
    return structure_ids


def materialize_candidate_prior_residues(
    authority_source: Mapping[str, Any],
    *,
    residue_by_position: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and emit Wang/Ec86 direct-contact mask-prior rows."""

    candidate_sets = authority_source.get("candidate_authority_sets", [])
    if not isinstance(candidate_sets, list):
        raise ValueError("candidate_authority_sets must be a list")

    rows: list[dict[str, Any]] = []
    seen_positions: set[int] = set()
    for candidate_set in candidate_sets:
        if not isinstance(candidate_set, Mapping):
            raise ValueError("candidate_authority_sets entries must be mappings")
        set_id = _require_text(candidate_set, "id")
        policy = _require_text(candidate_set, "policy")
        status = _require_text(candidate_set, "status")
        if status != WANG_DIRECT_CONTACT_MASK_PRIOR_POLICY:
            raise ValueError(f"candidate authority set {set_id!r} must be a direct-contact mask prior")
        if policy != WANG_DIRECT_CONTACT_MASK_PRIOR_POLICY:
            raise ValueError(f"candidate authority set {set_id!r} has invalid policy {policy!r}")
        for candidate in _as_list(candidate_set.get("residues"), f"candidate_authority_sets[{set_id}].residues"):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"candidate_authority_sets[{set_id}].residues entries must be mappings")
            position = _require_int(candidate, "canonical_position")
            if position in seen_positions:
                raise ValueError(f"candidate prior position {position} is duplicated")
            seen_positions.add(position)
            residue = residue_by_position.get(position)
            if residue is None:
                raise ValueError(f"candidate prior position {position} is absent from residue_map.parquet")
            if residue.get("mapping_status") != "mapped":
                raise ValueError(f"candidate prior position {position} is not mapped")
            expected = {
                "wt_aa": residue["wt_aa"],
                "structure_chain_id": residue["structure_chain_id"],
                "structure_residue_id": residue["structure_residue_id"],
            }
            for field, value in expected.items():
                if candidate.get(field) != value:
                    raise ValueError(f"candidate prior position {position} has mismatched {field}")
            rows.append(
                {
                    "candidate_authority_set_id": set_id,
                    "canonical_position": position,
                    "wt_aa": residue["wt_aa"],
                    "structure_chain_id": residue["structure_chain_id"],
                    "structure_residue_id": residue["structure_residue_id"],
                    "design_position": residue["design_position"],
                    "mapping_status": residue["mapping_status"],
                    "policy": policy,
                    "reason": _require_text(candidate, "reason"),
                    "source_locator": _require_text(candidate, "source_locator"),
                    "evidence_basis": _as_list(candidate.get("evidence_basis"), "evidence_basis"),
                    "nearest_context_summary": _require_text(candidate, "nearest_context_summary"),
                }
            )
    return sorted(rows, key=lambda row: int(row["canonical_position"]))


def validate_deferred_authority(authority_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate deferred RT-region records without making them mask-authoritative."""

    deferred = authority_source.get("deferred_authority", [])
    if not isinstance(deferred, list):
        raise ValueError("deferred_authority must be a list")
    validated: list[dict[str, Any]] = []
    for entry in deferred:
        if not isinstance(entry, Mapping):
            raise ValueError("deferred_authority entries must be mappings")
        policy = _require_text(entry, "policy")
        if policy != "not_mask_authoritative_until_materialized":
            raise ValueError(f"deferred authority {entry.get('id')!r} has invalid policy {policy!r}")
        validated.append(dict(entry))
    return validated


def _as_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value
