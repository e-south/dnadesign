"""Position protection and open-set rules for Eco1 RT generation policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .constants import (
    C_TERMINAL_THUMB_CONTEXT,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    CONSERVATION_PROFILE_ID,
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    DISTAL_SCAFFOLD_POLICY_ID,
    GENERATION_POLICY_VERSION,
    MOTIF_CONTEXTS,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    NEAR_REGION_MAX_INCLUSIVE_ANGSTROM,
    NEAR_REGION_MIN_EXCLUSIVE_ANGSTROM,
    WANG_THUMB_TRACK_POSITIONS,
)
from .models import GenerationPolicyConfig, GenerationPolicySpec


def build_position_rows(
    *,
    config: GenerationPolicyConfig,
    inputs: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build one protected/open position row per policy and canonical residue."""

    base_rows = _base_position_rows(
        residue_rows=inputs["residue_rows"],
        contact_rows=inputs["contact_rows"],
        conservation_rows=inputs["conservation_rows"],
    )
    rows: list[dict[str, Any]] = []
    for policy in config.enabled_policies:
        open_positions = _open_positions_for_policy(policy=policy, base_rows=base_rows)
        for base_row in base_rows:
            rows.append(
                {
                    "policy_id": policy.policy_id,
                    "policy_version": GENERATION_POLICY_VERSION,
                    "open_set_id": policy.open_set_id,
                    **base_row,
                    "is_open_position": int(base_row["eco1_position"]) in open_positions,
                }
            )
    return rows


def _base_position_rows(
    *,
    residue_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contact_by_position = {int(row["canonical_position"]): row for row in contact_rows}
    conserved_positions = {
        int(row["canonical_position"])
        for row in conservation_rows
        if row.get("profile_id") == CONSERVATION_PROFILE_ID and row.get("passes_conservation_mask") is True
    }
    rows: list[dict[str, Any]] = []
    for residue in sorted(residue_rows, key=lambda row: int(row["canonical_position"])):
        position = int(residue["canonical_position"])
        contact = contact_by_position[position]
        distance = _optional_float(contact.get("nearest_context_atom_distance_angstrom"))
        is_mapped = residue.get("mapping_status") == "mapped"
        is_direct_contact = is_mapped and distance is not None and distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM
        is_near_region = (
            is_mapped
            and distance is not None
            and NEAR_REGION_MIN_EXCLUSIVE_ANGSTROM < distance <= NEAR_REGION_MAX_INCLUSIVE_ANGSTROM
        )
        motif_contexts = _motif_context_codes(position)
        is_wang = position in WANG_THUMB_TRACK_POSITIONS
        is_c_terminal = _in_range(position, C_TERMINAL_THUMB_CONTEXT) and is_mapped
        is_conserved = position in conserved_positions
        structure_residue_id = residue.get("structure_residue_id")
        structure_chain_id = str(residue.get("structure_chain_id") or "")
        rows.append(
            {
                "eco1_position": position,
                "wt_aa": residue["wt_aa"],
                "structure_position": structure_residue_id,
                "chain_position": (
                    "" if structure_residue_id is None else f"{structure_chain_id}:{int(structure_residue_id)}"
                ),
                "is_mapped": is_mapped,
                "is_designable_backbone_position": is_mapped,
                "protected_reason_codes": _protected_reason_codes(
                    motif_contexts=motif_contexts,
                    is_direct_contact=is_direct_contact,
                    is_wang=is_wang,
                    is_c_terminal=is_c_terminal,
                    is_conserved=is_conserved,
                ),
                "distance_to_retained_dna_rna": distance,
                "is_direct_contact_le_5a": is_direct_contact,
                "is_near_region_gt5_le10a": is_near_region,
                "is_wang_thumb_track": is_wang,
                "is_c_terminal_thumb_context": is_c_terminal,
                "is_conserved_core": is_conserved,
                "motif_context_codes": motif_contexts,
            }
        )
    return rows


def _open_positions_for_policy(*, policy: GenerationPolicySpec, base_rows: list[dict[str, Any]]) -> set[int]:
    if policy.policy_id == DISTAL_SCAFFOLD_POLICY_ID:
        return _open_positions(base_rows, near_region=False)
    if policy.policy_id == NEAR_DNA_RNA_ACID_FREE_POLICY_ID:
        return _open_positions(base_rows, near_region=True)
    if policy.policy_id == COMBINED_NEAR_PLUS_DISTAL_POLICY_ID:
        return _open_positions(base_rows, near_region=False) | _open_positions(base_rows, near_region=True)
    raise ValueError(f"unknown generation policy id {policy.policy_id!r}")


def _open_positions(base_rows: list[dict[str, Any]], *, near_region: bool) -> set[int]:
    return {
        int(row["eco1_position"])
        for row in base_rows
        if row["is_designable_backbone_position"]
        and not row["protected_reason_codes"]
        and bool(row["is_near_region_gt5_le10a"]) is near_region
    }


def _protected_reason_codes(
    *,
    motif_contexts: list[str],
    is_direct_contact: bool,
    is_wang: bool,
    is_c_terminal: bool,
    is_conserved: bool,
) -> list[str]:
    reasons = [f"motif_context_{context}" for context in motif_contexts]
    if is_direct_contact:
        reasons.append("direct_retained_dna_rna_contact_le5a")
    if is_wang:
        reasons.append("wang_thumb_contact_track")
    if is_c_terminal:
        reasons.append("c_terminal_thumb_context_255_311")
    if is_conserved:
        reasons.append("conserved_core_clade9_25pct_plurality")
    return reasons


def _motif_context_codes(position: int) -> list[str]:
    return [code for code, span in MOTIF_CONTEXTS.items() if _in_range(position, span)]


def _in_range(position: int, span: tuple[int, int]) -> bool:
    return span[0] <= position <= span[1]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
