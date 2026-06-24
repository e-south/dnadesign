"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/masking/rows.py

Deterministic mask-row composition for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_DIRECT_CONTACT_THRESHOLD_ANGSTROM = 5.0
_CONSERVATION_PROFILE_ID = "ec86_clade9_conservation_v1"
_MOTIF_REASON = "motif_anchor"
_WANG_REASON = "wang_ec86_direct_contact_prior"
_CONSERVATION_REASON = "evolutionarily_conserved_clade9_25pct_plurality"
_DIRECT_CONTACT_REASON = "direct_retained_dna_rna_contact_5a"


def compose_mask_rows(
    *,
    residue_rows: list[dict[str, Any]],
    contact_geometry_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
    manual_authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compose per-residue mask rows under the simple Eco1 protection rule."""

    contact_by_position = {int(row["canonical_position"]): row for row in contact_geometry_rows}
    conservation_by_position: dict[int, list[dict[str, Any]]] = {}
    for row in conservation_rows:
        conservation_by_position.setdefault(int(row["canonical_position"]), []).append(row)
    manual_by_position = _manual_authority_by_position(manual_authority)
    candidate_prior_by_position = _candidate_prior_by_position(manual_authority)
    rt_review_label_by_position = _rt_review_label_by_position(manual_authority)

    rows: list[dict[str, Any]] = []
    for residue in residue_rows:
        position = int(residue["canonical_position"])
        contact = contact_by_position[position]
        profile_rows = conservation_by_position.get(position, [])
        mapped = residue["mapping_status"] == "mapped"
        manual_row = manual_by_position.get(position)
        candidate_prior = candidate_prior_by_position.get(position)
        motif_protected = manual_row is not None
        wang_prior = candidate_prior is not None
        selected_profile_rows = [row for row in profile_rows if row.get("profile_id") == _CONSERVATION_PROFILE_ID]
        evolutionarily_conserved = any(row["passes_conservation_mask"] is True for row in selected_profile_rows)
        nearest_distance = contact.get("nearest_context_atom_distance_angstrom")
        direct_contact = (
            mapped and nearest_distance is not None and float(nearest_distance) <= _DIRECT_CONTACT_THRESHOLD_ANGSTROM
        )
        protection_reasons = _protection_reasons(
            motif_protected=motif_protected,
            wang_prior=wang_prior,
            evolutionarily_conserved=evolutionarily_conserved,
            direct_contact=direct_contact,
        )
        protected = bool(protection_reasons)
        non_fixed_missing_backbone = not mapped and not protected
        wt_plurality = _wt_plurality_summary(selected_profile_rows)
        rows.append(
            {
                "canonical_position": position,
                "wt_aa": residue["wt_aa"],
                "structure_chain_id": residue["structure_chain_id"],
                "structure_residue_id": residue["structure_residue_id"],
                "design_position": residue["design_position"],
                "mapping_status": residue["mapping_status"],
                "has_backbone_coordinates": mapped,
                "min_distance_to_retained_dna_rna_angstrom": nearest_distance,
                "direct_contact_threshold_angstrom": _DIRECT_CONTACT_THRESHOLD_ANGSTROM,
                "direct_retained_dna_rna_contact_5a": direct_contact,
                "motif_protected": motif_protected,
                "wang_ec86_direct_contact_prior": wang_prior,
                "evolutionarily_conserved_clade9_25pct_plurality": evolutionarily_conserved,
                "wt_plurality_frequency": wt_plurality["frequency"],
                "wt_plurality_aa": wt_plurality["aa"],
                "conservation_profile_ids": sorted(
                    row["profile_id"] for row in selected_profile_rows if row["passes_conservation_mask"] is True
                ),
                "manual_mask_reason": "" if manual_row is None else str(manual_row["manual_mask_reason"]),
                "wang_ec86_direct_contact_reason": "" if candidate_prior is None else str(candidate_prior["reason"]),
                "rt_interval_review_label": rt_review_label_by_position.get(position, ""),
                "protected": protected,
                "non_fixed": mapped and not protected,
                "non_fixed_missing_backbone": non_fixed_missing_backbone,
                "protection_reasons": protection_reasons,
                "conflict_status": "none",
                "conflict_reason": "",
            }
        )
    return rows


def summarize_mask_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize mask rows using the current simple-policy vocabulary."""

    source_protected_counts = {
        _MOTIF_REASON: sum(1 for row in rows if row["motif_protected"]),
        _WANG_REASON: sum(1 for row in rows if row["wang_ec86_direct_contact_prior"]),
        _CONSERVATION_REASON: sum(1 for row in rows if row["evolutionarily_conserved_clade9_25pct_plurality"]),
        _DIRECT_CONTACT_REASON: sum(1 for row in rows if row["direct_retained_dna_rna_contact_5a"]),
    }
    non_fixed_mapped_positions = [row["canonical_position"] for row in rows if row["non_fixed"]]
    non_fixed_missing_backbone_positions = [
        row["canonical_position"] for row in rows if row["non_fixed_missing_backbone"]
    ]
    return {
        "total_positions": len(rows),
        "mapped_position_count": sum(1 for row in rows if row["mapping_status"] == "mapped"),
        "missing_backbone_position_count": sum(1 for row in rows if row["mapping_status"] != "mapped"),
        "protected_position_count": sum(1 for row in rows if row["protected"]),
        "non_fixed_mapped_position_count": len(non_fixed_mapped_positions),
        "non_fixed_missing_backbone_position_count": len(non_fixed_missing_backbone_positions),
        "total_unprotected_position_count": len(non_fixed_mapped_positions) + len(non_fixed_missing_backbone_positions),
        "source_protected_counts": source_protected_counts,
        "non_fixed_mapped_positions": non_fixed_mapped_positions,
        "non_fixed_missing_backbone_positions": non_fixed_missing_backbone_positions,
    }


def _manual_authority_by_position(manual_authority: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = manual_authority.get("residues")
    if not isinstance(rows, list):
        raise ValueError("manual_mask_authority.yaml must contain a residues list")
    return {int(row["canonical_position"]): row for row in rows if isinstance(row, Mapping)}


def _candidate_prior_by_position(manual_authority: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = manual_authority.get("candidate_prior_residues")
    if not isinstance(rows, list):
        raise ValueError("manual_mask_authority.yaml must contain a candidate_prior_residues list")
    return {int(row["canonical_position"]): row for row in rows if isinstance(row, Mapping)}


def _rt_review_label_by_position(manual_authority: Mapping[str, Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    features = manual_authority.get("features")
    if not isinstance(features, list):
        return labels
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        if feature.get("authority_type") != "rt_core_interval":
            continue
        label = str(feature.get("label", "")).strip()
        for position in feature.get("canonical_positions", []):
            if isinstance(position, int):
                labels[position] = label
    return labels


def _protection_reasons(
    *,
    motif_protected: bool,
    wang_prior: bool,
    evolutionarily_conserved: bool,
    direct_contact: bool,
) -> list[str]:
    reasons: list[str] = []
    if motif_protected:
        reasons.append(_MOTIF_REASON)
    if wang_prior:
        reasons.append(_WANG_REASON)
    if evolutionarily_conserved:
        reasons.append(_CONSERVATION_REASON)
    if direct_contact:
        reasons.append(_DIRECT_CONTACT_REASON)
    return reasons


def _wt_plurality_summary(profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not profile_rows:
        return {"frequency": None, "aa": ""}
    best = max(profile_rows, key=lambda row: float(row.get("wt_frequency") or 0.0))
    return {"frequency": best.get("wt_frequency"), "aa": best.get("plurality_aa", "")}
