"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/design_classes/masking.py

Mask-row composition for named Eco1 RT design classes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.models import (
    DesignClassSpec,
)

_MOTIF_REASON = "motif_anchor"
_WANG_REASON = "wang_ec86_direct_contact_prior"
_SELECTED_CONSERVATION_REASON = "selected_wt_plurality_rule"
_SELECTED_CONTACT_REASON = "selected_retained_dna_rna_contact"
_CLade9_P25_REASON = "evolutionarily_conserved_clade9_25pct_plurality"
_CONTACT5_REASON = "direct_retained_dna_rna_contact_5a"


def compose_design_class_mask_rows(
    *,
    spec: DesignClassSpec,
    residue_rows: list[dict[str, Any]],
    contact_geometry_rows: list[dict[str, Any]],
    conservation_rows: list[dict[str, Any]],
    manual_authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compose mask rows for one named Eco1 design class."""

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
        selected_profile_rows = [row for row in profile_rows if row.get("profile_id") == spec.conservation_profile_id]
        mapped = residue["mapping_status"] == "mapped"
        nearest_distance = contact.get("nearest_context_atom_distance_angstrom")
        selected_contact = _within_distance(
            mapped=mapped,
            nearest_distance=nearest_distance,
            threshold=spec.contact_threshold_angstrom,
        )
        contact5 = _within_distance(mapped=mapped, nearest_distance=nearest_distance, threshold=5.0)
        selected_conservation = any(
            _passes_plurality_rule(row, threshold=spec.conservation_threshold) for row in selected_profile_rows
        )
        clade9_p25 = any(
            _passes_plurality_rule(row, threshold=0.25)
            for row in profile_rows
            if row.get("profile_id") == "ec86_clade9_conservation_v1"
        )
        manual_row = manual_by_position.get(position)
        candidate_prior = candidate_prior_by_position.get(position)
        motif_protected = manual_row is not None
        wang_prior = candidate_prior is not None
        protection_reasons = _protection_reasons(
            motif_protected=motif_protected,
            wang_prior=wang_prior,
            selected_conservation=selected_conservation,
            selected_contact=selected_contact,
        )
        protected = bool(protection_reasons)
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
                "mask_policy_id": spec.design_class_id,
                "selected_conservation_profile_id": spec.conservation_profile_id,
                "selected_conservation_threshold": spec.conservation_threshold,
                "selected_conservation_rule_passed": selected_conservation,
                "selected_contact_threshold_angstrom": spec.contact_threshold_angstrom,
                "selected_retained_dna_rna_contact": selected_contact,
                "min_distance_to_retained_dna_rna_angstrom": nearest_distance,
                "direct_contact_threshold_angstrom": spec.contact_threshold_angstrom,
                "direct_retained_dna_rna_contact_5a": contact5,
                "motif_protected": motif_protected,
                "wang_ec86_direct_contact_prior": wang_prior,
                "evolutionarily_conserved_clade9_25pct_plurality": clade9_p25,
                "wt_plurality_frequency": wt_plurality["frequency"],
                "wt_plurality_aa": wt_plurality["aa"],
                "conservation_profile_ids": [spec.conservation_profile_id] if selected_conservation else [],
                "manual_mask_reason": "" if manual_row is None else str(manual_row["manual_mask_reason"]),
                "wang_ec86_direct_contact_reason": "" if candidate_prior is None else str(candidate_prior["reason"]),
                "rt_interval_review_label": rt_review_label_by_position.get(position, ""),
                "protected": protected,
                "non_fixed": mapped and not protected,
                "non_fixed_missing_backbone": (not mapped) and not protected,
                "protection_reasons": protection_reasons,
                "conflict_status": "none",
                "conflict_reason": "",
            }
        )
    return rows


def summarize_design_class_mask_rows(rows: list[dict[str, Any]], *, spec: DesignClassSpec) -> dict[str, Any]:
    """Summarize design-class mask rows with explicit selected-policy fields."""

    non_fixed_mapped_positions = [int(row["canonical_position"]) for row in rows if row["non_fixed"]]
    non_fixed_missing_backbone_positions = [
        int(row["canonical_position"]) for row in rows if row["non_fixed_missing_backbone"]
    ]
    return {
        "design_class_id": spec.design_class_id,
        "selected_conservation_profile_id": spec.conservation_profile_id,
        "selected_conservation_threshold": spec.conservation_threshold,
        "selected_contact_threshold_angstrom": spec.contact_threshold_angstrom,
        "total_positions": len(rows),
        "mapped_position_count": sum(1 for row in rows if row["mapping_status"] == "mapped"),
        "missing_backbone_position_count": sum(1 for row in rows if row["mapping_status"] != "mapped"),
        "protected_position_count": sum(1 for row in rows if row["protected"]),
        "non_fixed_mapped_position_count": len(non_fixed_mapped_positions),
        "non_fixed_missing_backbone_position_count": len(non_fixed_missing_backbone_positions),
        "total_unprotected_position_count": len(non_fixed_mapped_positions) + len(non_fixed_missing_backbone_positions),
        "source_protected_counts": {
            _MOTIF_REASON: sum(1 for row in rows if row["motif_protected"]),
            _WANG_REASON: sum(1 for row in rows if row["wang_ec86_direct_contact_prior"]),
            _SELECTED_CONSERVATION_REASON: sum(1 for row in rows if row["selected_conservation_rule_passed"]),
            _SELECTED_CONTACT_REASON: sum(1 for row in rows if row["selected_retained_dna_rna_contact"]),
        },
        "legacy_source_counts": {
            _CLade9_P25_REASON: sum(1 for row in rows if row["evolutionarily_conserved_clade9_25pct_plurality"]),
            _CONTACT5_REASON: sum(1 for row in rows if row["direct_retained_dna_rna_contact_5a"]),
        },
        "non_fixed_mapped_positions": non_fixed_mapped_positions,
        "non_fixed_missing_backbone_positions": non_fixed_missing_backbone_positions,
    }


def _within_distance(*, mapped: bool, nearest_distance: object, threshold: float) -> bool:
    return mapped and nearest_distance is not None and float(nearest_distance) <= threshold


def _passes_plurality_rule(row: Mapping[str, Any], *, threshold: float) -> bool:
    if row.get("wt_is_plurality") is not None:
        return bool(row.get("wt_is_plurality")) and float(row.get("wt_frequency") or 0.0) >= threshold
    if threshold == 0.25 and row.get("passes_conservation_mask") is not None:
        return bool(row.get("passes_conservation_mask"))
    return False


def _protection_reasons(
    *,
    motif_protected: bool,
    wang_prior: bool,
    selected_conservation: bool,
    selected_contact: bool,
) -> list[str]:
    reasons: list[str] = []
    if motif_protected:
        reasons.append(_MOTIF_REASON)
    if wang_prior:
        reasons.append(_WANG_REASON)
    if selected_conservation:
        reasons.append(_SELECTED_CONSERVATION_REASON)
    if selected_contact:
        reasons.append(_SELECTED_CONTACT_REASON)
    return reasons


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
        if not isinstance(feature, Mapping) or feature.get("authority_type") != "rt_core_interval":
            continue
        label = str(feature.get("label", "")).strip()
        for position in feature.get("canonical_positions", []):
            if isinstance(position, int):
                labels[position] = label
    return labels


def _wt_plurality_summary(profile_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not profile_rows:
        return {"frequency": None, "aa": ""}
    best = max(profile_rows, key=lambda row: float(row.get("wt_frequency") or 0.0))
    return {"frequency": best.get("wt_frequency"), "aa": best.get("plurality_aa", "")}
