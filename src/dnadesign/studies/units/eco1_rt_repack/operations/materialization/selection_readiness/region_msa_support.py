"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/region_msa_support.py

Region-wise MSA support evidence for Eco1 RT panel review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .review_axes import (
    C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS,
    CLADE9_PROFILE_ID,
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    NA_FACING_DISTANCE_ANGSTROM,
    SUBTYPE_PROFILE_ID,
    WANG_THUMB_CONTACT_TRACK_POSITIONS,
    Mutation,
    _float_or_none,
    _natural_support,
    _parse_mutations,
    _profile_support,
)
from .selection_policy_context import resolve_selection_policy_context


@dataclass(frozen=True)
class RegionMsaSupportSpec:
    """One mutation region used for region-wise natural-sequence support review."""

    region_id: str
    label: str
    role: str


REGION_MSA_SUPPORT_SPECS = (
    RegionMsaSupportSpec(
        region_id="catalytic_or_direct_contact",
        label="Catalytic or direct contact",
        role="protected_core_or_contact",
    ),
    RegionMsaSupportSpec(
        region_id="near_retained_dna_rna_region",
        label="Near retained DNA/RNA region",
        role="substrate_proximal_review",
    ),
    RegionMsaSupportSpec(
        region_id="thumb_contact_track",
        label="Thumb-contact track",
        role="thumb_contact_review",
    ),
    RegionMsaSupportSpec(
        region_id="c_terminal_primer_rna_recognition_region",
        label="C-terminal primer-RNA recognition region",
        role="c_terminal_primer_rna_recognition_review",
    ),
    RegionMsaSupportSpec(
        region_id="distal_scaffold",
        label="Distal scaffold",
        role="distal_scaffold_context",
    ),
)
REGION_MSA_SUPPORT_REGION_IDS = tuple(spec.region_id for spec in REGION_MSA_SUPPORT_SPECS)


def build_region_msa_support_rows(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    conservation_profile_rows: Sequence[Mapping[str, object]],
    clade9_alignment_path: Path,
    subtype_alignment_path: Path,
    contact_geometry_rows: Sequence[Mapping[str, object]],
    mask_residues: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build candidate-by-region MSA support rows for the selected profile denominator.

    The C-terminal primer-RNA recognition region is an overlapping review
    context. Mutations there still remain in their distance/contact bucket.
    """

    profile_support = {
        CLADE9_PROFILE_ID: _profile_support(
            profile_id=CLADE9_PROFILE_ID,
            conservation_profile_rows=conservation_profile_rows,
            alignment_path=clade9_alignment_path,
        ),
        SUBTYPE_PROFILE_ID: _profile_support(
            profile_id=SUBTYPE_PROFILE_ID,
            conservation_profile_rows=conservation_profile_rows,
            alignment_path=subtype_alignment_path,
        ),
    }
    contact_by_position = {int(row["canonical_position"]): row for row in contact_geometry_rows}
    mask_by_position = {int(row["canonical_position"]): row for row in mask_residues}
    rows: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if str(candidate.get("status")) != "accepted":
            continue
        candidate_id = str(candidate["candidate_id"])
        policy_context = resolve_selection_policy_context(candidate)
        profile_id = policy_context.support_profile_id
        mutations = _parse_mutations(candidate.get("canonical_mutations"), candidate_id=candidate_id)
        mutations_by_region = {spec.region_id: [] for spec in REGION_MSA_SUPPORT_SPECS}
        for mutation in mutations:
            mutations_by_region[
                _region_id_for_mutation(
                    mutation,
                    contact_by_position=contact_by_position,
                    mask_by_position=mask_by_position,
                )
            ].append(mutation)
            if mutation.position in C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS:
                mutations_by_region["c_terminal_primer_rna_recognition_region"].append(mutation)
        for spec in REGION_MSA_SUPPORT_SPECS:
            support = _natural_support(mutations_by_region[spec.region_id], profile_support[profile_id])
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "policy_id": policy_context.policy_id,
                    "selection_support_policy_id": policy_context.policy_id,
                    "selection_support_policy_source": policy_context.source_field,
                    "selection_support_profile_id": profile_id,
                    "region_id": spec.region_id,
                    "region_label": spec.label,
                    "region_role": spec.role,
                    "mutation_count": len(mutations_by_region[spec.region_id]),
                    "supportable_mutation_count": support["supportable_mutation_count"],
                    "alt_observed_fraction": support["alt_observed_fraction"],
                    "alt_frequency_mean": support["alt_frequency_mean"],
                    "unobserved_mutation_count": support["unobserved_mutation_count"],
                    "rare_or_unobserved_mutation_count": support["rare_or_unobserved_mutation_count"],
                    "mutation_positions": _position_spec(mutations_by_region[spec.region_id]),
                }
            )
    return rows


def _region_id_for_mutation(
    mutation: Mutation,
    *,
    contact_by_position: Mapping[int, Mapping[str, object]],
    mask_by_position: Mapping[int, Mapping[str, object]],
) -> str:
    position = mutation.position
    mask = mask_by_position.get(position, {})
    contact = contact_by_position.get(position, {})
    distance = _float_or_none(contact.get("nearest_context_atom_distance_angstrom"))
    is_catalytic_or_direct = (
        bool(mask.get("motif_protected"))
        or bool(mask.get("wang_ec86_direct_contact_prior"))
        or bool(mask.get("direct_retained_dna_rna_contact_5a"))
        or (distance is not None and distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM)
    )
    if is_catalytic_or_direct:
        return "catalytic_or_direct_contact"
    if position in WANG_THUMB_CONTACT_TRACK_POSITIONS:
        return "thumb_contact_track"
    if distance is not None and distance <= NA_FACING_DISTANCE_ANGSTROM:
        return "near_retained_dna_rna_region"
    return "distal_scaffold"


def _position_spec(mutations: Sequence[Mutation]) -> str:
    ordered = sorted({mutation.position for mutation in mutations})
    if not ordered:
        return ""
    return ",".join(str(position) for position in ordered)


__all__ = [
    "REGION_MSA_SUPPORT_REGION_IDS",
    "REGION_MSA_SUPPORT_SPECS",
    "RegionMsaSupportSpec",
    "build_region_msa_support_rows",
]
