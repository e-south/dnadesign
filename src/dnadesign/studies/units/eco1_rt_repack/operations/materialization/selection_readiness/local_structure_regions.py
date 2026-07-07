"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/local_structure_regions.py

Eco1 RT local-structure region and threshold contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS,
    DIRECT_CONTACT_DISTANCE_ANGSTROM,
    NA_FACING_DISTANCE_ANGSTROM,
    WANG_THUMB_CONTACT_TRACK_POSITIONS,
)

LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID = "eco1_rt_local_structure_rmsd_gate_v1"
LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE = (
    "Declared local C-alpha RMSD limits set from the current all-candidate distribution and enforced as "
    "selection-readiness preservation gates, not activity evidence."
)
LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM = {
    "catalytic_initiation_context": 1.50,
    "retron_x_naxxh_context": 1.25,
    "retron_y_vtg_context": 1.60,
    "thumb_contact_track_context": 3.00,
    "c_terminal_primer_rna_recognition_context": 3.50,
    "near_retained_dna_rna_annulus": 3.00,
    "distal_scaffold_control": 4.75,
}


@dataclass(frozen=True)
class LocalStructureRegionSpec:
    """One named Eco1 RT region for local backbone-shift review."""

    region_id: str
    label: str
    role: str
    positions: tuple[int, ...]
    position_source: str
    source_basis_ids: tuple[str, ...]


_STATIC_REGION_SPECS = (
    LocalStructureRegionSpec(
        region_id="catalytic_initiation_context",
        label="Catalytic YADD context",
        role="catalytic_initiation_review",
        positions=tuple(range(189, 205)),
        position_source="explicit Eco1 residues 189-204 around the YADD catalytic motif",
        source_basis_ids=(
            "tao_et_al_2026_functional_residue_preservation",
            "simon_et_al_2019_retron_rt_motif_grammar",
            "wang_et_al_2022_ec86_cryoem_structure_priors",
        ),
    ),
    LocalStructureRegionSpec(
        region_id="retron_x_naxxh_context",
        label="Retron X NAxxH context",
        role="retron_motif_review",
        positions=tuple(range(99, 116)),
        position_source="explicit Eco1 residues 99-115 around the NAxxH retron X motif",
        source_basis_ids=(
            "simon_et_al_2019_retron_rt_motif_grammar",
            "wang_et_al_2022_ec86_cryoem_structure_priors",
        ),
    ),
    LocalStructureRegionSpec(
        region_id="retron_y_vtg_context",
        label="Retron Y VTG context",
        role="retron_motif_review",
        positions=tuple(range(237, 252)),
        position_source="explicit Eco1 residues 237-251 around the VTG retron Y motif",
        source_basis_ids=(
            "simon_et_al_2019_retron_rt_motif_grammar",
            "wang_et_al_2022_ec86_cryoem_structure_priors",
        ),
    ),
    LocalStructureRegionSpec(
        region_id="thumb_contact_track_context",
        label="Wang thumb-contact track",
        role="thumb_contact_review",
        positions=tuple(sorted(WANG_THUMB_CONTACT_TRACK_POSITIONS)),
        position_source="explicit Wang/Ec86 thumb-contact positions 238,239,240,249,257,261,264,298",
        source_basis_ids=("wang_et_al_2022_ec86_cryoem_structure_priors",),
    ),
    LocalStructureRegionSpec(
        region_id="c_terminal_primer_rna_recognition_context",
        label="C-terminal primer-RNA recognition region",
        role="c_terminal_primer_rna_recognition_review",
        positions=tuple(sorted(C_TERMINAL_PRIMER_RNA_RECOGNITION_POSITIONS)),
        position_source=(
            "explicit mapped Eco1 residues 255-311 from the RT-Ec86 C-terminal primer-RNA recognition "
            "context; canonical residues 312-320 are missing backbone in the current 7V9U-backed scope"
        ),
        source_basis_ids=(
            "inouye_et_al_1999_ec86_primer_template_recognition",
            "inouye_et_al_2004_ec86_thumb_primer_rna_binding",
            "wang_et_al_2022_ec86_cryoem_structure_priors",
        ),
    ),
)

LOCAL_STRUCTURE_REGION_IDS = tuple(
    spec.region_id
    for spec in (
        *_STATIC_REGION_SPECS,
        LocalStructureRegionSpec(
            region_id="near_retained_dna_rna_annulus",
            label="Near retained DNA/RNA region",
            role="substrate_proximal_review",
            positions=(),
            position_source="derived from retained DNA/RNA distance shell after exclusions",
            source_basis_ids=("wang_et_al_2022_ec86_cryoem_structure_priors",),
        ),
        LocalStructureRegionSpec(
            region_id="distal_scaffold_control",
            label="Distal scaffold control",
            role="distal_scaffold_control",
            positions=(),
            position_source=(
                "derived from mapped residues not assigned to motif, direct-contact, near DNA/RNA, "
                "or thumb-track regions"
            ),
            source_basis_ids=("wang_et_al_2022_ec86_cryoem_structure_priors",),
        ),
    )
)


def local_structure_region_specs(
    *,
    mapped_positions: Sequence[int],
    contact_geometry_rows: Sequence[Mapping[str, Any]],
) -> tuple[LocalStructureRegionSpec, ...]:
    """Return static and derived Eco1 local-structure regions."""

    mapped = set(int(position) for position in mapped_positions)
    exclusive_static_positions = {
        position
        for spec in _STATIC_REGION_SPECS
        if spec.region_id != "c_terminal_primer_rna_recognition_context"
        for position in spec.positions
    }
    direct_contact_positions: set[int] = set()
    near_na_positions: set[int] = set()
    for row in contact_geometry_rows:
        position = int(row["canonical_position"])
        distance = _retained_na_distance(row)
        if distance is None:
            continue
        if distance <= DIRECT_CONTACT_DISTANCE_ANGSTROM:
            direct_contact_positions.add(position)
        elif distance <= NA_FACING_DISTANCE_ANGSTROM:
            near_na_positions.add(position)
    thumb_positions = set(WANG_THUMB_CONTACT_TRACK_POSITIONS)
    near_na_positions = (
        (near_na_positions & mapped) - direct_contact_positions - exclusive_static_positions - thumb_positions
    )
    distal_positions = (
        mapped - exclusive_static_positions - thumb_positions - near_na_positions - direct_contact_positions
    )
    return (
        *_STATIC_REGION_SPECS,
        LocalStructureRegionSpec(
            region_id="near_retained_dna_rna_annulus",
            label="Near retained DNA/RNA region",
            role="substrate_proximal_review",
            positions=tuple(sorted(near_na_positions)),
            position_source=(
                f"derived Eco1 residues with retained DNA/RNA distance >{DIRECT_CONTACT_DISTANCE_ANGSTROM:g} A "
                f"and <={NA_FACING_DISTANCE_ANGSTROM:g} A, excluding motif contexts, direct contacts, "
                "and Wang thumb-contact positions"
            ),
            source_basis_ids=("wang_et_al_2022_ec86_cryoem_structure_priors",),
        ),
        LocalStructureRegionSpec(
            region_id="distal_scaffold_control",
            label="Distal scaffold control",
            role="distal_scaffold_control",
            positions=tuple(sorted(distal_positions)),
            position_source=(
                "derived mapped Eco1 residues outside motif contexts, direct contacts, the near retained DNA/RNA "
                "region, and Wang thumb-contact positions"
            ),
            source_basis_ids=("wang_et_al_2022_ec86_cryoem_structure_priors",),
        ),
    )


def position_spec(positions: Sequence[int]) -> str:
    """Serialize explicit Eco1 positions as compact one-indexed ranges."""

    ordered = sorted({int(position) for position in positions})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = ordered[0]
    previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        ranges.append(_format_range(start, previous))
        start = previous = position
    ranges.append(_format_range(start, previous))
    return ",".join(ranges)


def _retained_na_distance(row: Mapping[str, Any]) -> float | None:
    for field in ("nearest_context_atom_distance_angstrom", "distance_to_retained_na_angstrom"):
        value = row.get(field)
        if value is not None:
            return float(value)
    return None


def _format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


__all__ = [
    "LOCAL_STRUCTURE_REGION_IDS",
    "LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID",
    "LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE",
    "LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM",
    "LocalStructureRegionSpec",
    "local_structure_region_specs",
    "position_spec",
]
