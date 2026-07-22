"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/rows.py

Per-residue contact-geometry row construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.constants import (
    _CHAIN_COUNT_THRESHOLDS,
    _CONTACT_THRESHOLDS,
    threshold_id,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.models import (
    AtomSite,
    NearestAtomResult,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.structure_io import (
    heavy_atoms_for_residue,
    split_backbone_sidechain_atoms,
)


def geometry_row(
    *,
    residue: Mapping[str, Any],
    residue_index: Mapping[tuple[str, int, str], Any],
    context_atoms: list[AtomSite],
) -> dict[str, Any]:
    """Build one canonical-position atom-class contact-geometry row."""

    position = int(residue["canonical_position"])
    if residue.get("mapping_status") != "mapped":
        return unresolved_geometry_row(residue)

    chain_id = str(residue["structure_chain_id"])
    residue_id = int(residue["structure_residue_id"])
    insertion_code = str(residue.get("pdb_insertion_code") or "")
    protein_residue = residue_index.get((chain_id, residue_id, insertion_code))
    if protein_residue is None:
        raise ValueError(f"mapped canonical position {position} is absent from selected mmCIF model")

    residue_name = str(protein_residue.get_resname()).upper()
    all_atoms = heavy_atoms_for_residue(protein_residue)
    backbone_atoms, sidechain_atoms = split_backbone_sidechain_atoms(all_atoms)
    if not all_atoms:
        raise ValueError(f"mapped canonical position {position} has no heavy atoms in selected mmCIF model")
    sidechain_status = _sidechain_status(residue_name=residue_name, sidechain_atoms=sidechain_atoms)

    nearest_all = nearest_atom(all_atoms, context_atoms)
    nearest_sidechain = nearest_atom(sidechain_atoms, context_atoms)
    nearest_backbone = nearest_atom(backbone_atoms, context_atoms)
    nearest_dna = nearest_atom(all_atoms, [atom for atom in context_atoms if atom.molecule_type == "dna"])
    nearest_rna = nearest_atom(all_atoms, [atom for atom in context_atoms if atom.molecule_type == "rna"])
    contact_counts_by_threshold = contact_counts(all_atoms, context_atoms, thresholds=_CONTACT_THRESHOLDS)
    chain_counts_by_threshold = chain_counts(all_atoms, context_atoms, thresholds=_CHAIN_COUNT_THRESHOLDS)

    row: dict[str, Any] = {
        "canonical_position": position,
        "wt_aa": str(residue["wt_aa"]),
        "structure_chain_id": chain_id,
        "structure_residue_id": residue_id,
        "pdb_insertion_code": insertion_code,
        "mapping_status": "mapped",
        "nearest_context_atom_distance_angstrom": _round_distance(nearest_all.distance),
        "nearest_sidechain_context_distance_angstrom": _round_distance(nearest_sidechain.distance),
        "nearest_backbone_context_distance_angstrom": _round_distance(nearest_backbone.distance),
        "nearest_dna_distance_angstrom": _round_distance(nearest_dna.distance),
        "nearest_rna_distance_angstrom": _round_distance(nearest_rna.distance),
        "nearest_context_chain_id": "" if nearest_all.atom is None else nearest_all.atom.chain_id,
        "nearest_context_molecule_type": "" if nearest_all.atom is None else nearest_all.atom.molecule_type,
        "nearest_context_residue_id": None if nearest_all.atom is None else nearest_all.atom.residue_id,
        "nearest_context_residue_name": "" if nearest_all.atom is None else nearest_all.atom.residue_name,
        "nearest_context_atom_name": "" if nearest_all.atom is None else nearest_all.atom.atom_name,
        "sidechain_atom_status": sidechain_status,
    }
    for threshold, count in contact_counts_by_threshold.items():
        row[f"contact_atom_count_within_{threshold_id(threshold)}"] = count
    for threshold, count in chain_counts_by_threshold.items():
        row[f"retained_context_chain_count_within_{threshold_id(threshold)}"] = count
    return row


def unresolved_geometry_row(residue: Mapping[str, Any]) -> dict[str, Any]:
    """Build an explicit no-geometry row for unresolved structure positions."""

    row: dict[str, Any] = {
        "canonical_position": int(residue["canonical_position"]),
        "wt_aa": str(residue["wt_aa"]),
        "structure_chain_id": "",
        "structure_residue_id": None,
        "pdb_insertion_code": "",
        "mapping_status": "unresolved_structure",
        "nearest_context_atom_distance_angstrom": None,
        "nearest_sidechain_context_distance_angstrom": None,
        "nearest_backbone_context_distance_angstrom": None,
        "nearest_dna_distance_angstrom": None,
        "nearest_rna_distance_angstrom": None,
        "nearest_context_chain_id": "",
        "nearest_context_molecule_type": "",
        "nearest_context_residue_id": None,
        "nearest_context_residue_name": "",
        "nearest_context_atom_name": "",
        "sidechain_atom_status": "unresolved_structure",
    }
    for threshold in _CONTACT_THRESHOLDS:
        row[f"contact_atom_count_within_{threshold_id(threshold)}"] = 0
    for threshold in _CHAIN_COUNT_THRESHOLDS:
        row[f"retained_context_chain_count_within_{threshold_id(threshold)}"] = 0
    return row


def nearest_atom(residue_atoms: list[AtomSite], context_atoms: list[AtomSite]) -> NearestAtomResult:
    """Return the nearest retained-context atom for a residue atom class."""

    if not residue_atoms or not context_atoms:
        return NearestAtomResult(distance=None, atom=None)
    distances = distance_matrix(residue_atoms, context_atoms)
    flat_index = int(np.argmin(distances))
    _, context_index = np.unravel_index(flat_index, distances.shape)
    return NearestAtomResult(distance=float(distances.ravel()[flat_index]), atom=context_atoms[int(context_index)])


def contact_counts(
    residue_atoms: list[AtomSite],
    context_atoms: list[AtomSite],
    *,
    thresholds: Iterable[float],
) -> dict[float, int]:
    """Count retained-context atoms within each threshold of any residue atom."""

    if not residue_atoms or not context_atoms:
        return {threshold: 0 for threshold in thresholds}
    distances = distance_matrix(residue_atoms, context_atoms)
    min_per_context_atom = distances.min(axis=0)
    return {threshold: int(np.sum(min_per_context_atom <= threshold)) for threshold in thresholds}


def chain_counts(
    residue_atoms: list[AtomSite],
    context_atoms: list[AtomSite],
    *,
    thresholds: Iterable[float],
) -> dict[float, int]:
    """Count retained context chains within each threshold of any residue atom."""

    if not residue_atoms or not context_atoms:
        return {threshold: 0 for threshold in thresholds}
    distances = distance_matrix(residue_atoms, context_atoms)
    min_per_context_atom = distances.min(axis=0)
    counts: dict[float, int] = {}
    for threshold in thresholds:
        counts[threshold] = len(
            {
                atom.chain_id
                for atom, within_threshold in zip(context_atoms, min_per_context_atom <= threshold, strict=True)
                if bool(within_threshold)
            }
        )
    return counts


def distance_matrix(residue_atoms: list[AtomSite], context_atoms: list[AtomSite]) -> np.ndarray:
    """Compute all-by-all residue/context atom Euclidean distances."""

    residue_coords = np.stack([atom.coord for atom in residue_atoms])
    context_coords = np.stack([atom.coord for atom in context_atoms])
    deltas = residue_coords[:, None, :] - context_coords[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _sidechain_status(*, residue_name: str, sidechain_atoms: list[AtomSite]) -> str:
    if sidechain_atoms:
        return "materialized"
    if residue_name == "GLY":
        return "glycine_no_sidechain"
    return "no_sidechain_heavy_atoms"


def _round_distance(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)
