"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/surface_accessibility/rows.py

Per-residue SASA row construction for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.surface_accessibility.constants import (
    _BACKBONE_ATOM_NAMES,
)


def surface_row(*, residue: Mapping[str, Any], residue_index: Mapping[tuple[str, int, str], Any]) -> dict[str, Any]:
    """Build one canonical-position surface-accessibility row."""

    if residue.get("mapping_status") != "mapped":
        return unresolved_surface_row(residue)
    position = int(residue["canonical_position"])
    chain_id = str(residue["structure_chain_id"])
    residue_id = int(residue["structure_residue_id"])
    insertion_code = str(residue.get("pdb_insertion_code") or "")
    protein_residue = residue_index.get((chain_id, residue_id, insertion_code))
    if protein_residue is None:
        raise ValueError(f"mapped canonical position {position} is absent from selected mmCIF model")
    residue_sasa = _residue_sasa(protein_residue)
    backbone_sasa, sidechain_sasa = _atom_class_sasa(protein_residue)
    sidechain_status = _sidechain_status(protein_residue)
    return {
        "canonical_position": position,
        "wt_aa": str(residue["wt_aa"]),
        "structure_chain_id": chain_id,
        "structure_residue_id": residue_id,
        "pdb_insertion_code": insertion_code,
        "mapping_status": "mapped",
        "residue_sasa_angstrom2": _round_sasa(residue_sasa),
        "sidechain_sasa_angstrom2": _round_sasa(sidechain_sasa),
        "backbone_sasa_angstrom2": _round_sasa(backbone_sasa),
        "surface_accessibility_class": _surface_class(
            residue_sasa=residue_sasa,
            sidechain_sasa=sidechain_sasa,
            sidechain_status=sidechain_status,
        ),
        "sidechain_surface_status": sidechain_status,
    }


def unresolved_surface_row(residue: Mapping[str, Any]) -> dict[str, Any]:
    """Build an explicit no-SASA row for unresolved structure positions."""

    return {
        "canonical_position": int(residue["canonical_position"]),
        "wt_aa": str(residue["wt_aa"]),
        "structure_chain_id": "",
        "structure_residue_id": None,
        "pdb_insertion_code": "",
        "mapping_status": "unresolved_structure",
        "residue_sasa_angstrom2": None,
        "sidechain_sasa_angstrom2": None,
        "backbone_sasa_angstrom2": None,
        "surface_accessibility_class": "unresolved_structure",
        "sidechain_surface_status": "unresolved_structure",
    }


def _residue_sasa(residue: Any) -> float:
    value = getattr(residue, "sasa", None)
    if value is None:
        return sum(float(getattr(atom, "sasa", 0.0) or 0.0) for atom in residue.get_atoms())
    return float(value)


def _atom_class_sasa(residue: Any) -> tuple[float, float]:
    backbone = 0.0
    sidechain = 0.0
    for atom in residue.get_atoms():
        atom_sasa = float(getattr(atom, "sasa", 0.0) or 0.0)
        if str(atom.get_name()).strip() in _BACKBONE_ATOM_NAMES:
            backbone += atom_sasa
        else:
            sidechain += atom_sasa
    return backbone, sidechain


def _sidechain_status(residue: Any) -> str:
    residue_name = str(residue.get_resname()).upper()
    sidechain_atoms = [atom for atom in residue.get_atoms() if str(atom.get_name()).strip() not in _BACKBONE_ATOM_NAMES]
    if sidechain_atoms:
        return "materialized"
    if residue_name == "GLY":
        return "glycine_no_sidechain"
    return "no_sidechain_atoms"


def _surface_class(*, residue_sasa: float, sidechain_sasa: float, sidechain_status: str) -> str:
    if sidechain_status == "glycine_no_sidechain" and residue_sasa >= 30.0:
        return "glycine_surface_by_backbone"
    if residue_sasa >= 30.0 or sidechain_sasa >= 25.0:
        return "surface_exposed"
    return "buried_or_limited_access"


def _round_sasa(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)
