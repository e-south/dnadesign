"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/contact_geometry/structure_io.py

Structure parsing and chain inventory helpers for contact geometry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.constants import (
    _BACKBONE_ATOM_NAMES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.models import AtomSite


def load_first_model(path: Path) -> Any:
    """Load the first model from the selected mmCIF structure."""

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("ec86_protomer1", str(path))
    return next(structure.get_models())


def retained_context_inventory(backbone_bundle: Mapping[str, Any]) -> dict[str, str]:
    """Return retained DNA/RNA chain ids from the materialized backbone bundle."""

    rows = backbone_bundle.get("chain_inventory")
    if not isinstance(rows, list):
        raise ValueError("backbone_bundle.yaml must declare chain_inventory before contact geometry materialization")
    retained: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("retention") != "retained" or row.get("thread_role") != "retained_context":
            continue
        chain_id = str(row.get("chain_id", "")).strip()
        molecule_type = str(row.get("molecule_type", "")).strip()
        if chain_id:
            retained[chain_id] = molecule_type
    if set(retained) != {"D", "E", "F"}:
        raise ValueError("contact geometry materialization requires retained context chains D/E/F")
    return retained


def validate_preprocessing_manifest(manifest: Mapping[str, Any], *, selected_source: Mapping[str, Any]) -> None:
    """Validate that preprocessing provenance still matches the selected structure source."""

    selected = manifest.get("selected_protomer")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_protomer must be a mapping")
    if selected.get("source_id") != selected_source.get("source_id"):
        raise ValueError("structure preprocessing manifest source_id must match selected structure source")
    if selected.get("rt_chain_id") != selected_source.get("rt_chain_id"):
        raise ValueError("structure preprocessing manifest rt_chain_id must match selected structure source")


def protein_residue_index(model: Any, *, rt_chain_id: str) -> dict[tuple[str, int, str], Any]:
    """Index protein residues by chain id, residue id, and insertion code."""

    chain = model[rt_chain_id]
    indexed: dict[tuple[str, int, str], Any] = {}
    for residue in chain.get_residues():
        insertion_code = "" if str(residue.id[2]).strip() == "" else str(residue.id[2]).strip()
        indexed[(rt_chain_id, int(residue.id[1]), insertion_code)] = residue
    return indexed


def context_atoms(model: Any, *, retained_context: Mapping[str, str]) -> list[AtomSite]:
    """Extract retained DNA/RNA heavy atoms from selected context chains."""

    atoms: list[AtomSite] = []
    for chain in model:
        chain_id = str(chain.id)
        molecule_type = retained_context.get(chain_id)
        if molecule_type not in {"dna", "rna"}:
            continue
        for residue in chain.get_residues():
            residue_id = int(residue.id[1])
            residue_name = str(residue.get_resname()).upper()
            for atom in residue.get_atoms():
                element = str(getattr(atom, "element", "") or "").strip().upper()
                atom_name = str(atom.get_name()).strip()
                if element == "H" or atom_name.startswith("H"):
                    continue
                atoms.append(
                    AtomSite(
                        coord=np.asarray(atom.get_coord(), dtype=float),
                        chain_id=chain_id,
                        molecule_type=str(molecule_type),
                        residue_id=residue_id,
                        residue_name=residue_name,
                        atom_name=atom_name,
                    )
                )
    return atoms


def heavy_atoms_for_residue(residue: Any) -> list[AtomSite]:
    """Extract protein heavy atoms from one Biopython residue."""

    chain_id = str(residue.get_parent().id)
    residue_id = int(residue.id[1])
    residue_name = str(residue.get_resname()).upper()
    atoms: list[AtomSite] = []
    for atom in residue.get_atoms():
        element = str(getattr(atom, "element", "") or "").strip().upper()
        atom_name = str(atom.get_name()).strip()
        if element == "H" or atom_name.startswith("H"):
            continue
        atoms.append(
            AtomSite(
                coord=np.asarray(atom.get_coord(), dtype=float),
                chain_id=chain_id,
                molecule_type="protein",
                residue_id=residue_id,
                residue_name=residue_name,
                atom_name=atom_name,
            )
        )
    return atoms


def split_backbone_sidechain_atoms(atoms: list[AtomSite]) -> tuple[list[AtomSite], list[AtomSite]]:
    """Split protein heavy atoms into backbone and side-chain classes."""

    backbone_atoms = [atom for atom in atoms if atom.atom_name in _BACKBONE_ATOM_NAMES]
    sidechain_atoms = [atom for atom in atoms if atom.atom_name not in _BACKBONE_ATOM_NAMES]
    return backbone_atoms, sidechain_atoms
