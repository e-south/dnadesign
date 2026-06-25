"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/structure.py

Protein-only backbone export helpers for ProteinMPNN requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from Bio.SeqUtils import seq1

from dnadesign.thread.adapters.proteinmpnn.models import ProteinMpnnBackboneExport

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def export_chain_backbone(
    *,
    model: Any,
    mapped_residue_rows: Sequence[Mapping[str, Any]],
    chain_id: str,
    output_path: Path,
    target_name: str,
) -> ProteinMpnnBackboneExport:
    """Write a protein-only PDB and a helper-compatible parsed JSON payload."""

    residue_index = protein_residue_index(model, chain_id=chain_id)
    sequence: list[str] = []
    coords_by_atom = {atom_name: [] for atom_name in BACKBONE_ATOMS}
    canonical_to_mpnn: dict[int, int] = {}
    pdb_lines: list[str] = []
    serial = 1

    for proteinmpnn_position, row in enumerate(mapped_residue_rows, start=1):
        canonical_position = _require_int(row, "canonical_position")
        structure_residue_id = _require_int(row, "structure_residue_id")
        insertion_code = str(row.get("pdb_insertion_code") or "")
        residue = residue_index.get((chain_id, structure_residue_id, insertion_code))
        if residue is None:
            raise ValueError(
                f"missing chain {chain_id} residue {structure_residue_id} for position {canonical_position}"
            )
        residue_name = str(residue.get_resname()).upper()
        sequence.append(seq1(residue_name, undef_code="X"))
        canonical_to_mpnn[canonical_position] = proteinmpnn_position
        for atom_name in BACKBONE_ATOMS:
            if atom_name not in residue:
                raise ValueError(f"residue {canonical_position} lacks backbone atom {atom_name}")
            atom = residue[atom_name]
            coord = [float(value) for value in atom.get_coord()]
            coords_by_atom[atom_name].append(coord)
            pdb_lines.append(_format_pdb_atom(serial, atom_name, residue_name, chain_id, proteinmpnn_position, coord))
            serial += 1
    output_path.write_text("".join(pdb_lines) + "TER\nEND\n", encoding="utf-8")
    sequence_text = "".join(sequence)
    parsed_payload = {
        "name": target_name,
        "num_of_chains": 1,
        "seq": sequence_text,
        f"seq_chain_{chain_id}": sequence_text,
        f"coords_chain_{chain_id}": {f"{atom}_chain_{chain_id}": coords for atom, coords in coords_by_atom.items()},
    }
    return ProteinMpnnBackboneExport(
        parsed_payload=parsed_payload,
        canonical_to_proteinmpnn_position=canonical_to_mpnn,
    )


def protein_residue_index(model: Any, *, chain_id: str) -> dict[tuple[str, int, str], Any]:
    """Index protein residues by chain id, residue id, and insertion code."""

    chain = model[chain_id]
    indexed: dict[tuple[str, int, str], Any] = {}
    for residue in chain.get_residues():
        insertion_code = "" if str(residue.id[2]).strip() == "" else str(residue.id[2]).strip()
        indexed[(chain_id, int(residue.id[1]), insertion_code)] = residue
    return indexed


def _format_pdb_atom(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    proteinmpnn_position: int,
    coord: Sequence[float],
) -> str:
    element = atom_name[0]
    return (
        f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} {chain_id:1s}"
        f"{proteinmpnn_position:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
    )


def _require_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value
