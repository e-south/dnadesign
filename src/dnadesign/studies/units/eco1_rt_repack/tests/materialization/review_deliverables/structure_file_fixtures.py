"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/structure_file_fixtures.py

PDB and mmCIF fixture writers for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_pdb(
    path: Path,
    *,
    residue_count: int,
    coordinate_offset: float = 0.0,
    include_sidechains: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    atom_index = 1
    for residue_index in range(1, residue_count + 1):
        x_coord = float(residue_index) + coordinate_offset
        lines.append(
            f"ATOM  {atom_index:5d}  CA  GLY A{residue_index:4d}    "
            f"{x_coord:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C"
        )
        atom_index += 1
        if include_sidechains:
            lines.append(
                f"ATOM  {atom_index:5d}  CB  ALA A{residue_index:4d}    "
                f"{x_coord:8.3f}{1.5:8.3f}{0.0:8.3f}  1.00 80.00           C"
            )
            atom_index += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mmcif_all_atom_reference(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["data_ec86_fixture", "#", "loop_"]
    atom_index = 1
    for residue_index in range(1, 310):
        x_coord = float(residue_index)
        for atom_name, y_coord in (("N", -0.8), ("CA", 0.0), ("C", 0.8), ("O", 1.2), ("CB", -1.2)):
            lines.append(
                " ".join(
                    [
                        "ATOM",
                        str(atom_index),
                        atom_name[0],
                        atom_name,
                        ".",
                        "ALA",
                        "A",
                        "1",
                        str(residue_index),
                        f"{x_coord:.3f}",
                        f"{y_coord:.3f}",
                        "0.000",
                        "A",
                        str(residue_index),
                        "?",
                        "1.00",
                        "80.00",
                        "1",
                    ]
                )
            )
            atom_index += 1
    lines.append(f"HETATM {atom_index} P P . DA D 2 1 0.000 0.000 0.000 D 1 ? 1.00 80.00 1")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
