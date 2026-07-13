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
    lines = [
        "data_ec86_fixture",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_PDB_model_num",
    ]
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
    for chain_id, residue_name, y_offset in (("D", "DA", 0.0), ("E", "U", 6.0), ("F", "U", 12.0)):
        for residue_index in range(1, 4):
            x_coord = float(residue_index) * 3.0
            for atom_name, element, dx, dy in (
                ("P", "P", 0.0, 0.0),
                ("O5'", "O", 0.6, 0.2),
                ("C5'", "C", 1.1, 0.3),
                ("C4'", "C", 1.5, 0.8),
                ("C3'", "C", 2.0, 0.4),
                ("O3'", "O", 2.5, 0.1),
                ("N1", "N", 1.5, 1.8),
            ):
                lines.append(
                    " ".join(
                        [
                            "HETATM",
                            str(atom_index),
                            element,
                            atom_name,
                            ".",
                            residue_name,
                            chain_id,
                            "2",
                            str(residue_index),
                            f"{x_coord + dx:.3f}",
                            f"{y_offset + dy:.3f}",
                            "0.000",
                            chain_id,
                            str(residue_index),
                            "?",
                            "1.00",
                            "80.00",
                            "1",
                        ]
                    )
                )
                atom_index += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
