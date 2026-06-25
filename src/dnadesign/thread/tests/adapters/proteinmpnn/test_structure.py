"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/proteinmpnn/test_structure.py

ProteinMPNN backbone export structure tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from Bio.PDB import PDBParser

from dnadesign.thread.adapters.proteinmpnn.structure import export_chain_backbone


def test_export_chain_backbone_ignores_hetero_residue_collisions(tmp_path: Path) -> None:
    parser = PDBParser(QUIET=True)
    model = parser.get_structure(
        "hetero_collision",
        StringIO(
            "\n".join(
                [
                    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
                    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C",
                    "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O",
                    "HETATM    5  O   HOH A   1       9.000   9.000   9.000  1.00  0.00           O",
                    "END",
                ]
            )
        ),
    )[0]

    result = export_chain_backbone(
        model=model,
        mapped_residue_rows=[
            {
                "canonical_position": 10,
                "structure_residue_id": 1,
                "pdb_insertion_code": "",
            }
        ],
        chain_id="A",
        output_path=tmp_path / "chain_a_backbone.pdb",
        target_name="target",
    )

    assert result.parsed_payload["seq"] == "A"
    assert result.canonical_to_proteinmpnn_position == {10: 1}
