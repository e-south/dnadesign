"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_local_structure_fixtures.py

Local-structure fixture helpers for Eco1 selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def write_local_structure_inputs(class_root: Path, candidates: list[dict[str, object]]) -> None:
    """Write compact reference and candidate C-alpha PDB fixtures."""

    structure_root = class_root / "foldcheck_review/structures"
    model_root = structure_root / "full_fold_set"
    model_root.mkdir(parents=True, exist_ok=True)
    rows = [(position, float(position), float(position % 17), float(position % 29)) for position in range(1, 321)]
    _write_ca_pdb(structure_root / "ec86kit_chain_a_backbone_reference.pdb", rows)
    for index, candidate in enumerate(candidates, start=1):
        shift = float(index) / 100.0
        _write_ca_pdb(
            model_root / f"{candidate['candidate_id']}.pdb",
            [(position, x + shift, y - shift, z + shift) for position, x, y, z in rows],
        )


def _write_ca_pdb(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        (f"ATOM  {index:5d}  CA  ALA A{residue:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
        for index, (residue, x, y, z) in enumerate(rows, start=1)
    ]
    path.write_text("".join(lines) + "END\n", encoding="utf-8")
