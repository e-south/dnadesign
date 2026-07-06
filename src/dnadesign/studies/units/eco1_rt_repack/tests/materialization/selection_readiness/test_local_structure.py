"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_local_structure.py

Local-structure review metric tests for Eco1 RT repack selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    build_local_structure_region_rows,
)


def test_local_structure_region_rows_use_one_global_alignment(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.pdb"
    model_root = tmp_path / "models"
    model_root.mkdir()
    mapped_positions = list(range(1, 321))
    reference_rows = [(index, float(index), float(index % 17), float(index % 31)) for index in mapped_positions]
    _write_ca_pdb(reference_path, reference_rows)
    _write_ca_pdb(
        model_root / "candidate_a.pdb",
        [(index, x + 10.0, y - 4.0, z + 2.0) for index, x, y, z in reference_rows],
    )

    rows = build_local_structure_region_rows(
        fold_review_rows=[
            {
                "candidate_id": "candidate_a",
                "design_class_id": "class_a",
                "model_artifact_path": "candidate_a.pdb",
            }
        ],
        reference_backbone_path=reference_path,
        model_root=model_root,
        mapped_positions=mapped_positions,
        contact_geometry_rows=_contact_geometry_rows(mapped_positions),
    )

    available = [row for row in rows if row["status"] == "available"]
    assert {row["region_id"] for row in rows} == set(LOCAL_STRUCTURE_REGION_IDS)
    assert len(available) == len(LOCAL_STRUCTURE_REGION_IDS)
    assert max(float(row["local_ca_rmsd_angstrom"]) for row in available) == pytest.approx(0.0, abs=1e-6)
    assert max(float(row["mean_ca_displacement_angstrom"]) for row in available) == pytest.approx(0.0, abs=1e-6)


def test_local_structure_region_rows_report_missing_models(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.pdb"
    model_root = tmp_path / "models"
    model_root.mkdir()
    mapped_positions = list(range(1, 321))
    _write_ca_pdb(reference_path, [(index, float(index), 0.0, 0.0) for index in mapped_positions])

    rows = build_local_structure_region_rows(
        fold_review_rows=[
            {
                "candidate_id": "candidate_missing",
                "design_class_id": "class_a",
                "model_artifact_path": "candidate_missing.pdb",
            }
        ],
        reference_backbone_path=reference_path,
        model_root=model_root,
        mapped_positions=mapped_positions,
        contact_geometry_rows=_contact_geometry_rows(mapped_positions),
    )

    assert {row["status"] for row in rows} == {"model_structure_missing"}
    assert all(row["local_ca_rmsd_angstrom"] is None for row in rows)
    assert all("candidate_missing.pdb" in str(row["status_reason"]) for row in rows)


def test_local_structure_region_rows_report_insufficient_overlap(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.pdb"
    model_root = tmp_path / "models"
    model_root.mkdir()
    mapped_positions = list(range(1, 321))
    _write_ca_pdb(reference_path, [(index, float(index), 0.0, 0.0) for index in mapped_positions])
    _write_ca_pdb(model_root / "candidate_short.pdb", [(index, float(index), 0.0, 0.0) for index in range(1, 3)])

    rows = build_local_structure_region_rows(
        fold_review_rows=[
            {
                "candidate_id": "candidate_short",
                "design_class_id": "class_a",
                "model_artifact_path": "candidate_short.pdb",
            }
        ],
        reference_backbone_path=reference_path,
        model_root=model_root,
        mapped_positions=mapped_positions,
        contact_geometry_rows=_contact_geometry_rows(mapped_positions),
    )

    assert {row["status"] for row in rows} == {"insufficient_alignment_overlap"}
    assert all(row["n_shared_ca"] < 3 for row in rows)


def _write_ca_pdb(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        (f"ATOM  {index:5d}  CA  ALA A{residue:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n")
        for index, (residue, x, y, z) in enumerate(rows, start=1)
    ]
    path.write_text("".join(lines) + "END\n", encoding="utf-8")


def _contact_geometry_rows(mapped_positions: list[int]) -> list[dict[str, object]]:
    return [
        {
            "canonical_position": position,
            "nearest_context_atom_distance_angstrom": 8.0 if position in {70, 71, 72, 73} else 14.0,
        }
        for position in mapped_positions
    ]
