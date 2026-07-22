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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_GATE_REGION_IDS,
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
    build_local_structure_region_rows,
)


def test_local_structure_gate_uses_one_declared_cutoff_and_keeps_distal_rmsd_for_review() -> None:
    assert {LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM[region_id] for region_id in LOCAL_STRUCTURE_GATE_REGION_IDS} == {
        2.5
    }
    assert LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM["distal_scaffold_control"] is None


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
                "policy_id": COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
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
    assert all(str(row["region_position_spec"]) for row in rows)
    assert all(int(row["region_position_count"]) > 0 for row in rows)
    assert all(str(row["region_position_source"]) for row in rows)
    assert all(str(row["region_source_basis_ids_json"]) for row in rows)
    assert all(row["local_ca_rmsd_threshold_policy_id"] == LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID for row in rows)
    assert all(
        row["local_ca_rmsd_threshold_status"] == "passed"
        for row in rows
        if row["region_id"] in LOCAL_STRUCTURE_GATE_REGION_IDS
    )
    distal = next(row for row in rows if row["region_id"] == "distal_scaffold_control")
    assert distal["local_ca_rmsd_threshold_status"] == "review_only"
    catalytic = next(row for row in rows if row["region_id"] == "catalytic_initiation_context")
    c_terminal = next(row for row in rows if row["region_id"] == "c_terminal_primer_rna_recognition_context")
    assert catalytic["region_position_spec"] == "189-204"
    assert "YADD" in str(catalytic["region_position_source"])
    assert "tao_et_al_2026_functional_residue_preservation" in str(catalytic["region_source_basis_ids_json"])
    assert (
        catalytic["local_ca_rmsd_threshold_angstrom"]
        == LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM["catalytic_initiation_context"]
    )
    assert c_terminal["region_position_spec"] == "255-311"
    assert "primer-template recognition" in str(c_terminal["region_position_source"])
    assert "inouye_et_al_2004_ec86_thumb_primer_rna_binding" in str(c_terminal["region_source_basis_ids_json"])


def test_local_structure_region_rows_do_not_refit_each_region(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.pdb"
    model_root = tmp_path / "models"
    model_root.mkdir()
    mapped_positions = list(range(1, 321))
    reference_rows = [(index, float(index), float(index % 17), float(index % 31)) for index in mapped_positions]
    shifted_rows = [
        (index, x, y + 3.0, z) if 189 <= index <= 204 else (index, x, y, z) for index, x, y, z in reference_rows
    ]
    _write_ca_pdb(reference_path, reference_rows)
    _write_ca_pdb(model_root / "candidate_local_shift.pdb", shifted_rows)

    rows = build_local_structure_region_rows(
        fold_review_rows=[
            {
                "candidate_id": "candidate_local_shift",
                "policy_id": COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
                "model_artifact_path": "candidate_local_shift.pdb",
            }
        ],
        reference_backbone_path=reference_path,
        model_root=model_root,
        mapped_positions=mapped_positions,
        contact_geometry_rows=_contact_geometry_rows(mapped_positions),
    )

    catalytic = next(row for row in rows if row["region_id"] == "catalytic_initiation_context")
    distal = next(row for row in rows if row["region_id"] == "distal_scaffold_control")
    assert catalytic["coordinate_scope"] == "mapped_rt_chain_ca_after_global_fit"
    assert float(catalytic["local_ca_rmsd_angstrom"]) > 2.0
    assert float(distal["local_ca_rmsd_angstrom"]) < float(catalytic["local_ca_rmsd_angstrom"])


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
                "policy_id": COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
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
    assert all(row["local_ca_rmsd_threshold_status"] == "not_evaluated" for row in rows)
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
                "policy_id": COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
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
