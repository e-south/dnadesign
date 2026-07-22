"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_report/test_materialization.py

Eco1 fold-check report materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.foldcheck import (
    validate_foldcheck_report_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_report import (
    materialize_foldcheck_report,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request import (
    materialize_foldcheck_request,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_request._fixtures import (
    write_minimal_foldcheck_inputs,
)


def test_foldcheck_report_materializes_normalized_rows_from_colabfold_outputs(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    output_root = request.request_manifest_path.parent / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(
        tmp_path / "proteinmpnn_request/chain_a_backbone.pdb",
        bfactor=0.0,
        y_offset=1.0,
        residue_count=309,
    )
    _write_ca_pdb(
        output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=90.0,
        residue_count=320,
    )
    _write_ca_pdb(
        output_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        y_offset=0.2,
        residue_count=320,
    )

    result = materialize_foldcheck_report(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        colabfold_output_root=output_root,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "sequence_limit": 2},
    )

    rows = pq.read_table(result.foldcheck_report_path).to_pylist()
    assert [row["candidate_id"] for row in rows] == ["wild_type", "thread_candidate_test"]
    assert {row["status"] for row in rows} == {"accepted"}
    assert rows[0]["backbone_rmsd_to_reference"] > 0.0
    assert rows[0]["backbone_rmsd_to_wt_baseline"] == 0.0
    assert rows[1]["backbone_rmsd_to_wt_baseline"] > 0.0
    assert pq.read_schema(result.foldcheck_report_path).metadata[b"schema_version"] == b"2"
    assert (
        validate_foldcheck_report_content(
            result.foldcheck_report_path,
            repo_root=Path.cwd(),
            output_root=tmp_path,
        )
        == []
    )

    reference_path = tmp_path / "proteinmpnn_request/chain_a_backbone.pdb"
    original_reference = reference_path.read_bytes()
    reference_path.write_bytes(original_reference + b"REMARK drift\n")
    drift_issues = validate_foldcheck_report_content(
        result.foldcheck_report_path,
        repo_root=Path.cwd(),
        output_root=tmp_path,
    )
    assert "eco1_rt.foldcheck_report.reference_structure_hash_mismatch" in {issue.check_id for issue in drift_issues}

    reference_path.write_bytes(original_reference)
    residue_map_path = tmp_path / "residue_map.parquet"
    residue_map_rows = pq.read_table(residue_map_path).to_pylist()
    mapped_row = next(row for row in residue_map_rows if row["mapping_status"] == "mapped")
    mapped_row["mapping_status"] = "unresolved"
    pq.write_table(pa.Table.from_pylist(residue_map_rows), residue_map_path)
    position_drift_issues = validate_foldcheck_report_content(
        result.foldcheck_report_path,
        repo_root=Path.cwd(),
        output_root=tmp_path,
    )
    assert "eco1_rt.foldcheck_report.reference_mobile_positions_hash_mismatch" in {
        issue.check_id for issue in position_drift_issues
    }


def test_foldcheck_report_requires_explicit_reference_backbone(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    output_root = request.request_manifest_path.parent / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(
        output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=90.0,
        residue_count=320,
    )

    with pytest.raises(FileNotFoundError, match="chain_a_backbone.pdb"):
        materialize_foldcheck_report(
            repo_root=Path.cwd(),
            output_root=tmp_path,
            source_output_root=tmp_path,
            colabfold_output_root=output_root,
            runtime_version="colabfold-test",
        )


def test_foldcheck_report_resolves_reference_authority_from_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "policy"
    write_minimal_foldcheck_inputs(source_root)
    _write_ca_pdb(
        source_root / "proteinmpnn_request/chain_a_backbone.pdb",
        bfactor=0.0,
        y_offset=1.0,
        residue_count=309,
    )
    write_minimal_foldcheck_inputs(output_root)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=output_root)
    (output_root / "residue_map.parquet").unlink()
    colabfold_root = request.request_manifest_path.parent / "colabfold_outputs"
    colabfold_root.mkdir()
    _write_ca_pdb(
        colabfold_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=90.0,
        residue_count=320,
    )
    _write_ca_pdb(
        colabfold_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        y_offset=0.2,
        residue_count=320,
    )

    result = materialize_foldcheck_report(
        repo_root=Path.cwd(),
        output_root=output_root,
        source_output_root=source_root,
        colabfold_output_root=colabfold_root,
        runtime_version="colabfold-test",
    )

    assert {row["status"] for row in pq.read_table(result.foldcheck_report_path).to_pylist()} == {"accepted"}


def _write_ca_pdb(
    path: Path,
    *,
    bfactor: float,
    y_offset: float = 0.0,
    residue_count: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bend = 0.4 if y_offset == 0.0 else 0.8
    coords = [
        (float(index), y_offset + (bend if index == residue_count - 1 else 0.0), 0.0) for index in range(residue_count)
    ]
    lines = []
    for index, (x_coord, y_coord, z_coord) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
            f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00{bfactor:6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
