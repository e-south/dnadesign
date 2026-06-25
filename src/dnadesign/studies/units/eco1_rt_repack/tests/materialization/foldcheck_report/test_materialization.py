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

import pyarrow.parquet as pq
import yaml

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
from dnadesign.thread.foldcheck import write_foldcheck_report


def test_foldcheck_report_materializes_normalized_rows_from_colabfold_outputs(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    output_root = request.request_manifest_path.parent / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=90.0)
    _write_ca_pdb(
        output_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        y_offset=0.2,
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
    assert validate_foldcheck_report_content(result.foldcheck_report_path, output_root=tmp_path) == []


def test_foldcheck_report_validator_rejects_sequence_hash_drift(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)
    request = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)
    output_root = request.request_manifest_path.parent / "colabfold_outputs"
    output_root.mkdir()
    _write_ca_pdb(output_root / "wild_type_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb", bfactor=90.0)
    _write_ca_pdb(
        output_root / "thread_candidate_test_unrelaxed_rank_001_alphafold2_model_1_seed_000.pdb",
        bfactor=82.0,
        y_offset=0.2,
    )
    result = materialize_foldcheck_report(
        repo_root=Path.cwd(),
        output_root=tmp_path,
        colabfold_output_root=output_root,
        runtime_version="colabfold-test",
        runtime_parameters={"command": "colabfold_batch", "sequence_limit": 2},
    )
    rows = pq.read_table(result.foldcheck_report_path).to_pylist()
    rows[1]["input_sequence_hash"] = "sha256:" + "9" * 64
    manifest = yaml.safe_load(request.request_manifest_path.read_text(encoding="utf-8"))
    write_foldcheck_report(result.foldcheck_report_path, rows, request_hash=str(manifest["request_hash"]))

    issues = validate_foldcheck_report_content(result.foldcheck_report_path, output_root=tmp_path)

    assert [issue.check_id for issue in issues] == ["eco1_rt.foldcheck_report.sequence_hash_mismatch"]


def _write_ca_pdb(path: Path, *, bfactor: float, y_offset: float = 0.0) -> None:
    bend = 0.4 if y_offset == 0.0 else 0.8
    coords = [(0.0, y_offset, 0.0), (1.0, y_offset, 0.0), (2.0, y_offset, 0.0), (3.0, y_offset + bend, 0.0)]
    lines = []
    for index, (x_coord, y_coord, z_coord) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
            f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}  1.00{bfactor:6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
