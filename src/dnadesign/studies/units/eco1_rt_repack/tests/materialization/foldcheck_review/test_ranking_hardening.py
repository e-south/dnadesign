"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_review/test_ranking_hardening.py

Foldcheck-review ranking hardening tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review import (
    materialize_foldcheck_review,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_review.fixtures import (
    write_review_inputs,
)
from dnadesign.thread.adapters.colabfold.manifest import file_sha256_uri, ordered_positions_hash
from dnadesign.thread.adapters.colabfold.outputs import MAPPED_REFERENCE_COORDINATE_BASIS
from dnadesign.thread.foldcheck import write_foldcheck_report


def test_foldcheck_review_rejects_reference_coordinate_basis_mismatch(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    reference_path = tmp_path / "proteinmpnn_request" / "chain_a_backbone.pdb"
    reference_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference backbone CA count"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)


def test_foldcheck_review_preserves_empty_model_artifact_as_missing(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    rows = pq.read_table(foldcheck_report_path).to_pylist()
    for row in rows:
        if row["candidate_id"] == "thread_candidate_best_rmsd":
            row["model_artifact_path"] = ""
            row["status"] = "errored"
            row["plddt"] = None
            row["backbone_rmsd_to_reference"] = None
    pq.write_table(pa.Table.from_pylist(rows), foldcheck_report_path)

    result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)

    ranking_rows = {
        row["candidate_id"]: row
        for row in pq.read_table(
            result.ranking_path, columns=["candidate_id", "cryoem_mapped_ca_rmsd_status"]
        ).to_pylist()
    }
    assert ranking_rows["thread_candidate_best_rmsd"]["cryoem_mapped_ca_rmsd_status"] == "model_artifact_missing"


def test_foldcheck_review_keeps_reference_and_wt_baseline_rmsd_distinct(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=False)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    reference_path = tmp_path / "proteinmpnn_request/chain_a_backbone.pdb"
    mapped_positions = list(range(3, 312))
    rows = pq.read_table(foldcheck_report_path).to_pylist()
    for row in rows:
        row["backbone_rmsd_to_wt_baseline"] = row["backbone_rmsd_to_reference"]
        row["reference_structure_hash"] = file_sha256_uri(reference_path)
        row["reference_mobile_positions_hash"] = ordered_positions_hash(mapped_positions)
        row["reference_coordinate_basis"] = MAPPED_REFERENCE_COORDINATE_BASIS
    candidate = next(row for row in rows if row["candidate_id"] == "thread_candidate_best_rmsd")
    candidate["backbone_rmsd_to_reference"] = 4.25
    candidate["backbone_rmsd_to_wt_baseline"] = 0.8
    write_foldcheck_report(foldcheck_report_path, rows, request_hash="sha256:" + "8" * 64)

    result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)

    ranking_rows = {row["candidate_id"]: row for row in pq.read_table(result.ranking_path).to_pylist()}
    ranked = ranking_rows["thread_candidate_best_rmsd"]
    assert ranked["wt_runtime_ca_rmsd"] == 0.8
    assert ranked["cryoem_mapped_ca_rmsd"] == 4.25
    assert ranked["cryoem_mapped_ca_rmsd_status"] == "available"
    assert ranked["review_class"] == "strong_fold_preserved"


def test_foldcheck_review_rejects_v2_reference_lineage_drift(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=False)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    rows = pq.read_table(foldcheck_report_path).to_pylist()
    for row in rows:
        row["backbone_rmsd_to_wt_baseline"] = row["backbone_rmsd_to_reference"]
        row["reference_structure_hash"] = "sha256:" + "1" * 64
        row["reference_mobile_positions_hash"] = ordered_positions_hash(range(3, 312))
        row["reference_coordinate_basis"] = MAPPED_REFERENCE_COORDINATE_BASIS
    write_foldcheck_report(foldcheck_report_path, rows, request_hash="sha256:" + "8" * 64)

    with pytest.raises(ValueError, match="does not match the current Eco1 authority"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)


def test_foldcheck_review_rejects_unknown_report_schema_version(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    table = pq.read_table(foldcheck_report_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"schema_version"] = b"3"
    pq.write_table(table.replace_schema_metadata(metadata), foldcheck_report_path)

    with pytest.raises(ValueError, match="unsupported foldcheck report schema version"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)


def test_foldcheck_review_rejects_v2_fields_without_v2_metadata(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=False)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    reference_path = tmp_path / "proteinmpnn_request/chain_a_backbone.pdb"
    rows = pq.read_table(foldcheck_report_path).to_pylist()
    for row in rows:
        row["backbone_rmsd_to_wt_baseline"] = row["backbone_rmsd_to_reference"]
        row["reference_structure_hash"] = file_sha256_uri(reference_path)
        row["reference_mobile_positions_hash"] = ordered_positions_hash(range(3, 312))
        row["reference_coordinate_basis"] = MAPPED_REFERENCE_COORDINATE_BASIS
    write_foldcheck_report(foldcheck_report_path, rows, request_hash="sha256:" + "8" * 64)
    table = pq.read_table(foldcheck_report_path)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"schema_version")
    pq.write_table(table.replace_schema_metadata(metadata), foldcheck_report_path)

    with pytest.raises(ValueError, match="v1 foldcheck report contains v2-only"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)
