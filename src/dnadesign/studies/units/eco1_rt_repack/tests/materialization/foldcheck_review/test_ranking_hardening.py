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
