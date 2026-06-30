"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_review/test_contract_failures.py

Negative contract tests for Eco1 fold-check review materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review import (
    materialize_foldcheck_review,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_review.fixtures import (
    write_review_inputs,
)


def test_foldcheck_review_rejects_duplicate_fold_rows(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    foldcheck_report_path = tmp_path / "foldcheck_report.parquet"
    table = pq.read_table(foldcheck_report_path)
    rows = table.to_pylist()
    rows.append(dict(rows[-1]))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), foldcheck_report_path)

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)


def test_foldcheck_review_rejects_unverified_stale_local_structure(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=False)
    stale_path = tmp_path / "foldcheck_review" / "structures" / "full_fold_set" / "thread_candidate_best_rmsd.pdb"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unverified staged model"):
        materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)


def test_foldcheck_review_reuses_verified_staged_structure_after_request_hash_change(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    first_result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)
    first_full_set = yaml.safe_load(first_result.full_structure_set_path.read_text(encoding="utf-8"))
    assert first_full_set["copy_summary"] == {"copied": 7}

    request_manifest_path = tmp_path / "foldcheck_request" / "foldcheck_request_manifest.yaml"
    request_manifest = yaml.safe_load(request_manifest_path.read_text(encoding="utf-8"))
    request_manifest["request_hash"] = "sha256:" + "9" * 64
    request_manifest_path.write_text(yaml.safe_dump(request_manifest, sort_keys=False), encoding="utf-8")
    shutil.rmtree(tmp_path / "colabfold_models")

    second_result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)
    second_full_set = yaml.safe_load(second_result.full_structure_set_path.read_text(encoding="utf-8"))

    assert second_full_set["source_request_hash"] == "sha256:" + "9" * 64
    assert second_full_set["copy_summary"] == {"already_local_verified": 7}
