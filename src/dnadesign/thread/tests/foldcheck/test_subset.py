"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/foldcheck/test_subset.py

Fold-check request subset tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.thread.foldcheck.hashes import sequence_hash
from dnadesign.thread.foldcheck.subset import materialize_foldcheck_sequence_subset


def test_materialize_foldcheck_sequence_subset_writes_fasta_and_manifest(tmp_path: Path) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    _write_request_fixture(request_dir)

    run_manifest = materialize_foldcheck_sequence_subset(
        request_manifest_path=request_dir / "foldcheck_request_manifest.yaml",
        sequence_limit="2",
        input_fasta_path=tmp_path / "run/input_sequences.fasta",
        run_manifest_path=tmp_path / "run/colabfold_run_manifest.yaml",
        output_dir=tmp_path / "run/colabfold_outputs",
        schema_id="eco1_rt.colabfold_scc_run_manifest",
        execution_status="submitted_to_external_colabfold_cli",
    )

    assert (tmp_path / "run/input_sequences.fasta").read_text(encoding="utf-8") == (
        ">wild_type\nACDE\n>candidate_a\nACDF\n"
    )
    manifest = yaml.safe_load((tmp_path / "run/colabfold_run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest == run_manifest
    assert manifest["schema_id"] == "eco1_rt.colabfold_scc_run_manifest"
    assert manifest["source_sequence_count"] == 3
    assert manifest["selected_sequence_count"] == 2
    assert manifest["selected_sequence_ids"] == ["wild_type", "candidate_a"]


def test_materialize_foldcheck_sequence_subset_rejects_hash_mismatch(tmp_path: Path) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    _write_request_fixture(request_dir, candidate_a_sequence="ACDG")

    try:
        materialize_foldcheck_sequence_subset(
            request_manifest_path=request_dir / "foldcheck_request_manifest.yaml",
            sequence_limit="all",
            input_fasta_path=tmp_path / "run/input_sequences.fasta",
            run_manifest_path=tmp_path / "run/colabfold_run_manifest.yaml",
            output_dir=tmp_path / "run/colabfold_outputs",
        )
    except ValueError as error:
        assert "sequence hash mismatch" in str(error)
    else:
        raise AssertionError("Expected stale FASTA hash to fail")


def test_materialize_foldcheck_sequence_subset_rejects_oversized_limit(tmp_path: Path) -> None:
    request_dir = tmp_path / "request"
    request_dir.mkdir()
    _write_request_fixture(request_dir)

    try:
        materialize_foldcheck_sequence_subset(
            request_manifest_path=request_dir / "foldcheck_request_manifest.yaml",
            sequence_limit="4",
            input_fasta_path=tmp_path / "run/input_sequences.fasta",
            run_manifest_path=tmp_path / "run/colabfold_run_manifest.yaml",
            output_dir=tmp_path / "run/colabfold_outputs",
        )
    except ValueError as error:
        assert "exceeds request sequence count" in str(error)
    else:
        raise AssertionError("Expected oversized sequence limit to fail")


def _write_request_fixture(request_dir: Path, *, candidate_a_sequence: str = "ACDF") -> None:
    fasta = request_dir / "input_sequences.fasta"
    fasta.write_text(
        f">wild_type\nACDE\n>candidate_a\n{candidate_a_sequence}\n>candidate_b\nACDG\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_id": "thread.foldcheck_request",
        "request_hash": "sha256:" + "1" * 64,
        "sequence_count": 3,
        "input_fasta_path": "input_sequences.fasta",
        "sequences": [
            {"sequence_id": "wild_type", "sequence_hash": sequence_hash("ACDE")},
            {"sequence_id": "candidate_a", "sequence_hash": sequence_hash("ACDF")},
            {"sequence_id": "candidate_b", "sequence_hash": sequence_hash("ACDG")},
        ],
    }
    (request_dir / "foldcheck_request_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
