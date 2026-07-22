"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/structure_predictions/test_registry.py

Structure-prediction registry contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.thread.structure_predictions import (
    file_sha256_uri,
    validate_structure_prediction_registry,
    write_structure_prediction_registry,
)


def test_structure_prediction_registry_validates_empty_lookup_only_runs(tmp_path: Path) -> None:
    request_hash = "sha256:" + "1" * 64

    artifacts = write_structure_prediction_registry(output_root=tmp_path, rows=[], request_hash=request_hash)

    issues = validate_structure_prediction_registry(registry_path=artifacts.registry_path, request_hash=request_hash)

    assert issues == []


def test_structure_prediction_registry_validates_local_structure_hashes(tmp_path: Path) -> None:
    request_hash = "sha256:" + "1" * 64
    pdb_path = tmp_path / "structures" / "candidate_a.pdb"
    pdb_path.parent.mkdir()
    pdb_path.write_text(_PDB_TEXT, encoding="utf-8")
    row = {
        "candidate_id": "candidate_a",
        "sequence_hash": "sha256:" + "2" * 64,
        "prediction_id": "prediction_a",
        "prediction_set_id": "prediction_set_a",
        "backend_kind": "esm_atlas",
        "model_family": "esmfold_family",
        "model_name": "esm_atlas_structure_prediction",
        "model_version": "v1alpha1",
        "runtime_or_endpoint": "https://biohub.ai",
        "parameters_hash": "sha256:" + "3" * 64,
        "request_hash": request_hash,
        "source_request_hash": "sha256:" + "4" * 64,
        "raw_response_hash": "sha256:" + "5" * 64,
        "structure_hash": file_sha256_uri(pdb_path),
        "structure_source_uri": "atlas://fixture",
        "local_structure_path": str(pdb_path),
        "plddt": 86.0,
        "ptm": 0.91,
        "pae_summary_hash": "",
        "status": "accepted",
        "failure_reason": "",
    }

    artifacts = write_structure_prediction_registry(output_root=tmp_path, rows=[row], request_hash=request_hash)

    issues = validate_structure_prediction_registry(registry_path=artifacts.registry_path, request_hash=request_hash)

    assert issues == []


_PDB_TEXT = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 86.00           C\nEND\n"
