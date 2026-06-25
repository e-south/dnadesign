"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/test_proteinmpnn_candidates.py

ProteinMPNN candidate table tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.thread.adapters.proteinmpnn.samples import write_sample_table
from dnadesign.thread.candidates.proteinmpnn import (
    build_proteinmpnn_candidate_rows,
    validate_candidate_table,
    write_candidate_table,
)


def test_candidate_rows_use_canonical_position_mapping(tmp_path: Path) -> None:
    manifest_path = _write_request_manifest(tmp_path)
    sample_table = _write_sample_table(
        tmp_path,
        [
            _sample_row("sample-1", "BBC"),
            _sample_row("sample-2", "AXC"),
        ],
    )

    rows = build_proteinmpnn_candidate_rows(
        sample_table_path=sample_table,
        request_manifest_path=manifest_path,
    )

    by_sample = {row["source_sample_id"]: row for row in rows}
    assert by_sample["sample-1"]["canonical_mutations"] == ["A10B"]
    assert by_sample["sample-1"]["protected_mutation_count"] == 0
    assert by_sample["sample-1"]["status"] == "accepted"
    assert by_sample["sample-2"]["canonical_mutations"] == ["B11X"]
    assert by_sample["sample-2"]["protected_mutation_count"] == 1
    assert by_sample["sample-2"]["status"] == "rejected_protected_mutation"


def test_candidate_table_validator_rejects_protected_mutations(tmp_path: Path) -> None:
    manifest_path = _write_request_manifest(tmp_path)
    sample_table = _write_sample_table(tmp_path, [_sample_row("sample-1", "AXC")])
    candidate_table = tmp_path / "candidate_table.parquet"
    rows = build_proteinmpnn_candidate_rows(sample_table_path=sample_table, request_manifest_path=manifest_path)
    write_candidate_table(candidate_table, rows, request_hash="sha256:request")

    issues = validate_candidate_table(
        candidate_table,
        request_hash="sha256:request",
        sample_table_path=sample_table,
    )

    assert [issue.check_id for issue in issues] == ["thread.candidate_table.protected_mutation"]


def _write_request_manifest(tmp_path: Path) -> Path:
    parsed_path = tmp_path / "parsed_pdbs.jsonl"
    parsed_path.write_text(json.dumps({"name": "target", "seq_chain_A": "ABC"}) + "\n", encoding="utf-8")
    manifest = {
        "request_hash": "sha256:request",
        "proteinmpnn_name": "target",
        "proteinmpnn_design_chain": "A",
        "canonical_to_proteinmpnn_position": {"10": 1, "11": 2, "12": 3},
        "fixed_positions_jsonl": {"target": {"A": [2]}},
        "mutable_positions_by_chain": {"A": [1, 3]},
        "sidecar_paths": {"parsed_pdbs_jsonl": str(parsed_path)},
    }
    manifest_path = tmp_path / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_sample_table(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "sample_table.parquet"
    write_sample_table(path, rows, request_hash="sha256:request")
    return path


def _sample_row(sample_id: str, sequence: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "backend_run_id": "backend-1",
        "request_hash": "sha256:request",
        "seed": 101,
        "temperature": 0.1,
        "sample_index": 1,
        "sequence": sequence,
        "sequence_hash": "sha256:" + sample_id,
        "score": 1.0,
        "global_score": 2.0,
        "seq_recovery": 0.5,
        "backend_result_hash": "sha256:result",
        "status": "accepted",
    }
