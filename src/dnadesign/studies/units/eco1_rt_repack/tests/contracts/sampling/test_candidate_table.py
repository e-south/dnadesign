"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/sampling/test_candidate_table.py

Candidate-table contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import (
    validate_candidate_table_content,
)
from dnadesign.thread.adapters.proteinmpnn.samples import write_sample_table
from dnadesign.thread.candidates.proteinmpnn import build_proteinmpnn_candidate_rows, write_candidate_table


def test_candidate_table_contract_rejects_protected_mutation(tmp_path: Path) -> None:
    request_manifest = _write_request_manifest(tmp_path)
    sample_table = tmp_path / "sample_table.parquet"
    write_sample_table(
        sample_table,
        [_sample_row(sequence="AXC")],
        request_hash="sha256:request",
    )
    rows = build_proteinmpnn_candidate_rows(
        sample_table_path=sample_table,
        request_manifest_path=request_manifest,
    )
    candidate_table = tmp_path / "candidate_table.parquet"
    write_candidate_table(candidate_table, rows, request_hash="sha256:request")

    issues = validate_candidate_table_content(candidate_table, output_root=tmp_path)

    assert [issue.check_id for issue in issues] == ["eco1_rt.sampling.candidate_table_protected_mutation"]


def _write_request_manifest(tmp_path: Path) -> Path:
    request_dir = tmp_path / "proteinmpnn_request"
    request_dir.mkdir()
    parsed_path = request_dir / "parsed_pdbs.jsonl"
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
    manifest_path = request_dir / "request_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _sample_row(*, sequence: str) -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "backend_run_id": "backend-1",
        "request_hash": "sha256:request",
        "seed": 101,
        "temperature": 0.1,
        "sample_index": 1,
        "sequence": sequence,
        "sequence_hash": "sha256:sample",
        "score": 1.0,
        "global_score": 2.0,
        "seq_recovery": 0.5,
        "backend_result_hash": "sha256:result",
        "status": "accepted",
    }
