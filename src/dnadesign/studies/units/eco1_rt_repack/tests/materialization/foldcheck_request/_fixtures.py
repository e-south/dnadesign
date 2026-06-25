"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_request/_fixtures.py

Fixtures for Eco1 fold-check request materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.thread.candidates.proteinmpnn import write_candidate_table


def write_minimal_foldcheck_inputs(output_root: Path) -> None:
    """Write a compact candidate table, residue map, and request manifest."""

    output_root.mkdir(parents=True, exist_ok=True)
    _write_residue_map(output_root / "residue_map.parquet")
    candidate_rows = [
        {
            "candidate_id": "thread_candidate_test",
            "source_sample_id": "sample-1",
            "backend_run_id": "backend-1",
            "request_hash": "sha256:" + "1" * 64,
            "sequence_hash": "sha256:" + "2" * 64,
            "sequence": "A" * 309,
            "score": 1.0,
            "global_score": 2.0,
            "seq_recovery": 0.5,
            "seed": 101,
            "temperature": 0.1,
            "sample_index": 1,
            "duplicate_sample_count": 1,
            "mutation_count": 1,
            "mutable_mutation_count": 1,
            "protected_mutation_count": 0,
            "outside_mutable_positions": [],
            "canonical_mutations": ["A4E"],
            "status": "accepted",
            "rank": 1,
        }
    ]
    write_candidate_table(output_root / "candidate_table.parquet", candidate_rows, request_hash="sha256:" + "1" * 64)
    request_root = output_root / "proteinmpnn_request"
    request_root.mkdir(parents=True, exist_ok=True)
    (request_root / "request_manifest.yaml").write_text(
        yaml.safe_dump({"request_hash": "sha256:" + "1" * 64}, sort_keys=False),
        encoding="utf-8",
    )


def _write_residue_map(path: Path) -> None:
    rows = [
        {
            "canonical_position": position,
            "wt_aa": "A",
            "mapping_status": "mapped" if 3 <= position <= 311 else "unresolved_structure",
        }
        for position in range(1, 321)
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
