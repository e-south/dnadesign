"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/_candidate_tables.py

Candidate-table helpers for Eco1 generation-policy tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_candidate_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def candidate_row(candidate_id: str, sequence_hash: str, mutations: list[str], rank: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_sample_id": f"sample_{candidate_id}",
        "backend_run_id": "test_run",
        "request_hash": "sha256:test",
        "sequence_hash": sequence_hash,
        "sequence": "TEST",
        "score": 1.0,
        "global_score": 1.0,
        "seq_recovery": 0.5,
        "seed": 101,
        "temperature": 0.1,
        "sample_index": rank,
        "duplicate_sample_count": 1,
        "mutation_count": len(mutations),
        "mutable_mutation_count": len(mutations),
        "protected_mutation_count": 0,
        "outside_mutable_positions": [],
        "canonical_mutations": mutations,
        "status": "accepted",
        "rank": rank,
    }
