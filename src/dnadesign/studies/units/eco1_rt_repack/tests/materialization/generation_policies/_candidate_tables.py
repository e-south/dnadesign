"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/_candidate_tables.py

Candidate-table helpers for Eco1 generation-policy tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._source_fixtures import (
    write_selection_source_inputs,
)
from dnadesign.thread.adapters.proteinmpnn import write_sample_table
from dnadesign.thread.candidates import write_candidate_table as write_proteinmpnn_candidate_table


def write_generation_policy_source_inputs(source_root: Path) -> None:
    write_selection_source_inputs(source_root)
    _add_structure_columns_to_residue_map(source_root / "residue_map.parquet")
    _neutralize_conservation_support(source_root / "conservation_profile.parquet")


def write_candidate_table(path: Path, rows: list[dict[str, object]], *, request_hash: str) -> None:
    write_proteinmpnn_candidate_table(path, rows, request_hash=request_hash)


def candidate_row(
    candidate_id: str,
    sequence: str,
    mutations: list[str],
    rank: int,
    *,
    request_hash: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_sample_id": f"sample_{candidate_id}",
        "backend_run_id": "test_run",
        "request_hash": request_hash,
        "sequence_hash": _sequence_hash(sequence),
        "sequence": sequence,
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


def request_hash_for_policy(root: Path, policy_id: str) -> str:
    manifest_path = root / policy_id / "proteinmpnn_request/request_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return str(manifest["request_hash"])


def write_policy_sample_table(
    root: Path,
    policy_id: str,
    candidate_rows: list[dict[str, object]],
) -> None:
    request_hash = request_hash_for_policy(root, policy_id)
    rows: list[dict[str, object]] = []
    for index in range(336):
        candidate = candidate_rows[index] if index < len(candidate_rows) else None
        sequence = str(candidate["sequence"]) if candidate is not None else sequence_for_index(index + 100)
        sample_id = str(candidate["source_sample_id"]) if candidate is not None else f"sample_{policy_id}_{index:03d}"
        rows.append(
            {
                "sample_id": sample_id,
                "backend_run_id": f"run_{policy_id}",
                "request_hash": request_hash,
                "seed": 101,
                "temperature": 0.1,
                "sample_index": index + 1,
                "sequence": sequence,
                "sequence_hash": _sequence_hash(sequence),
                "score": 1.0,
                "global_score": 1.0,
                "seq_recovery": 0.5,
                "backend_result_hash": "sha256:" + "0" * 64,
                "status": "accepted",
            }
        )
    write_sample_table(root / policy_id / "sample_table.parquet", rows, request_hash=request_hash)


def sequence_for_index(index: int) -> str:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    suffix = ""
    value = index
    for _ in range(3):
        suffix = alphabet[value % len(alphabet)] + suffix
        value //= len(alphabet)
    return "A" * 306 + suffix


def _sequence_hash(sequence: str) -> str:
    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _add_structure_columns_to_residue_map(path: Path) -> None:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        position = int(row["canonical_position"])
        row["structure_chain_id"] = "A"
        row["structure_residue_id"] = position
        row["design_position"] = position
        row["mapping_status"] = "mapped" if 3 <= position <= 311 else "unresolved_structure"
    pq.write_table(pa.Table.from_pylist(rows), path)


def _neutralize_conservation_support(path: Path) -> None:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["wt_frequency"] = 0.0
        row["wt_is_plurality"] = False
        row["passes_conservation_mask"] = False
    pq.write_table(pa.Table.from_pylist(rows), path)
