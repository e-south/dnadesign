"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/candidates/proteinmpnn.py

ProteinMPNN sample-to-candidate normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.thread.adapters.proteinmpnn.models import ProteinMpnnRequestIssue
from dnadesign.thread.adapters.proteinmpnn.sidecars import resolve_manifest_sidecar_path

CANDIDATE_TABLE_SCHEMA_ID = "thread.proteinmpnn_candidate_table"
_REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "source_sample_id",
    "backend_run_id",
    "request_hash",
    "sequence_hash",
    "sequence",
    "score",
    "global_score",
    "seq_recovery",
    "seed",
    "temperature",
    "sample_index",
    "duplicate_sample_count",
    "mutation_count",
    "mutable_mutation_count",
    "protected_mutation_count",
    "outside_mutable_positions",
    "canonical_mutations",
    "status",
}


def build_proteinmpnn_candidate_rows(
    *,
    sample_table_path: Path,
    request_manifest_path: Path,
) -> list[dict[str, Any]]:
    """Build one candidate row per unique ProteinMPNN output sequence."""

    sample_rows = pq.read_table(sample_table_path).to_pylist()
    manifest = _load_yaml(request_manifest_path)
    context = _CandidateContext.from_manifest(manifest, request_manifest_path=request_manifest_path)
    rows_by_hash: dict[str, list[Mapping[str, Any]]] = {}
    for row in sample_rows:
        rows_by_hash.setdefault(str(row["sequence_hash"]), []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    for sequence_hash, duplicate_rows in sorted(rows_by_hash.items()):
        best = min(
            duplicate_rows,
            key=lambda row: (
                float(row["score"]),
                float(row["global_score"]),
                int(row["seed"]),
                float(row["temperature"]),
                int(row["sample_index"]),
            ),
        )
        mutations = _canonical_mutations(str(best["sequence"]), context=context)
        outside_mutable = [
            mutation["canonical_position"]
            for mutation in mutations
            if mutation["proteinmpnn_position"] not in context.mutable_positions
            and mutation["proteinmpnn_position"] not in context.fixed_positions
        ]
        protected = [
            mutation["canonical_position"]
            for mutation in mutations
            if mutation["proteinmpnn_position"] in context.fixed_positions
        ]
        mutable_mutation_count = sum(
            1 for mutation in mutations if mutation["proteinmpnn_position"] in context.mutable_positions
        )
        status = _candidate_status(
            source_status=str(best.get("status")),
            outside_mutable=outside_mutable,
            protected=protected,
        )
        candidate_rows.append(
            {
                "candidate_id": "thread_candidate_" + sequence_hash.removeprefix("sha256:")[:12],
                "source_sample_id": best["sample_id"],
                "backend_run_id": best["backend_run_id"],
                "request_hash": best["request_hash"],
                "sequence_hash": sequence_hash,
                "sequence": best["sequence"],
                "score": float(best["score"]),
                "global_score": float(best["global_score"]),
                "seq_recovery": float(best["seq_recovery"]),
                "seed": int(best["seed"]),
                "temperature": float(best["temperature"]),
                "sample_index": int(best["sample_index"]),
                "duplicate_sample_count": len(duplicate_rows),
                "mutation_count": len(mutations),
                "mutable_mutation_count": mutable_mutation_count,
                "protected_mutation_count": len(protected),
                "outside_mutable_positions": outside_mutable,
                "canonical_mutations": [mutation["label"] for mutation in mutations],
                "status": status,
            }
        )
    candidate_rows.sort(
        key=lambda row: (row["status"] != "accepted", row["score"], row["global_score"], row["candidate_id"])
    )
    for rank, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank
    return candidate_rows


def _candidate_status(*, source_status: str, outside_mutable: Sequence[int], protected: Sequence[int]) -> str:
    if protected:
        return "rejected_protected_mutation"
    if outside_mutable:
        return "rejected_outside_mutable_position"
    if source_status != "accepted":
        return "rejected_source_sample"
    return "accepted"


def write_candidate_table(path: Path, rows: Sequence[Mapping[str, Any]], *, request_hash: str) -> None:
    """Write ProteinMPNN candidate rows to Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    metadata = {
        b"schema_id": CANDIDATE_TABLE_SCHEMA_ID.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"request_hash": request_hash.encode("utf-8"),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def validate_candidate_table(
    path: Path,
    *,
    request_hash: str,
    sample_table_path: Path,
) -> list[ProteinMpnnRequestIssue]:
    """Validate a generic ProteinMPNN candidate table."""

    issues: list[ProteinMpnnRequestIssue] = []
    table = pq.read_table(path)
    missing_columns = sorted(_REQUIRED_CANDIDATE_COLUMNS - set(table.column_names))
    if missing_columns:
        return [
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.missing_columns",
                message=f"Candidate table is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != CANDIDATE_TABLE_SCHEMA_ID.encode("utf-8"):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.metadata_mismatch",
                message="Candidate table must declare the generic candidate-table schema id",
                path=str(path),
            )
        )
    if metadata.get(b"request_hash") != request_hash.encode("utf-8"):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.metadata_request_hash_mismatch",
                message=f"Candidate table metadata must carry request hash {request_hash}",
                path=str(path),
            )
        )
    sample_rows = pq.read_table(sample_table_path).to_pylist()
    sample_count = len(sample_rows)
    rows = table.to_pylist()
    if not rows or len(rows) > sample_count:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.count_mismatch",
                message="Candidate table must contain one or more rows and no more rows than the source sample table",
                path=str(path),
            )
        )
    if any(row["request_hash"] != request_hash for row in rows):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.request_hash_mismatch",
                message=f"Candidate rows must carry request hash {request_hash}",
                path=str(path),
            )
        )
    sample_ids = {str(row["sample_id"]) for row in sample_rows}
    missing_source_sample_ids = sorted(
        str(row["source_sample_id"]) for row in rows if str(row["source_sample_id"]) not in sample_ids
    )
    if missing_source_sample_ids:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.missing_source_sample",
                message="Candidate rows must reference sample ids in the source sample table",
                path=str(path),
            )
        )
    if any(int(row["protected_mutation_count"]) > 0 for row in rows):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.protected_mutation",
                message="Candidate rows must not mutate fixed/protected ProteinMPNN positions",
                path=str(path),
            )
        )
    if any(row["outside_mutable_positions"] for row in rows):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.candidate_table.outside_mutable_position",
                message="Candidate rows must not mutate positions outside the declared mutable set",
                path=str(path),
            )
        )
    return issues


class _CandidateContext:
    def __init__(
        self,
        *,
        wt_sequence: str,
        proteinmpnn_to_canonical: Mapping[int, int],
        fixed_positions: set[int],
        mutable_positions: set[int],
    ) -> None:
        self.wt_sequence = wt_sequence
        self.proteinmpnn_to_canonical = proteinmpnn_to_canonical
        self.fixed_positions = fixed_positions
        self.mutable_positions = mutable_positions

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, request_manifest_path: Path) -> "_CandidateContext":
        chain_id = str(manifest["proteinmpnn_design_chain"])
        target_name = str(manifest["proteinmpnn_name"])
        canonical_to_mpnn = {
            int(canonical): int(position)
            for canonical, position in dict(manifest["canonical_to_proteinmpnn_position"]).items()
        }
        fixed_payload = manifest["fixed_positions_jsonl"][target_name][chain_id]
        mutable_payload = manifest["mutable_positions_by_chain"][chain_id]
        parsed_path = resolve_manifest_sidecar_path(
            request_manifest_path,
            manifest["sidecar_paths"]["parsed_pdbs_jsonl"],
        )
        parsed_record = _load_jsonl_record(parsed_path)
        wt_sequence = str(parsed_record[f"seq_chain_{chain_id}"])
        return cls(
            wt_sequence=wt_sequence,
            proteinmpnn_to_canonical={position: canonical for canonical, position in canonical_to_mpnn.items()},
            fixed_positions={int(position) for position in fixed_payload},
            mutable_positions={int(position) for position in mutable_payload},
        )


def _canonical_mutations(sequence: str, *, context: _CandidateContext) -> list[dict[str, Any]]:
    if len(sequence) != len(context.wt_sequence):
        raise ValueError(
            f"Candidate sequence length {len(sequence)} does not match WT length {len(context.wt_sequence)}"
        )
    mutations: list[dict[str, Any]] = []
    for zero_index, (wt_aa, new_aa) in enumerate(zip(context.wt_sequence, sequence, strict=True)):
        if wt_aa == new_aa:
            continue
        proteinmpnn_position = zero_index + 1
        canonical_position = context.proteinmpnn_to_canonical[proteinmpnn_position]
        mutations.append(
            {
                "proteinmpnn_position": proteinmpnn_position,
                "canonical_position": canonical_position,
                "label": f"{wt_aa}{canonical_position}{new_aa}",
            }
        )
    return mutations


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _load_jsonl_record(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return records[0]
