"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/proteinmpnn/samples.py

ProteinMPNN backend output normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.adapters.proteinmpnn.models import ProteinMpnnRequestIssue

SAMPLE_TABLE_SCHEMA_ID = "thread.proteinmpnn_sample_table"
BACKEND_RUN_SCHEMA_ID = "thread.proteinmpnn_backend_run"
_REQUIRED_SAMPLE_COLUMNS = {
    "sample_id",
    "backend_run_id",
    "request_hash",
    "seed",
    "temperature",
    "sample_index",
    "sequence",
    "sequence_hash",
    "score",
    "global_score",
    "seq_recovery",
    "backend_result_hash",
    "status",
}


def parse_proteinmpnn_outputs(
    *,
    run_outputs: Sequence[Mapping[str, Any]],
    backend_run_id: str,
    request_hash: str,
    target_name: str,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Parse official ProteinMPNN run directories into normalized sample rows."""

    rows: list[dict[str, Any]] = []
    for run in run_outputs:
        output_dir = Path(str(run["output_dir"]))
        fasta_path = output_dir / "seqs" / f"{target_name}.fa"
        rows.extend(
            parse_proteinmpnn_fasta_samples(
                fasta_path,
                backend_run_id=backend_run_id,
                request_hash=request_hash,
                seed=int(run["seed"]),
                sequence_length=sequence_length,
            )
        )
    rows.sort(key=lambda row: (int(row["seed"]), float(row["temperature"]), int(row["sample_index"])))
    return rows


def parse_proteinmpnn_fasta_samples(
    fasta_path: Path,
    *,
    backend_run_id: str,
    request_hash: str,
    seed: int,
    sequence_length: int,
) -> list[dict[str, Any]]:
    """Parse one ProteinMPNN `seqs/<target>.fa` file."""

    records = _read_fasta_records(fasta_path)
    sample_rows: list[dict[str, Any]] = []
    result_hash = sha256_uri(fasta_path)
    for header, sequence in records:
        if header.startswith("T="):
            fields = _parse_header_fields(header)
            temperature = float(fields["T"])
            sample_index = int(fields["sample"])
            if len(sequence) != sequence_length:
                raise ValueError(
                    f"ProteinMPNN sequence length mismatch in {fasta_path}: {len(sequence)} != {sequence_length}"
                )
            sample_rows.append(
                {
                    "sample_id": _sample_id(backend_run_id, seed, temperature, sample_index),
                    "backend_run_id": backend_run_id,
                    "request_hash": request_hash,
                    "seed": seed,
                    "temperature": temperature,
                    "sample_index": sample_index,
                    "sequence": sequence,
                    "sequence_hash": _sequence_hash(sequence),
                    "score": float(fields["score"]),
                    "global_score": float(fields["global_score"]),
                    "seq_recovery": float(fields["seq_recovery"]),
                    "backend_result_hash": result_hash,
                    "status": "accepted",
                }
            )
    return sample_rows


def write_sample_table(path: Path, rows: Sequence[Mapping[str, Any]], *, request_hash: str) -> None:
    """Write normalized ProteinMPNN samples to Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    metadata = {
        b"schema_id": SAMPLE_TABLE_SCHEMA_ID.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"request_hash": request_hash.encode("utf-8"),
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path)


def validate_sample_table(
    path: Path,
    *,
    request_hash: str,
    expected_sample_count: int,
    sequence_length: int,
) -> list[ProteinMpnnRequestIssue]:
    """Validate a generic ProteinMPNN sample table."""

    issues: list[ProteinMpnnRequestIssue] = []
    table = pq.read_table(path)
    missing_columns = sorted(_REQUIRED_SAMPLE_COLUMNS - set(table.column_names))
    if missing_columns:
        return [
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_missing_columns",
                message=f"ProteinMPNN sample table is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != SAMPLE_TABLE_SCHEMA_ID.encode("utf-8"):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_metadata_mismatch",
                message="ProteinMPNN sample table must declare the generic sample-table schema id",
                path=str(path),
            )
        )
    rows = table.to_pylist()
    if len(rows) != expected_sample_count:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_count_mismatch",
                message=f"ProteinMPNN sample table must contain {expected_sample_count} rows",
                path=str(path),
            )
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_duplicate_ids",
                message="ProteinMPNN sample ids must be unique",
                path=str(path),
            )
        )
    bad_hash_rows = [row["sample_id"] for row in rows if row["request_hash"] != request_hash]
    if bad_hash_rows:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_request_hash_mismatch",
                message=f"ProteinMPNN sample rows must carry request hash {request_hash}",
                path=str(path),
            )
        )
    bad_lengths = [row["sample_id"] for row in rows if len(str(row["sequence"])) != sequence_length]
    if bad_lengths:
        issues.append(
            ProteinMpnnRequestIssue(
                check_id="thread.proteinmpnn.sample_table_sequence_length_mismatch",
                message=f"ProteinMPNN sample sequences must have length {sequence_length}",
                path=str(path),
            )
        )
    return issues


def write_backend_run_manifest(
    path: Path,
    *,
    request_manifest_path: Path,
    request_hash: str,
    proteinmpnn_root: Path,
    proteinmpnn_git_commit: str,
    runs: Sequence[Mapping[str, Any]],
    batch_id: str,
    num_seq_per_target: int,
    batch_size: int,
    expected_sample_count: int,
) -> None:
    """Write a backend-run manifest without changing the request manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": BACKEND_RUN_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "backend_kind": "proteinmpnn",
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_hash": sha256_uri(request_manifest_path),
        "request_hash": request_hash,
        "batch_id": batch_id,
        "num_seq_per_target": num_seq_per_target,
        "batch_size": batch_size,
        "expected_sample_count": expected_sample_count,
        "proteinmpnn_root": str(proteinmpnn_root),
        "proteinmpnn_git_commit": proteinmpnn_git_commit,
        "runs": list(runs),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_sequence)))
            current_header = line[1:]
            current_sequence = []
        else:
            current_sequence.append(line.strip())
    if current_header is not None:
        records.append((current_header, "".join(current_sequence)))
    return records


def _parse_header_fields(header: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in re.split(r",\s*", header):
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key.strip()] = value.strip()
    required = {"T", "sample", "score", "global_score", "seq_recovery"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"ProteinMPNN sample header is missing fields {missing}: {header}")
    return fields


def _sequence_hash(sequence: str) -> str:
    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _sample_id(backend_run_id: str, seed: int, temperature: float, sample_index: int) -> str:
    temp = f"{temperature:g}"
    return f"{backend_run_id}__seed{seed}__temp{temp}__sample{sample_index}"
