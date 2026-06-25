"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/request.py

Generic fold-check request manifest and FASTA helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.thread.foldcheck.models import FoldCheckSequenceRecord

FOLDCHECK_REQUEST_SCHEMA_ID = "thread.foldcheck_request"
_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")


def write_foldcheck_fasta(path: Path, records: Sequence[FoldCheckSequenceRecord]) -> None:
    """Write unique fold-check sequence records to FASTA."""

    if not records:
        raise ValueError("fold-check FASTA requires at least one sequence record")
    seen: set[str] = set()
    lines: list[str] = []
    for record in records:
        sequence_id = _require_identifier(record.sequence_id)
        if sequence_id in seen:
            raise ValueError(f"duplicate fold-check sequence id {sequence_id!r}")
        seen.add(sequence_id)
        sequence = _require_sequence(record.sequence, sequence_id)
        lines.append(f">{sequence_id}")
        lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_foldcheck_request_manifest(
    *,
    artifact_id: str,
    created_by: str,
    created_at: str,
    backend_kind: str,
    runtime_kind: str,
    execution_status: str,
    input_fasta_path: Path,
    output_root: Path,
    sequence_records: Sequence[FoldCheckSequenceRecord],
    wt_sequence_id: str,
    reference_structure_id: str,
    threshold_policy_id: str,
    threshold_values: Mapping[str, Any],
    upstream_artifact_hashes: Mapping[str, str],
    storage_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a model-agnostic fold-check request manifest."""

    _require_identifier(artifact_id)
    _require_identifier(created_by)
    _require_identifier(backend_kind)
    _require_identifier(runtime_kind)
    _require_identifier(execution_status)
    _require_identifier(wt_sequence_id)
    _require_identifier(reference_structure_id)
    _require_identifier(threshold_policy_id)
    if not sequence_records:
        raise ValueError("fold-check request requires at least one sequence record")
    if wt_sequence_id not in {record.sequence_id for record in sequence_records}:
        raise ValueError("fold-check request must include the declared WT baseline sequence id")
    if not threshold_values:
        raise ValueError("fold-check request requires explicit threshold values")
    if not upstream_artifact_hashes:
        raise ValueError("fold-check request requires upstream artifact hashes")

    manifest_without_hash = {
        "schema_id": FOLDCHECK_REQUEST_SCHEMA_ID,
        "schema_version": 1,
        "artifact_id": artifact_id,
        "status": "materialized",
        "created_by": created_by,
        "created_at": created_at,
        "backend_kind": backend_kind,
        "runtime_kind": runtime_kind,
        "execution_status": execution_status,
        "input_fasta_path": str(input_fasta_path),
        "output_root": str(output_root),
        "sequence_count": len(sequence_records),
        "wt_sequence_id": wt_sequence_id,
        "reference_structure_id": reference_structure_id,
        "threshold_policy_id": threshold_policy_id,
        "threshold_values": dict(threshold_values),
        "sequences": [
            {
                "sequence_id": record.sequence_id,
                "sequence_hash": record.sequence_hash,
                "source_kind": record.source_kind,
                "length": len(record.sequence),
            }
            for record in sequence_records
        ],
        "upstream_artifact_hashes": dict(upstream_artifact_hashes),
        "storage_policy": dict(storage_policy),
    }
    return {"request_hash": request_hash(manifest_without_hash), **manifest_without_hash}


def request_hash(payload: Mapping[str, Any]) -> str:
    """Return the canonical hash URI for a fold-check request payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fold-check identifiers must be non-empty strings")
    return value.strip()


def _require_sequence(sequence: str, sequence_id: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"fold-check sequence {sequence_id!r} is empty")
    invalid = sorted(set(normalized) - _ALLOWED_AA)
    if invalid:
        raise ValueError(f"fold-check sequence {sequence_id!r} has unsupported residues: {invalid}")
    return normalized
