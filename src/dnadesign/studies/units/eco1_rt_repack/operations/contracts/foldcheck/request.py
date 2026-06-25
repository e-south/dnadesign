"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/foldcheck/request.py

Eco1 fold-check request validator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import _load_yaml
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    BACKEND_KIND,
    EXECUTION_STATUS,
    REFERENCE_STRUCTURE_ID,
    RUNTIME_KIND,
    WT_SEQUENCE_ID,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri
from dnadesign.thread.foldcheck import FOLDCHECK_REQUEST_SCHEMA_ID, request_hash


def validate_foldcheck_request_content(path: Path, *, output_root: Path) -> list[ContractIssue]:
    """Validate the Eco1 fold-check request manifest and FASTA."""

    issues: list[ContractIssue] = []
    manifest = _load_yaml(path)
    expected_pairs = {
        "schema_id": FOLDCHECK_REQUEST_SCHEMA_ID,
        "status": "materialized",
        "backend_kind": BACKEND_KIND,
        "runtime_kind": RUNTIME_KIND,
        "execution_status": EXECUTION_STATUS,
        "wt_sequence_id": WT_SEQUENCE_ID,
        "reference_structure_id": REFERENCE_STRUCTURE_ID,
    }
    for field, expected in expected_pairs.items():
        if manifest.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.foldcheck_request.field_mismatch",
                    message=f"fold-check request field {field!r} must equal {expected!r}",
                    path=f"{path}:{field}",
                )
            )
    observed_hash = manifest.get("request_hash")
    payload_without_hash = dict(manifest)
    payload_without_hash.pop("request_hash", None)
    if observed_hash != request_hash(payload_without_hash):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.request_hash_mismatch",
                message="fold-check request_hash must match manifest content",
                path=str(path),
            )
        )

    fasta_path = Path(str(manifest.get("input_fasta_path", "")))
    if not fasta_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.input_fasta_missing",
                message="fold-check request input_fasta_path must exist",
                path=str(fasta_path),
            )
        )
        return issues

    fasta_sequences = _read_fasta_ids(fasta_path)
    manifest_sequences = _manifest_sequences(manifest)
    if sorted(fasta_sequences) != sorted(manifest_sequences):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.sequence_id_mismatch",
                message="fold-check FASTA ids must match manifest sequence ids",
                path=str(path),
            )
        )
    bad_lengths = [sequence_id for sequence_id, sequence in fasta_sequences.items() if len(sequence) != 320]
    if bad_lengths:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.sequence_length_mismatch",
                message=f"Eco1 fold-check sequences must be full canonical 320-aa sequences: {bad_lengths}",
                path=str(fasta_path),
            )
        )

    candidate_table = output_root / "candidate_table.parquet"
    expected_candidate_ids = {WT_SEQUENCE_ID}
    if candidate_table.exists():
        expected_candidate_ids.update(
            str(row["candidate_id"])
            for row in pq.read_table(candidate_table).to_pylist()
            if str(row.get("status")) == "accepted"
        )
    missing_candidates = sorted(expected_candidate_ids - set(fasta_sequences))
    if missing_candidates:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.missing_candidate_sequences",
                message=f"fold-check request is missing accepted candidate ids: {missing_candidates}",
                path=str(fasta_path),
            )
        )

    upstream_hashes = manifest.get("upstream_artifact_hashes")
    if not isinstance(upstream_hashes, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.foldcheck_request.missing_upstream_hashes",
                message="fold-check request must declare upstream_artifact_hashes",
                path=f"{path}:upstream_artifact_hashes",
            )
        )
        return issues
    expected_hashes = {
        "candidate_table": output_root / "candidate_table.parquet",
        "residue_map": output_root / "residue_map.parquet",
        "proteinmpnn_request": output_root / "proteinmpnn_request/request_manifest.yaml",
    }
    for field, artifact_path in expected_hashes.items():
        if artifact_path.exists() and upstream_hashes.get(field) != sha256_uri(artifact_path):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.foldcheck_request.upstream_hash_mismatch",
                    message=f"fold-check request upstream hash {field!r} must match current artifact",
                    path=f"{path}:upstream_artifact_hashes.{field}",
                )
            )
    return issues


def _manifest_sequences(manifest: Mapping[str, Any]) -> dict[str, str]:
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list):
        return {}
    result: dict[str, str] = {}
    for row in sequences:
        if isinstance(row, Mapping) and isinstance(row.get("sequence_id"), str):
            result[str(row["sequence_id"])] = str(row.get("sequence_hash", ""))
    return result


def _read_fasta_ids(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].strip().split()[0]
            records[current] = []
        elif current is None:
            raise ValueError(f"FASTA sequence line before header in {path}")
        else:
            records[current].append(line.strip())
    return {sequence_id: "".join(parts) for sequence_id, parts in records.items()}
