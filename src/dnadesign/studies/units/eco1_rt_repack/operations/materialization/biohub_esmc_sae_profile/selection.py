"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/selection.py

Select Eco1 fold-accepted sequences for Biohub ESMC SAE annotation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.constants import (
    FOLDCHECK_REPORT_FILE_NAME,
    REQUEST_MANIFEST_RELATIVE_PATH,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request.constants import (
    WT_SEQUENCE_ID,
)
from dnadesign.thread.foldcheck import sequence_hash


@dataclass(frozen=True)
class BiohubEsmcSequenceRecord:
    """One fold-accepted sequence selected for Biohub ESMC annotation."""

    sequence_id: str
    sequence: str
    sequence_hash: str
    source_kind: str


@dataclass(frozen=True)
class BiohubEsmcSequenceSelection:
    """Selected sequences and the upstream fold-check request hash."""

    records: tuple[BiohubEsmcSequenceRecord, ...]
    source_request_hash: str


def select_fold_accepted_biohub_esmc_sequences(
    *,
    output_root: Path,
    sequence_limit: str,
) -> BiohubEsmcSequenceSelection:
    """Select WT plus fold-accepted candidates from Eco1 fold-check artifacts."""

    request_manifest_path = output_root / REQUEST_MANIFEST_RELATIVE_PATH
    report_path = output_root / FOLDCHECK_REPORT_FILE_NAME
    manifest = _load_manifest(request_manifest_path)
    records_by_id = _read_request_fasta(request_manifest_path, manifest)
    accepted_ids = _accepted_foldcheck_ids(report_path)
    if WT_SEQUENCE_ID not in accepted_ids:
        raise ValueError("Biohub ESMC profile requires an accepted WT fold-check row")
    rows = manifest.get("sequences")
    if not isinstance(rows, list):
        raise ValueError("fold-check request manifest requires sequences")
    ordered_records: list[BiohubEsmcSequenceRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fold-check request sequence rows must be mappings")
        sequence_id = str(row.get("sequence_id", ""))
        if sequence_id not in accepted_ids:
            continue
        fasta_record = records_by_id.get(sequence_id)
        if fasta_record is None:
            raise ValueError(f"fold-check FASTA is missing sequence {sequence_id!r}")
        if fasta_record.sequence_hash != str(row.get("sequence_hash", "")):
            raise ValueError(f"fold-check FASTA sequence hash mismatch for {sequence_id!r}")
        ordered_records.append(fasta_record)
    return BiohubEsmcSequenceSelection(
        records=tuple(_apply_sequence_limit(ordered_records, sequence_limit)),
        source_request_hash=str(manifest.get("request_hash", "")),
    )


def _load_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"fold-check request manifest must be a mapping: {path}")
    if payload.get("schema_id") != "thread.foldcheck_request":
        raise ValueError("Biohub ESMC selection requires a thread.foldcheck_request manifest")
    if not str(payload.get("request_hash", "")).strip():
        raise ValueError("fold-check request manifest requires request_hash")
    return payload


def _read_request_fasta(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, BiohubEsmcSequenceRecord]:
    fasta_path = _resolve_manifest_path(manifest_path, manifest.get("input_fasta_path"))
    parsed = _parse_fasta(fasta_path)
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or len(rows) != len(parsed):
        raise ValueError("fold-check FASTA record count does not match request manifest")
    records: dict[str, BiohubEsmcSequenceRecord] = {}
    for row, (sequence_id, sequence) in zip(rows, parsed, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("fold-check request sequence rows must be mappings")
        expected_id = str(row.get("sequence_id", ""))
        expected_hash = str(row.get("sequence_hash", ""))
        if sequence_id != expected_id:
            raise ValueError(f"fold-check FASTA id {sequence_id!r} does not match manifest id {expected_id!r}")
        observed_hash = sequence_hash(sequence)
        if observed_hash != expected_hash:
            raise ValueError(f"fold-check FASTA sequence hash mismatch for {sequence_id!r}")
        records[sequence_id] = BiohubEsmcSequenceRecord(
            sequence_id=sequence_id,
            sequence=sequence,
            sequence_hash=observed_hash,
            source_kind=str(row.get("source_kind", "")),
        )
    return records


def _resolve_manifest_path(manifest_path: Path, raw_value: Any) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("fold-check request manifest requires input_fasta_path")
    path = Path(raw_value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_parts).upper()))
            header = line[1:].strip()
            sequence_parts = []
        else:
            sequence_parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(sequence_parts).upper()))
    if not records:
        raise ValueError(f"fold-check input FASTA has no records: {path}")
    return records


def _accepted_foldcheck_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        str(row.get("candidate_id", "")) for row in pq.read_table(path).to_pylist() if row.get("status") == "accepted"
    }


def _apply_sequence_limit(
    records: Sequence[BiohubEsmcSequenceRecord],
    sequence_limit: str,
) -> list[BiohubEsmcSequenceRecord]:
    if not records:
        raise ValueError("Biohub ESMC profile requires at least one selected sequence")
    if sequence_limit.lower() == "all":
        return list(records)
    try:
        limit = int(sequence_limit)
    except ValueError as error:
        raise ValueError("sequence_limit must be a positive integer or 'all'") from error
    if limit < 1:
        raise ValueError("sequence_limit must be a positive integer or 'all'")
    if limit > len(records):
        raise ValueError(f"sequence_limit {limit} exceeds fold-accepted sequence count {len(records)}")
    return list(records[:limit])
