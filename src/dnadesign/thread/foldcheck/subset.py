"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/subset.py

Fold-check request subsetting for external runtimes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dnadesign.thread.foldcheck.hashes import sequence_hash

DEFAULT_RUN_MANIFEST_SCHEMA_ID = "thread.foldcheck_external_run_manifest"
DEFAULT_EXECUTION_STATUS = "prepared_for_external_fold_cli"


@dataclass(frozen=True)
class FastaRecord:
    """One FASTA record selected for a fold-check runtime."""

    sequence_id: str
    sequence: str


def materialize_foldcheck_sequence_subset(
    *,
    request_manifest_path: Path,
    sequence_limit: str,
    input_fasta_path: Path,
    run_manifest_path: Path,
    output_dir: Path,
    schema_id: str = DEFAULT_RUN_MANIFEST_SCHEMA_ID,
    execution_status: str = DEFAULT_EXECUTION_STATUS,
) -> dict[str, Any]:
    """Write a subset FASTA and compact run manifest from a fold-check request."""

    manifest = _read_manifest(request_manifest_path)
    source_records = _read_request_fasta(request_manifest_path, manifest)
    _validate_records_against_manifest(source_records, manifest)
    selected_records = _select_records(source_records, sequence_limit)

    input_fasta_path.parent.mkdir(parents=True, exist_ok=True)
    input_fasta_path.write_text(_fasta_text(selected_records), encoding="utf-8")

    run_payload = {
        "schema_id": _require_nonempty(schema_id, "schema_id"),
        "schema_version": 1,
        "source_request_manifest": str(request_manifest_path),
        "source_request_hash": _require_nonempty(str(manifest.get("request_hash", "")), "request_hash"),
        "source_sequence_count": int(manifest["sequence_count"]),
        "selected_sequence_count": len(selected_records),
        "selected_sequence_ids": [record.sequence_id for record in selected_records],
        "input_fasta": str(input_fasta_path),
        "output_dir": str(output_dir),
        "execution_status": _require_nonempty(execution_status, "execution_status"),
    }
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(yaml.safe_dump(run_payload, sort_keys=False), encoding="utf-8")
    return run_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fold-check subset FASTA from a request manifest.")
    parser.add_argument("--request-manifest", required=True, type=Path)
    parser.add_argument("--sequence-limit", required=True)
    parser.add_argument("--input-fasta", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--schema-id", default=DEFAULT_RUN_MANIFEST_SCHEMA_ID)
    parser.add_argument("--execution-status", default=DEFAULT_EXECUTION_STATUS)
    args = parser.parse_args(argv)

    payload = materialize_foldcheck_sequence_subset(
        request_manifest_path=args.request_manifest,
        sequence_limit=args.sequence_limit,
        input_fasta_path=args.input_fasta,
        run_manifest_path=args.run_manifest,
        output_dir=args.output_dir,
        schema_id=args.schema_id,
        execution_status=args.execution_status,
    )
    print(f"wrote {payload['selected_sequence_count']} fold-check sequences to {args.input_fasta}")
    print(f"wrote run manifest to {args.run_manifest}")
    return 0


def _read_manifest(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"fold-check request manifest must be a mapping: {path}")
    if payload.get("schema_id") != "thread.foldcheck_request":
        raise ValueError("fold-check request subset requires schema_id thread.foldcheck_request")
    if not isinstance(payload.get("sequence_count"), int):
        raise ValueError("fold-check request manifest requires integer sequence_count")
    return payload


def _read_request_fasta(manifest_path: Path, manifest: Mapping[str, Any]) -> list[FastaRecord]:
    input_path_value = manifest.get("input_fasta_path")
    if not isinstance(input_path_value, str) or not input_path_value.strip():
        raise ValueError("fold-check request manifest requires input_fasta_path")
    source_fasta = Path(input_path_value)
    if not source_fasta.is_absolute():
        source_fasta = manifest_path.parent / source_fasta
    if not source_fasta.exists():
        raise FileNotFoundError(source_fasta)
    return _parse_fasta(source_fasta)


def _parse_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append(FastaRecord(header, "".join(sequence_parts)))
            header = line[1:].strip()
            sequence_parts = []
        else:
            sequence_parts.append(line.strip())
    if header is not None:
        records.append(FastaRecord(header, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"fold-check input FASTA has no records: {path}")
    return records


def _validate_records_against_manifest(records: Sequence[FastaRecord], manifest: Mapping[str, Any]) -> None:
    sequence_rows = manifest.get("sequences")
    if not isinstance(sequence_rows, list):
        raise ValueError("fold-check request manifest requires sequences")
    if len(records) != int(manifest["sequence_count"]) or len(records) != len(sequence_rows):
        raise ValueError("fold-check FASTA record count does not match request manifest")
    for record, row in zip(records, sequence_rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("fold-check request sequences must be mappings")
        expected_id = str(row.get("sequence_id", ""))
        expected_hash = str(row.get("sequence_hash", ""))
        if record.sequence_id != expected_id:
            raise ValueError(
                f"fold-check FASTA id {record.sequence_id!r} does not match request manifest id {expected_id!r}"
            )
        actual_hash = sequence_hash(record.sequence)
        if actual_hash != expected_hash:
            raise ValueError(f"fold-check FASTA sequence hash mismatch for {record.sequence_id!r}")


def _select_records(records: Sequence[FastaRecord], sequence_limit: str) -> list[FastaRecord]:
    if sequence_limit.lower() == "all":
        return list(records)
    try:
        limit = int(sequence_limit)
    except ValueError as error:
        raise ValueError("sequence_limit must be a positive integer or 'all'") from error
    if limit < 1:
        raise ValueError("sequence_limit must be a positive integer or 'all'")
    if limit > len(records):
        raise ValueError(f"sequence_limit {limit} exceeds request sequence count {len(records)}")
    return list(records[:limit])


def _fasta_text(records: Sequence[FastaRecord]) -> str:
    return "".join(f">{record.sequence_id}\n{record.sequence}\n" for record in records)


def _require_nonempty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
