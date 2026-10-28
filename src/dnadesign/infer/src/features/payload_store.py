"""Immutable payload commits and explicit migration/compaction for Infer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.contracts.feature_payloads import (
    FORMAT,
    iter_payload_batches,
    load_manifest,
    manifest_path,
    parts_path,
    payload_files,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_manifest(path: Path, document: dict) -> None:
    destination = manifest_path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _part_entry(part: Path, schema: pa.Schema) -> dict:
    parquet = pq.ParquetFile(part)
    if not parquet.schema_arrow.equals(schema, check_metadata=False):
        raise ValueError(f"Feature payload schema mismatch: {part}")
    with part.open("rb") as handle:
        os.fsync(handle.fileno())
    return {"file": part.name, "rows": parquet.metadata.num_rows, "bytes": part.stat().st_size, "sha256": _sha256(part)}


def _new_part(path: Path) -> Path:
    directory = parts_path(path)
    if directory.is_symlink():
        raise ValueError(f"Feature parts directory must not be a symlink: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"part-{uuid.uuid4().hex}.parquet"


def _discard_uncommitted(path: Path, part: Path) -> None:
    # A failed call may have published its manifest before raising. Retain any
    # referenced part, and leave ambiguous state intact for explicit inspection.
    try:
        document = load_manifest(path)
    except (ValueError, OSError):
        return
    if document is None or part.name not in {entry["file"] for entry in document["parts"]}:
        part.unlink(missing_ok=True)


def require_parts_for_write(path: Path) -> dict | None:
    document = load_manifest(path)
    if document is None and path.exists():
        raise ValueError(
            "Existing feature payloads require explicit migration: infer features migrate --root ROOT --dataset ID"
        )
    return document


def same_numeric_payload(left, right) -> bool:
    """Compare persisted numeric values, including corresponding NaNs."""
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_numeric_payload(a, b) for a, b in zip(left, right, strict=True))
    return left == right or (math.isnan(left) and math.isnan(right))


def _verified_values(expected: pa.Table, observed: pa.Table) -> bool:
    if expected.equals(observed):
        return True
    if not expected.drop(["value"]).equals(observed.drop(["value"])):
        return False
    left, right = (table.column("value").combine_chunks() for table in (expected, observed))
    if pa.types.is_list(left.type):
        if not left.offsets.equals(right.offsets) or not pc.is_null(left).equals(pc.is_null(right)):
            return False
        left, right = left.values, right.values
    left_nan, right_nan = (pc.fill_null(pc.is_nan(values), False) for values in (left, right))
    return left_nan.equals(right_nan) and pc.if_else(left_nan, None, left).equals(pc.if_else(right_nan, None, right))


def append_payload(path: Path, table: pa.Table) -> Path:
    """Commit only new payload rows; the caller holds the sidecar write lock."""
    document = require_parts_for_write(path)
    document = document or {"format": FORMAT, "key_column": table.schema.names[0], "parts": []}
    part = _new_part(path)
    try:
        pq.write_table(table, part)
        entry = _part_entry(part, table.schema)
        if not _verified_values(table, pq.read_table(part)):
            raise ValueError(f"Feature part verification failed: {part}")
        _write_manifest(path, {**document, "parts": [*document["parts"], entry]})
        return part
    except BaseException:
        _discard_uncommitted(path, part)
        raise


def migrate_payload(path: Path, *, schema: pa.Schema) -> dict[str, int]:
    """Copy and verify one single-file payload before publishing its manifest."""
    document = load_manifest(path)
    if document is not None:
        return {"migrated_files": 0, "migrated_rows": 0, "parts": len(document["parts"])}
    if not path.exists():
        return {"migrated_files": 0, "migrated_rows": 0, "parts": 0}
    if path.is_symlink():
        raise ValueError(f"Feature migration source must not be a symlink: {path}")
    keys = pq.read_table(path, columns=[schema.names[0]]).column(0)
    if keys.null_count or len(pc.unique(keys)) != len(keys) or any(not str(key).strip() for key in keys.to_pylist()):
        raise ValueError(f"Feature migration requires unique, nonempty payload keys: {path}")
    original_digest = _sha256(path)
    part = _new_part(path)
    try:
        shutil.copyfile(path, part)
        entry = _part_entry(part, schema)
        if entry["sha256"] != original_digest:
            raise ValueError(f"Feature migration verification failed: {path}")
        _write_manifest(path, {"format": FORMAT, "key_column": schema.names[0], "parts": [entry]})
    except BaseException:
        _discard_uncommitted(path, part)
        raise
    path.unlink()
    return {"migrated_files": 1, "migrated_rows": entry["rows"], "parts": 1}


def compact_payload(path: Path, *, schema: pa.Schema, keep_keys: set[str], batch_size: int) -> dict[str, int]:
    """Publish one verified part, then remove superseded parts under the write lock."""
    document = require_parts_for_write(path)
    if document is None:
        return {"rows_before": 0, "rows_after": 0}
    source_files = payload_files(path)
    for file, entry in zip(source_files, document["parts"], strict=True):
        if _sha256(file) != entry["sha256"]:
            raise ValueError(f"Feature part checksum mismatch: {file}")
    part = _new_part(path)
    keep = pa.array(sorted(keep_keys), type=pa.string())
    before = sum(entry["rows"] for entry in document["parts"])
    try:
        with pq.ParquetWriter(part, schema) as writer:
            for batch in iter_payload_batches(path, batch_size=batch_size, lock=False):
                table = pa.Table.from_batches([batch]).cast(schema)
                writer.write_table(table.filter(pc.is_in(table.column(schema.names[0]), value_set=keep)))
        entry = _part_entry(part, schema)
        _write_manifest(path, {**document, "parts": [entry]})
    except BaseException:
        _discard_uncommitted(path, part)
        raise
    for old in parts_path(path).iterdir():
        if old != part and not old.is_symlink() and re.fullmatch(r"part-[a-f0-9]{32}\.parquet", old.name):
            old.unlink()
    return {"rows_before": before, "rows_after": entry["rows"]}
