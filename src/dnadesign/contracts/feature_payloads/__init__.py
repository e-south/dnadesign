"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/feature_payloads/__init__.py

Read committed Infer feature payloads without importing the inference runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

#
# A manifest selects immutable Parquet parts. Unlisted files are never visible.
# Single-file Parquet payloads remain readable until explicitly migrated.

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "dnadesign.infer-feature-parts/v1"
VECTOR_SCHEMA = pa.schema(
    [("feature_vector_key", pa.string()), ("value", pa.list_(pa.float64())), ("created_at", pa.string())]
)
SCALAR_SCHEMA = pa.schema([("feature_scalar_key", pa.string()), ("value", pa.float64()), ("created_at", pa.string())])
ReadMode = Literal["live", "snapshot"]


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def parts_path(path: Path) -> Path:
    return path.with_suffix(".parts")


@contextmanager
def payload_lock(path: Path, *, exclusive: bool = False):
    """Coordinate payload readers and writers with Infer's dataset-sidecar lock."""
    lock_path = path.parent / ".sidecar.lock"
    if exclusive:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    else:
        try:
            descriptor = os.open(lock_path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        except FileNotFoundError:
            yield
            return
    with os.fdopen(descriptor, "r") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_manifest(path: Path) -> dict | None:
    """Validate the committed file inventory; callers hold the sidecar lock."""
    source = manifest_path(path)
    if source.is_symlink():
        raise ValueError(f"Feature manifest must not be a symlink: {source}")
    if not source.exists():
        return None
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"format", "key_column", "parts"}:
        raise ValueError(f"Invalid feature manifest fields: {source}")
    expected_key = {"feature_vectors": "feature_vector_key", "feature_scalars": "feature_scalar_key"}.get(path.stem)
    if document["format"] != FORMAT or document["key_column"] != expected_key:
        raise ValueError(f"Unsupported feature manifest contract: {source}")
    if not isinstance(document["parts"], list):
        raise ValueError(f"Feature manifest parts must be a list: {source}")
    directory = parts_path(path)
    if directory.is_symlink():
        raise ValueError(f"Feature parts directory must not be a symlink: {directory}")
    seen = set()
    for entry in document["parts"]:
        if not isinstance(entry, dict) or set(entry) != {"file", "rows", "bytes", "sha256"}:
            raise ValueError(f"Invalid feature part entry: {source}")
        name = entry["file"]
        if not isinstance(name, str) or not re.fullmatch(r"part-[a-f0-9]{32}\.parquet", name) or name in seen:
            raise ValueError(f"Invalid or duplicate feature part name: {name!r}")
        seen.add(name)
        if any(type(entry[field]) is not int or entry[field] < 0 for field in ("rows", "bytes")):
            raise ValueError(f"Invalid feature part dimensions: {name}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError(f"Invalid feature part checksum: {name}")
        part = directory / name
        if part.is_symlink() or not part.is_file() or part.stat().st_size != entry["bytes"]:
            raise ValueError(f"Committed feature part is missing or changed: {part}")
    return document


def payload_exists(path: Path) -> bool:
    return manifest_path(path).exists() or path.exists()


def payload_signature(path: Path) -> tuple[int, int]:
    current = manifest_path(path) if manifest_path(path).exists() else path
    stat = current.stat() if current.exists() else None
    return (stat.st_mtime_ns, stat.st_size) if stat is not None else (0, 0)


def _file_identity(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=512)
def _verify_part(path: Path, identity: tuple[int, ...], digest: str, rows: int, kind: str) -> None:
    with path.open("rb") as handle:
        if _file_identity(os.fstat(handle.fileno())) != identity:
            raise ValueError(f"Feature part changed during verification: {path}")
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError(f"Feature part checksum mismatch: {path}")
        handle.seek(0)
        parquet = pq.ParquetFile(handle)
        schema = VECTOR_SCHEMA if kind == "feature_vectors" else SCALAR_SCHEMA
        if not parquet.schema_arrow.equals(schema, check_metadata=False):
            raise ValueError(f"Feature payload schema mismatch: {path}")
        if parquet.metadata.num_rows != rows:
            raise ValueError(f"Feature part row count mismatch: {path}")
        if _file_identity(os.fstat(handle.fileno())) != identity:
            raise ValueError(f"Feature part changed during verification: {path}")


def payload_files(path: Path, *, include_manifest: bool = False) -> tuple[Path, ...]:
    """Resolve and verify active parts under a caller-held lock or immutable snapshot."""
    document = load_manifest(path)
    if document is None:
        return (path,) if path.is_file() else ()
    for entry in document["parts"]:
        part = parts_path(path) / entry["file"]
        _verify_part(part, _file_identity(part.stat()), entry["sha256"], entry["rows"], path.stem)
    prefix = (manifest_path(path),) if include_manifest else ()
    return prefix + tuple(parts_path(path) / entry["file"] for entry in document["parts"])


@contextmanager
def _read_files(path: Path, *, lock: bool, read_mode: ReadMode):
    if read_mode not in {"live", "snapshot"}:
        raise ValueError(f"Unsupported feature read_mode: {read_mode!r}")
    with payload_lock(path) if lock and read_mode == "live" else nullcontext():
        inventory = manifest_path(path) if manifest_path(path).exists() else path
        before = _file_identity(inventory.stat()) if inventory.exists() else None
        files = payload_files(path)
        identities = {part: _file_identity(part.stat()) for part in files}
        yield files
        after = _file_identity(inventory.stat()) if inventory.exists() else None
        if after != before or any(
            not p.exists() or _file_identity(p.stat()) != identity for p, identity in identities.items()
        ):
            raise ValueError(f"Feature payload changed during read: {path}")


def iter_payload_batches(
    path: Path, *, columns=None, batch_size: int = 4096, lock: bool = True, read_mode: ReadMode = "live"
):
    """Read one committed snapshot, excluding staged and orphaned parts."""
    with _read_files(path, lock=lock, read_mode=read_mode) as files:
        for part in files:
            yield from pq.ParquetFile(part).iter_batches(columns=columns, batch_size=batch_size)


def read_payload_table(path: Path, *, columns=None, lock: bool = True, read_mode: ReadMode = "live") -> pa.Table:
    with _read_files(path, lock=lock, read_mode=read_mode) as files:
        tables = [pq.read_table(part, columns=columns) for part in files]
    if tables:
        return pa.concat_tables(tables)
    schema = VECTOR_SCHEMA if path.stem == "feature_vectors" else SCALAR_SCHEMA
    if columns is not None:
        schema = pa.schema([schema.field(name) for name in columns])
    return pa.Table.from_batches([], schema=schema)
