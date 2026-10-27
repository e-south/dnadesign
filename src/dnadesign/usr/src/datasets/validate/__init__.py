"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/validate/__init__.py

Streaming validation helpers for USR dataset schema and content integrity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from ...contracts import (
    META_REGISTRY_HASH,
    AlphabetError,
    DuplicateIDError,
    NamespaceError,
    SchemaError,
    compute_id,
    normalize_sequence,
    validate_alphabet,
    validate_bio_type,
)
from ...overlays import list_overlays
from ...registry import load_registry, load_registry_file, registry_hash_for_entries, validate_overlay_schema
from ...storage.parquet import iter_parquet_batches
from .registry_modes import normalize_registry_mode, validate_overlays_for_registry_mode


class DatasetValidateHost(Protocol):
    dir: Path
    root: Path
    records_path: Path

    def _load_overlays(self, *, include_tombstone: bool = False, namespaces=None): ...

    def _require_exists(self) -> None: ...

    def _tombstone_path(self) -> Path: ...

    def _frozen_registry_path(self) -> Path: ...


def _strict_registry_candidates(dataset: DatasetValidateHost, mode: str) -> list[tuple[str, dict, bool]]:
    if mode in {"current", "namespace-current"}:
        return [("current", load_registry(dataset.root, required=True), mode == "current")]
    if mode in {"frozen", "namespace-frozen"}:
        return [("frozen", load_registry_file(dataset._frozen_registry_path()), mode == "frozen")]

    candidates: list[tuple[str, dict, bool]] = []
    current_error: SchemaError | None = None
    frozen_error: SchemaError | None = None
    try:
        candidates.append(("current", load_registry(dataset.root, required=True), mode == "either"))
    except SchemaError as exc:
        current_error = exc
    try:
        candidates.append(("frozen", load_registry_file(dataset._frozen_registry_path()), mode == "either"))
    except SchemaError as exc:
        frozen_error = exc
    if candidates:
        return candidates
    if current_error is not None:
        raise current_error
    if frozen_error is not None:
        raise frozen_error
    raise SchemaError(f"Unsupported registry_mode '{mode}'.")


def _validate_materialized_base_registry_contract(
    *,
    dataset: DatasetValidateHost,
    schema: pa.Schema,
    mode: str,
    essential: set[str],
    reserved_namespaces: set[str],
) -> None:
    materialized_by_namespace: dict[str, list[pa.Field]] = {}
    id_field = schema.field("id")
    for field in schema:
        if field.name in essential or "__" not in field.name:
            continue
        namespace = field.name.split("__", 1)[0]
        if namespace in reserved_namespaces:
            continue
        materialized_by_namespace.setdefault(namespace, []).append(field)

    if not materialized_by_namespace:
        return

    metadata = schema.metadata or {}
    base_registry_hash_raw = metadata.get(META_REGISTRY_HASH.encode("utf-8"))
    errors: list[str] = []
    for label, registry, check_full_registry_hash in _strict_registry_candidates(dataset, mode):
        try:
            if check_full_registry_hash:
                if base_registry_hash_raw is None:
                    raise SchemaError("records.parquet missing registry_hash metadata.")
                expected_hash = registry_hash_for_entries(registry)
                actual_hash = base_registry_hash_raw.decode("utf-8")
                if actual_hash != expected_hash:
                    raise SchemaError(
                        f"records.parquet registry_hash mismatch: expected {expected_hash}, got {actual_hash}."
                    )
            for namespace, fields in materialized_by_namespace.items():
                validate_overlay_schema(namespace, pa.schema([id_field, *fields]), registry=registry, key="id")
            return
        except SchemaError as exc:
            errors.append(f"{label}: {exc}")
    joined = " | ".join(errors)
    raise SchemaError(f"Materialized base columns failed strict registry validation. {joined}")


def validate_dataset(
    dataset: DatasetValidateHost,
    *,
    strict: bool = False,
    registry_mode: str = "current",
    required_columns: tuple[tuple[str, pa.DataType], ...],
    reserved_namespaces: set[str],
) -> None:
    """
    Validate schema, IDs, alphabet constraints, and namespacing policy.
    """
    dataset._require_exists()
    mode = normalize_registry_mode(registry_mode)
    pf = pq.ParquetFile(str(dataset.records_path))
    schema = pf.schema_arrow
    names = set(schema.names)

    for req, dtype in required_columns:
        if req not in names:
            raise SchemaError(f"Missing required column: {req}")
        if schema.field(req).type != dtype:
            raise SchemaError(f"Column '{req}' has type {schema.field(req).type}, expected {dtype}.")

    essential = {k for k, _ in required_columns}
    derived = [c for c in schema.names if c not in essential]
    bad_ns = [c for c in derived if "__" not in c or c.split("__", 1)[0] == ""]
    if bad_ns:
        msg = f"Derived columns must be namespaced as '<tool>__<field>'. Offending columns: {', '.join(sorted(bad_ns))}"
        raise NamespaceError(msg)

    if dataset._tombstone_path().exists():
        dataset._load_overlays(include_tombstone=True, namespaces=["usr"])
        tomb_pf = pq.ParquetFile(str(dataset._tombstone_path()))
        tomb_schema = tomb_pf.schema_arrow
        if "id" not in tomb_schema.names:
            raise SchemaError("Tombstone overlay missing required 'id' column.")
        if tomb_schema.field("id").type != pa.string():
            raise SchemaError("Tombstone overlay 'id' must be string.")
        if "usr__deleted" not in tomb_schema.names:
            raise SchemaError("Tombstone overlay missing 'usr__deleted' column.")
        if tomb_schema.field("usr__deleted").type != pa.bool_():
            raise SchemaError("Tombstone overlay 'usr__deleted' must be bool.")
        if "usr__deleted_at" not in tomb_schema.names:
            raise SchemaError("Tombstone overlay missing 'usr__deleted_at' column.")
        if tomb_schema.field("usr__deleted_at").type != pa.timestamp("us", tz="UTC"):
            raise SchemaError("Tombstone overlay 'usr__deleted_at' must be timestamp(us, UTC).")
        if "usr__deleted_reason" not in tomb_schema.names:
            raise SchemaError("Tombstone overlay missing 'usr__deleted_reason' column.")
        if tomb_schema.field("usr__deleted_reason").type != pa.string():
            raise SchemaError("Tombstone overlay 'usr__deleted_reason' must be string.")

    overlays = list_overlays(dataset.dir)
    if overlays:
        validate_overlays_for_registry_mode(
            dataset=dataset,
            overlays=overlays,
            mode=mode,
            reserved_namespaces=reserved_namespaces,
        )

    if strict:
        _validate_materialized_base_registry_contract(
            dataset=dataset,
            schema=schema,
            mode=mode,
            essential=essential,
            reserved_namespaces=reserved_namespaces,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "validate.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE seen (val TEXT PRIMARY KEY)")
            dup_count = 0
            dup_samples: list[str] = []
            row_idx = 0
            for batch in iter_parquet_batches(
                dataset.records_path,
                columns=["id", "bio_type", "sequence", "alphabet", "length"],
            ):
                ids = batch.column("id").to_pylist()
                bios = batch.column("bio_type").to_pylist()
                seqs = batch.column("sequence").to_pylist()
                alphs = batch.column("alphabet").to_pylist()
                lens = batch.column("length").to_pylist()
                for rid, bt, seq, ab, ln in zip(ids, bios, seqs, alphs, lens):
                    row_idx += 1
                    if rid is None or str(rid).strip() == "":
                        raise SchemaError(f"Row {row_idx}: missing id.")
                    cur = conn.execute("INSERT OR IGNORE INTO seen(val) VALUES (?)", (str(rid),))
                    if cur.rowcount == 0:
                        dup_count += 1
                        if len(dup_samples) < 5:
                            dup_samples.append(str(rid))

                    if bt is None or str(bt).strip() == "":
                        raise SchemaError(f"Row {row_idx}: missing bio_type.")
                    try:
                        bt_norm = validate_bio_type(str(bt))
                    except ValueError as e:
                        raise SchemaError(f"Row {row_idx}: {e}") from e

                    if ab is None or str(ab).strip() == "":
                        raise AlphabetError(f"Row {row_idx}: missing alphabet.")
                    try:
                        ab_norm = validate_alphabet(bt_norm, str(ab))
                    except ValueError as e:
                        raise AlphabetError(f"Row {row_idx}: {e}") from e

                    if seq is None or str(seq).strip() == "":
                        raise SchemaError(f"Row {row_idx}: missing sequence.")
                    try:
                        seq_norm = normalize_sequence(str(seq), bt_norm, ab_norm, validate=False)
                    except ValueError as e:
                        raise AlphabetError(f"Row {row_idx}: {e}") from e

                    if ln is None:
                        raise SchemaError(f"Row {row_idx}: missing length.")
                    if int(ln) != len(seq_norm):
                        raise SchemaError(f"Row {row_idx}: length {ln} does not match sequence length {len(seq_norm)}.")
                    if compute_id(bt_norm, seq_norm) != str(rid):
                        raise SchemaError(f"Row {row_idx}: id does not match bio_type+sequence.")
            if dup_count:
                sample = ", ".join(dup_samples)
                raise DuplicateIDError(f"Duplicate ids detected: {dup_count} duplicate row(s). Sample ids: {sample}.")
        finally:
            conn.close()
