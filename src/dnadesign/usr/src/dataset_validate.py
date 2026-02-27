"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_validate.py

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

from .dataset_registry_modes import normalize_registry_mode, validate_overlays_for_registry_mode
from .errors import AlphabetError, DuplicateIDError, NamespaceError, SchemaError
from .normalize import compute_id, normalize_sequence, validate_alphabet, validate_bio_type
from .overlays import list_overlays
from .storage.parquet import iter_parquet_batches


class DatasetValidateHost(Protocol):
    dir: Path
    records_path: Path

    def _load_overlays(self, *, include_tombstone: bool = False, namespaces=None): ...

    def _require_exists(self) -> None: ...

    def _tombstone_path(self) -> Path: ...


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
    del strict
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
