"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_views.py

Read and export helpers for Dataset scan/query views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import SchemaError, SequencesError
from .overlays import list_overlays
from .storage.parquet import PARQUET_COMPRESSION

if TYPE_CHECKING:
    from .dataset import Dataset


_HEAD_CACHE: dict[tuple[str, int, tuple[str, ...] | None, bool, bool], tuple[tuple[object, ...], pd.DataFrame]] = {}
_HEAD_CACHE_MAX = 64


def _path_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _head_state_signature(dataset: Dataset, *, include_derived: bool) -> tuple[object, ...]:
    records_sig = _path_signature(dataset.records_path)
    if not include_derived:
        return (records_sig,)
    overlay_sigs: list[tuple[str, int, int]] = []
    for overlay in list_overlays(dataset.dir):
        sig = _path_signature(overlay)
        overlay_sigs.append((str(overlay), sig[0], sig[1]))
    return (records_sig, tuple(overlay_sigs))


def scan_dataset(
    dataset: Dataset,
    *,
    columns: Optional[List[str]] = None,
    include_overlays: Union[bool, Sequence[str]] = True,
    include_deleted: bool = False,
    batch_size: int = 65536,
    limit: int | None = None,
):
    """
    Stream record batches with optional overlay merge.
    """
    if batch_size < 1:
        raise SequencesError("batch_size must be >= 1.")
    con, query, params = dataset._duckdb_query(  # noqa: SLF001
        columns=columns,
        include_overlays=include_overlays,
        include_deleted=include_deleted,
        limit=limit,
    )
    try:
        con.execute(query, params)
        reader = con.fetch_record_batch(int(batch_size))
        for batch in reader:
            yield batch
    finally:
        con.close()


def head_dataset(
    dataset: Dataset,
    n: int = 10,
    columns: Optional[List[str]] = None,
    *,
    include_derived: bool = True,
    include_deleted: bool = False,
):
    """Return the first N rows as a pandas DataFrame."""
    columns_key = tuple(columns) if columns is not None else None
    cache_key = (
        str(dataset.dir.resolve()),
        int(n),
        columns_key,
        bool(include_derived),
        bool(include_deleted),
    )
    state_sig = _head_state_signature(dataset, include_derived=include_derived)
    cached = _HEAD_CACHE.get(cache_key)
    if cached is not None and cached[0] == state_sig:
        return cached[1].copy(deep=True)

    batches = []
    rows = 0
    for batch in scan_dataset(
        dataset,
        columns=columns,
        include_overlays=include_derived,
        include_deleted=include_deleted,
        batch_size=max(int(n), 1),
        limit=int(n),
    ):
        batches.append(batch)
        rows += batch.num_rows
        if rows >= n:
            break
    if not batches:
        empty = pd.DataFrame(columns=columns or dataset.schema().names)
        _HEAD_CACHE[cache_key] = (state_sig, empty.copy(deep=True))
        if len(_HEAD_CACHE) > _HEAD_CACHE_MAX:
            _HEAD_CACHE.clear()
        return empty
    tbl = pa.Table.from_batches(batches)
    if tbl.num_rows > n:
        tbl = tbl.slice(0, n)
    out = tbl.to_pandas()
    _HEAD_CACHE[cache_key] = (state_sig, out.copy(deep=True))
    if len(_HEAD_CACHE) > _HEAD_CACHE_MAX:
        _HEAD_CACHE.clear()
    return out


def get_dataset(
    dataset: Dataset, record_id: str, columns: Optional[List[str]] = None, *, include_deleted: bool = False
):
    """Return a single record by id (as a pandas DataFrame row)."""
    con, query, params = dataset._duckdb_query(  # noqa: SLF001
        columns=columns,
        include_overlays=True,
        include_deleted=include_deleted,
        where="b.id = ?",
        params=[str(record_id)],
        limit=1,
    )
    try:
        con.execute(query, params)
        reader = con.fetch_record_batch(1)
        batch = reader.read_next_batch()
        if batch is None or batch.num_rows == 0:
            return pd.DataFrame(columns=columns or dataset.schema().names)
        tbl = pa.Table.from_batches([batch])
        return tbl.to_pandas()
    finally:
        con.close()


def grep_dataset(
    dataset: Dataset,
    pattern: str,
    limit: int = 20,
    batch_size: int = 65536,
    *,
    include_deleted: bool = False,
):
    """Regex search across sequences, returning first `limit` hits."""
    if batch_size < 1:
        raise SequencesError("batch_size must be >= 1.")
    pattern_ci = f"(?i){pattern}"
    con, query, params = dataset._duckdb_query(  # noqa: SLF001
        columns=["id", "sequence", "length"],
        include_overlays=True,
        include_deleted=include_deleted,
        where="regexp_matches(b.sequence, ?)",
        params=[pattern_ci],
        limit=int(limit),
    )
    try:
        con.execute(query, params)
        reader = con.fetch_record_batch(int(batch_size))
        batches = []
        for batch in reader:
            batches.append(batch)
        if not batches:
            return pd.DataFrame(columns=["id", "sequence", "length"])
        tbl = pa.Table.from_batches(batches)
        return tbl.to_pandas().head(limit)
    finally:
        con.close()


def export_dataset(
    dataset: Dataset,
    fmt: str,
    out_path: Path,
    columns: Optional[List[str]] = None,
    *,
    include_deleted: bool = False,
) -> None:
    """Export current table to CSV, JSONL, or Parquet."""
    fmt_norm = str(fmt or "").strip().lower()
    if fmt_norm not in {"csv", "jsonl", "parquet"}:
        raise SequencesError("Unsupported export format. Use csv|jsonl|parquet.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt_norm == "csv":
        first = True
        wrote = False
        for batch in scan_dataset(
            dataset,
            columns=columns,
            include_overlays=True,
            include_deleted=include_deleted,
            batch_size=65536,
        ):
            df = batch.to_pandas()
            df.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
            first = False
            wrote = True
        if not wrote:
            empty_cols = columns if columns else dataset.schema().names
            pd.DataFrame(columns=list(empty_cols)).to_csv(out_path, index=False)
    elif fmt_norm == "jsonl":
        wrote = False
        with out_path.open("w", encoding="utf-8") as handle:
            for batch in scan_dataset(
                dataset,
                columns=columns,
                include_overlays=True,
                include_deleted=include_deleted,
                batch_size=65536,
            ):
                df = batch.to_pandas()
                text = df.to_json(orient="records", lines=True)
                if text and not text.endswith("\n"):
                    text += "\n"
                if text:
                    handle.write(text)
                    wrote = True
        if not wrote:
            out_path.write_text("", encoding="utf-8")
    else:
        writer: pq.ParquetWriter | None = None
        try:
            for batch in scan_dataset(
                dataset,
                columns=columns,
                include_overlays=True,
                include_deleted=include_deleted,
                batch_size=65536,
            ):
                table = pa.Table.from_batches([batch], schema=batch.schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema=table.schema, compression=PARQUET_COMPRESSION)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            schema = dataset.schema()
            if columns:
                fields = []
                for name in columns:
                    idx = schema.get_field_index(str(name))
                    if idx < 0:
                        raise SchemaError(f"Unknown column '{name}' in export selection.")
                    fields.append(schema.field(idx))
                schema = pa.schema(fields)
            arrays = [pa.array([], type=field.type) for field in schema]
            empty = pa.Table.from_arrays(arrays, schema=schema)
            pq.write_table(empty, out_path, compression=PARQUET_COMPRESSION)
