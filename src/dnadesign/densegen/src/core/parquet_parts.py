"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/parquet_parts.py

Helpers for consolidating parquet part files into finalized tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def consolidate_parquet_parts(tables_root: Path, *, part_glob: str, final_name: str) -> bool:
    parts = sorted(tables_root.glob(part_glob))
    if not parts:
        return False
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to consolidate parquet parts.") from exc
    final_path = tables_root / final_name
    sources = [str(p) for p in parts]
    if final_path.exists():
        sources.insert(0, str(final_path))
    schemas = [pq.read_schema(source) for source in sources]
    dataset = ds.dataset(sources, format="parquet", schema=pa.unify_schemas(schemas))
    tmp_path = tables_root / f".{final_name}.tmp"
    writer = pq.ParquetWriter(tmp_path, schema=dataset.schema)
    scanner = ds.Scanner.from_dataset(dataset, batch_size=4096)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        writer.write_table(pa.Table.from_batches([batch], schema=dataset.schema))
    writer.close()
    tmp_path.replace(final_path)
    for part in parts:
        part.unlink()
    return True
