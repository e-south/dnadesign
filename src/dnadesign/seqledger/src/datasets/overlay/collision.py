"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/src/datasets/overlay/collision.py

Cell occupancy checks used while holding the dataset write lock.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from ...contracts import SchemaError
from ..query import sql_ident


def reject_occupied_cells(dataset, *, namespace: str, incoming: pa.Table, key: str) -> None:
    """Reject writes to non-null logical values, including materialized overlays."""
    available = set(pq.read_schema(dataset.records_path).names)
    for overlay in dataset._load_overlays(include_tombstone=False, namespaces=[namespace]):
        available.update(overlay["cols"])
    columns = [name for name in incoming.column_names if name != key and name in available]
    if not columns:
        return
    base_key = "id" if key == "id" else "sequence"
    con, query, params = dataset._duckdb_query(
        columns=list(dict.fromkeys(["id", base_key, *columns])),
        include_overlays=[namespace],
        include_deleted=True,
    )
    try:
        con.register("overwrite_incoming", incoming.select([key]))

        def normalized(expr: str) -> str:
            expr = f"TRIM(CAST({expr} AS VARCHAR))"
            if key == "sequence_ci":
                expr = f"UPPER({expr})"
            return f"NULLIF({expr}, '')"

        join = f"{normalized(f'e.{sql_ident(base_key)}')} = {normalized(f'i.{sql_ident(key)}')}"
        for column in columns:
            occupied = con.execute(
                f"SELECT e.id FROM ({query}) e JOIN overwrite_incoming i ON {join} "
                f"WHERE e.{sql_ident(column)} IS NOT NULL LIMIT 5",
                params,
            ).fetchall()
            if occupied:
                sample = ", ".join(str(row[0]) for row in occupied)
                raise SchemaError(
                    f"Refusing overwrite for existing values in column '{column}' (sample ids: {sample}). "
                    "Re-run with overwrite=true."
                )
    finally:
        con.close()
