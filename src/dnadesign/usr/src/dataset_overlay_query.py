"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_overlay_query.py

Overlay-aware DuckDB query planning for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import pyarrow.parquet as pq

from .duckdb_runtime import connect_duckdb_utc
from .errors import NamespaceError, SchemaError


class DatasetOverlayQueryHost(Protocol):
    records_path: Path

    def _require_exists(self) -> None: ...

    def _tombstone_path(self) -> Path: ...

    def _load_overlays(
        self,
        *,
        include_tombstone: bool = True,
        namespaces: Sequence[str] | None = None,
    ): ...

    @staticmethod
    def _sql_ident(name: str) -> str: ...

    def _create_overlay_view(
        self,
        con,
        *,
        view_name: str,
        path: Path,
        key: str,
    ) -> None: ...


def build_overlay_query(
    dataset: DatasetOverlayQueryHost,
    *,
    columns: list[str] | None,
    include_overlays: bool | Sequence[str],
    include_deleted: bool,
    where: str | None,
    params: list | None,
    limit: int | None,
    required_columns: Sequence[tuple[str, object]],
    tombstone_columns: Sequence[str],
    tombstone_namespace: str,
):
    dataset._require_exists()

    if columns is not None and not include_deleted and any(col in tombstone_columns for col in columns):
        raise SchemaError("Tombstone columns require include_deleted=True.")

    base_pf = pq.ParquetFile(str(dataset.records_path))
    base_cols = list(base_pf.schema_arrow.names)

    if include_overlays is True:
        overlays = dataset._load_overlays(include_tombstone=False)
    elif include_overlays is False:
        overlays = []
    else:
        overlays = dataset._load_overlays(include_tombstone=False, namespaces=include_overlays)

    requested = set(columns) if columns is not None else None
    selected_cols: list[str] = []

    def _key_expr(expr: str, *, key: str) -> str:
        if key == "sequence_ci":
            return f"NULLIF(UPPER(TRIM(CAST({expr} AS VARCHAR))), '')"
        return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"

    con = connect_duckdb_utc(
        missing_dependency_message="duckdb is required for overlay joins (install duckdb).",
        error_context="overlay joins",
    )
    base_sql = str(dataset.records_path).replace("'", "''")
    con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{base_sql}')")
    base_view = "base"

    tomb_path = dataset._tombstone_path()
    tomb_exists = tomb_path.exists()
    if not include_deleted and tomb_exists:
        tomb_sql = str(tomb_path).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW tombstone_filter AS SELECT id, usr__deleted FROM read_parquet('{tomb_sql}')")
        con.execute(
            "CREATE TEMP VIEW base_filtered AS "
            "SELECT b.* FROM base b LEFT JOIN tombstone_filter t ON b.id = t.id "
            "WHERE COALESCE(t.usr__deleted, FALSE) = FALSE"
        )
        base_view = "base_filtered"

    include_tombstone_cols = False
    if include_deleted and tomb_exists:
        if include_overlays is True and columns is None:
            include_tombstone_cols = True
        elif requested and any(col in tombstone_columns for col in requested):
            include_tombstone_cols = True

    if include_tombstone_cols:
        tomb_pf = pq.ParquetFile(str(tomb_path))
        overlays.append(
            {
                "namespace": tombstone_namespace,
                "key": "id",
                "cols": list(tombstone_columns),
                "schema": tomb_pf.schema_arrow,
                "path": tomb_path,
                "read_path": str(tomb_path),
            }
        )

    select_expr_by_col = {col: f"b.{dataset._sql_ident(col)}" for col in base_cols}
    select_order = list(base_cols)

    join_clauses: list[str] = []
    essential = {key for key, _ in required_columns}

    for idx, overlay in enumerate(overlays):
        namespace = overlay["namespace"]
        key = overlay["key"]
        overlay_cols = overlay["cols"]
        if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
            raise SchemaError(f"Unsupported overlay key '{key}': {overlay['path']}")

        derived_cols = overlay_cols
        if requested is not None:
            derived_cols = [col for col in overlay_cols if col in requested]
        if not derived_cols:
            continue

        for col in derived_cols:
            if namespace != tombstone_namespace and col in essential:
                raise NamespaceError(f"Overlay cannot modify required column '{col}'.")
            if "__" not in col:
                raise NamespaceError(f"Derived columns must be namespaced (got '{col}').")

        view_name = f"overlay_{idx}"
        dataset._create_overlay_view(con, view_name=view_name, path=overlay["path"], key=key)

        if key in {"sequence", "sequence_norm", "sequence_ci"}:
            bio_type_count = int(con.execute(f"SELECT COUNT(DISTINCT bio_type) FROM {base_view}").fetchone()[0])
            if bio_type_count > 1:
                raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
            if key == "sequence_ci":
                invalid_alphabet = int(
                    con.execute(f"SELECT COUNT(*) FROM {base_view} WHERE alphabet != 'dna_4'").fetchone()[0]
                )
                if invalid_alphabet:
                    raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
            base_key_expr = _key_expr(f"b.{dataset._sql_ident('sequence')}", key=key)
            dup_base = int(
                con.execute(
                    "SELECT COUNT(*) FROM "
                    f"(SELECT {base_key_expr} AS k FROM {base_view} b GROUP BY k HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            if dup_base:
                raise SchemaError(f"Attach key requires unique base keys; duplicate base keys detected for '{key}'.")
        else:
            base_key_expr = _key_expr(f"b.{dataset._sql_ident('id')}", key=key)

        overlay_key_expr = _key_expr(f"o{idx}.{dataset._sql_ident(key)}", key=key)
        join_clauses.append(f"LEFT JOIN {view_name} o{idx} ON {base_key_expr} = {overlay_key_expr}")
        for col in derived_cols:
            col_ident = dataset._sql_ident(col)
            overlay_expr = f"o{idx}.{col_ident}"
            existing_expr = select_expr_by_col.get(col)
            if existing_expr is None:
                select_expr_by_col[col] = overlay_expr
                select_order.append(col)
            else:
                select_expr_by_col[col] = f"COALESCE({overlay_expr}, {existing_expr})"

    select_exprs: list[str] = []
    for col in select_order:
        if requested is None or col in requested:
            select_exprs.append(f"{select_expr_by_col[col]} AS {dataset._sql_ident(col)}")
            selected_cols.append(col)

    if requested is not None:
        missing = [col for col in columns if col not in selected_cols]
        if missing:
            raise SchemaError(f"Requested columns not found after overlay merge: {', '.join(missing)}")

    query = "SELECT " + ", ".join(select_exprs) + f" FROM {base_view} b " + " ".join(join_clauses)
    if where:
        query += f" WHERE {where}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return con, query, (params or [])
