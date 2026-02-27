"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_materialize.py

Materialize operation logic for merging overlays into base USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pyarrow.parquet as pq

from .duckdb_runtime import connect_duckdb_utc
from .errors import NamespaceError, SchemaError
from .events import fingerprint_parquet
from .maintenance import require_maintenance
from .overlays import list_overlays, overlay_metadata, overlay_schema
from .registry import validate_overlay_schema
from .schema import REQUIRED_COLUMNS
from .storage.locking import dataset_write_lock
from .storage.parquet import PARQUET_COMPRESSION, iter_parquet_batches, now_utc, write_parquet_atomic_batches
from .types import Fingerprint


def materialize_dataset(
    *,
    dataset: Any,
    namespaces: Optional[Sequence[str]],
    keep_overlays: bool,
    archive_overlays: bool,
    drop_deleted: bool,
    reserved_namespaces: set[str],
) -> None:
    """Merge overlays into base records for a dataset instance."""
    ctx = require_maintenance("materialize")
    with dataset_write_lock(dataset.dir):
        dataset._require_exists()
        dataset._auto_freeze_registry()
        overlays_all = list_overlays(dataset.dir)
        if not overlays_all and not drop_deleted:
            return

        def _overlay_ns(path: Path) -> str:
            meta = overlay_metadata(path)
            return meta.get("namespace") or path.stem

        if namespaces is None:
            overlays = [p for p in overlays_all if _overlay_ns(p) not in reserved_namespaces]
        else:
            ns_set = {str(n) for n in namespaces}
            overlays = [p for p in overlays_all if _overlay_ns(p) in ns_set]

        if not overlays and not drop_deleted:
            return

        require_registry = any(_overlay_ns(p) not in reserved_namespaces for p in overlays)
        registry = dataset._registry(required=require_registry) if require_registry else {}

        def _key_expr(expr: str, *, key: str) -> str:
            if key == "sequence_ci":
                return f"NULLIF(UPPER(TRIM(CAST({expr} AS VARCHAR))), '')"
            return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"

        base_pf = pq.ParquetFile(str(dataset.records_path))
        base_cols = list(base_pf.schema_arrow.names)
        essential = {k for k, _ in REQUIRED_COLUMNS}

        tmp_path = dataset.records_path.with_suffix(".materialize.parquet")
        con = connect_duckdb_utc(
            missing_dependency_message="materialize requires duckdb (install duckdb).",
            error_context="materialize",
        )
        try:
            base_sql = str(dataset.records_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{base_sql}')")
            base_view = "base"

            if drop_deleted and dataset._tombstone_path().exists():
                tomb_path = dataset._tombstone_path()
                meta = overlay_metadata(tomb_path)
                key = meta.get("key")
                if key and key != "id":
                    raise SchemaError("Tombstone overlay must use key 'id'.")
                tomb_sql = str(tomb_path).replace("'", "''")
                con.execute(f"CREATE TEMP VIEW tombstone AS SELECT id, usr__deleted FROM read_parquet('{tomb_sql}')")
                con.execute(
                    "CREATE TEMP VIEW base_filtered AS "
                    "SELECT b.* FROM base b LEFT JOIN tombstone t ON b.id = t.id "
                    "WHERE COALESCE(t.usr__deleted, FALSE) = FALSE"
                )
                base_view = "base_filtered"

            select_expr_by_col = {col: f"b.{dataset._sql_ident(col)}" for col in base_cols}
            select_order = list(base_cols)
            join_clauses: List[str] = []

            for idx, path in enumerate(overlays):
                meta = overlay_metadata(path)
                key = meta.get("key")
                if not key:
                    raise SchemaError(f"Overlay missing required metadata key: {path}")
                if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
                    raise SchemaError(f"Unsupported overlay key '{key}': {path}")

                schema = overlay_schema(path)
                overlay_cols = list(schema.names)
                if key not in overlay_cols:
                    raise SchemaError(f"Overlay missing key column '{key}': {path}")
                validate_overlay_schema(_overlay_ns(path), schema, registry=registry, key=key)

                derived_cols = [c for c in overlay_cols if c != key]
                if not derived_cols:
                    raise SchemaError(f"Overlay '{path.name}' has no derived columns.")

                for col in derived_cols:
                    if col in essential:
                        raise NamespaceError(f"Overlay cannot modify required column '{col}'.")
                    if "__" not in col:
                        raise NamespaceError(f"Derived columns must be namespaced (got '{col}').")

                view_name = f"overlay_{idx}"
                dataset._create_overlay_view(con, view_name=view_name, path=path, key=key)

                if key in {"sequence", "sequence_norm", "sequence_ci"}:
                    bt_count = int(con.execute(f"SELECT COUNT(DISTINCT bio_type) FROM {base_view}").fetchone()[0])
                    if bt_count > 1:
                        raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
                    if key == "sequence_ci":
                        bad = int(
                            con.execute(f"SELECT COUNT(*) FROM {base_view} WHERE alphabet != 'dna_4'").fetchone()[0]
                        )
                        if bad:
                            raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                    base_key_expr = _key_expr(f"b.{dataset._sql_ident('sequence')}", key=key)
                    dup_base = int(
                        con.execute(
                            "SELECT COUNT(*) FROM "
                            f"(SELECT {base_key_expr} AS k FROM {base_view} b GROUP BY k HAVING COUNT(*) > 1)"
                        ).fetchone()[0]
                    )
                    if dup_base:
                        raise SchemaError(
                            f"Attach key requires unique base keys; duplicate base keys detected for '{key}'."
                        )
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

            select_exprs = [f"{select_expr_by_col[col]} AS {dataset._sql_ident(col)}" for col in select_order]
            query = "SELECT " + ", ".join(select_exprs) + f" FROM {base_view} b " + " ".join(join_clauses)
            tmp_sql = str(tmp_path).replace("'", "''")
            compression = PARQUET_COMPRESSION.upper()
            con.execute(f"COPY ({query}) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION '{compression}')")
        finally:
            con.close()

        pf_tmp = pq.ParquetFile(str(tmp_path))
        schema = pf_tmp.schema_arrow
        metadata = dataset._base_metadata(created_at=now_utc())

        def _batches():
            for batch in iter_parquet_batches(tmp_path, columns=None):
                yield batch

        write_parquet_atomic_batches(
            _batches(),
            schema,
            dataset.records_path,
            dataset.snapshot_dir,
            metadata=metadata,
        )
        tmp_path.unlink(missing_ok=True)

        overlay_fingerprints = []
        for p in overlays:
            meta = overlay_metadata(p)
            if p.is_dir():
                parts = sorted(p.glob("part-*.parquet"))
                if not parts:
                    raise SchemaError(f"Overlay has no parquet parts: {p}")
                schema = overlay_schema(p)
                rows = 0
                size_bytes = 0
                for part in parts:
                    pf_part = pq.ParquetFile(str(part))
                    rows += pf_part.metadata.num_rows
                    size_bytes += int(part.stat().st_size)
                fp = Fingerprint(rows=int(rows), cols=int(len(schema.names)), size_bytes=int(size_bytes)).to_dict()
            else:
                fp = fingerprint_parquet(p).to_dict()
            overlay_fingerprints.append(
                {
                    "namespace": meta.get("namespace") or p.stem,
                    "key": meta.get("key"),
                    "fingerprint": fp,
                }
            )
        tomb_fp = None
        if drop_deleted and dataset._tombstone_path().exists():
            tomb_fp = fingerprint_parquet(dataset._tombstone_path()).to_dict()

        if archive_overlays:
            archive_dir = dataset.dir / "_derived" / "_archived"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
            for p in overlays:
                if p.is_dir():
                    archived = archive_dir / f"{p.name}-{stamp}"
                else:
                    archived = archive_dir / f"{p.stem}-{stamp}.parquet"
                p.replace(archived)
        elif not keep_overlays:
            for p in overlays:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

        dataset._record_event(
            "materialize",
            args={
                "namespaces": list(namespaces) if namespaces is not None else None,
                "drop_overlays": bool(not keep_overlays or archive_overlays),
                "archive_overlays": bool(archive_overlays),
                "drop_deleted": bool(drop_deleted),
                "overlays": overlay_fingerprints,
                "tombstone": tomb_fp,
                "maintenance_reason": ctx.reason,
            },
            artifacts={"overlays": overlay_fingerprints, "tombstone": tomb_fp},
            maintenance={"reason": ctx.reason},
            actor=ctx.actor,
        )
