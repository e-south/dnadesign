"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_overlay_ops.py

Overlay attach and write operations extracted from Dataset methods.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .duckdb_runtime import connect_duckdb_utc
from .errors import NamespaceError, SchemaError
from .overlays import (
    OVERLAY_META_CREATED,
    OVERLAY_META_KEY,
    OVERLAY_META_NAMESPACE,
    OVERLAY_META_REGISTRY_HASH,
    overlay_dir_path,
    overlay_metadata,
    overlay_path,
    with_overlay_metadata,
)
from .schema import REQUIRED_COLUMNS
from .storage.locking import dataset_write_lock
from .storage.parquet import PARQUET_COMPRESSION, now_utc, read_parquet, write_parquet_atomic_batches


def attach_dataset(
    *,
    dataset: Any,
    path: Path,
    namespace: str,
    key: str,
    key_col: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    allow_overwrite: bool = False,
    allow_missing: bool = False,
    parse_json: bool = True,
    backend: str = "pyarrow",
    note: str = "",
    namespace_pattern: Any,
    reserved_namespaces: set[str],
) -> int:
    """Attach derived columns into an overlay keyed by an explicit join key."""
    dataset._require_exists()
    if not namespace_pattern.match(namespace):
        raise NamespaceError(
            "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
        )
    if namespace in reserved_namespaces:
        raise NamespaceError(f"Namespace '{namespace}' is reserved.")
    if backend not in {"pyarrow", "duckdb"}:
        raise SchemaError(f"Unsupported backend '{backend}'.")
    if backend == "duckdb" and parse_json:
        raise SchemaError("duckdb backend does not support JSON parsing. Use --no-parse-json or the pyarrow backend.")
    key = str(key).strip()
    if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
        raise SchemaError(f"Unsupported join key '{key}'.")
    if key_col is None:
        key_col = "sequence" if key in {"sequence", "sequence_norm", "sequence_ci"} else key
    part_dir = overlay_dir_path(dataset.dir, namespace)
    if part_dir.exists():
        raise SchemaError(
            f"Overlay parts already exist for namespace '{namespace}'. "
            "Use write_overlay_part or compact the parts first."
        )

    if backend == "duckdb":
        return attach_duckdb_dataset(
            dataset=dataset,
            path=path,
            namespace=namespace,
            key=key,
            key_col=key_col,
            columns=columns,
            allow_overwrite=allow_overwrite,
            allow_missing=allow_missing,
            note=note,
        )

    if path.suffix.lower() == ".parquet":
        inc = pq.read_table(path).to_pandas()
    elif path.suffix.lower() in {".csv"}:
        inc = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        inc = pd.read_json(path, lines=(path.suffix.lower() == ".jsonl"))
    else:
        raise SchemaError("Unsupported input format. Use parquet|csv|jsonl.")
    if key_col not in inc.columns:
        raise SchemaError(f"Missing key column '{key_col}' in incoming data.")

    rows_incoming = int(len(inc))
    row_nums = list(range(1, rows_incoming + 1))

    attach_cols = [c for c in inc.columns if c != key_col] if columns is None else list(columns)
    if not attach_cols:
        return 0

    work = inc[[key_col] + attach_cols].copy()

    def _normalize_optional_str(x: object) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.lower() in {"nan", "none"}:
            return None
        return s

    def _parse_jsonish(v: object, col: str, row_idx: int) -> object:
        if not parse_json:
            return v
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return v
        if s.startswith("[") or s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                if s.startswith("[") and ("'" in s) and ('"' not in s):
                    try:
                        return json.loads(s.replace("'", '"'))
                    except json.JSONDecodeError:
                        pass
                raise SchemaError(
                    f"Column '{col}' row {row_idx}: invalid JSON-like value. "
                    "Provide valid JSON or pass --no-parse-json."
                )
        return v

    if parse_json:
        for col in attach_cols:
            vals = work[col].tolist()
            parsed = [_parse_jsonish(v, col, i) for i, v in enumerate(vals, start=1)]
            work[col] = parsed

    def target_name(c: str) -> str:
        if c.startswith(namespace + "__"):
            return c
        if "__" in c:
            raise NamespaceError(f"Column '{c}' does not belong to namespace '{namespace}'.")
        return f"{namespace}__{c}"

    targets = [target_name(c) for c in attach_cols]

    key_vals_raw = [_normalize_optional_str(v) for v in work[key_col].tolist()]
    if key in {"sequence", "sequence_norm"}:
        key_vals = [None if v is None else str(v).strip() for v in key_vals_raw]
    elif key == "sequence_ci":
        key_vals = [None if v is None else str(v).strip().upper() for v in key_vals_raw]
    else:
        key_vals = [None if v is None else str(v) for v in key_vals_raw]

    missing_key_rows = [i for i, v in enumerate(key_vals, start=1) if v is None or str(v).strip() == ""]
    if missing_key_rows:
        sample = ", ".join(str(i) for i in missing_key_rows[:5])
        raise SchemaError(f"{len(missing_key_rows)} row(s) have missing key values (rows: {sample}).")

    dup_map: Dict[str, List[int]] = defaultdict(list)
    for k, row_num in zip(key_vals, row_nums):
        dup_map[str(k)].append(row_num)
    dup = {k: rows for k, rows in dup_map.items() if len(rows) > 1}
    if dup:
        preview = []
        for k, rows in list(dup.items())[:3]:
            rows_str = ",".join(str(r) for r in rows[:5])
            preview.append(f"{k} (rows {rows_str})")
        sample = "; ".join(preview)
        raise SchemaError(f"Duplicate keys in attachment input: {len(dup)} key(s) repeated. Sample: {sample}.")

    work.columns = [key] + targets

    essential = {k for k, _ in REQUIRED_COLUMNS}
    for t in targets:
        if t in essential:
            raise NamespaceError(f"Refusing to write essential column: {t}")
        if "__" not in t:
            raise NamespaceError(f"Derived columns must be namespaced (got '{t}').")

    def _write_overlay() -> int:
        dataset._auto_freeze_registry()
        base_cols = {"id"} if key == "id" else {"sequence", "alphabet", "bio_type"}
        base_tbl = read_parquet(dataset.records_path, columns=list(base_cols))
        key_vals_local = list(key_vals)
        work_local = work.copy()
        if key == "id":
            base_keys_list = [str(r) for r in base_tbl.column("id").to_pylist()]
            base_keys = set(base_keys_list)
        elif key in {"sequence", "sequence_norm", "sequence_ci"}:
            bio_vals = [str(b) for b in base_tbl.column("bio_type").to_pylist()]
            if any(b.strip() == "" for b in bio_vals):
                raise SchemaError("Missing bio_type values in base dataset.")
            if len(set(bio_vals)) != 1:
                raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
            seq_vals = [str(s).strip() for s in base_tbl.column("sequence").to_pylist()]
            if key == "sequence_ci":
                alph = [str(a) for a in base_tbl.column("alphabet").to_pylist()]
                if any(a != "dna_4" for a in alph):
                    raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                base_keys_list = [s.upper() for s in seq_vals]
            else:
                base_keys_list = seq_vals
            if len(base_keys_list) != len(set(base_keys_list)):
                raise SchemaError(f"Attach key requires unique base keys; duplicate base keys detected for '{key}'.")
            base_keys = set(base_keys_list)
        else:
            raise SchemaError(f"Unsupported join key '{key}'.")

        rows_missing_local = 0
        missing_keys = [k for k in key_vals_local if k not in base_keys]
        if missing_keys:
            if not allow_missing:
                sample = ", ".join(str(k) for k in missing_keys[:5])
                raise SchemaError(
                    f"{len(missing_keys)} row(s) reference keys not present in the dataset (sample: {sample})."
                )
            rows_missing_local = len(missing_keys)
            keep_mask = [k in base_keys for k in key_vals_local]
            work_local = work_local[keep_mask].reset_index(drop=True)
            key_vals_local = [k for k in key_vals_local if k in base_keys]

        overlay_df = work_local.copy()
        overlay_df[key] = key_vals_local

        out_path = overlay_path(dataset.dir, namespace)
        if out_path.exists():
            existing_df = pq.read_table(out_path).to_pandas()
            meta = overlay_metadata(out_path)
            if meta.get("key") != key:
                raise SchemaError(
                    f"Overlay key mismatch for namespace '{namespace}': existing={meta.get('key')} new={key}"
                )
            if key not in existing_df.columns:
                raise SchemaError(f"Existing overlay missing key column '{key}'.")
            if existing_df[key].duplicated().any():
                raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")
            existing_df = existing_df.set_index(key, drop=False)
            new_df = overlay_df.set_index(key, drop=False)
            overlap_cols = sorted((set(existing_df.columns) & set(new_df.columns)) - {key})
            if overlap_cols and not allow_overwrite:
                raise NamespaceError(f"Columns already exist: {', '.join(overlap_cols)}. Use --allow-overwrite.")
            combined = existing_df
            for col in new_df.columns:
                if col == key:
                    continue
                if col in combined.columns:
                    combined = combined.reindex(combined.index.union(new_df.index))
                    combined.loc[new_df.index, col] = new_df[col]
                else:
                    combined = combined.join(new_df[[col]], how="outer")
            combined[key] = combined.index
            overlay_df = combined.reset_index(drop=True)

        tbl = pa.Table.from_pandas(overlay_df, preserve_index=False)
        dataset._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
        reg_hash = dataset._registry_hash(required=True)
        tbl = with_overlay_metadata(
            tbl,
            namespace=namespace,
            key=key,
            created_at=now_utc(),
            registry_hash=reg_hash,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        pq.write_table(tbl, tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, out_path)

        rows_matched = int(overlay_df.shape[0])
        dataset._record_event(
            "attach",
            args={
                "namespace": namespace,
                "key": key,
                "rows_incoming": rows_incoming,
                "rows_matched": rows_matched,
                "rows_missing": rows_missing_local,
                "allow_overwrite": allow_overwrite,
                "note": note,
            },
            target_path=out_path,
        )
        return rows_matched

    with dataset_write_lock(dataset.dir):
        return _write_overlay()


def attach_duckdb_dataset(
    *,
    dataset: Any,
    path: Path,
    namespace: str,
    key: str,
    key_col: str,
    columns: Optional[Iterable[str]],
    allow_overwrite: bool,
    allow_missing: bool,
    note: str,
) -> int:
    """Attach derived columns using DuckDB for large parquet inputs."""
    if path.suffix.lower() != ".parquet":
        raise SchemaError("duckdb backend requires parquet input.")

    pf_in = pq.ParquetFile(str(path))
    incoming_cols = list(pf_in.schema_arrow.names)
    if key_col not in incoming_cols:
        raise SchemaError(f"Missing key column '{key_col}' in incoming data.")

    if columns is None:
        attach_cols = [c for c in incoming_cols if c != key_col]
    else:
        attach_cols = [c for c in columns]
        missing = [c for c in attach_cols if c not in incoming_cols]
        if missing:
            raise SchemaError(f"Requested columns not found in input: {', '.join(missing)}")
        if key_col in attach_cols:
            raise SchemaError(f"Key column '{key_col}' cannot be attached as a derived column.")

    if not attach_cols:
        return 0

    def _sql_ident(name: str) -> str:
        escaped = str(name).replace('"', '""')
        return f'"{escaped}"'

    def _key_expr(col: str) -> str:
        ident = _sql_ident(col)
        if key == "sequence_ci":
            return f"NULLIF(UPPER(TRIM(CAST({ident} AS VARCHAR))), '')"
        return f"NULLIF(TRIM(CAST({ident} AS VARCHAR)), '')"

    def target_name(c: str) -> str:
        if c.startswith(namespace + "__"):
            return c
        if "__" in c:
            raise NamespaceError(f"Column '{c}' does not belong to namespace '{namespace}'.")
        return f"{namespace}__{c}"

    targets = [target_name(c) for c in attach_cols]

    essential = {k for k, _ in REQUIRED_COLUMNS}
    for t in targets:
        if t in essential:
            raise NamespaceError(f"Refusing to write essential column: {t}")
        if "__" not in t:
            raise NamespaceError(f"Derived columns must be namespaced (got '{t}').")

    key_q = _sql_ident(key)
    select_exprs = [f"{_key_expr(key_col)} AS {key_q}"]
    for src_col, tgt in zip(attach_cols, targets):
        select_exprs.append(f"{_sql_ident(src_col)} AS {_sql_ident(tgt)}")
    incoming_select = ", ".join(select_exprs)

    rows_incoming = int(pf_in.metadata.num_rows)

    out_path = overlay_path(dataset.dir, namespace)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_overlay_duckdb() -> int:
        dataset._auto_freeze_registry()
        con = connect_duckdb_utc(
            missing_dependency_message="duckdb backend requires duckdb (install duckdb).",
            error_context="attach duckdb backend",
        )
        try:
            base_sql = str(dataset.records_path).replace("'", "''")
            con.execute(
                f"CREATE TEMP VIEW base AS SELECT id, sequence, alphabet, bio_type FROM read_parquet('{base_sql}')"
            )

            if key in {"sequence", "sequence_norm", "sequence_ci"}:
                bt_count = int(con.execute("SELECT COUNT(DISTINCT bio_type) FROM base").fetchone()[0])
                if bt_count > 1:
                    raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
            if key == "sequence_ci":
                bad = int(con.execute("SELECT COUNT(*) FROM base WHERE alphabet != 'dna_4'").fetchone()[0])
                if bad:
                    raise SchemaError("sequence_ci is only valid for dna_4 datasets.")

            incoming_sql = str(path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW incoming AS SELECT {incoming_select} FROM read_parquet('{incoming_sql}')")

            missing_keys = int(
                con.execute(f"SELECT COUNT(*) FROM incoming WHERE {key_q} IS NULL OR {key_q} = ''").fetchone()[0]
            )
            if missing_keys:
                raise SchemaError(f"{missing_keys} row(s) have missing key values in attachment input.")

            dup_keys = int(
                con.execute(
                    f"SELECT COUNT(*) FROM (SELECT {key_q} FROM incoming GROUP BY {key_q} HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            if dup_keys:
                preview = con.execute(
                    f"SELECT {key_q}, COUNT(*) AS cnt FROM incoming GROUP BY {key_q} HAVING cnt > 1 LIMIT 3"
                ).fetchall()
                sample = "; ".join(f"{row[0]} (count {row[1]})" for row in preview)
                raise SchemaError(f"Duplicate keys in attachment input: {dup_keys} key(s) repeated. Sample: {sample}.")

            if key in {"sequence", "sequence_norm", "sequence_ci"}:
                base_key_expr = _key_expr("sequence")
                dup_base = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM (SELECT {base_key_expr} AS k FROM base GROUP BY k HAVING COUNT(*) > 1)"
                    ).fetchone()[0]
                )
                if dup_base:
                    raise SchemaError(
                        f"Attach key requires unique base keys; duplicate base keys detected for '{key}'."
                    )
            else:
                base_key_expr = _key_expr("id")
            con.execute(f"CREATE TEMP VIEW base_keys AS SELECT {base_key_expr} AS k FROM base")

            rows_missing = int(
                con.execute(
                    f"SELECT COUNT(*) FROM incoming i LEFT JOIN base_keys b ON i.{key_q} = b.k WHERE b.k IS NULL"
                ).fetchone()[0]
            )
            if rows_missing and not allow_missing:
                raise SchemaError(
                    f"{rows_missing} row(s) reference keys not present in the dataset. Use --allow-missing to skip."
                )

            if rows_missing and allow_missing:
                con.execute(
                    "CREATE TEMP VIEW incoming_filtered AS "
                    f"SELECT i.* FROM incoming i JOIN base_keys b ON i.{key_q} = b.k"
                )
            else:
                con.execute("CREATE TEMP VIEW incoming_filtered AS SELECT * FROM incoming")

            tmp_path = out_path.with_suffix(".duckdb.tmp.parquet")
            tmp_sql = str(tmp_path).replace("'", "''")
            compression = PARQUET_COMPRESSION.upper()

            if out_path.exists():
                meta = overlay_metadata(out_path)
                if meta.get("key") != key:
                    raise SchemaError(
                        f"Overlay key mismatch for namespace '{namespace}': existing={meta.get('key')} new={key}"
                    )
                pf_existing = pq.ParquetFile(str(out_path))
                existing_cols = list(pf_existing.schema_arrow.names)
                if key not in existing_cols:
                    raise SchemaError(f"Existing overlay missing key column '{key}'.")

                existing_sql = str(out_path).replace("'", "''")
                con.execute(f"CREATE TEMP VIEW existing_overlay AS SELECT * FROM read_parquet('{existing_sql}')")
                dup_query = (
                    f"SELECT COUNT(*) FROM (SELECT {key_q} FROM existing_overlay GROUP BY {key_q} HAVING COUNT(*) > 1)"
                )
                dup_existing = int(con.execute(dup_query).fetchone()[0])
                if dup_existing:
                    raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")

                existing_set = set(existing_cols)
                overlap_cols = sorted((existing_set & set(targets)) - {key})
                if overlap_cols and not allow_overwrite:
                    raise NamespaceError(f"Columns already exist: {', '.join(overlap_cols)}. Use --allow-overwrite.")

                ordered_cols = (
                    [key] + [c for c in existing_cols if c != key] + [c for c in targets if c not in existing_set]
                )
                select_cols: List[str] = [f"COALESCE(e.{key_q}, n.{key_q}) AS {key_q}"]
                for col in ordered_cols[1:]:
                    col_q = _sql_ident(col)
                    if col in existing_set and col in targets:
                        select_cols.append(
                            f"CASE WHEN n.{key_q} IS NOT NULL THEN n.{col_q} ELSE e.{col_q} END AS {col_q}"
                        )
                    elif col in existing_set:
                        select_cols.append(f"e.{col_q} AS {col_q}")
                    else:
                        select_cols.append(f"n.{col_q} AS {col_q}")

                merge_query = (
                    "SELECT "
                    + ", ".join(select_cols)
                    + " FROM existing_overlay e FULL OUTER JOIN incoming_filtered n "
                    + f"ON e.{key_q} = n.{key_q}"
                )
                rows_matched = int(con.execute(f"SELECT COUNT(*) FROM ({merge_query})").fetchone()[0])
                con.execute(f"COPY ({merge_query}) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION '{compression}')")
            else:
                select_cols = [key_q] + [_sql_ident(c) for c in targets]
                merge_query = f"SELECT {', '.join(select_cols)} FROM incoming_filtered"
                rows_matched = int(con.execute(f"SELECT COUNT(*) FROM ({merge_query})").fetchone()[0])
                con.execute(f"COPY ({merge_query}) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION '{compression}')")

            pf_tmp = pq.ParquetFile(str(tmp_path))
            schema = pf_tmp.schema_arrow
            dataset._validate_registry_schema(namespace=namespace, schema=schema, key=key)
            metadata = dict(schema.metadata or {})
            metadata[OVERLAY_META_NAMESPACE.encode("utf-8")] = str(namespace).encode("utf-8")
            metadata[OVERLAY_META_KEY.encode("utf-8")] = str(key).encode("utf-8")
            metadata[OVERLAY_META_CREATED.encode("utf-8")] = str(now_utc()).encode("utf-8")
            reg_hash = dataset._registry_hash(required=True)
            if reg_hash:
                metadata[OVERLAY_META_REGISTRY_HASH.encode("utf-8")] = str(reg_hash).encode("utf-8")

            def _batches():
                for batch in pf_tmp.iter_batches(batch_size=65536):
                    yield batch

            write_parquet_atomic_batches(_batches(), schema, out_path, snapshot_dir=None, metadata=metadata)
            tmp_path.unlink(missing_ok=True)

            dataset._record_event(
                "attach",
                args={
                    "namespace": namespace,
                    "key": key,
                    "rows_incoming": rows_incoming,
                    "rows_matched": rows_matched,
                    "rows_missing": rows_missing if allow_missing else 0,
                    "allow_overwrite": allow_overwrite,
                    "note": note,
                },
                target_path=out_path,
            )
            return rows_matched
        finally:
            con.close()

    with dataset_write_lock(dataset.dir):
        return _write_overlay_duckdb()


def attach_columns_dataset(
    *,
    dataset: Any,
    path: Path,
    namespace: str,
    key: str,
    key_col: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    allow_overwrite: bool = False,
    allow_missing: bool = False,
    parse_json: bool = True,
    backend: str = "pyarrow",
    note: str = "",
    namespace_pattern: Any,
    reserved_namespaces: set[str],
) -> int:
    return attach_dataset(
        dataset=dataset,
        path=path,
        namespace=namespace,
        key=key,
        key_col=key_col,
        columns=columns,
        allow_overwrite=allow_overwrite,
        allow_missing=allow_missing,
        parse_json=parse_json,
        backend=backend,
        note=note,
        namespace_pattern=namespace_pattern,
        reserved_namespaces=reserved_namespaces,
    )


def write_overlay_dataset(
    *,
    dataset: Any,
    namespace: str,
    table_or_batches: Any,
    key: str = "id",
    overwrite: bool = False,
    allow_missing: bool = False,
    namespace_pattern: Any,
    reserved_namespaces: set[str],
) -> int:
    """Attach a derived overlay from an Arrow/Pandas table or batches."""
    if isinstance(table_or_batches, pa.Table):
        tbl = table_or_batches
    elif isinstance(table_or_batches, pd.DataFrame):
        tbl = pa.Table.from_pandas(table_or_batches, preserve_index=False)
    else:
        tbl = pa.Table.from_batches(list(table_or_batches))

    dataset._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
    if key not in tbl.schema.names:
        raise SchemaError(f"Overlay table missing key column '{key}'.")
    attach_cols = [c for c in tbl.schema.names if c != key]
    if not attach_cols:
        return 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "overlay.parquet"
        pq.write_table(tbl, tmp_path, compression=PARQUET_COMPRESSION)
        return attach_dataset(
            dataset=dataset,
            path=tmp_path,
            namespace=namespace,
            key=key,
            key_col=key,
            columns=attach_cols,
            allow_overwrite=overwrite,
            allow_missing=allow_missing,
            parse_json=False,
            backend="pyarrow",
            note="",
            namespace_pattern=namespace_pattern,
            reserved_namespaces=reserved_namespaces,
        )


def write_overlay_part_dataset(
    *,
    dataset: Any,
    namespace: str,
    table_or_batches: Any,
    key: str = "id",
    key_col: Optional[str] = None,
    allow_missing: bool = False,
    actor: Optional[dict] = None,
) -> int:
    """Append an overlay part file under _derived/<namespace>/part-*.parquet."""
    dataset._require_exists()
    key = str(key or "").strip()
    if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
        raise SchemaError(f"Unsupported overlay key '{key}'.")
    key_col = str(key_col or key)

    file_path = overlay_path(dataset.dir, namespace)
    dir_path = overlay_dir_path(dataset.dir, namespace)
    if file_path.exists():
        raise SchemaError(
            f"Overlay file already exists for namespace '{namespace}'. Remove it or compact it before writing parts."
        )

    if isinstance(table_or_batches, pa.Table):
        tbl = table_or_batches
    elif isinstance(table_or_batches, pd.DataFrame):
        tbl = pa.Table.from_pandas(table_or_batches, preserve_index=False)
    else:
        batches = list(table_or_batches)
        if not batches:
            return 0
        tbl = pa.Table.from_batches(batches)

    if key_col not in tbl.schema.names:
        raise SchemaError(f"Overlay table missing key column '{key_col}'.")
    if key_col != key:
        if key in tbl.schema.names:
            raise SchemaError(f"Overlay table already contains a '{key}' column; cannot rename '{key_col}'.")
        cols = [key if c == key_col else c for c in tbl.schema.names]
        tbl = tbl.rename_columns(cols)

    attach_cols = [c for c in tbl.schema.names if c != key]
    if not attach_cols:
        return 0

    essential = {k for k, _ in REQUIRED_COLUMNS}
    for col in attach_cols:
        if col in essential:
            raise NamespaceError(f"Overlay cannot modify required column '{col}'.")
        if "__" not in col:
            raise NamespaceError(f"Derived columns must be namespaced (got '{col}').")

    dataset._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)

    def _write_part() -> int:
        def _sql_ident(name: str) -> str:
            escaped = str(name).replace('"', '""')
            return f'"{escaped}"'

        def _key_expr(expr: str, *, key_name: str) -> str:
            if key_name == "sequence_ci":
                return f"NULLIF(UPPER(TRIM(CAST({expr} AS VARCHAR))), '')"
            return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"

        con = connect_duckdb_utc(
            missing_dependency_message="write_overlay_part requires duckdb (install duckdb).",
            error_context="write_overlay_part",
        )
        try:
            base_sql = str(dataset.records_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{base_sql}')")
            con.register("incoming", tbl)

            incoming_key_expr = _key_expr(f"i.{_sql_ident(key)}", key_name=key)

            dup_incoming = int(
                con.execute(
                    "SELECT COUNT(*) FROM "
                    f"(SELECT {incoming_key_expr} AS k FROM incoming i "
                    "GROUP BY k HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            if dup_incoming:
                raise SchemaError(f"Overlay part has duplicate keys for '{key}'.")

            if key in {"sequence", "sequence_norm", "sequence_ci"}:
                bt_count = int(con.execute("SELECT COUNT(DISTINCT bio_type) FROM base").fetchone()[0])
                if bt_count > 1:
                    raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
                if key == "sequence_ci":
                    bad = int(con.execute("SELECT COUNT(*) FROM base WHERE alphabet != 'dna_4'").fetchone()[0])
                    if bad:
                        raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                base_key_expr = _key_expr(f"b.{_sql_ident('sequence')}", key_name=key)
                dup_base = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM (SELECT {base_key_expr} AS k FROM base b GROUP BY k HAVING COUNT(*) > 1)"
                    ).fetchone()[0]
                )
                if dup_base:
                    raise SchemaError(
                        f"Attach key requires unique base keys; duplicate base keys detected for '{key}'."
                    )
            else:
                base_key_expr = _key_expr(f"b.{_sql_ident('id')}", key_name=key)

            missing = int(
                con.execute(
                    "SELECT COUNT(*) FROM incoming i "
                    f"LEFT JOIN base b ON {base_key_expr} = {incoming_key_expr} "
                    "WHERE b.id IS NULL"
                ).fetchone()[0]
            )
            if missing and not allow_missing:
                raise SchemaError(f"{missing} row(s) reference keys not present in the dataset.")

            if allow_missing:
                tbl_out = con.execute(
                    f"SELECT i.* FROM incoming i JOIN base b ON {base_key_expr} = {incoming_key_expr}"
                ).fetch_arrow_table()
            else:
                tbl_out = tbl
        finally:
            con.close()

        rows_incoming = int(tbl.num_rows)
        rows_written = int(tbl_out.num_rows)
        rows_missing = rows_incoming - rows_written
        if rows_written == 0:
            return 0

        reg_hash = dataset._registry_hash(required=True)
        tbl_out = with_overlay_metadata(
            tbl_out,
            namespace=namespace,
            key=key,
            created_at=now_utc(),
            registry_hash=reg_hash,
        )

        dir_path.mkdir(parents=True, exist_ok=True)
        stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
        part_path = dir_path / f"part-{stamp}-{uuid.uuid4().hex}.parquet"
        tmp_path = part_path.with_suffix(".parquet.tmp")
        try:
            pq.write_table(tbl_out, tmp_path, compression=PARQUET_COMPRESSION)
            os.replace(tmp_path, part_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        dataset._record_event(
            "write_overlay_part",
            args={
                "namespace": namespace,
                "key": key,
                "rows_incoming": rows_incoming,
                "rows_written": rows_written,
                "rows_missing": rows_missing,
                "allow_missing": allow_missing,
            },
            metrics={
                "rows_incoming": rows_incoming,
                "rows_written": rows_written,
                "rows_missing": rows_missing,
            },
            artifacts={"overlay": {"namespace": namespace, "key": key}},
            target_path=part_path,
            actor=actor,
        )
        return rows_written

    with dataset_write_lock(dataset.dir):
        dataset._auto_freeze_registry()
        return _write_part()
