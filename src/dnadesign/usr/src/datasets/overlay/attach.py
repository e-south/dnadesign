"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/overlay/attach.py

Overlay attach helpers for dataset overlay mutations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...duckdb_runtime import connect_duckdb_utc
from ...errors import NamespaceError, SchemaError
from ...overlays import (
    OVERLAY_META_CREATED,
    OVERLAY_META_KEY,
    OVERLAY_META_NAMESPACE,
    OVERLAY_META_NAMESPACE_CONTRACT_HASH,
    OVERLAY_META_REGISTRY_HASH,
    overlay_metadata,
    overlay_path,
    with_overlay_metadata,
)
from ...registry import namespace_contract_hash_for_entries, registry_entry
from ...storage.locking import dataset_write_lock
from ...storage.parquet import PARQUET_COMPRESSION, now_utc, read_parquet, write_parquet_atomic_batches
from .policy import (
    _overlay_table_from_registry,
    ensure_overlay_columns_allowed,
    normalize_overlay_targets,
    validate_overlay_target,
)


def _read_attach_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pq.read_table(path).to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    raise SchemaError("Unsupported input format. Use parquet|csv|jsonl.")


def _merge_attach_event_args(
    *,
    namespace: str,
    key: str,
    rows_incoming: int,
    rows_matched: int,
    rows_missing: int,
    allow_overwrite: bool,
    note: str,
    event_args: Mapping[str, object] | None,
) -> dict[str, object]:
    args: dict[str, object] = {
        "namespace": namespace,
        "key": key,
        "rows_incoming": rows_incoming,
        "rows_matched": rows_matched,
        "rows_missing": rows_missing,
        "allow_overwrite": allow_overwrite,
        "note": note,
    }
    if event_args is None:
        return args
    for event_key, event_value in event_args.items():
        if event_key in args:
            raise SchemaError(f"Attach event_args cannot override reserved key '{event_key}'.")
        args[event_key] = event_value
    return args


def _merge_overlay_frame(
    *,
    existing_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    key: str,
    allow_overwrite: bool,
    fail_on_non_null_overwrite: bool = False,
) -> pd.DataFrame:
    if key not in existing_df.columns:
        raise SchemaError(f"Existing overlay missing key column '{key}'.")
    if existing_df[key].duplicated().any():
        raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")

    existing_indexed = existing_df.set_index(key, drop=False)
    incoming_indexed = incoming_df.set_index(key, drop=False)
    overlap_cols = sorted((set(existing_indexed.columns) & set(incoming_indexed.columns)) - {key})
    if overlap_cols and not allow_overwrite:
        raise NamespaceError(f"Columns already exist: {', '.join(overlap_cols)}. Use --allow-overwrite.")
    if fail_on_non_null_overwrite and overlap_cols:
        existing_for_incoming = existing_indexed.reindex(incoming_indexed.index)
        for col in overlap_cols:
            occupied = existing_for_incoming[col].notna()
            if not occupied.any():
                continue
            collision_ids = [str(row_id) for row_id in existing_for_incoming.index[occupied].tolist()]
            sample = ", ".join(collision_ids[:5])
            raise SchemaError(
                f"Refusing overwrite for existing values in column '{col}' (sample ids: {sample}). "
                "Re-run with overwrite=true."
            )

    all_index = existing_indexed.index.union(incoming_indexed.index)
    combined = existing_indexed.reindex(all_index)
    incoming_cols = [col for col in incoming_indexed.columns if col != key]
    for col in incoming_cols:
        if col not in combined.columns:
            combined[col] = pd.NA
    if incoming_cols:
        combined.loc[incoming_indexed.index, incoming_cols] = incoming_indexed[incoming_cols]
    combined[key] = combined.index

    ordered_cols = [key, *[col for col in existing_df.columns if col != key]]
    for col in incoming_df.columns:
        if col != key and col not in ordered_cols:
            ordered_cols.append(col)
    return combined.loc[:, ordered_cols].reset_index(drop=True)


def _attach_frame_dataset(
    *,
    dataset: Any,
    incoming: pd.DataFrame,
    namespace: str,
    key: str,
    key_col: str,
    columns: Optional[Iterable[str]] = None,
    allow_overwrite: bool = False,
    allow_missing: bool = False,
    parse_json: bool = True,
    fail_on_non_null_overwrite: bool = False,
    note: str = "",
    actor: Optional[dict] = None,
    event_args: Mapping[str, object] | None = None,
    reserved_namespaces: set[str],
    write_lock=dataset_write_lock,
) -> int:
    if key_col not in incoming.columns:
        raise SchemaError(f"Missing key column '{key_col}' in incoming data.")

    rows_incoming = int(len(incoming))
    row_nums = list(range(1, rows_incoming + 1))

    attach_cols = [c for c in incoming.columns if c != key_col] if columns is None else list(columns)
    if not attach_cols:
        return 0

    work = incoming[[key_col] + attach_cols].copy()

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

    targets = normalize_overlay_targets(namespace, attach_cols)

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
    for resolved_key, row_num in zip(key_vals, row_nums):
        dup_map[str(resolved_key)].append(row_num)
    dup = {resolved_key: rows for resolved_key, rows in dup_map.items() if len(rows) > 1}
    if dup:
        preview = []
        for resolved_key, rows in list(dup.items())[:3]:
            rows_str = ",".join(str(r) for r in rows[:5])
            preview.append(f"{resolved_key} (rows {rows_str})")
        sample = "; ".join(preview)
        raise SchemaError(f"Duplicate keys in attachment input: {len(dup)} key(s) repeated. Sample: {sample}.")

    work.columns = [key] + targets
    ensure_overlay_columns_allowed(targets)

    def _write_overlay() -> int:
        dataset._auto_freeze_registry()
        base_cols = {"id"} if key == "id" else {"sequence", "alphabet", "bio_type"}
        base_tbl = read_parquet(dataset.records_path, columns=list(base_cols))
        key_vals_local = list(key_vals)
        work_local = work.copy()
        if key == "id":
            base_keys_list = [str(record_id) for record_id in base_tbl.column("id").to_pylist()]
            base_keys = set(base_keys_list)
        elif key in {"sequence", "sequence_norm", "sequence_ci"}:
            bio_vals = [str(bio_type) for bio_type in base_tbl.column("bio_type").to_pylist()]
            if any(bio_type.strip() == "" for bio_type in bio_vals):
                raise SchemaError("Missing bio_type values in base dataset.")
            if len(set(bio_vals)) != 1:
                raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
            seq_vals = [str(sequence).strip() for sequence in base_tbl.column("sequence").to_pylist()]
            if key == "sequence_ci":
                alph = [str(alphabet) for alphabet in base_tbl.column("alphabet").to_pylist()]
                if any(alphabet != "dna_4" for alphabet in alph):
                    raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                base_keys_list = [sequence.upper() for sequence in seq_vals]
            else:
                base_keys_list = seq_vals
            if len(base_keys_list) != len(set(base_keys_list)):
                raise SchemaError(f"Attach key requires unique base keys; duplicate base keys detected for '{key}'.")
            base_keys = set(base_keys_list)
        else:
            raise SchemaError(f"Unsupported join key '{key}'.")

        rows_missing_local = 0
        missing_keys = [resolved_key for resolved_key in key_vals_local if resolved_key not in base_keys]
        if missing_keys:
            if not allow_missing:
                sample = ", ".join(str(resolved_key) for resolved_key in missing_keys[:5])
                raise SchemaError(
                    f"{len(missing_keys)} row(s) reference keys not present in the dataset (sample: {sample})."
                )
            rows_missing_local = len(missing_keys)
            keep_mask = [resolved_key in base_keys for resolved_key in key_vals_local]
            work_local = work_local[keep_mask].reset_index(drop=True)
            key_vals_local = [resolved_key for resolved_key in key_vals_local if resolved_key in base_keys]

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
            overlay_df = _merge_overlay_frame(
                existing_df=existing_df,
                incoming_df=overlay_df,
                key=key,
                allow_overwrite=allow_overwrite,
                fail_on_non_null_overwrite=fail_on_non_null_overwrite,
            )

        if namespace in reserved_namespaces:
            tbl = pa.Table.from_pandas(overlay_df, preserve_index=False)
        else:
            registry = dataset._registry(required=True)
            entry = registry_entry(registry, namespace)
            tbl = _overlay_table_from_registry(overlay_df, entry=entry, key=key)
        dataset._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
        registry = dataset._registry(required=True)
        reg_hash = dataset._registry_hash(required=True)
        namespace_hash = namespace_contract_hash_for_entries(registry, namespace)
        tbl = with_overlay_metadata(
            tbl,
            namespace=namespace,
            key=key,
            created_at=now_utc(),
            registry_hash=reg_hash,
            namespace_contract_hash=namespace_hash,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        pq.write_table(tbl, tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, out_path)

        rows_matched = int(overlay_df.shape[0])
        dataset._record_event(
            "attach",
            args=_merge_attach_event_args(
                namespace=namespace,
                key=key,
                rows_incoming=rows_incoming,
                rows_matched=rows_matched,
                rows_missing=rows_missing_local,
                allow_overwrite=allow_overwrite,
                note=note,
                event_args=event_args,
            ),
            target_path=out_path,
            actor=actor,
        )
        return rows_matched

    with write_lock(dataset.dir):
        return _write_overlay()


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
    actor: Optional[dict] = None,
    event_args: Mapping[str, object] | None = None,
    namespace_pattern: Any,
    reserved_namespaces: set[str],
    write_lock=dataset_write_lock,
) -> int:
    """Attach derived columns into an overlay keyed by an explicit join key."""
    key = validate_overlay_target(
        dataset=dataset,
        namespace=namespace,
        key=key,
        namespace_pattern=namespace_pattern,
        reserved_namespaces=reserved_namespaces,
    )
    if backend not in {"pyarrow", "duckdb"}:
        raise SchemaError(f"Unsupported backend '{backend}'.")
    if backend == "duckdb" and parse_json:
        raise SchemaError("duckdb backend does not support JSON parsing. Use --no-parse-json or the pyarrow backend.")
    if key_col is None:
        key_col = "sequence" if key in {"sequence", "sequence_norm", "sequence_ci"} else key

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
            event_args=event_args,
            write_lock=write_lock,
        )

    return _attach_frame_dataset(
        dataset=dataset,
        incoming=_read_attach_input(path),
        namespace=namespace,
        key=key,
        key_col=key_col,
        columns=columns,
        allow_overwrite=allow_overwrite,
        allow_missing=allow_missing,
        parse_json=parse_json,
        note=note,
        actor=actor,
        event_args=event_args,
        reserved_namespaces=reserved_namespaces,
        write_lock=write_lock,
    )


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
    event_args: Mapping[str, object] | None,
    write_lock=dataset_write_lock,
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

    targets = normalize_overlay_targets(namespace, attach_cols)
    ensure_overlay_columns_allowed(targets)

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
            registry = dataset._registry(required=True)
            reg_hash = dataset._registry_hash(required=True)
            if reg_hash:
                metadata[OVERLAY_META_REGISTRY_HASH.encode("utf-8")] = str(reg_hash).encode("utf-8")
            metadata[OVERLAY_META_NAMESPACE_CONTRACT_HASH.encode("utf-8")] = str(
                namespace_contract_hash_for_entries(registry, namespace)
            ).encode("utf-8")

            def _batches():
                for batch in pf_tmp.iter_batches(batch_size=65536):
                    yield batch

            write_parquet_atomic_batches(_batches(), schema, out_path, snapshot_dir=None, metadata=metadata)
            tmp_path.unlink(missing_ok=True)

            dataset._record_event(
                "attach",
                args=_merge_attach_event_args(
                    namespace=namespace,
                    key=key,
                    rows_incoming=rows_incoming,
                    rows_matched=rows_matched,
                    rows_missing=rows_missing if allow_missing else 0,
                    allow_overwrite=allow_overwrite,
                    note=note,
                    event_args=event_args,
                ),
                target_path=out_path,
            )
            return rows_matched
        finally:
            con.close()

    with write_lock(dataset.dir):
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
    actor: Optional[dict] = None,
    event_args: Mapping[str, object] | None = None,
    namespace_pattern: Any,
    reserved_namespaces: set[str],
    write_lock=dataset_write_lock,
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
        actor=actor,
        event_args=event_args,
        namespace_pattern=namespace_pattern,
        reserved_namespaces=reserved_namespaces,
        write_lock=write_lock,
    )
