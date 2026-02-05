"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/merge_datasets.py

USR dataset merge logic and conflict handling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import Dataset
from .errors import SchemaError, ValidationError
from .events import record_event
from .io import PARQUET_COMPRESSION, iter_parquet_batches, now_utc, write_parquet_atomic_batches
from .locks import dataset_write_lock
from .schema import REQUIRED_COLUMNS

# -------------------- public enums & preview --------------------


class MergePolicy(Enum):
    ERROR = "error"
    SKIP = "skip"
    PREFER_SRC = "prefer-src"
    PREFER_DEST = "prefer-dest"


class MergeColumnsMode(Enum):
    REQUIRE_SAME = "require-same"
    UNION = "union"


@dataclass(frozen=True)
class MergePreview:
    dest_rows_before: int
    src_rows: int
    # duplicates accounting by id (present in dest ∩ src)
    duplicates_total: int
    duplicates_skipped: int
    duplicates_replaced: int
    duplicate_policy: MergePolicy
    # final row math
    new_rows: int
    dest_rows_after: int
    # column summary (names count)
    columns_total: int
    overlapping_columns: int


# -------------------- helpers --------------------

_ESSENTIAL = [k for k, _ in REQUIRED_COLUMNS]
_ESSENTIAL_SET = set(_ESSENTIAL)


def _ordered_fields(schema: pa.Schema) -> List[pa.Field]:
    return [schema.field(i) for i in range(schema.num_fields)]


def _field_map(schema: pa.Schema):
    return {f.name: f for f in schema}


def _casefold_keys(tbl: pa.Table) -> set[tuple[str, str]]:
    bt = [str(x) for x in tbl.column("bio_type").to_pylist()]
    sq = [str(x) for x in tbl.column("sequence").to_pylist()]
    return {(b.lower(), s.upper()) for b, s in zip(bt, sq)}


def _coerce_jsonish_to_type(col: "pa.ChunkedArray | pa.Array", target_type: pa.DataType, colname: str) -> pa.Array:
    """
    Best-effort coercion used during USR↔USR merge when overlapping columns disagree
    on type and the user has requested `--coerce-overlap to-dest`.

    Handles:
      - numeric strings → numbers (int/float)
      - truthy strings → bool
      - JSON arrays/objects in strings → list/struct
      - empty string / 'null' / None → NULL

    Returns a pyarrow.Array with the *destination* type.
    """
    import math

    vals = col.to_pylist()

    def _is_nullish(v) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            s = v.strip().lower()
            return s == "" or s == "null" or s == "none" or s == "nan"
        return False

    def _to_int(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int,)):
            return int(v)
        if isinstance(v, float):
            # Avoid arrow casting surprises on 3.0 → 3
            if math.isnan(v):
                return None
            return int(v)
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            try:
                return int(s)
            except (TypeError, ValueError, OverflowError):
                return int(float(s))  # allow "3.0"
        return None if _is_nullish(v) else int(v)

    def _to_float(v):
        if isinstance(v, (int, float, bool)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return None
            return float(s)
        return None if _is_nullish(v) else float(v)

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "t", "yes", "y", "1"}:
                return True
            if s in {"false", "f", "no", "n", "0"}:
                return False
            # fallthrough: non-empty strings → True, but that’s surprising; treat as error
            raise SchemaError(f"Cannot coerce value '{v}' to bool for column '{colname}'.")
        if _is_nullish(v):
            return None
        return bool(v)

    def _parse_jsonish(v):
        if v is None:
            return None
        if isinstance(v, (list, dict)):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in {"null", "none"}:
                return None
            try:
                return json.loads(s)
            except (TypeError, ValueError) as e:
                raise SchemaError(f"Column '{colname}': failed to parse JSON: {e}")
        return v

    # Elementwise normalization to match target_type
    if pa.types.is_integer(target_type):
        norm = [None if _is_nullish(v) else _to_int(v) for v in vals]
        return pa.array(norm, type=target_type)
    if pa.types.is_floating(target_type):
        norm = [None if _is_nullish(v) else _to_float(v) for v in vals]
        return pa.array(norm, type=target_type)
    if pa.types.is_boolean(target_type):
        norm = [None if _is_nullish(v) else _to_bool(v) for v in vals]
        return pa.array(norm, type=target_type)
    if pa.types.is_string(target_type):
        norm = [None if _is_nullish(v) else str(v) for v in vals]
        return pa.array(norm, type=target_type)
    if pa.types.is_list(target_type) or pa.types.is_large_list(target_type):
        norm = [_parse_jsonish(v) for v in vals]
        return pa.array(norm, type=target_type)
    if pa.types.is_struct(target_type):
        norm = [_parse_jsonish(v) for v in vals]
        return pa.array(norm, type=target_type)
    # For timestamps and other scalar types, let Arrow do the heavy lifting
    return pa.array([None if _is_nullish(v) else v for v in vals], type=target_type)


def _ensure_columns_same(dest: pa.Schema, src: pa.Schema) -> List[pa.Field]:
    """
    Require exact same column names and Arrow types (order can differ).
    Return a consistent ordered schema (use destination order).
    """
    dm, sm = _field_map(dest), _field_map(src)
    if set(dm.keys()) != set(sm.keys()):
        missing_in_src = sorted(set(dm.keys()) - set(sm.keys()))
        missing_in_dest = sorted(set(sm.keys()) - set(dm.keys()))
        raise SchemaError(
            "Column sets differ.\n"
            + (f"  Missing in src : {', '.join(missing_in_src)}\n" if missing_in_src else "")
            + (f"  Missing in dest: {', '.join(missing_in_dest)}\n" if missing_in_dest else "")
        )
    # types match
    for name in dm.keys():
        if dm[name].type != sm[name].type:
            raise SchemaError(f"Column type mismatch for '{name}': {dm[name].type} (dest) vs {sm[name].type} (src)")
    # use dest order
    return _ordered_fields(dest)


def _union_columns(dest: pa.Schema, src: pa.Schema) -> Tuple[List[pa.Field], int, int]:
    """
    Allow column union. Overlapping names must have the same Arrow type.
    Schema order: essential columns first (in USR order), then the rest
    in destination order, then remaining from source.

    Returns (fields, total_column_count, overlapping_column_count).
    """
    dm, sm = _field_map(dest), _field_map(src)
    all_names = list(dict.fromkeys(list(dm.keys()) + list(sm.keys())))
    overlap = set(dm.keys()).intersection(set(sm.keys()))
    # check overlapping type equality
    for n in overlap:
        if dm[n].type != sm[n].type:
            raise SchemaError(f"Overlapping column '{n}' has different types: {dm[n].type} vs {sm[n].type}")

    # Order: essential first (USR order), then others
    def essential_first(names: List[str]) -> List[str]:
        essentials = [n for n in _ESSENTIAL if n in names]
        others = [n for n in names if n not in _ESSENTIAL_SET]
        return essentials + others

    ordered_names = essential_first(all_names)

    fields: List[pa.Field] = []
    for n in ordered_names:
        if n in dm:
            fields.append(dm[n])
        else:
            fields.append(sm[n])
    return fields, len(ordered_names), len(overlap)


def _restrict_to_subset(fields: List[pa.Field], subset: Optional[Sequence[str]]) -> List[pa.Field]:
    if not subset:
        return fields
    subset_set = set(subset).union(_ESSENTIAL_SET)
    keep = [f for f in fields if f.name in subset_set]
    # sanity: essential still present
    for e in _ESSENTIAL:
        if e not in {f.name for f in keep}:
            raise SchemaError("Required essential columns were dropped by --columns subset.")
    return keep


def _add_missing_as_nulls(tbl: pa.Table, out_fields: List[pa.Field]) -> pa.Table:
    """
    Ensure `tbl` has **all** fields in out_fields (same order and types).
    Missing columns are added as NULL arrays of the field type.
    Extra columns (not in out_fields) are dropped.
    """
    existing = _field_map(tbl)
    arrays: List[pa.Array] = []
    n = tbl.num_rows
    for f in out_fields:
        if f.name in existing:
            arr = tbl.column(f.name)
            # cast if necessary (shouldn't be with earlier checks)
            if arr.type != f.type:
                arr = arr.cast(f.type)
            arrays.append(arr)
        else:
            arrays.append(pa.nulls(n, type=f.type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(out_fields))


def _index_map(values: List[str]) -> dict:
    return {v: i for i, v in enumerate(values)}


def _apply_schema_casts(schema: pa.Schema, casts: Dict[str, pa.DataType]) -> pa.Schema:
    if not casts:
        return schema
    fields: List[pa.Field] = []
    for f in schema:
        if f.name in casts:
            fields.append(pa.field(f.name, casts[f.name], nullable=True))
        else:
            fields.append(f)
    return pa.schema(fields)


def _duckdb_type_for_arrow(dtype: pa.DataType) -> str:
    if pa.types.is_string(dtype):
        return "VARCHAR"
    if pa.types.is_boolean(dtype):
        return "BOOLEAN"
    if pa.types.is_int8(dtype):
        return "TINYINT"
    if pa.types.is_int16(dtype):
        return "SMALLINT"
    if pa.types.is_int32(dtype):
        return "INTEGER"
    if pa.types.is_int64(dtype):
        return "BIGINT"
    if pa.types.is_uint8(dtype):
        return "UTINYINT"
    if pa.types.is_uint16(dtype):
        return "USMALLINT"
    if pa.types.is_uint32(dtype):
        return "UINTEGER"
    if pa.types.is_uint64(dtype):
        return "UBIGINT"
    if pa.types.is_float32(dtype):
        return "FLOAT"
    if pa.types.is_float64(dtype):
        return "DOUBLE"
    if pa.types.is_timestamp(dtype):
        return "TIMESTAMP"
    raise SchemaError(f"Unsupported Arrow type for DuckDB cast: {dtype}")


def _sql_ident(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


# -------------------- core merge --------------------
def merge_usr_to_usr(
    *,
    root: Path,
    dest: str,
    src: str,
    columns_mode: MergeColumnsMode,
    duplicate_policy: MergePolicy,
    columns_subset: Optional[Sequence[str]] = None,
    dry_run: bool = False,
    assume_yes: bool = False,
    note: str = "",
    overlap_coercion: str = "none",  # "none" | "to-dest"
    avoid_casefold_dups: bool = True,
    maintenance: bool = False,
) -> MergePreview:
    """
    Merge rows from a source USR dataset into a destination dataset.
    See CLI help for options. Returns a MergePreview with counts.
    """
    if not maintenance:
        raise SchemaError("merge is a maintenance-only operation.")
    ds_dest = Dataset(root, dest)
    ds_src = Dataset(root, src)
    lock_ctx = dataset_write_lock(ds_dest.dir) if not dry_run else nullcontext()
    with lock_ctx:
        dest_pf = pq.ParquetFile(str(ds_dest.records_path))
        src_pf = pq.ParquetFile(str(ds_src.records_path))
        dest_schema = dest_pf.schema_arrow
        src_schema = src_pf.schema_arrow

        dest_rows_before = int(dest_pf.metadata.num_rows)
        src_rows = int(src_pf.metadata.num_rows)

        if overlap_coercion not in {"none", "to-dest"}:
            raise SchemaError(f"Unsupported overlap coercion '{overlap_coercion}'.")

        coercion_notes: List[str] = []
        overlap = set(dest_schema.names).intersection(set(src_schema.names))
        overlap_casts: Dict[str, pa.DataType] = {}

        for name in overlap:
            d_field = dest_schema.field(name)
            s_field = src_schema.field(name)
            if d_field.type.equals(s_field.type):
                continue
            if overlap_coercion == "none":
                raise SchemaError(f"Overlapping column '{name}' has different types: {d_field.type} vs {s_field.type}")
            _duckdb_type_for_arrow(d_field.type)
            overlap_casts[name] = d_field.type
            coercion_notes.append(f"{name}: {s_field.type} → {d_field.type}")

        src_schema_adjusted = _apply_schema_casts(src_schema, overlap_casts)

        if columns_mode == MergeColumnsMode.REQUIRE_SAME:
            fields = _ensure_columns_same(dest_schema, src_schema_adjusted)
        else:
            fields, _, _ = _union_columns(dest_schema, src_schema_adjusted)
        fields = _restrict_to_subset(fields, columns_subset)

        columns_total = len(fields)
        overlapping_columns = len(set(dest_schema.names).intersection(set(src_schema.names)))

        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise SchemaError("duckdb is required for merge-datasets (install duckdb).") from e

        con = duckdb.connect()
        try:
            dest_sql = str(ds_dest.records_path).replace("'", "''")
            src_sql = str(ds_src.records_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW dest AS SELECT * FROM read_parquet('{dest_sql}')")
            src_exprs: List[str] = []
            for name in src_schema.names:
                ident = _sql_ident(name)
                if name in overlap_casts:
                    dtype = _duckdb_type_for_arrow(overlap_casts[name])
                    src_exprs.append(f"CAST({ident} AS {dtype}) AS {ident}")
                else:
                    src_exprs.append(ident)
            src_select = ", ".join(src_exprs) if src_exprs else "*"
            con.execute(f"CREATE TEMP VIEW src AS SELECT {src_select} FROM read_parquet('{src_sql}')")

            if avoid_casefold_dups:
                con.execute(
                    "CREATE TEMP VIEW dest_casefold AS SELECT lower(bio_type) AS bt, upper(sequence) AS seq FROM dest"
                )
                con.execute(
                    "CREATE TEMP VIEW src_filtered AS "
                    "SELECT s.* FROM src s "
                    "LEFT JOIN dest_casefold d "
                    "ON lower(s.bio_type)=d.bt AND upper(s.sequence)=d.seq "
                    "WHERE d.bt IS NULL"
                )
            else:
                con.execute("CREATE TEMP VIEW src_filtered AS SELECT * FROM src")

            duplicates_total = int(
                con.execute("SELECT COUNT(*) FROM src_filtered s JOIN dest d USING (id)").fetchone()[0]
            )

            duplicates_skipped = 0
            duplicates_replaced = 0

            if duplicate_policy == MergePolicy.ERROR and duplicates_total:
                raise ValidationError(f"{duplicates_total} duplicate id(s) present in src.")

            if duplicate_policy == MergePolicy.PREFER_SRC:
                con.execute(
                    "CREATE TEMP VIEW dest_final AS "
                    "SELECT d.* FROM dest d LEFT JOIN src_filtered s USING (id) "
                    "WHERE s.id IS NULL"
                )
                con.execute("CREATE TEMP VIEW src_final AS SELECT * FROM src_filtered")
                duplicates_replaced = duplicates_total
                dest_rel = "dest_final"
            else:
                con.execute(
                    "CREATE TEMP VIEW src_final AS "
                    "SELECT s.* FROM src_filtered s LEFT JOIN dest d USING (id) "
                    "WHERE d.id IS NULL"
                )
                if duplicate_policy in (MergePolicy.SKIP, MergePolicy.PREFER_DEST):
                    duplicates_skipped = duplicates_total
                dest_rel = "dest"

            def _select_exprs(existing: set[str]) -> str:
                exprs: List[str] = []
                for f in fields:
                    if f.name in existing:
                        exprs.append(_sql_ident(f.name))
                    else:
                        exprs.append(f"NULL AS {_sql_ident(f.name)}")
                return ", ".join(exprs)

            dest_select = _select_exprs(set(dest_schema.names))
            src_select = _select_exprs(set(src_schema_adjusted.names))
            union_query = f"SELECT {dest_select} FROM {dest_rel} UNION ALL SELECT {src_select} FROM src_final"

            dest_rows_after = int(con.execute(f"SELECT COUNT(*) FROM ({union_query})").fetchone()[0])
            new_rows = int(dest_rows_after - dest_rows_before)

            if not dry_run:
                tmp_path = ds_dest.records_path.with_suffix(".merge.parquet")
                tmp_sql = str(tmp_path).replace("'", "''")
                compression = PARQUET_COMPRESSION.upper()
                con.execute(f"COPY ({union_query}) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION '{compression}')")

                pf_tmp = pq.ParquetFile(str(tmp_path))
                schema = pf_tmp.schema_arrow
                metadata = ds_dest._base_metadata(created_at=now_utc())

                def _batches():
                    for batch in iter_parquet_batches(tmp_path, columns=None):
                        yield batch

                write_parquet_atomic_batches(
                    _batches(),
                    schema,
                    ds_dest.records_path,
                    ds_dest.snapshot_dir,
                    metadata=metadata,
                )
                tmp_path.unlink(missing_ok=True)

                payload = {
                    "dest": dest,
                    "src": src,
                    "duplicate_policy": duplicate_policy.value,
                    "columns_mode": columns_mode.value,
                    "rows_src": src_rows,
                    "duplicates_total": duplicates_total,
                    "duplicates_skipped": duplicates_skipped,
                    "duplicates_replaced": duplicates_replaced,
                    "rows_added": new_rows,
                    "columns_total": columns_total,
                    "overlapping_columns": overlapping_columns,
                    "note": note,
                }
                if coercion_notes:
                    payload["overlap_coercions"] = coercion_notes
                record_event(
                    ds_dest.events_path,
                    "merge_datasets",
                    dataset=ds_dest.name,
                    args=payload,
                    target_path=ds_dest.records_path,
                )
        finally:
            con.close()

        return MergePreview(
            dest_rows_before=dest_rows_before,
            src_rows=src_rows,
            duplicates_total=duplicates_total,
            duplicates_skipped=duplicates_skipped,
            duplicates_replaced=duplicates_replaced,
            duplicate_policy=duplicate_policy,
            new_rows=new_rows,
            dest_rows_after=dest_rows_after,
            columns_total=columns_total,
            overlapping_columns=overlapping_columns,
        )
