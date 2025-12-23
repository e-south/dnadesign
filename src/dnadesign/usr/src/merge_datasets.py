"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/merge_datasets.py

USR ↔ USR MERGE (records.parquet → records.parquet)

Combine rows from a **source** USR dataset into a **destination** USR dataset.
Features:
- Duplicate policy: error | skip | prefer-src | prefer-dest
- Column policy: require-same OR union (fill missing cols with NULLs)
- Optional column subset (essential columns always included)
- **Clear reporting**: counts for src rows, duplicates encountered, skipped/replaced,
  and rows actually added
- Atomic write with snapshotting to _snapshots/ and events log entry

Use via CLI:
  usr merge-datasets --dest <name> --src <name> --union-columns --if-duplicate skip
or programmatically via merge_usr_to_usr(...).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pyarrow as pa

from .dataset import Dataset
from .errors import SchemaError, ValidationError
from .io import append_event, read_parquet, write_parquet_atomic
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


def _ordered_fields(tbl: pa.Table) -> List[pa.Field]:
    return [tbl.schema.field(i) for i in range(tbl.num_columns)]


def _field_map(tbl: pa.Table):
    return {f.name: f for f in tbl.schema}


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
            except Exception:
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
            except Exception as e:
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


def _ensure_columns_same(dest: pa.Table, src: pa.Table) -> List[pa.Field]:
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


def _union_columns(dest: pa.Table, src: pa.Table) -> Tuple[List[pa.Field], int, int]:
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
) -> MergePreview:
    """
    Merge rows from a source USR dataset into a destination dataset.
    See CLI help for options. Returns a MergePreview with counts.
    """
    ds_dest = Dataset(root, dest)
    ds_src = Dataset(root, src)
    dest_tbl = read_parquet(ds_dest.records_path)
    src_tbl = read_parquet(ds_src.records_path)

    coercion_notes: List[str] = []

    # ---- Optional: coerce overlapping columns in src to dest types ----
    if overlap_coercion == "to-dest":
        overlap = set(dest_tbl.schema.names).intersection(set(src_tbl.schema.names))
        for name in overlap:
            d_field = dest_tbl.schema.field(name)
            s_field = src_tbl.schema.field(name)
            if d_field.type.equals(s_field.type):
                continue
            try:
                coerced = _coerce_jsonish_to_type(src_tbl.column(name), d_field.type, name)
                idx = src_tbl.schema.get_field_index(name)
                src_tbl = src_tbl.set_column(idx, pa.field(name, d_field.type, nullable=True), coerced)
                coercion_notes.append(f"{name}: {s_field.type} → {d_field.type}")
            except Exception as e:
                raise SchemaError(
                    f"Overlapping column '{name}' has different types ({s_field.type} vs {d_field.type}) "
                    f"and coercion failed: {e}"
                )

    # ------------- (existing) schema reconciliation -------------
    if columns_mode == MergeColumnsMode.REQUIRE_SAME:
        fields = _ensure_columns_same(dest_tbl, src_tbl)
    else:
        fields, total_cols, overlap_count = _union_columns(dest_tbl, src_tbl)
        # we keep these for the preview later
    fields = _restrict_to_subset(fields, columns_subset)

    # Build aligned tables with the chosen field order
    names = [f.name for f in fields]
    d_aligned = dest_tbl.select(names)

    # For missing columns in either side, add NULL arrays
    def _ensure_all_columns(tbl: pa.Table, fields: List[pa.Field]) -> pa.Table:
        present = set(tbl.schema.names)
        out = tbl
        for f in fields:
            if f.name not in present:
                out = out.add_column(out.num_columns, f, pa.nulls(tbl.num_rows, type=f.type))
        return out.select([f.name for f in fields])

    d_aligned = _ensure_all_columns(d_aligned, fields)
    s_aligned = _ensure_all_columns(src_tbl.select([c for c in src_tbl.schema.names if c in names]), fields)

    # -------- drop source rows whose (bio_type, upper(sequence)) already exist in dest --------
    if avoid_casefold_dups:
        dest_cf = _casefold_keys(d_aligned)
        s_bt = [str(x) for x in s_aligned.column("bio_type").to_pylist()]
        s_sq = [str(x) for x in s_aligned.column("sequence").to_pylist()]
        keep_mask_cf = [(b.lower(), s.upper()) not in dest_cf for b, s in zip(s_bt, s_sq)]
        if not all(keep_mask_cf):
            s_aligned = s_aligned.filter(pa.array(keep_mask_cf))

    # ------- duplicate policy by id -------
    dest_ids = set(d_aligned.column("id").to_pylist())
    src_ids = s_aligned.column("id").to_pylist()
    keep_mask = [rid not in dest_ids for rid in src_ids]
    duplicates_total = len(src_ids) - sum(keep_mask)
    duplicates_skipped = duplicates_replaced = 0

    if duplicate_policy == MergePolicy.ERROR and duplicates_total:
        raise ValidationError(f"{duplicates_total} duplicate id(s) present in src.")
    if duplicate_policy in (MergePolicy.SKIP, MergePolicy.ERROR):
        # simple filter
        s_kept = s_aligned.filter(pa.array(keep_mask))
        duplicates_skipped = duplicates_total
        combined = pa.concat_tables([d_aligned, s_kept], promote_options="default")
    else:
        # prefer-src or prefer-dest (replace semantics)
        # Build a map from id -> row index for destination
        pos = {rid: i for i, rid in enumerate(d_aligned.column("id").to_pylist())}
        base = d_aligned
        if duplicate_policy == MergePolicy.PREFER_SRC:
            # replace dest rows with src rows where duplicate
            for i, rid in enumerate(src_ids):
                j = pos.get(rid)
                if j is not None:
                    # overwrite each column value at j with src row i
                    for col_idx, f in enumerate(fields):
                        col = base.column(col_idx).to_pylist()
                        col[j] = s_aligned.column(col_idx)[i].as_py()
                        base = base.set_column(col_idx, f, pa.array(col, type=f.type))
            duplicates_replaced = duplicates_total
            # append only new rows from src
            keep_new = [rid not in dest_ids for rid in src_ids]
            combined = pa.concat_tables([base, s_aligned.filter(pa.array(keep_new))], promote_options="default")
        else:  # prefer-dest: drop duplicates from src entirely
            s_kept = s_aligned.filter(pa.array(keep_mask))
            duplicates_skipped = duplicates_total
            combined = pa.concat_tables([base, s_kept], promote_options="default")

    # ----- finalize + write (or dry-run) -----
    dest_rows_before = d_aligned.num_rows
    src_rows = src_tbl.num_rows
    new_rows = combined.num_rows - dest_rows_before
    dest_rows_after = combined.num_rows
    columns_total = len(fields)
    overlapping_columns = len(set(dest_tbl.schema.names).intersection(set(src_tbl.schema.names)))

    if not dry_run:
        write_parquet_atomic(combined, ds_dest.records_path, ds_dest.snapshot_dir)
        payload = {
            "action": "merge_datasets",
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
        append_event(ds_dest.events_path, payload)

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
