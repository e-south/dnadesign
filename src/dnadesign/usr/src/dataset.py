"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset.py

USR dataset lifecycle operations and validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .errors import (
    AlphabetError,
    DuplicateGroup,
    DuplicateIDError,
    NamespaceError,
    SchemaError,
    SequencesError,
)
from .events import fingerprint_parquet, record_event
from .io import (
    PARQUET_COMPRESSION,
    iter_parquet_batches,
    now_utc,
    read_parquet,
    snapshot_parquet_file,
    write_parquet_atomic,
    write_parquet_atomic_batches,
)
from .locks import dataset_write_lock
from .normalize import compute_id, normalize_sequence, validate_alphabet, validate_bio_type  # case-preserving
from .overlays import (
    OVERLAY_META_CREATED,
    OVERLAY_META_KEY,
    OVERLAY_META_NAMESPACE,
    list_overlays,
    overlay_metadata,
    overlay_path,
    with_overlay_metadata,
)
from .registry import load_registry, validate_overlay_schema
from .schema import ARROW_SCHEMA, REQUIRED_COLUMNS, merge_base_metadata, with_base_metadata
from .types import AddSequencesResult, DatasetInfo, Manifest, OverlayInfo

RECORDS = "records.parquet"
SNAPDIR = "_snapshots"  # standardized
META_MD = "meta.md"
EVENTS_LOG = ".events.log"

_NS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TOMBSTONE_NAMESPACE = "usr"
TOMBSTONE_COLUMNS = ("usr__deleted", "usr__deleted_at", "usr__deleted_reason")
RESERVED_NAMESPACES = {TOMBSTONE_NAMESPACE}


@dataclass(frozen=True)
class DedupeStats:
    rows_total: int
    rows_dropped: int
    groups: int
    key: str
    keep: str


def normalize_dataset_id(name: str) -> str:
    ds = str(name or "").strip()
    if not ds:
        raise SequencesError("Dataset name cannot be empty.")
    p = Path(ds)
    if p.is_absolute():
        raise SequencesError("Dataset name must be a relative path.")
    if any(part in {".", ".."} for part in p.parts):
        raise SequencesError("Dataset name must not contain '.' or '..'.")
    return Path(*p.parts).as_posix()


@dataclass
class Dataset:
    """A concrete, local dataset located at `<root>/<name>/`."""

    root: Path  # the usr/datasets/ folder
    name: str

    def __post_init__(self) -> None:
        self.name = normalize_dataset_id(self.name)

    @classmethod
    def open(cls, root: Path, name_or_path: str) -> "Dataset":
        root_path = Path(root).resolve()
        target = Path(str(name_or_path)).expanduser()
        if target.exists():
            if target.is_file() and target.name == RECORDS:
                dataset_dir = target.parent
            elif target.is_dir() and (target / RECORDS).exists():
                dataset_dir = target
            else:
                raise SequencesError(f"Path does not point to a dataset: {target}")
            try:
                rel = dataset_dir.resolve().relative_to(root_path)
            except ValueError as e:
                raise SequencesError(f"Dataset path must live under root: {root_path}") from e
            return cls(root_path, rel.as_posix())
        return cls(root_path, normalize_dataset_id(str(name_or_path)))

    @property
    def dir(self) -> Path:
        return self.root / self.name

    @property
    def records_path(self) -> Path:
        return self.dir / RECORDS

    @property
    def snapshot_dir(self) -> Path:
        return self.dir / SNAPDIR

    @property
    def meta_path(self) -> Path:
        return self.dir / META_MD

    @property
    def events_path(self) -> Path:
        return self.dir / EVENTS_LOG

    # ---- quick stats for CLI and sync ----
    def stats(self):
        """Return local primary FileStat (sha/size/rows/cols/mtime)."""
        from .diff import parquet_stats

        return parquet_stats(self.records_path)

    # ---- lifecycle ----

    def init(self, source: str = "", notes: str = "") -> None:
        """Create a new, empty dataset directory with canonical schema."""
        with dataset_write_lock(self.dir):
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.records_path.exists():
                raise SequencesError(f"Dataset already initialized: {self.records_path}")
            ts = now_utc()
            empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in ARROW_SCHEMA], schema=ARROW_SCHEMA)
            empty = with_base_metadata(empty, created_at=ts)
            write_parquet_atomic(empty, self.records_path, self.snapshot_dir)
            date = ts.split("T")[0]
            meta_md = (
                f"name: {self.name}\n"
                f"created_at: {ts}\n"
                f"source: {source}\n"
                f"notes: {notes}\n"
                f"schema: USR v1\n\n"
                f"### Updates ({date})\n"
                f"- {ts}: initialized dataset.\n"
            )
            self.meta_path.write_text(meta_md, encoding="utf-8")
            record_event(
                self.events_path,
                "init",
                dataset=self.name,
                args={"source": source},
                target_path=self.records_path,
            )

    # --- lightweight, best-effort scratch-pad logging in meta.md ---
    def append_meta_note(self, title: str, code_block: Optional[str] = None) -> None:
        ts = now_utc()
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            # create minimal header if missing
            hdr = f"name: {self.name}\ncreated_at: {ts}\nsource: \nnotes: \nschema: USR v1\n\n### Updates ({ts.split('T')[0]})\n"  # noqa
            self.meta_path.write_text(hdr, encoding="utf-8")
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(f"- {ts}: {title}\n")
            if code_block:
                f.write("```bash\n")
                f.write(code_block.strip() + "\n")
                f.write("```\n")

    def _require_exists(self) -> None:
        if not self.records_path.exists():
            raise SequencesError(f"Dataset not initialized: {self.records_path}")

    def _base_metadata(self, *, created_at: Optional[str] = None) -> Dict[bytes, bytes]:
        md = None
        if self.records_path.exists():
            md = pq.ParquetFile(str(self.records_path)).schema_arrow.metadata
        return merge_base_metadata(md, created_at)

    def _tombstone_path(self) -> Path:
        return overlay_path(self.dir, TOMBSTONE_NAMESPACE)

    def _registry(self, *, required: bool) -> dict:
        return load_registry(self.root, required=required)

    def _validate_registry_schema(self, *, namespace: str, schema: pa.Schema, key: str) -> None:
        if namespace in RESERVED_NAMESPACES:
            return
        registry = self._registry(required=True)
        validate_overlay_schema(namespace, schema, registry=registry, key=key)

    # ---- info ----

    def info(self) -> DatasetInfo:
        """Basic dataset metadata plus discovered namespaces."""
        self._require_exists()
        pf = pq.ParquetFile(str(self.records_path))
        cols = list(pf.schema_arrow.names)
        derived_cols = []
        try:
            overlay_data = self._load_overlays()
            for ov in overlay_data:
                derived_cols.extend(ov["cols"])
        except FileNotFoundError:
            overlay_data = []
        all_cols = cols + derived_cols
        namespaces = sorted(
            {c.split("__", 1)[0] for c in all_cols if c not in {k for k, _ in REQUIRED_COLUMNS} and "__" in c}
        )
        return DatasetInfo(
            name=self.name,
            path=str(self.records_path),
            rows=int(pf.metadata.num_rows),
            columns=all_cols,
            namespaces=namespaces,
        )

    def info_dict(self) -> dict:
        return self.info().to_dict()

    def schema(self):
        """Return the Arrow schema of the current table (base + overlays)."""
        self._require_exists()
        base_schema = pq.ParquetFile(str(self.records_path)).schema_arrow
        for ov in self._load_overlays():
            for field in ov["schema"]:
                if field.name == ov["key"]:
                    continue
                if base_schema.get_field_index(field.name) >= 0:
                    raise NamespaceError(f"Derived column already exists in schema: {field.name}")
                base_schema = base_schema.append(field)
        return base_schema

    def scan(
        self,
        *,
        columns: Optional[List[str]] = None,
        include_overlays: Union[bool, Sequence[str]] = True,
        include_deleted: bool = False,
        batch_size: int = 65536,
    ):
        """
        Stream record batches with optional overlay merge.
        """
        if batch_size < 1:
            raise SequencesError("batch_size must be >= 1.")
        con, query, params = self._duckdb_query(
            columns=columns,
            include_overlays=include_overlays,
            include_deleted=include_deleted,
        )
        try:
            con.execute(query, params)
            reader = con.fetch_record_batch(int(batch_size))
            for batch in reader:
                yield batch
        finally:
            con.close()

    def head(
        self,
        n: int = 10,
        columns: Optional[List[str]] = None,
        *,
        include_derived: bool = True,
        include_deleted: bool = False,
    ):
        """Return the first N rows as a pandas DataFrame."""
        batches = []
        rows = 0
        for batch in self.scan(
            columns=columns,
            include_overlays=include_derived,
            include_deleted=include_deleted,
            batch_size=max(int(n), 1),
        ):
            batches.append(batch)
            rows += batch.num_rows
            if rows >= n:
                break
        if not batches:
            return pd.DataFrame(columns=columns or self.schema().names)
        tbl = pa.Table.from_batches(batches)
        if tbl.num_rows > n:
            tbl = tbl.slice(0, n)
        return tbl.to_pandas()

    def _key_list_from_batch(self, batch: pa.RecordBatch, key: str) -> List[str]:
        def _col(name: str) -> pa.Array:
            idx = batch.schema.get_field_index(name)
            if idx < 0:
                raise SchemaError(f"Missing required column '{name}' for key '{key}'.")
            return batch.column(idx)

        if key == "id":
            vals = _col("id").to_pylist()
            if any(v is None or str(v).strip() == "" for v in vals):
                raise SchemaError("Missing id values while computing dedupe key.")
            return [str(v) for v in vals]
        if key in {"sequence", "sequence_norm"}:
            vals = _col("sequence").to_pylist()
            if any(v is None or str(v).strip() == "" for v in vals):
                raise SchemaError("Missing sequence values while computing dedupe key.")
            return [str(v).strip() for v in vals]
        if key == "sequence_ci":
            alph = _col("alphabet").to_pylist()
            if any(v is None or str(v).strip() == "" for v in alph):
                raise SchemaError("Missing alphabet values while computing dedupe key.")
            if any(str(v) != "dna_4" for v in alph):
                raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
            seqs = _col("sequence").to_pylist()
            if any(v is None or str(v).strip() == "" for v in seqs):
                raise SchemaError("Missing sequence values while computing dedupe key.")
            return [str(v).strip().upper() for v in seqs]
        raise SchemaError(f"Unsupported join key '{key}'.")

    def _load_overlays(
        self,
        *,
        include_tombstone: bool = True,
        namespaces: Optional[Sequence[str]] = None,
    ):
        overlays = []
        paths = list_overlays(self.dir)
        require_registry = any(
            (overlay_metadata(p).get("namespace") or p.stem) not in RESERVED_NAMESPACES for p in paths
        )
        registry = self._registry(required=require_registry) if require_registry else {}
        for path in paths:
            meta = overlay_metadata(path)
            key = meta.get("key")
            if not key:
                raise SchemaError(f"Overlay missing required metadata key: {path}")
            ns = meta.get("namespace") or path.stem
            if not include_tombstone and ns in RESERVED_NAMESPACES:
                continue
            if namespaces and ns not in set(namespaces):
                continue
            pf = pq.ParquetFile(str(path))
            schema = pf.schema_arrow
            if ns not in RESERVED_NAMESPACES:
                validate_overlay_schema(ns, schema, registry=registry, key=key)
            if key not in schema.names:
                raise SchemaError(f"Overlay missing key column '{key}': {path}")
            overlay_cols = [c for c in schema.names if c != key]
            overlays.append(
                {
                    "namespace": ns,
                    "key": key,
                    "cols": overlay_cols,
                    "schema": schema,
                    "path": path,
                }
            )
        return overlays

    def _duckdb_query(
        self,
        *,
        columns: Optional[List[str]],
        include_overlays: Union[bool, Sequence[str]],
        include_deleted: bool,
        where: str | None = None,
        params: Optional[list] = None,
        limit: int | None = None,
    ):
        self._require_exists()
        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise SchemaError("duckdb is required for overlay joins (install duckdb).") from e

        if columns is not None and not include_deleted and any(c in TOMBSTONE_COLUMNS for c in columns):
            raise SchemaError("Tombstone columns require include_deleted=True.")

        base_pf = pq.ParquetFile(str(self.records_path))
        base_cols = list(base_pf.schema_arrow.names)

        if include_overlays is True:
            overlays = self._load_overlays(include_tombstone=False)
        elif include_overlays is False:
            overlays = []
        else:
            overlays = self._load_overlays(include_tombstone=False, namespaces=include_overlays)

        requested = set(columns) if columns is not None else None
        selected_cols: List[str] = []

        def _sql_ident(name: str) -> str:
            escaped = str(name).replace('"', '""')
            return f'"{escaped}"'

        def _key_expr(expr: str, *, key: str) -> str:
            if key == "sequence_ci":
                return f"NULLIF(UPPER(TRIM(CAST({expr} AS VARCHAR))), '')"
            return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"

        con = duckdb.connect()
        base_sql = str(self.records_path).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{base_sql}')")
        base_view = "base"

        tomb_path = self._tombstone_path()
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
            elif requested and any(c in TOMBSTONE_COLUMNS for c in requested):
                include_tombstone_cols = True

        if include_tombstone_cols:
            tomb_pf = pq.ParquetFile(str(tomb_path))
            overlays.append(
                {
                    "namespace": TOMBSTONE_NAMESPACE,
                    "key": "id",
                    "cols": list(TOMBSTONE_COLUMNS),
                    "schema": tomb_pf.schema_arrow,
                    "path": tomb_path,
                }
            )

        select_exprs = []
        existing_cols = set(base_cols)
        for col in base_cols:
            if requested is None or col in requested:
                select_exprs.append(f"b.{_sql_ident(col)} AS {_sql_ident(col)}")
                selected_cols.append(col)

        join_clauses: List[str] = []
        essential = {k for k, _ in REQUIRED_COLUMNS}

        for idx, ov in enumerate(overlays):
            ns = ov["namespace"]
            key = ov["key"]
            overlay_cols = ov["cols"]
            if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
                raise SchemaError(f"Unsupported overlay key '{key}': {ov['path']}")

            derived_cols = overlay_cols
            if requested is not None:
                derived_cols = [c for c in overlay_cols if c in requested]
            if not derived_cols:
                continue

            for col in derived_cols:
                if ns != TOMBSTONE_NAMESPACE and col in essential:
                    raise NamespaceError(f"Overlay cannot modify required column '{col}'.")
                if "__" not in col:
                    raise NamespaceError(f"Derived columns must be namespaced (got '{col}').")
                if col in existing_cols:
                    raise NamespaceError(f"Derived columns already exist: {col}")

            view_name = f"overlay_{idx}"
            overlay_sql = str(ov["path"]).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW {view_name} AS SELECT * FROM read_parquet('{overlay_sql}')")

            key_q = _sql_ident(key)
            dup_overlay = int(
                con.execute(
                    f"SELECT COUNT(*) FROM (SELECT {key_q} FROM {view_name} GROUP BY {key_q} HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
            if dup_overlay:
                raise SchemaError(f"Overlay has duplicate keys for '{key}': {ov['path']}")

            if key in {"sequence", "sequence_norm", "sequence_ci"}:
                bt_count = int(con.execute(f"SELECT COUNT(DISTINCT bio_type) FROM {base_view}").fetchone()[0])
                if bt_count > 1:
                    raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
                if key == "sequence_ci":
                    bad = int(con.execute(f"SELECT COUNT(*) FROM {base_view} WHERE alphabet != 'dna_4'").fetchone()[0])
                    if bad:
                        raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                base_key_expr = _key_expr(f"b.{_sql_ident('sequence')}", key=key)
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
                base_key_expr = _key_expr(f"b.{_sql_ident('id')}", key=key)

            overlay_key_expr = _key_expr(f"o{idx}.{_sql_ident(key)}", key=key)
            join_clauses.append(f"LEFT JOIN {view_name} o{idx} ON {base_key_expr} = {overlay_key_expr}")
            for col in derived_cols:
                select_exprs.append(f"o{idx}.{_sql_ident(col)} AS {_sql_ident(col)}")
                selected_cols.append(col)
            existing_cols.update(derived_cols)

        if requested is not None:
            missing = [c for c in columns if c not in selected_cols]
            if missing:
                raise SchemaError(f"Requested columns not found after overlay merge: {', '.join(missing)}")

        query = "SELECT " + ", ".join(select_exprs) + f" FROM {base_view} b " + " ".join(join_clauses)
        if where:
            query += f" WHERE {where}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        return con, query, (params or [])

    # ---- ingest ----

    def _prepare_import_rows(
        self,
        rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
        *,
        default_bio_type: str,
        default_alphabet: str,
        source: Optional[str],
        strict_id_check: bool,
        created_at_override: Optional[str] = None,
    ) -> pd.DataFrame:
        # Normalize input → DataFrame
        if isinstance(rows, pd.DataFrame):
            df_in = rows.copy()
        else:
            df_in = pd.DataFrame(list(rows) if rows else [])

        if "sequence" not in df_in.columns:
            raise SchemaError("Missing required column: 'sequence'.")

        def _is_missing_scalar(x: object) -> bool:
            if x is None:
                return True
            if isinstance(x, (list, tuple, dict, np.ndarray)):
                return False
            try:
                res = pd.isna(x)
            except (TypeError, ValueError) as e:
                raise SchemaError(f"Unable to check missingness for value: {x!r}") from e
            if isinstance(res, (list, tuple, np.ndarray)):
                return False
            return bool(res)

        seq_raw = df_in["sequence"].tolist()
        bad_seq = [i for i, s in enumerate(seq_raw, start=1) if _is_missing_scalar(s) or str(s).strip() == ""]
        if bad_seq:
            sample = ", ".join(str(i) for i in bad_seq[:5])
            raise SchemaError(
                f"{len(bad_seq)} row(s) have missing/empty 'sequence' (rows: {sample}). "
                "Provide a non-empty sequence string."
            )

        # Default columns (fail fast on missing/empty values if provided)
        if "bio_type" in df_in.columns:
            bio_raw = df_in["bio_type"].tolist()
            bad_bt = [i for i, v in enumerate(bio_raw, start=1) if _is_missing_scalar(v) or str(v).strip() == ""]
            if bad_bt:
                sample = ", ".join(str(i) for i in bad_bt[:5])
                raise SchemaError(
                    f"{len(bad_bt)} row(s) have missing/empty 'bio_type' (rows: {sample}). "
                    "Either provide values or omit the column to use the default."
                )
            bio_vals = [str(v).strip() for v in bio_raw]
        else:
            bio_vals = [default_bio_type] * len(df_in)

        if "alphabet" in df_in.columns:
            alph_raw = df_in["alphabet"].tolist()
            bad_ab = [i for i, v in enumerate(alph_raw, start=1) if _is_missing_scalar(v) or str(v).strip() == ""]
            if bad_ab:
                sample = ", ".join(str(i) for i in bad_ab[:5])
                raise SchemaError(
                    f"{len(bad_ab)} row(s) have missing/empty 'alphabet' (rows: {sample}). "
                    "Either provide values or omit the column to use the default."
                )
            alph_vals = [str(v).strip() for v in alph_raw]
        else:
            alph_vals = [default_alphabet] * len(df_in)

        # Compute normalized sequences + ids + lengths with strict validation
        ids, seqs, lens, bio_out, alph_out = [], [], [], [], []
        for i, (s, bt, ab) in enumerate(zip(seq_raw, bio_vals, alph_vals), start=1):
            try:
                bt_norm = validate_bio_type(str(bt))
            except ValueError as e:
                raise SchemaError(f"Row {i}: {e}") from e
            try:
                ab_norm = validate_alphabet(bt_norm, str(ab))
            except ValueError as e:
                raise AlphabetError(f"Row {i}: {e}") from e
            try:
                s_norm = normalize_sequence(str(s), bt_norm, ab_norm, validate=False)
            except ValueError as e:
                raise AlphabetError(f"Row {i}: {e}") from e
            seqs.append(s_norm)
            ids.append(compute_id(bt_norm, s_norm))
            lens.append(len(s_norm))
            bio_out.append(bt_norm)
            alph_out.append(ab_norm)

        # Optional user-supplied ids must match
        if "id" in df_in.columns and strict_id_check:
            bad_id = [
                i for i, v in enumerate(df_in["id"].tolist(), start=1) if _is_missing_scalar(v) or str(v).strip() == ""
            ]
            if bad_id:
                sample = ", ".join(str(i) for i in bad_id[:5])
                raise SchemaError(
                    f"{len(bad_id)} row(s) have missing/empty 'id' (rows: {sample}). "
                    "Drop the column or provide valid ids."
                )
            bad_idx = [
                i for i, (given, want) in enumerate(zip(df_in["id"].astype(str), ids)) if str(given) != str(want)
            ]
            if bad_idx:
                raise SchemaError(
                    f"'id' mismatch for {len(bad_idx)} row(s); ids must equal sha1(bio_type|sequence_norm)."
                )

        # created_at: robust tz-aware timestamp
        if created_at_override is not None:
            created = pd.to_datetime([created_at_override] * len(df_in), utc=True)
        elif "created_at" in df_in.columns:
            created = pd.to_datetime(df_in["created_at"], utc=True)
        else:
            ts_utc = pd.Timestamp.now(tz="UTC")
            created = pd.Series([ts_utc] * len(df_in), dtype="datetime64[ns, UTC]")

        # source column: prefer explicit param; fall back to per-row source or ""
        src_str = source if source is not None else ""
        if "source" in df_in.columns:
            src_col = df_in["source"].astype(str)
            src_vals = [src_str if src_str else s for s in src_col.tolist()]
        else:
            src_vals = [src_str] * len(df_in)

        out_df = pd.DataFrame(
            {
                "id": ids,
                "bio_type": bio_out,
                "sequence": seqs,
                "alphabet": alph_out,
                "length": lens,
                "source": src_vals,
                "created_at": created,
            }
        )

        # ---------- Duplicate diagnostics (EXACT) ----------
        by_id: Dict[str, List[int]] = defaultdict(list)
        for i, rid in enumerate(ids, start=1):
            by_id[str(rid)].append(i)
        exact_groups = [
            DuplicateGroup(
                id=rid,
                count=len(rows_idx),
                rows=rows_idx,
                sequence=seqs[rows_idx[0] - 1],
            )
            for rid, rows_idx in by_id.items()
            if len(rows_idx) > 1
        ]
        exact_groups.sort(key=lambda g: (-g.count, g.id))

        if exact_groups:
            raise DuplicateIDError(
                "Duplicate sequences detected in incoming data (same canonical id).",
                groups=exact_groups[:5],
                hint=(
                    "Remove repeated rows before importing. If you need to keep a single copy, "
                    "deduplicate by the 'sequence' column (case preserved)."
                ),
            )
        return out_df

    def _write_import_df(
        self,
        out_df: pd.DataFrame,
        *,
        source: Optional[str],
        on_conflict: str,
        return_ids: bool = False,
    ) -> int | tuple[int, list[str], list[str]]:
        incoming = pa.Table.from_pandas(out_df, schema=ARROW_SCHEMA, preserve_index=False)
        ids_all = out_df["id"].astype(str).tolist()

        def _write_dataset() -> int | tuple[int, list[str], list[str]]:
            # Merge with existing (append-only; reject collisions)
            if self.records_path.exists():
                with tempfile.TemporaryDirectory() as tmpdir:
                    db_path = Path(tmpdir) / "import.sqlite"
                    conn = sqlite3.connect(db_path)
                    try:
                        conn.execute("CREATE TABLE seen (val TEXT PRIMARY KEY)")
                        for batch in iter_parquet_batches(self.records_path, columns=["id"]):
                            for rid in batch.column("id").to_pylist():
                                conn.execute("INSERT OR IGNORE INTO seen(val) VALUES (?)", (str(rid),))

                        conflicts = []
                        keep_mask: List[bool] = []
                        for rid in ids_all:
                            cur = conn.execute("INSERT OR IGNORE INTO seen(val) VALUES (?)", (str(rid),))
                            if cur.rowcount == 0:
                                if on_conflict == "ignore":
                                    keep_mask.append(False)
                                else:
                                    conflicts.append(str(rid))
                                    if len(conflicts) >= 5:
                                        break
                            else:
                                keep_mask.append(True)
                        if conflicts and on_conflict != "ignore":
                            sample = ", ".join(conflicts[:5])
                            raise DuplicateIDError(
                                f"Duplicate ids already present in dataset (sample: {sample}).",
                                hint=(
                                    "These sequences already exist in this dataset. "
                                    "Remove them from your import file or put new rows in a separate dataset."
                                ),
                            )
                        if on_conflict == "ignore":
                            if not any(keep_mask):
                                if return_ids:
                                    return 0, [], ids_all
                                return 0
                            out_df_local = out_df.loc[keep_mask].reset_index(drop=True)
                            incoming_local = pa.Table.from_pandas(
                                out_df_local, schema=ARROW_SCHEMA, preserve_index=False
                            )
                            ids_added = [rid for rid, keep in zip(ids_all, keep_mask) if keep]
                            ids_skipped = [rid for rid, keep in zip(ids_all, keep_mask) if not keep]
                        else:
                            out_df_local = out_df
                            incoming_local = incoming
                            ids_added = list(ids_all)
                            ids_skipped = []
                    finally:
                        conn.close()

                def _batch_iter():
                    for batch in iter_parquet_batches(self.records_path):
                        yield batch
                    for batch in incoming_local.to_batches():
                        yield batch

                metadata = self._base_metadata(created_at=now_utc())
                write_parquet_atomic_batches(
                    _batch_iter(),
                    ARROW_SCHEMA,
                    self.records_path,
                    self.snapshot_dir,
                    metadata=metadata,
                )
                out_count = int(len(out_df_local))
            else:
                metadata = self._base_metadata(created_at=now_utc())
                write_parquet_atomic_batches(
                    incoming.to_batches(),
                    ARROW_SCHEMA,
                    self.records_path,
                    self.snapshot_dir,
                    metadata=metadata,
                )
                out_count = int(len(out_df))
                ids_added = list(ids_all)
                ids_skipped = []

            record_event(
                self.events_path,
                "import_rows",
                dataset=self.name,
                args={"n": int(out_count), "source_param": source or "", "on_conflict": on_conflict},
                target_path=self.records_path,
            )
            if return_ids:
                return int(out_count), ids_added, ids_skipped
            return int(out_count)

        with dataset_write_lock(self.dir):
            return _write_dataset()

    def import_rows(
        self,
        rows: Union[pd.DataFrame, Sequence[Dict[str, object]]],
        *,
        default_bio_type: str = "dna",
        default_alphabet: str = "dna_4",
        source: Optional[str] = None,
        strict_id_check: bool = True,
    ) -> int:
        """
        Import sequence rows (DataFrame or sequence of dicts).

        Expectations per row (case-preserving):
          - 'sequence' (required): string; trimmed; validated by (bio_type, alphabet)
          - 'bio_type'  (optional): defaults to `default_bio_type`
          - 'alphabet'  (optional): defaults to `default_alphabet`
          - 'id'        (optional): if present and `strict_id_check`, must equal
                                   sha1(bio_type|sequence_norm)
          - 'created_at'(optional): if missing, set to now (UTC)
          - 'source'    (optional): defaults to `source` param or ""

        Behavior:
          - computes 'length' from normalized sequence
          - rejects duplicate ids within incoming
          - rejects ids that already exist on disk (append-only semantics)
          - atomic write + snapshot + event log
        """
        self._require_exists()
        out_df = self._prepare_import_rows(
            rows,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source,
            strict_id_check=strict_id_check,
        )
        return self._write_import_df(out_df, source=source, on_conflict="error")

    def add_sequences(
        self,
        rows_or_sequences: Union[pd.DataFrame, Sequence[Dict[str, object]], Sequence[str]],
        *,
        bio_type: str,
        alphabet: str,
        source: str = "",
        created_at: Optional[str] = None,
        on_conflict: str = "error",
    ) -> AddSequencesResult:
        """
        Append sequences with deterministic ids.

        rows_or_sequences can be:
          - list[str]: raw sequences
          - list[dict]: dicts with 'sequence' plus optional fields
          - pandas.DataFrame
        """
        if on_conflict not in {"error", "ignore"}:
            raise SchemaError(f"Unsupported on_conflict '{on_conflict}'.")

        if isinstance(rows_or_sequences, pd.DataFrame):
            rows = rows_or_sequences
        else:
            rows_list = list(rows_or_sequences) if rows_or_sequences is not None else []
            if rows_list and all(isinstance(v, str) for v in rows_list):
                rows = [{"sequence": s} for s in rows_list]
            else:
                rows = rows_list

        out_df = self._prepare_import_rows(
            rows,
            default_bio_type=bio_type,
            default_alphabet=alphabet,
            source=source,
            strict_id_check=True,
            created_at_override=created_at,
        )
        out_count, ids_added, ids_skipped = self._write_import_df(
            out_df,
            source=source,
            on_conflict=on_conflict,
            return_ids=True,
        )
        return AddSequencesResult(
            added=int(out_count),
            skipped=len(ids_skipped),
            ids=list(ids_added),
        )

    # Legacy file import entry points now route to import_rows (no special logic)
    def import_csv(
        self,
        csv_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        df = pd.read_csv(csv_path)
        return self.import_rows(
            df,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source or str(csv_path),
        )

    def import_jsonl(
        self,
        jsonl_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        df = pd.read_json(jsonl_path, lines=True)
        return self.import_rows(
            df,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source or str(jsonl_path),
        )

    # ---- generic attach ----

    def attach(
        self,
        path: Path,
        namespace: str,
        *,
        key: str,
        key_col: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        backend: str = "pyarrow",
        note: str = "",
    ) -> int:
        """
        Attach derived columns into an overlay keyed by an explicit join key.
        """
        self._require_exists()
        if not _NS_RE.match(namespace):
            raise NamespaceError(
                "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
            )
        if namespace in RESERVED_NAMESPACES:
            raise NamespaceError(f"Namespace '{namespace}' is reserved.")
        if backend not in {"pyarrow", "duckdb"}:
            raise SchemaError(f"Unsupported backend '{backend}'.")
        if backend == "duckdb" and parse_json:
            raise SchemaError(
                "duckdb backend does not support JSON parsing. Use --no-parse-json or the pyarrow backend."
            )
        key = str(key).strip()
        if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
            raise SchemaError(f"Unsupported join key '{key}'.")
        if key_col is None:
            key_col = "sequence" if key in {"sequence", "sequence_norm", "sequence_ci"} else key

        if backend == "duckdb":
            return self._attach_duckdb(
                path=path,
                namespace=namespace,
                key=key,
                key_col=key_col,
                columns=columns,
                allow_overwrite=allow_overwrite,
                allow_missing=allow_missing,
                note=note,
            )

        # Load incoming attachment (usually small)
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

        # Choose + prefix targets
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
                    # try a small fix for single-quoted CSV-y lists
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

        # Normalize key values for attachment input
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

        # Reject duplicate keys in attachment input
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

        # Read existing once; check policy (full schema)
        essential = {k for k, _ in REQUIRED_COLUMNS}
        for t in targets:
            if t in essential:
                raise NamespaceError(f"Refusing to write essential column: {t}")
            if "__" not in t:
                raise NamespaceError(f"Derived columns must be namespaced (got '{t}').")

        def _write_overlay() -> int:
            # Validate keys exist in base dataset
            base_cols = {"id"} if key == "id" else {"sequence", "alphabet", "bio_type"}
            base_tbl = read_parquet(self.records_path, columns=list(base_cols))
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
                    raise SchemaError(
                        f"Attach key requires unique base keys; duplicate base keys detected for '{key}'."
                    )
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

            out_path = overlay_path(self.dir, namespace)
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
                        combined.loc[new_df.index, col] = new_df[col]
                    else:
                        combined = combined.join(new_df[[col]], how="outer")
                combined[key] = combined.index
                overlay_df = combined.reset_index(drop=True)

            tbl = pa.Table.from_pandas(overlay_df, preserve_index=False)
            self._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
            tbl = with_overlay_metadata(tbl, namespace=namespace, key=key, created_at=now_utc())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp.parquet")
            pq.write_table(tbl, tmp, compression=PARQUET_COMPRESSION)
            os.replace(tmp, out_path)

            rows_matched = int(overlay_df.shape[0])
            record_event(
                self.events_path,
                "attach",
                dataset=self.name,
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

        with dataset_write_lock(self.dir):
            return _write_overlay()

    def _attach_duckdb(
        self,
        *,
        path: Path,
        namespace: str,
        key: str,
        key_col: str,
        columns: Optional[Iterable[str]],
        allow_overwrite: bool,
        allow_missing: bool,
        note: str,
    ) -> int:
        """
        Attach derived columns using DuckDB for large parquet inputs.
        """
        if path.suffix.lower() != ".parquet":
            raise SchemaError("duckdb backend requires parquet input.")
        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise SchemaError("duckdb backend requires duckdb (install duckdb).") from e

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

        out_path = overlay_path(self.dir, namespace)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_overlay_duckdb() -> int:
            con = duckdb.connect()
            try:
                base_sql = str(self.records_path).replace("'", "''")
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
                con.execute(
                    f"CREATE TEMP VIEW incoming AS SELECT {incoming_select} FROM read_parquet('{incoming_sql}')"
                )

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
                    raise SchemaError(
                        f"Duplicate keys in attachment input: {dup_keys} key(s) repeated. Sample: {sample}."
                    )

                if key in {"sequence", "sequence_norm", "sequence_ci"}:
                    base_key_expr = _key_expr("sequence")
                    dup_base = int(
                        con.execute(
                            "SELECT COUNT(*) FROM "
                            f"(SELECT {base_key_expr} AS k FROM base GROUP BY k HAVING COUNT(*) > 1)"
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
                        f"SELECT COUNT(*) FROM (SELECT {key_q} FROM existing_overlay "
                        f"GROUP BY {key_q} HAVING COUNT(*) > 1)"
                    )
                    dup_existing = int(con.execute(dup_query).fetchone()[0])
                    if dup_existing:
                        raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")

                    existing_set = set(existing_cols)
                    overlap_cols = sorted((existing_set & set(targets)) - {key})
                    if overlap_cols and not allow_overwrite:
                        raise NamespaceError(
                            f"Columns already exist: {', '.join(overlap_cols)}. Use --allow-overwrite."
                        )

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
                self._validate_registry_schema(namespace=namespace, schema=schema, key=key)
                metadata = dict(schema.metadata or {})
                metadata[OVERLAY_META_NAMESPACE.encode("utf-8")] = str(namespace).encode("utf-8")
                metadata[OVERLAY_META_KEY.encode("utf-8")] = str(key).encode("utf-8")
                metadata[OVERLAY_META_CREATED.encode("utf-8")] = str(now_utc()).encode("utf-8")

                def _batches():
                    for batch in pf_tmp.iter_batches(batch_size=65536):
                        yield batch

                write_parquet_atomic_batches(
                    _batches(),
                    schema,
                    out_path,
                    snapshot_dir=None,
                    metadata=metadata,
                )
                tmp_path.unlink(missing_ok=True)

                record_event(
                    self.events_path,
                    "attach",
                    dataset=self.name,
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

        with dataset_write_lock(self.dir):
            return _write_overlay_duckdb()

    # Friendly alias for didactic API name in README/examples
    def attach_columns(
        self,
        path: Path,
        namespace: str,
        *,
        key: str,
        key_col: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
        allow_overwrite: bool = False,
        allow_missing: bool = False,
        parse_json: bool = True,
        backend: str = "pyarrow",
        note: str = "",
    ) -> int:
        return self.attach(
            path,
            namespace,
            key=key,
            key_col=key_col,
            columns=columns,
            allow_overwrite=allow_overwrite,
            allow_missing=allow_missing,
            parse_json=parse_json,
            backend=backend,
            note=note,
        )

    def write_overlay(
        self,
        namespace: str,
        table_or_batches,
        *,
        key: str = "id",
        overwrite: bool = False,
        allow_missing: bool = False,
    ) -> int:
        """
        Attach a derived overlay from an Arrow/Pandas table or batches.
        """
        if isinstance(table_or_batches, pa.Table):
            tbl = table_or_batches
        elif isinstance(table_or_batches, pd.DataFrame):
            tbl = pa.Table.from_pandas(table_or_batches, preserve_index=False)
        else:
            tbl = pa.Table.from_batches(list(table_or_batches))

        self._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
        if key not in tbl.schema.names:
            raise SchemaError(f"Overlay table missing key column '{key}'.")
        attach_cols = [c for c in tbl.schema.names if c != key]
        if not attach_cols:
            return 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "overlay.parquet"
            pq.write_table(tbl, tmp_path, compression=PARQUET_COMPRESSION)
            return self.attach(
                tmp_path,
                namespace=namespace,
                key=key,
                key_col=key,
                columns=attach_cols,
                allow_overwrite=overwrite,
                allow_missing=allow_missing,
                parse_json=False,
                backend="pyarrow",
            )

    def list_overlays(self) -> List[OverlayInfo]:
        """Return overlay metadata summaries."""
        self._require_exists()
        overlays = []
        for path in list_overlays(self.dir):
            meta = overlay_metadata(path)
            pf = pq.ParquetFile(str(path))
            overlays.append(
                OverlayInfo(
                    namespace=meta.get("namespace") or path.stem,
                    key=meta.get("key"),
                    created_at=meta.get("created_at"),
                    path=str(path),
                    columns=list(pf.schema_arrow.names),
                    fingerprint=fingerprint_parquet(path),
                )
            )
        return overlays

    def remove_overlay(self, namespace: str, *, mode: str = "error") -> dict:
        """
        Remove or archive an overlay namespace.
        """
        if mode not in {"error", "delete", "archive"}:
            raise SchemaError(f"Unsupported remove_overlay mode '{mode}'.")
        path = overlay_path(self.dir, namespace)
        if not path.exists():
            if mode == "error":
                raise SchemaError(f"Overlay '{namespace}' not found.")
            return {"removed": False, "namespace": namespace}

        with dataset_write_lock(self.dir):
            if mode == "archive":
                archive_dir = path.parent / "_archived"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = now_utc().replace(":", "").replace("-", "")
                archived = archive_dir / f"{path.stem}-{stamp}.parquet"
                path.replace(archived)
                record_event(
                    self.events_path,
                    "archive_overlay",
                    dataset=self.name,
                    args={"namespace": namespace, "archived": str(archived)},
                    target_path=self.records_path,
                )
                return {"removed": True, "namespace": namespace, "archived_path": str(archived)}

            path.unlink()
            record_event(
                self.events_path,
                "remove_overlay",
                dataset=self.name,
                args={"namespace": namespace},
                target_path=self.records_path,
            )
            return {"removed": True, "namespace": namespace}

    # ---- validation & utils ----

    def validate(self, strict: bool = False) -> None:
        """
        Validate schema, ID uniqueness, alphabet constraints, and namespacing policy.
        In strict mode, warnings become errors for alphabet/namespacing issues.
        """
        self._require_exists()
        pf = pq.ParquetFile(str(self.records_path))
        schema = pf.schema_arrow
        names = set(schema.names)

        # required columns present + type checks
        for req, dtype in REQUIRED_COLUMNS:
            if req not in names:
                raise SchemaError(f"Missing required column: {req}")
            if schema.field(req).type != dtype:
                raise SchemaError(f"Column '{req}' has type {schema.field(req).type}, expected {dtype}.")

        # namespacing policy (no legacy exceptions)
        essential = {k for k, _ in REQUIRED_COLUMNS}
        derived = [c for c in schema.names if c not in essential]
        bad_ns = [c for c in derived if "__" not in c or c.split("__", 1)[0] == ""]
        if bad_ns:
            msg = (
                "Derived columns must be namespaced as '<tool>__<field>'. "
                f"Offending columns: {', '.join(sorted(bad_ns))}"
            )
            raise NamespaceError(msg)

        if self._tombstone_path().exists():
            _ = self._load_overlays(include_tombstone=True, namespaces=[TOMBSTONE_NAMESPACE])
            tomb_pf = pq.ParquetFile(str(self._tombstone_path()))
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

        # streaming validation + uniqueness check
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "validate.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE seen (val TEXT PRIMARY KEY)")
                dup_count = 0
                dup_samples: List[str] = []
                row_idx = 0
                for batch in iter_parquet_batches(
                    self.records_path,
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
                            raise SchemaError(
                                f"Row {row_idx}: length {ln} does not match sequence length {len(seq_norm)}."
                            )
                        if compute_id(bt_norm, seq_norm) != str(rid):
                            raise SchemaError(f"Row {row_idx}: id does not match bio_type+sequence.")
                if dup_count:
                    sample = ", ".join(dup_samples)
                    raise DuplicateIDError(
                        f"Duplicate ids detected: {dup_count} duplicate row(s). Sample ids: {sample}."
                    )
            finally:
                conn.close()

    def dedupe(
        self,
        *,
        key: str,
        keep: str,
        batch_size: int = 65536,
        dry_run: bool = False,
        maintenance: bool = False,
    ) -> DedupeStats:
        """
        Deduplicate rows by key, keeping either the first or last occurrence.
        """
        self._require_exists()
        if not maintenance:
            raise SchemaError("dedupe is a maintenance-only operation.")
        if batch_size < 1:
            raise SequencesError("batch_size must be >= 1.")
        key = str(key or "").strip()
        keep = str(keep or "").strip()
        if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
            raise SchemaError(f"Unsupported dedupe key '{key}'.")
        if keep not in {"keep-first", "keep-last"}:
            raise SchemaError(f"Unsupported dedupe keep policy '{keep}'.")

        key_cols = ["id"] if key == "id" else ["sequence"]
        if key == "sequence_ci":
            key_cols = ["sequence", "alphabet"]

        def _chunked(items: List[str], size: int = 900) -> Iterable[List[str]]:
            for i in range(0, len(items), size):
                yield items[i : i + size]

        def _fetch_existing(conn: sqlite3.Connection, keys: List[str]) -> set:
            existing: set[str] = set()
            if not keys:
                return existing
            for chunk in _chunked(keys):
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(f"SELECT k FROM seen WHERE k IN ({placeholders})", chunk)
                existing.update(row[0] for row in cur.fetchall())
            return existing

        def _fetch_last_indices(conn: sqlite3.Connection, keys: List[str]) -> Dict[str, int]:
            last: Dict[str, int] = {}
            if not keys:
                return last
            for chunk in _chunked(keys):
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(f"SELECT k, idx FROM last WHERE k IN ({placeholders})", chunk)
                for k, idx in cur.fetchall():
                    last[str(k)] = int(idx)
            return last

        def _filter_batch(batch: pa.RecordBatch, keep_mask: List[bool]) -> Iterable[pa.RecordBatch]:
            if not keep_mask or not any(keep_mask):
                return []
            tbl = pa.Table.from_batches([batch])
            filtered = tbl.filter(pa.array(keep_mask))
            return filtered.to_batches()

        with dataset_write_lock(self.dir):
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "dedupe.sqlite"
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("CREATE TABLE counts (k TEXT PRIMARY KEY, cnt INTEGER NOT NULL)")
                    if keep == "keep-last":
                        conn.execute("CREATE TABLE last (k TEXT PRIMARY KEY, idx INTEGER NOT NULL)")

                    row_idx = 0
                    for batch in iter_parquet_batches(self.records_path, columns=key_cols, batch_size=int(batch_size)):
                        keys = self._key_list_from_batch(batch, key)
                        conn.executemany(
                            "INSERT INTO counts (k, cnt) VALUES (?, 1) ON CONFLICT(k) DO UPDATE SET cnt = cnt + 1",
                            [(k,) for k in keys],
                        )
                        if keep == "keep-last":
                            conn.executemany(
                                "INSERT INTO last (k, idx) VALUES (?, ?) "
                                "ON CONFLICT(k) DO UPDATE SET idx = excluded.idx",
                                [(k, row_idx + i) for i, k in enumerate(keys)],
                            )
                        row_idx += len(keys)

                    groups = conn.execute("SELECT COUNT(*) FROM counts WHERE cnt > 1").fetchone()[0]
                    dropped = conn.execute("SELECT COALESCE(SUM(cnt - 1), 0) FROM counts WHERE cnt > 1").fetchone()[0]
                    stats = DedupeStats(
                        rows_total=int(row_idx),
                        rows_dropped=int(dropped),
                        groups=int(groups),
                        key=key,
                        keep=keep,
                    )

                    if dry_run or stats.rows_dropped == 0:
                        return stats

                    pf = pq.ParquetFile(str(self.records_path))
                    schema = pf.schema_arrow

                    def _iter_keep_first():
                        conn.execute("CREATE TABLE seen (k TEXT PRIMARY KEY)")
                        for batch in iter_parquet_batches(self.records_path, columns=None, batch_size=int(batch_size)):
                            keys = self._key_list_from_batch(batch, key)
                            existing = _fetch_existing(conn, keys)
                            batch_seen: set[str] = set()
                            keep_mask: List[bool] = []
                            new_keys: List[tuple] = []
                            for k in keys:
                                if k in existing or k in batch_seen:
                                    keep_mask.append(False)
                                    continue
                                keep_mask.append(True)
                                batch_seen.add(k)
                                new_keys.append((k,))
                            if new_keys:
                                conn.executemany("INSERT OR IGNORE INTO seen(k) VALUES (?)", new_keys)
                            for out_batch in _filter_batch(batch, keep_mask):
                                yield out_batch

                    def _iter_keep_last():
                        row_at = 0
                        for batch in iter_parquet_batches(self.records_path, columns=None, batch_size=int(batch_size)):
                            keys = self._key_list_from_batch(batch, key)
                            last_idx = _fetch_last_indices(conn, keys)
                            keep_mask = [(last_idx.get(k) == row_at + i) for i, k in enumerate(keys)]
                            for out_batch in _filter_batch(batch, keep_mask):
                                yield out_batch
                            row_at += len(keys)

                    batches = _iter_keep_first() if keep == "keep-first" else _iter_keep_last()
                    metadata = self._base_metadata(created_at=now_utc())
                    write_parquet_atomic_batches(
                        batches,
                        schema,
                        self.records_path,
                        self.snapshot_dir,
                        compression=PARQUET_COMPRESSION,
                        metadata=metadata,
                    )
                    record_event(
                        self.events_path,
                        "dedupe",
                        dataset=self.name,
                        args={
                            "key": key,
                            "keep": keep,
                            "rows_total": stats.rows_total,
                            "rows_dropped": stats.rows_dropped,
                            "groups": stats.groups,
                        },
                        target_path=self.records_path,
                    )
                    return stats
                finally:
                    conn.close()

    def get(self, record_id: str, columns: Optional[List[str]] = None, *, include_deleted: bool = False):
        """Return a single record by id (as a pandas DataFrame row)."""
        con, query, params = self._duckdb_query(
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
                return pd.DataFrame(columns=columns or self.schema().names)
            tbl = pa.Table.from_batches([batch])
            return tbl.to_pandas()
        finally:
            con.close()

    def grep(self, pattern: str, limit: int = 20, batch_size: int = 65536, *, include_deleted: bool = False):
        """Regex search across sequences, returning first `limit` hits."""
        if batch_size < 1:
            raise SequencesError("batch_size must be >= 1.")
        pattern_ci = f"(?i){pattern}"
        con, query, params = self._duckdb_query(
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

    def export(
        self,
        fmt: str,
        out_path: Path,
        columns: Optional[List[str]] = None,
        *,
        include_deleted: bool = False,
    ) -> None:
        """Export current table to CSV or JSONL."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            first = True
            wrote = False
            for batch in self.scan(
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
                empty_cols = columns if columns else self.schema().names
                pd.DataFrame(columns=list(empty_cols)).to_csv(out_path, index=False)
        elif fmt == "jsonl":
            wrote = False
            with out_path.open("w", encoding="utf-8") as f:
                for batch in self.scan(
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
                        f.write(text)
                        wrote = True
            if not wrote:
                out_path.write_text("", encoding="utf-8")
        else:
            raise SequencesError("Unsupported export format. Use csv|jsonl.")

    def _write_reserved_overlay(self, namespace: str, key: str, overlay_df: pd.DataFrame) -> int:
        if not _NS_RE.match(namespace):
            raise NamespaceError(
                "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
            )
        if overlay_df[key].duplicated().any():
            raise SchemaError(f"Overlay has duplicate keys for '{key}'.")
        out_path = overlay_path(self.dir, namespace)
        if out_path.exists():
            meta = overlay_metadata(out_path)
            if meta.get("key") != key:
                raise SchemaError(
                    f"Overlay key mismatch for namespace '{namespace}': existing={meta.get('key')} new={key}"
                )
            existing_df = pq.read_table(out_path).to_pandas()
            if existing_df[key].duplicated().any():
                raise SchemaError(f"Existing overlay has duplicate keys for '{key}'.")
            existing_df = existing_df.set_index(key, drop=False)
            new_df = overlay_df.set_index(key, drop=False)
            combined = existing_df
            for col in new_df.columns:
                if col == key:
                    continue
                if col in combined.columns:
                    combined.loc[new_df.index, col] = new_df[col]
                else:
                    combined = combined.join(new_df[[col]], how="outer")
            combined[key] = combined.index
            overlay_df = combined.reset_index(drop=True)

        tbl = pa.Table.from_pandas(overlay_df, preserve_index=False)
        tbl = with_overlay_metadata(tbl, namespace=namespace, key=key, created_at=now_utc())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.parquet")
        pq.write_table(tbl, tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, out_path)
        return int(overlay_df.shape[0])

    def tombstone(
        self,
        ids: Sequence[str],
        *,
        reason: Optional[str] = None,
        deleted_at: Optional[str] = None,
        allow_missing: bool = False,
    ) -> int:
        """
        Mark records as deleted using a tombstone overlay.
        """
        self._require_exists()
        ids_list = [str(i).strip() for i in ids if str(i).strip()]
        if not ids_list:
            raise SchemaError("Provide at least one id to tombstone.")
        if len(ids_list) != len(set(ids_list)):
            raise SchemaError("Duplicate ids provided to tombstone.")

        if not allow_missing:
            targets = set(ids_list)
            found: set[str] = set()
            for batch in iter_parquet_batches(self.records_path, columns=["id"]):
                for rid in batch.column("id").to_pylist():
                    s = str(rid)
                    if s in targets:
                        found.add(s)
                if len(found) == len(targets):
                    break
            missing = sorted(targets - found)
            if missing:
                sample = ", ".join(missing[:5])
                raise SchemaError(f"{len(missing)} id(s) not found in dataset (sample: {sample}).")

        ts = pd.to_datetime(deleted_at, utc=True) if deleted_at is not None else pd.Timestamp.now(tz="UTC")
        ts_series = pd.Series([ts] * len(ids_list), dtype="datetime64[ns, UTC]")
        reason_vals = [reason] * len(ids_list)
        overlay_df = pd.DataFrame(
            {
                "id": ids_list,
                "usr__deleted": [True] * len(ids_list),
                "usr__deleted_at": ts_series,
                "usr__deleted_reason": reason_vals,
            }
        )

        with dataset_write_lock(self.dir):
            rows = self._write_reserved_overlay(TOMBSTONE_NAMESPACE, "id", overlay_df)
            record_event(
                self.events_path,
                "tombstone",
                dataset=self.name,
                args={"rows": rows, "reason": reason or "", "allow_missing": allow_missing},
                target_path=self.records_path,
            )
            return rows

    def restore(self, ids: Sequence[str], *, allow_missing: bool = False) -> int:
        """
        Clear tombstones for provided ids.
        """
        self._require_exists()
        ids_list = [str(i).strip() for i in ids if str(i).strip()]
        if not ids_list:
            raise SchemaError("Provide at least one id to restore.")
        if len(ids_list) != len(set(ids_list)):
            raise SchemaError("Duplicate ids provided to restore.")

        if not allow_missing:
            targets = set(ids_list)
            found: set[str] = set()
            for batch in iter_parquet_batches(self.records_path, columns=["id"]):
                for rid in batch.column("id").to_pylist():
                    s = str(rid)
                    if s in targets:
                        found.add(s)
                if len(found) == len(targets):
                    break
            missing = sorted(targets - found)
            if missing:
                sample = ", ".join(missing[:5])
                raise SchemaError(f"{len(missing)} id(s) not found in dataset (sample: {sample}).")

        ts_series = pd.Series([pd.NaT] * len(ids_list), dtype="datetime64[ns, UTC]")
        overlay_df = pd.DataFrame(
            {
                "id": ids_list,
                "usr__deleted": [False] * len(ids_list),
                "usr__deleted_at": ts_series,
                "usr__deleted_reason": [None] * len(ids_list),
            }
        )

        with dataset_write_lock(self.dir):
            rows = self._write_reserved_overlay(TOMBSTONE_NAMESPACE, "id", overlay_df)
            record_event(
                self.events_path,
                "restore",
                dataset=self.name,
                args={"rows": rows, "allow_missing": allow_missing},
                target_path=self.records_path,
            )
            return rows

    def manifest(self, *, include_events: bool = False) -> Manifest:
        """
        Return a structured manifest of the dataset, overlays, and snapshots.
        """
        self._require_exists()
        base_pf = pq.ParquetFile(str(self.records_path))
        base_meta = base_pf.schema_arrow.metadata or {}
        meta_decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in base_meta.items()}
        snapshots = []
        if self.snapshot_dir.exists():
            snapshots = sorted(str(p) for p in self.snapshot_dir.glob("records-*.parquet"))
        events_count = None
        if include_events and self.events_path.exists():
            events_count = sum(1 for _ in self.events_path.open("r", encoding="utf-8"))
        return Manifest(
            name=self.name,
            path=str(self.records_path),
            metadata=meta_decoded,
            fingerprint=fingerprint_parquet(self.records_path),
            overlays=self.list_overlays(),
            snapshots=snapshots,
            events_count=events_count,
        )

    def manifest_dict(self, *, include_events: bool = False) -> dict:
        return self.manifest(include_events=include_events).to_dict()

    def describe(
        self,
        opts,
        *,
        columns: Optional[List[str]] = None,
        sample: int = 1024,
        batch_size: int = 65536,
        include_deleted: bool = False,
    ) -> List[dict]:
        """Profile columns with optional overlay merge."""
        from .pretty import profile_batches

        merged_schema = self.schema()
        cols = columns if columns else list(merged_schema.names)
        if not include_deleted:
            cols = [c for c in cols if c not in TOMBSTONE_COLUMNS]
        out_schema = pa.schema([merged_schema.field(name) for name in cols])

        return profile_batches(
            self.scan(
                columns=cols,
                include_overlays=True,
                include_deleted=include_deleted,
                batch_size=int(batch_size),
            ),
            out_schema,
            opts,
            columns=cols,
            sample=int(sample),
            total_rows=None,
        )

    def materialize(
        self,
        *,
        namespaces: Optional[Sequence[str]] = None,
        keep_overlays: bool = True,
        archive_overlays: bool = False,
        drop_deleted: bool = False,
        maintenance: bool = False,
    ) -> None:
        """Merge overlays into the base table (maintenance operation)."""
        if not maintenance:
            raise SchemaError("materialize is a maintenance operation. Pass maintenance=True to proceed.")
        with dataset_write_lock(self.dir):
            self._require_exists()
            overlays_all = list_overlays(self.dir)
            if not overlays_all and not drop_deleted:
                return

            def _overlay_ns(path: Path) -> str:
                meta = overlay_metadata(path)
                return meta.get("namespace") or path.stem

            if namespaces is None:
                overlays = [p for p in overlays_all if _overlay_ns(p) not in RESERVED_NAMESPACES]
            else:
                ns_set = {str(n) for n in namespaces}
                overlays = [p for p in overlays_all if _overlay_ns(p) in ns_set]

            if not overlays and not drop_deleted:
                return

            require_registry = any(_overlay_ns(p) not in RESERVED_NAMESPACES for p in overlays)
            registry = self._registry(required=require_registry) if require_registry else {}

            try:
                import duckdb  # type: ignore
            except ImportError as e:
                raise SchemaError("materialize requires duckdb (install duckdb).") from e

            def _sql_ident(name: str) -> str:
                escaped = str(name).replace('"', '""')
                return f'"{escaped}"'

            def _key_expr(expr: str, *, key: str) -> str:
                if key == "sequence_ci":
                    return f"NULLIF(UPPER(TRIM(CAST({expr} AS VARCHAR))), '')"
                return f"NULLIF(TRIM(CAST({expr} AS VARCHAR)), '')"

            base_pf = pq.ParquetFile(str(self.records_path))
            base_cols = list(base_pf.schema_arrow.names)
            essential = {k for k, _ in REQUIRED_COLUMNS}
            existing_cols = set(base_cols)

            tmp_path = self.records_path.with_suffix(".materialize.parquet")
            con = duckdb.connect()
            try:
                base_sql = str(self.records_path).replace("'", "''")
                con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{base_sql}')")
                base_view = "base"

                if drop_deleted and self._tombstone_path().exists():
                    tomb_path = self._tombstone_path()
                    meta = overlay_metadata(tomb_path)
                    key = meta.get("key")
                    if key and key != "id":
                        raise SchemaError("Tombstone overlay must use key 'id'.")
                    tomb_sql = str(tomb_path).replace("'", "''")
                    con.execute(
                        f"CREATE TEMP VIEW tombstone AS SELECT id, usr__deleted FROM read_parquet('{tomb_sql}')"
                    )
                    con.execute(
                        "CREATE TEMP VIEW base_filtered AS "
                        "SELECT b.* FROM base b LEFT JOIN tombstone t ON b.id = t.id "
                        "WHERE COALESCE(t.usr__deleted, FALSE) = FALSE"
                    )
                    base_view = "base_filtered"

                select_exprs = [f"b.{_sql_ident(col)} AS {_sql_ident(col)}" for col in base_cols]
                join_clauses: List[str] = []

                for idx, path in enumerate(overlays):
                    meta = overlay_metadata(path)
                    key = meta.get("key")
                    if not key:
                        raise SchemaError(f"Overlay missing required metadata key: {path}")
                    if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
                        raise SchemaError(f"Unsupported overlay key '{key}': {path}")

                    pf_overlay = pq.ParquetFile(str(path))
                    overlay_cols = list(pf_overlay.schema_arrow.names)
                    if key not in overlay_cols:
                        raise SchemaError(f"Overlay missing key column '{key}': {path}")
                    validate_overlay_schema(_overlay_ns(path), pf_overlay.schema_arrow, registry=registry, key=key)

                    derived_cols = [c for c in overlay_cols if c != key]
                    if not derived_cols:
                        raise SchemaError(f"Overlay '{path.name}' has no derived columns.")

                    for col in derived_cols:
                        if col in essential:
                            raise NamespaceError(f"Overlay cannot modify required column '{col}'.")
                        if "__" not in col:
                            raise NamespaceError(f"Derived columns must be namespaced (got '{col}').")
                        if col in existing_cols:
                            raise NamespaceError(f"Derived columns already exist: {col}")

                    view_name = f"overlay_{idx}"
                    overlay_sql = str(path).replace("'", "''")
                    con.execute(f"CREATE TEMP VIEW {view_name} AS SELECT * FROM read_parquet('{overlay_sql}')")

                    dup_overlay = int(
                        con.execute(
                            "SELECT COUNT(*) FROM "
                            f"(SELECT {_sql_ident(key)} FROM {view_name} "
                            f"GROUP BY {_sql_ident(key)} HAVING COUNT(*) > 1)"
                        ).fetchone()[0]
                    )
                    if dup_overlay:
                        raise SchemaError(f"Overlay has duplicate keys for '{key}': {path}")

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
                        base_key_expr = _key_expr(f"b.{_sql_ident('sequence')}", key=key)
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
                        base_key_expr = _key_expr(f"b.{_sql_ident('id')}", key=key)

                    overlay_key_expr = _key_expr(f"o{idx}.{_sql_ident(key)}", key=key)
                    join_clauses.append(f"LEFT JOIN {view_name} o{idx} ON {base_key_expr} = {overlay_key_expr}")
                    for col in derived_cols:
                        select_exprs.append(f"o{idx}.{_sql_ident(col)} AS {_sql_ident(col)}")
                    existing_cols.update(derived_cols)

                query = "SELECT " + ", ".join(select_exprs) + f" FROM {base_view} b " + " ".join(join_clauses)
                tmp_sql = str(tmp_path).replace("'", "''")
                compression = PARQUET_COMPRESSION.upper()
                con.execute(f"COPY ({query}) TO '{tmp_sql}' (FORMAT PARQUET, COMPRESSION '{compression}')")
            finally:
                con.close()

            pf_tmp = pq.ParquetFile(str(tmp_path))
            schema = pf_tmp.schema_arrow
            metadata = self._base_metadata(created_at=now_utc())

            def _batches():
                for batch in iter_parquet_batches(tmp_path, columns=None):
                    yield batch

            write_parquet_atomic_batches(
                _batches(),
                schema,
                self.records_path,
                self.snapshot_dir,
                metadata=metadata,
            )
            tmp_path.unlink(missing_ok=True)

            overlay_fingerprints = []
            for p in overlays:
                meta = overlay_metadata(p)
                overlay_fingerprints.append(
                    {
                        "namespace": meta.get("namespace") or p.stem,
                        "key": meta.get("key"),
                        "fingerprint": fingerprint_parquet(p).to_dict(),
                    }
                )
            tomb_fp = None
            if drop_deleted and self._tombstone_path().exists():
                tomb_fp = fingerprint_parquet(self._tombstone_path()).to_dict()

            if archive_overlays:
                archive_dir = self.dir / "_derived" / "_archived"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = now_utc().replace(":", "").replace("-", "")
                for p in overlays:
                    archived = archive_dir / f"{p.stem}-{stamp}.parquet"
                    p.replace(archived)
            elif not keep_overlays:
                for p in overlays:
                    p.unlink()

            record_event(
                self.events_path,
                "materialize",
                dataset=self.name,
                args={
                    "namespaces": list(namespaces) if namespaces is not None else None,
                    "drop_overlays": bool(not keep_overlays or archive_overlays),
                    "archive_overlays": bool(archive_overlays),
                    "drop_deleted": bool(drop_deleted),
                    "overlays": overlay_fingerprints,
                    "tombstone": tomb_fp,
                },
                target_path=self.records_path,
            )

    def snapshot(self) -> None:
        """Write a timestamped snapshot and atomically persist current table."""
        with dataset_write_lock(self.dir):
            self._require_exists()
            snapshot_parquet_file(self.records_path, self.snapshot_dir)
            record_event(
                self.events_path,
                "snapshot",
                dataset=self.name,
                args={},
                target_path=self.records_path,
            )
