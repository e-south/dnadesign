"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset.py

USR dataset lifecycle operations and validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import shutil
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

from .dataset_activity import append_meta_note as append_dataset_meta_note
from .dataset_activity import record_dataset_activity_event
from .dataset_dedupe import dedupe_dataset
from .dataset_materialize import materialize_dataset
from .dataset_overlay_ops import (
    attach_columns_dataset,
    attach_dataset,
    write_overlay_dataset,
    write_overlay_part_dataset,
)
from .dataset_query import create_overlay_view, sql_ident, sql_str
from .dataset_registry_modes import normalize_registry_mode, validate_overlays_for_registry_mode
from .dataset_state import (
    clear_state as dataset_clear_state,
)
from .dataset_state import (
    ensure_ids_exist as dataset_ensure_ids_exist,
)
from .dataset_state import (
    get_state as dataset_get_state,
)
from .dataset_state import (
    restore as dataset_restore,
)
from .dataset_state import (
    set_state as dataset_set_state,
)
from .dataset_state import (
    tombstone as dataset_tombstone,
)
from .errors import (
    AlphabetError,
    DuplicateGroup,
    DuplicateIDError,
    NamespaceError,
    SchemaError,
    SequencesError,
)
from .events import fingerprint_parquet
from .maintenance import maintenance as maintenance_context
from .maintenance import require_maintenance
from .normalize import compute_id, normalize_sequence, validate_alphabet, validate_bio_type  # case-preserving
from .overlays import (
    OVERLAY_META_CREATED,
    OVERLAY_META_REGISTRY_HASH,
    list_overlays,
    overlay_dir_path,
    overlay_metadata,
    overlay_path,
    overlay_schema,
    with_overlay_metadata,
)
from .registry import (
    USR_STATE_NAMESPACE,
    load_registry,
    registry_bytes,
    registry_hash,
    validate_overlay_schema,
)
from .schema import ARROW_SCHEMA, META_REGISTRY_HASH, REQUIRED_COLUMNS, merge_base_metadata, with_base_metadata
from .storage.locking import dataset_write_lock
from .storage.parquet import (
    PARQUET_COMPRESSION,
    iter_parquet_batches,
    now_utc,
    snapshot_parquet_file,
    write_parquet_atomic,
    write_parquet_atomic_batches,
)
from .types import AddSequencesResult, DatasetInfo, Fingerprint, Manifest, OverlayInfo

RECORDS = "records.parquet"
SNAPDIR = "_snapshots"  # standardized
META_MD = "meta.md"
EVENTS_LOG = ".events.log"

_NS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
TOMBSTONE_NAMESPACE = "usr"
TOMBSTONE_COLUMNS = ("usr__deleted", "usr__deleted_at", "usr__deleted_reason")
RESERVED_NAMESPACES = {TOMBSTONE_NAMESPACE}
LEGACY_DATASET_PREFIX = "archived"
USR_STATE_SCHEMA_TYPES = {
    "id": pa.string(),
    "usr_state__masked": pa.bool_(),
    "usr_state__qc_status": pa.string(),
    "usr_state__split": pa.string(),
    "usr_state__supersedes": pa.string(),
    "usr_state__lineage": pa.list_(pa.string()),
}
USR_STATE_QC_STATUS_ALLOWED = {"pass", "fail", "warn", "unknown"}
USR_STATE_SPLIT_ALLOWED = {"train", "val", "test", "holdout"}


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
    if p.parts and p.parts[0] == LEGACY_DATASET_PREFIX:
        raise SequencesError(
            "legacy dataset paths under 'archived/' are not supported. "
            "Use canonical datasets or datasets/_archive/<namespace>/<dataset>."
        )
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

    def maintenance(self, reason: Optional[str] = None, *, actor: Optional[dict] = None):
        return maintenance_context(reason=reason, actor=actor)

    def init(self, source: str = "", notes: str = "") -> None:
        """Create a new, empty dataset directory with canonical schema."""
        with dataset_write_lock(self.dir):
            self._require_registry_for_mutation("init")
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.records_path.exists():
                raise SequencesError(f"Dataset already initialized: {self.records_path}")
            ts = now_utc()
            empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in ARROW_SCHEMA], schema=ARROW_SCHEMA)
            reg_hash = self._registry_hash(required=True)
            empty = with_base_metadata(empty, created_at=ts, registry_hash=reg_hash)
            write_parquet_atomic(empty, self.records_path, self.snapshot_dir)
            self._auto_freeze_registry()
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
            self._record_event(
                "init",
                args={"source": source},
            )

    # --- lightweight, best-effort scratch-pad logging in meta.md ---
    def append_meta_note(self, title: str, code_block: Optional[str] = None) -> None:
        append_dataset_meta_note(
            dataset_dir=self.dir,
            dataset_name=self.name,
            meta_path=self.meta_path,
            title=title,
            code_block=code_block,
        )

    def _record_event(
        self,
        action: str,
        *,
        args: Optional[dict] = None,
        metrics: Optional[dict] = None,
        artifacts: Optional[dict] = None,
        maintenance: Optional[dict] = None,
        target_path: Optional[Path] = None,
        registry_hash: Optional[str] = None,
        actor: Optional[dict] = None,
    ) -> None:
        record_dataset_activity_event(
            events_path=self.events_path,
            action=action,
            dataset_name=self.name,
            dataset_root=self.root,
            records_path=self.records_path,
            args=args,
            metrics=metrics,
            artifacts=artifacts,
            maintenance=maintenance,
            target_path=target_path,
            registry_hash=registry_hash,
            actor=actor,
        )

    def log_event(
        self,
        action: str,
        *,
        args: Optional[dict] = None,
        metrics: Optional[dict] = None,
        artifacts: Optional[dict] = None,
        maintenance: Optional[dict] = None,
        target_path: Optional[Path] = None,
        actor: Optional[dict] = None,
    ) -> None:
        self._require_exists()
        self._record_event(
            action,
            args=args,
            metrics=metrics,
            artifacts=artifacts,
            maintenance=maintenance,
            target_path=target_path,
            actor=actor,
        )

    def freeze_registry(self) -> Path:
        """
        Snapshot the current registry into this dataset and persist its hash.
        """
        ctx = require_maintenance("freeze_registry")
        self._require_exists()
        with dataset_write_lock(self.dir):
            snap_path, reg_hash, updated = self._auto_freeze_registry(record_auto_event=False)
            self._record_event(
                "registry_freeze",
                args={"registry_hash": reg_hash, "snapshot": str(snap_path), "auto": False, "updated": updated},
                maintenance={"reason": ctx.reason},
                actor=ctx.actor,
            )
            return snap_path

    def _require_exists(self) -> None:
        if not self.records_path.exists():
            raise SequencesError(f"Dataset not initialized: {self.records_path}")

    def _require_registry_for_mutation(self, action: str) -> dict:
        try:
            return load_registry(self.root, required=True)
        except SchemaError as e:
            raise SchemaError(f"Registry required for {action}. Create registry.yaml before mutating datasets.") from e

    def _base_metadata(self, *, created_at: Optional[str] = None) -> Dict[bytes, bytes]:
        md = None
        if self.records_path.exists():
            md = pq.ParquetFile(str(self.records_path)).schema_arrow.metadata
        reg_hash = registry_hash(self.root, required=True)
        return merge_base_metadata(md, created_at, reg_hash)

    def _tombstone_path(self) -> Path:
        return overlay_path(self.dir, TOMBSTONE_NAMESPACE)

    def _registry(self, *, required: bool) -> dict:
        return load_registry(self.root, required=required)

    def _validate_registry_schema(self, *, namespace: str, schema: pa.Schema, key: str) -> None:
        if namespace in RESERVED_NAMESPACES:
            return
        registry = self._registry(required=True)
        validate_overlay_schema(namespace, schema, registry=registry, key=key)

    def _registry_hash(self, *, required: bool) -> Optional[str]:
        return registry_hash(self.root, required=required)

    def _dataset_registry_hash(self) -> str:
        pf = pq.ParquetFile(str(self.records_path))
        md = pf.schema_arrow.metadata or {}
        raw = md.get(META_REGISTRY_HASH.encode("utf-8"))
        if not raw:
            raise SchemaError("Dataset does not have a registry_hash; run `usr maintenance registry-freeze`.")
        return raw.decode("utf-8")

    def _frozen_registry_path(self) -> Path:
        reg_hash = self._dataset_registry_hash()
        return self.dir / "_registry" / f"registry.{reg_hash}.yaml"

    def _auto_freeze_registry(self, *, record_auto_event: bool = True) -> tuple[Path, str, bool]:
        reg_hash = registry_hash(self.root, required=True)
        reg_bytes = registry_bytes(self.root)
        snap_dir = self.dir / "_registry"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / f"registry.{reg_hash}.yaml"
        created = False
        if not snap_path.exists():
            snap_path.write_bytes(reg_bytes)
            created = True

        pf = pq.ParquetFile(str(self.records_path))
        md = pf.schema_arrow.metadata or {}
        if md.get(META_REGISTRY_HASH.encode("utf-8")) != reg_hash.encode("utf-8"):

            def _iter_batches():
                for batch in iter_parquet_batches(self.records_path):
                    yield batch

            metadata = merge_base_metadata(md, registry_hash=reg_hash)
            write_parquet_atomic_batches(
                _iter_batches(),
                pf.schema_arrow,
                self.records_path,
                self.snapshot_dir,
                metadata=metadata,
            )
            created = True

        if created and record_auto_event:
            self._record_event(
                "registry_freeze",
                args={"registry_hash": reg_hash, "snapshot": str(snap_path), "auto": True},
            )
        return snap_path, reg_hash, created

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
        all_cols = list(cols)
        for col in derived_cols:
            if col not in all_cols:
                all_cols.append(col)
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
                existing_idx = base_schema.get_field_index(field.name)
                if existing_idx >= 0:
                    existing_field = base_schema.field(existing_idx)
                    if existing_field.type != field.type:
                        raise NamespaceError(
                            f"Derived column type mismatch in schema: {field.name} "
                            f"(base={existing_field.type}, overlay={field.type})"
                        )
                    continue
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
        namespace_filter = set(namespaces) if namespaces else None
        require_registry = any(
            (overlay_metadata(p).get("namespace") or p.stem) not in RESERVED_NAMESPACES for p in paths
        )
        registry = self._registry(required=require_registry) if require_registry else {}
        seen: Dict[str, Path] = {}
        for path in paths:
            meta = overlay_metadata(path)
            key = meta.get("key")
            if not key:
                raise SchemaError(f"Overlay missing required metadata key: {path}")
            ns = meta.get("namespace") or path.stem
            if ns in seen:
                raise SchemaError(
                    f"Overlay namespace '{ns}' has multiple sources: {seen[ns]} and {path}. "
                    "Resolve by compacting or removing one source."
                )
            seen[ns] = path
            if not include_tombstone and ns in RESERVED_NAMESPACES:
                continue
            if namespace_filter and ns not in namespace_filter:
                continue
            schema = overlay_schema(path)
            if ns not in RESERVED_NAMESPACES:
                validate_overlay_schema(ns, schema, registry=registry, key=key)
            if key not in schema.names:
                raise SchemaError(f"Overlay missing key column '{key}': {path}")
            overlay_cols = [c for c in schema.names if c != key]
            read_path = str(path / "part-*.parquet") if path.is_dir() else str(path)
            overlays.append(
                {
                    "namespace": ns,
                    "key": key,
                    "cols": overlay_cols,
                    "schema": schema,
                    "path": path,
                    "read_path": read_path,
                }
            )
        return overlays

    @staticmethod
    def _sql_ident(name: str) -> str:
        return sql_ident(name)

    @staticmethod
    def _sql_str(value: str) -> str:
        return sql_str(value)

    def _create_overlay_view(
        self,
        con,
        *,
        view_name: str,
        path: Path,
        key: str,
    ) -> None:
        create_overlay_view(con, view_name=view_name, path=path, key=key)

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
                    "read_path": str(tomb_path),
                }
            )

        select_expr_by_col = {col: f"b.{self._sql_ident(col)}" for col in base_cols}
        select_order = list(base_cols)

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

            view_name = f"overlay_{idx}"
            self._create_overlay_view(con, view_name=view_name, path=ov["path"], key=key)

            if key in {"sequence", "sequence_norm", "sequence_ci"}:
                bt_count = int(con.execute(f"SELECT COUNT(DISTINCT bio_type) FROM {base_view}").fetchone()[0])
                if bt_count > 1:
                    raise SchemaError("Attach by sequence requires dataset with a single bio_type.")
                if key == "sequence_ci":
                    bad = int(con.execute(f"SELECT COUNT(*) FROM {base_view} WHERE alphabet != 'dna_4'").fetchone()[0])
                    if bad:
                        raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
                base_key_expr = _key_expr(f"b.{self._sql_ident('sequence')}", key=key)
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
                base_key_expr = _key_expr(f"b.{self._sql_ident('id')}", key=key)

            overlay_key_expr = _key_expr(f"o{idx}.{self._sql_ident(key)}", key=key)
            join_clauses.append(f"LEFT JOIN {view_name} o{idx} ON {base_key_expr} = {overlay_key_expr}")
            for col in derived_cols:
                col_ident = self._sql_ident(col)
                overlay_expr = f"o{idx}.{col_ident}"
                existing_expr = select_expr_by_col.get(col)
                if existing_expr is None:
                    select_expr_by_col[col] = overlay_expr
                    select_order.append(col)
                else:
                    select_expr_by_col[col] = f"COALESCE({overlay_expr}, {existing_expr})"

        select_exprs: List[str] = []
        for col in select_order:
            if requested is None or col in requested:
                select_exprs.append(f"{select_expr_by_col[col]} AS {self._sql_ident(col)}")
                selected_cols.append(col)

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
        actor: Optional[dict] = None,
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

            self._auto_freeze_registry()
            self._record_event(
                "import_rows",
                args={"n": int(out_count), "source_param": source or "", "on_conflict": on_conflict},
                metrics={"rows_written": int(out_count), "rows_skipped": len(ids_skipped)},
                actor=actor,
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
        actor: Optional[dict] = None,
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
        return self._write_import_df(
            out_df,
            source=source,
            on_conflict="error",
            actor=actor,
        )

    def add_sequences(
        self,
        rows_or_sequences: Union[pd.DataFrame, Sequence[Dict[str, object]], Sequence[str]],
        *,
        bio_type: str,
        alphabet: str,
        source: str = "",
        created_at: Optional[str] = None,
        on_conflict: str = "error",
        actor: Optional[dict] = None,
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
            actor=actor,
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
        return attach_dataset(
            dataset=self,
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
            namespace_pattern=_NS_RE,
            reserved_namespaces=RESERVED_NAMESPACES,
        )

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
        return attach_columns_dataset(
            dataset=self,
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
            namespace_pattern=_NS_RE,
            reserved_namespaces=RESERVED_NAMESPACES,
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
        return write_overlay_dataset(
            dataset=self,
            namespace=namespace,
            table_or_batches=table_or_batches,
            key=key,
            overwrite=overwrite,
            allow_missing=allow_missing,
            namespace_pattern=_NS_RE,
            reserved_namespaces=RESERVED_NAMESPACES,
        )

    def write_overlay_part(
        self,
        namespace: str,
        table_or_batches,
        *,
        key: str = "id",
        key_col: Optional[str] = None,
        allow_missing: bool = False,
        actor: Optional[dict] = None,
    ) -> int:
        return write_overlay_part_dataset(
            dataset=self,
            namespace=namespace,
            table_or_batches=table_or_batches,
            key=key,
            key_col=key_col,
            allow_missing=allow_missing,
            actor=actor,
        )

    def list_overlays(self) -> List[OverlayInfo]:
        """Return overlay metadata summaries."""
        self._require_exists()
        overlays = []
        for path in list_overlays(self.dir):
            meta = overlay_metadata(path)
            parts = []
            if path.is_dir():
                parts = sorted(path.glob("part-*.parquet"))
            else:
                parts = [path]
            if not parts:
                raise SchemaError(f"Overlay has no parquet parts: {path}")
            schema = overlay_schema(path)
            rows = 0
            size_bytes = 0
            for part in parts:
                pf_part = pq.ParquetFile(str(part))
                rows += pf_part.metadata.num_rows
                size_bytes += int(part.stat().st_size)
            overlays.append(
                OverlayInfo(
                    namespace=meta.get("namespace") or path.stem,
                    key=meta.get("key"),
                    created_at=meta.get("created_at"),
                    path=str(path),
                    columns=list(schema.names),
                    fingerprint=Fingerprint(
                        rows=int(rows),
                        cols=int(len(schema.names)),
                        size_bytes=int(size_bytes),
                    ),
                )
            )
        return overlays

    def remove_overlay(self, namespace: str, *, mode: str = "error") -> dict:
        """
        Remove or archive an overlay namespace.
        """
        if mode not in {"error", "delete", "archive"}:
            raise SchemaError(f"Unsupported remove_overlay mode '{mode}'.")
        self._require_registry_for_mutation("remove_overlay")
        file_path = overlay_path(self.dir, namespace)
        dir_path = overlay_dir_path(self.dir, namespace)
        if file_path.exists() and dir_path.exists():
            raise SchemaError(f"Overlay '{namespace}' has both file and directory sources; resolve manually.")
        path = dir_path if dir_path.exists() else file_path
        if not path.exists():
            if mode == "error":
                raise SchemaError(f"Overlay '{namespace}' not found.")
            return {"removed": False, "namespace": namespace}

        with dataset_write_lock(self.dir):
            if mode == "archive":
                archive_dir = path.parent / "_archived"
                archive_dir.mkdir(parents=True, exist_ok=True)
                stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
                suffix = ".parquet" if path.is_file() else ""
                archived = archive_dir / f"{path.stem}-{stamp}{suffix}"
                path.replace(archived)
                self._record_event(
                    "archive_overlay",
                    args={"namespace": namespace, "archived": str(archived)},
                )
                return {"removed": True, "namespace": namespace, "archived_path": str(archived)}

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self._record_event(
                "remove_overlay",
                args={"namespace": namespace},
            )
            return {"removed": True, "namespace": namespace}

    def compact_overlay(self, namespace: str) -> Path:
        """
        Compact overlay parts into a single parquet file and archive old parts.
        """
        ctx = require_maintenance("compact_overlay")
        with dataset_write_lock(self.dir):
            self._auto_freeze_registry()
            file_path = overlay_path(self.dir, namespace)
            dir_path = overlay_dir_path(self.dir, namespace)
            if not dir_path.exists():
                raise SchemaError(f"Overlay parts not found for namespace '{namespace}'.")
            if file_path.exists():
                raise SchemaError(f"Overlay file already exists for namespace '{namespace}'. Remove it first.")
            parts = sorted(dir_path.glob("part-*.parquet"))
            if not parts:
                raise SchemaError(f"Overlay parts not found for namespace '{namespace}'.")

            meta = overlay_metadata(dir_path)
            key = meta.get("key")
            if not key:
                raise SchemaError(f"Overlay missing required metadata key: {dir_path}")
            schema = overlay_schema(dir_path)
            if namespace not in RESERVED_NAMESPACES:
                registry = self._registry(required=True)
                validate_overlay_schema(namespace, schema, registry=registry, key=key)

            try:
                import pyarrow.dataset as ds
            except Exception as e:
                raise SchemaError(f"Parquet dataset support is required for compact_overlay: {e}") from e

            dataset = ds.dataset([str(p) for p in parts], format="parquet")
            batches = dataset.to_batches(batch_size=65536)

            metadata = dict(schema.metadata or {})
            metadata[OVERLAY_META_CREATED.encode("utf-8")] = str(now_utc()).encode("utf-8")
            reg_hash = self._registry_hash(required=namespace not in RESERVED_NAMESPACES)
            if reg_hash:
                metadata[OVERLAY_META_REGISTRY_HASH.encode("utf-8")] = str(reg_hash).encode("utf-8")

            write_parquet_atomic_batches(
                batches,
                schema,
                file_path,
                snapshot_dir=None,
                metadata=metadata,
            )

            archive_dir = dir_path.parent / "_archived" / namespace
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
            archived = archive_dir / stamp
            shutil.move(str(dir_path), str(archived))

            self._record_event(
                "compact_overlay",
                args={"namespace": namespace, "archived": str(archived), "maintenance_reason": ctx.reason},
                maintenance={"reason": ctx.reason},
                target_path=file_path,
                actor=ctx.actor,
            )
            return file_path

    # ---- validation & utils ----

    def validate(self, strict: bool = False, *, registry_mode: str = "current") -> None:
        """
        Validate schema, ID uniqueness, alphabet constraints, and namespacing policy.
        In strict mode, warnings become errors for alphabet/namespacing issues.
        """
        self._require_exists()
        mode = normalize_registry_mode(registry_mode)
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

        overlays = list_overlays(self.dir)
        if overlays:
            validate_overlays_for_registry_mode(
                dataset=self,
                overlays=overlays,
                mode=mode,
                reserved_namespaces=RESERVED_NAMESPACES,
            )

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
    ) -> DedupeStats:
        return dedupe_dataset(
            dataset=self,
            key=key,
            keep=keep,
            batch_size=batch_size,
            dry_run=dry_run,
        )

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
        """Export current table to CSV, JSONL, or Parquet."""
        fmt_norm = str(fmt or "").strip().lower()
        if fmt_norm not in {"csv", "jsonl", "parquet"}:
            raise SequencesError("Unsupported export format. Use csv|jsonl|parquet.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if fmt_norm == "csv":
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
        elif fmt_norm == "jsonl":
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
            writer: pq.ParquetWriter | None = None
            try:
                for batch in self.scan(
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
                schema = self.schema()
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

    def _write_reserved_overlay(
        self,
        namespace: str,
        key: str,
        overlay_df: pd.DataFrame,
        *,
        validate_registry: bool = False,
        schema_types: Optional[Dict[str, pa.DataType]] = None,
    ) -> int:
        self._auto_freeze_registry()
        if not _NS_RE.match(namespace):
            raise NamespaceError(
                "Invalid namespace. Use lowercase letters, digits, and underscores, starting with a letter."
            )
        if overlay_df[key].duplicated().any():
            raise SchemaError(f"Overlay has duplicate keys for '{key}'.")
        out_path = overlay_path(self.dir, namespace)
        dir_path = overlay_dir_path(self.dir, namespace)
        if dir_path.exists():
            raise SchemaError(f"Overlay parts already exist for namespace '{namespace}'. Remove them before writing.")
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

        schema = None
        if schema_types:
            fields = []
            for col in overlay_df.columns:
                if col in schema_types:
                    fields.append(pa.field(col, schema_types[col]))
            if fields:
                schema = pa.schema(fields)
        tbl = pa.Table.from_pandas(overlay_df, preserve_index=False, schema=schema)
        if validate_registry:
            self._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
        reg_hash = self._registry_hash(required=False)
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
        return int(overlay_df.shape[0])

    def _ensure_ids_exist(self, ids: list[str]) -> None:
        dataset_ensure_ids_exist(self, ids)

    def tombstone(
        self,
        ids: Sequence[str],
        *,
        reason: Optional[str] = None,
        deleted_at: Optional[str] = None,
        allow_missing: bool = False,
    ) -> int:
        return dataset_tombstone(
            self,
            ids,
            reason=reason,
            deleted_at=deleted_at,
            allow_missing=allow_missing,
            tombstone_namespace=TOMBSTONE_NAMESPACE,
        )

    def restore(self, ids: Sequence[str], *, allow_missing: bool = False) -> int:
        return dataset_restore(
            self,
            ids,
            allow_missing=allow_missing,
            tombstone_namespace=TOMBSTONE_NAMESPACE,
        )

    def set_state(
        self,
        ids: Sequence[str],
        *,
        masked: Optional[bool] = None,
        qc_status: Optional[str] = None,
        split: Optional[str] = None,
        supersedes: Optional[str] = None,
        lineage: Optional[Sequence[str] | str] = None,
        allow_missing: bool = False,
    ) -> int:
        return dataset_set_state(
            self,
            ids,
            masked=masked,
            qc_status=qc_status,
            split=split,
            supersedes=supersedes,
            lineage=lineage,
            allow_missing=allow_missing,
            state_namespace=USR_STATE_NAMESPACE,
            state_schema_types=USR_STATE_SCHEMA_TYPES,
            state_qc_status_allowed=USR_STATE_QC_STATUS_ALLOWED,
            state_split_allowed=USR_STATE_SPLIT_ALLOWED,
        )

    def clear_state(self, ids: Sequence[str], *, allow_missing: bool = False) -> int:
        return dataset_clear_state(
            self,
            ids,
            allow_missing=allow_missing,
            state_namespace=USR_STATE_NAMESPACE,
            state_schema_types=USR_STATE_SCHEMA_TYPES,
        )

    def get_state(
        self,
        ids: Sequence[str],
        *,
        allow_missing: bool = False,
    ) -> pd.DataFrame:
        return dataset_get_state(
            self,
            ids,
            allow_missing=allow_missing,
            state_namespace=USR_STATE_NAMESPACE,
        )

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
    ) -> None:
        """Merge overlays into the base table (maintenance operation)."""
        materialize_dataset(
            dataset=self,
            namespaces=namespaces,
            keep_overlays=keep_overlays,
            archive_overlays=archive_overlays,
            drop_deleted=drop_deleted,
            reserved_namespaces=RESERVED_NAMESPACES,
        )

    def snapshot(self) -> None:
        """Write a timestamped snapshot and atomically persist current table."""
        with dataset_write_lock(self.dir):
            self._require_exists()
            self._require_registry_for_mutation("snapshot")
            snapshot_parquet_file(self.records_path, self.snapshot_dir)
            self._record_event(
                "snapshot",
                args={},
            )
