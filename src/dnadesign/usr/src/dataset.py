"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset.py

USR dataset lifecycle operations and validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset_activity import append_meta_note as append_dataset_meta_note
from .dataset_activity import record_dataset_activity_event
from .dataset_dedupe import dedupe_dataset
from .dataset_ingest import (
    add_sequences_dataset,
    import_csv_dataset,
    import_jsonl_dataset,
    import_rows_dataset,
    prepare_import_rows_dataset,
    write_import_df_dataset,
)
from .dataset_materialize import materialize_dataset
from .dataset_overlay_maintenance import (
    compact_overlay_namespace,
    list_overlay_infos,
    remove_overlay_namespace,
)
from .dataset_overlay_ops import (
    attach_columns_dataset,
    attach_dataset,
    write_overlay_dataset,
    write_overlay_part_dataset,
)
from .dataset_overlay_query import build_overlay_query
from .dataset_query import create_overlay_view, sql_ident, sql_str
from .dataset_reporting import describe_dataset, manifest_dataset, manifest_dict_dataset
from .dataset_reserved_overlay import write_reserved_overlay
from .dataset_state_facade import (
    clear_dataset_state_fields,
    ensure_dataset_ids_exist,
    get_dataset_state_frame,
    restore_dataset_rows,
    set_dataset_state_fields,
    tombstone_dataset_rows,
)
from .dataset_validate import validate_dataset
from .dataset_views import export_dataset, get_dataset, grep_dataset, head_dataset, scan_dataset
from .errors import (
    NamespaceError,
    SchemaError,
    SequencesError,
)
from .maintenance import maintenance as maintenance_context
from .maintenance import require_maintenance
from .overlays import (
    list_overlays,
    overlay_metadata,
    overlay_path,
    overlay_schema,
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
    iter_parquet_batches,
    now_utc,
    snapshot_parquet_file,
    write_parquet_atomic,
    write_parquet_atomic_batches,
)
from .types import AddSequencesResult, DatasetInfo, Manifest, OverlayInfo

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
        return scan_dataset(
            self,
            columns=columns,
            include_overlays=include_overlays,
            include_deleted=include_deleted,
            batch_size=batch_size,
        )

    def head(
        self,
        n: int = 10,
        columns: Optional[List[str]] = None,
        *,
        include_derived: bool = True,
        include_deleted: bool = False,
    ):
        return head_dataset(
            self,
            n=n,
            columns=columns,
            include_derived=include_derived,
            include_deleted=include_deleted,
        )

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
        return build_overlay_query(
            self,
            columns=columns,
            include_overlays=include_overlays,
            include_deleted=include_deleted,
            where=where,
            params=params,
            limit=limit,
            required_columns=REQUIRED_COLUMNS,
            tombstone_columns=TOMBSTONE_COLUMNS,
            tombstone_namespace=TOMBSTONE_NAMESPACE,
        )

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
        return prepare_import_rows_dataset(
            self,
            rows,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source,
            strict_id_check=strict_id_check,
            created_at_override=created_at_override,
        )

    def _write_import_df(
        self,
        out_df: pd.DataFrame,
        *,
        source: Optional[str],
        on_conflict: str,
        actor: Optional[dict] = None,
        return_ids: bool = False,
    ) -> int | tuple[int, list[str], list[str]]:
        return write_import_df_dataset(
            self,
            out_df,
            source=source,
            on_conflict=on_conflict,
            actor=actor,
            return_ids=return_ids,
            write_lock=dataset_write_lock,
        )

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
        return import_rows_dataset(
            self,
            rows,
            default_bio_type=default_bio_type,
            default_alphabet=default_alphabet,
            source=source,
            strict_id_check=strict_id_check,
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
        return add_sequences_dataset(
            self,
            rows_or_sequences,
            bio_type=bio_type,
            alphabet=alphabet,
            source=source,
            created_at=created_at,
            on_conflict=on_conflict,
            actor=actor,
        )

    # Legacy file import entry points now route to import_rows (no special logic)
    def import_csv(
        self,
        csv_path: Path,
        default_bio_type="dna",
        default_alphabet="dna_4",
        source: Optional[str] = None,
    ) -> int:
        return import_csv_dataset(
            self,
            csv_path,
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
        return import_jsonl_dataset(
            self,
            jsonl_path,
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
        return list_overlay_infos(self)

    def remove_overlay(self, namespace: str, *, mode: str = "error") -> dict:
        return remove_overlay_namespace(self, namespace, mode=mode)

    def compact_overlay(self, namespace: str) -> Path:
        return compact_overlay_namespace(self, namespace, reserved_namespaces=RESERVED_NAMESPACES)

    # ---- validation & utils ----

    def validate(self, strict: bool = False, *, registry_mode: str = "current") -> None:
        """
        Validate schema, ID uniqueness, alphabet constraints, and namespacing policy.
        In strict mode, warnings become errors for alphabet/namespacing issues.
        """
        validate_dataset(
            self,
            strict=strict,
            registry_mode=registry_mode,
            required_columns=REQUIRED_COLUMNS,
            reserved_namespaces=RESERVED_NAMESPACES,
        )

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
        return get_dataset(self, record_id, columns=columns, include_deleted=include_deleted)

    def grep(self, pattern: str, limit: int = 20, batch_size: int = 65536, *, include_deleted: bool = False):
        return grep_dataset(
            self,
            pattern=pattern,
            limit=limit,
            batch_size=batch_size,
            include_deleted=include_deleted,
        )

    def export(
        self,
        fmt: str,
        out_path: Path,
        columns: Optional[List[str]] = None,
        *,
        include_deleted: bool = False,
    ) -> None:
        export_dataset(
            self,
            fmt=fmt,
            out_path=out_path,
            columns=columns,
            include_deleted=include_deleted,
        )

    def _write_reserved_overlay(
        self,
        namespace: str,
        key: str,
        overlay_df: pd.DataFrame,
        *,
        validate_registry: bool = False,
        schema_types: Optional[Dict[str, pa.DataType]] = None,
    ) -> int:
        return write_reserved_overlay(
            self,
            namespace,
            key,
            overlay_df,
            validate_registry=validate_registry,
            schema_types=schema_types,
            namespace_pattern=_NS_RE,
        )

    def _ensure_ids_exist(self, ids: list[str]) -> None:
        ensure_dataset_ids_exist(self, ids)

    def tombstone(
        self,
        ids: Sequence[str],
        *,
        reason: Optional[str] = None,
        deleted_at: Optional[str] = None,
        allow_missing: bool = False,
    ) -> int:
        return tombstone_dataset_rows(
            self,
            ids,
            reason=reason,
            deleted_at=deleted_at,
            allow_missing=allow_missing,
            tombstone_namespace=TOMBSTONE_NAMESPACE,
        )

    def restore(self, ids: Sequence[str], *, allow_missing: bool = False) -> int:
        return restore_dataset_rows(
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
        return set_dataset_state_fields(
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
        return clear_dataset_state_fields(
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
        return get_dataset_state_frame(
            self,
            ids,
            allow_missing=allow_missing,
            state_namespace=USR_STATE_NAMESPACE,
        )

    def manifest(self, *, include_events: bool = False) -> Manifest:
        return manifest_dataset(self, include_events=include_events)

    def manifest_dict(self, *, include_events: bool = False) -> dict:
        return manifest_dict_dataset(self, include_events=include_events)

    def describe(
        self,
        opts,
        *,
        columns: Optional[List[str]] = None,
        sample: int = 1024,
        batch_size: int = 65536,
        include_deleted: bool = False,
    ) -> List[dict]:
        return describe_dataset(
            self,
            opts,
            columns=columns,
            sample=sample,
            batch_size=batch_size,
            include_deleted=include_deleted,
            tombstone_columns=TOMBSTONE_COLUMNS,
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
