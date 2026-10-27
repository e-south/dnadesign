"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/overlay/write.py

Overlay write helpers for compact and parts-based overlay mutations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

from ...contracts import NamespaceError, SchemaError
from ...events import (
    EventAppendAttempt,
    EventAppendFailure,
    EventAppendState,
    prepare_event,
    validate_event_metadata,
)
from ...overlays import overlay_dir_path, overlay_path, with_overlay_metadata
from ...overlays.support.digest_ledger import overlay_digest_ledger_path, update_overlay_digest_ledger
from ...registry import namespace_contract_hash_for_entries
from ...runtime import connect_duckdb_utc
from ...storage.locking import dataset_write_lock
from ...storage.parquet import PARQUET_COMPRESSION, now_utc
from .attach import _attach_frame_dataset
from .policy import (
    SUPPORTED_OVERLAY_KEYS,
    coerce_null_overlay_columns_to_registry_schema,
    ensure_overlay_columns_allowed,
    validate_overlay_join_key,
    validate_overlay_target,
)

_WRITE_OVERLAY_PART_RESERVED_EVENT_KEYS = frozenset(
    {
        "namespace",
        "key",
        "columns",
        "rows_incoming",
        "rows_matched",
        "rows_written",
        "rows_missing",
        "allow_missing",
    }
)


def _entry_exists(path: Path) -> bool:
    """Return whether one directory entry exists, including broken symlinks."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _validate_write_overlay_part_event_args(event_args: Mapping[str, object] | None) -> None:
    if event_args is None:
        return
    for event_key in event_args:
        if event_key in _WRITE_OVERLAY_PART_RESERVED_EVENT_KEYS:
            raise SchemaError(f"write_overlay_part event_args cannot override reserved key '{event_key}'.")


def _merge_write_overlay_part_event_args(
    *,
    namespace: str,
    key: str,
    columns: list[str],
    rows_incoming: int,
    rows_written: int,
    rows_missing: int,
    allow_missing: bool,
    event_args: Mapping[str, object] | None,
) -> dict[str, object]:
    args: dict[str, object] = {
        "namespace": namespace,
        "key": key,
        "columns": list(columns),
        "rows_incoming": rows_incoming,
        "rows_matched": rows_written,
        "rows_written": rows_written,
        "rows_missing": rows_missing,
        "allow_missing": allow_missing,
    }
    if event_args is None:
        return args
    _validate_write_overlay_part_event_args(event_args)
    for event_key, event_value in event_args.items():
        args[event_key] = event_value
    return args


def write_overlay_dataset(
    *,
    dataset: Any,
    namespace: str,
    table_or_batches: Any,
    key: str = "id",
    overwrite: bool = False,
    allow_missing: bool = False,
    note: str = "",
    actor: Optional[dict] = None,
    namespace_pattern: Any,
    reserved_namespaces: set[str],
    write_lock=dataset_write_lock,
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
    validate_overlay_target(
        dataset=dataset,
        namespace=namespace,
        key=key,
        namespace_pattern=namespace_pattern,
        reserved_namespaces=reserved_namespaces,
    )
    return _attach_frame_dataset(
        dataset=dataset,
        incoming=tbl.to_pandas(),
        namespace=namespace,
        key=key,
        key_col=key,
        columns=attach_cols,
        allow_overwrite=overwrite,
        allow_missing=allow_missing,
        parse_json=False,
        note=note,
        actor=actor,
        reserved_namespaces=reserved_namespaces,
        write_lock=write_lock,
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
    event_args: Mapping[str, object] | None = None,
    create_only: bool = False,
    reserved_namespaces: set[str],
    write_lock=dataset_write_lock,
) -> int:
    """Write an overlay part, optionally requiring an absent namespace."""
    dataset._require_exists()
    if namespace in reserved_namespaces:
        raise NamespaceError(f"Namespace '{namespace}' is reserved.")
    key = validate_overlay_join_key(key or "", context_label="overlay key")
    key_col = str(key_col or key)

    file_path = overlay_path(dataset.dir, namespace)
    dir_path = overlay_dir_path(dataset.dir, namespace)
    if file_path.exists() and dir_path.exists():
        raise SchemaError(
            f"Overlay for namespace '{namespace}' has both file and directory sources. "
            "Resolve by compacting or removing one source."
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

    ensure_overlay_columns_allowed(attach_cols)

    tbl = coerce_null_overlay_columns_to_registry_schema(dataset=dataset, namespace=namespace, tbl=tbl, key=key)
    dataset._validate_registry_schema(namespace=namespace, schema=tbl.schema, key=key)
    _validate_write_overlay_part_event_args(event_args)
    validate_event_metadata(
        "write_overlay_part",
        args=_merge_write_overlay_part_event_args(
            namespace=namespace,
            key=key,
            columns=attach_cols,
            rows_incoming=0,
            rows_written=0,
            rows_missing=0,
            allow_missing=allow_missing,
            event_args=event_args,
        ),
        metrics={"rows_incoming": 0, "rows_written": 0, "rows_missing": 0},
        artifacts={"overlay": {"namespace": namespace, "key": key}},
        actor=actor,
    )

    def _write_part(
        *,
        output_dir: Path = dir_path,
        promote_existing_file: bool = True,
        verify_written: bool = False,
    ) -> tuple[int, int, int, Path] | None:
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

            if key in SUPPORTED_OVERLAY_KEYS - {"id"}:
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
            return None

        registry = dataset._registry(required=True)
        reg_hash = dataset._registry_hash(required=True)
        namespace_hash = namespace_contract_hash_for_entries(registry, namespace)
        tbl_out = with_overlay_metadata(
            tbl_out,
            namespace=namespace,
            key=key,
            created_at=now_utc(),
            registry_hash=reg_hash,
            namespace_contract_hash=namespace_hash,
        )

        output_dir = Path(output_dir)
        if promote_existing_file and file_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
            promoted_path = output_dir / f"part-{stamp}-{uuid.uuid4().hex}.parquet"
            os.replace(file_path, promoted_path)
            new_parts = [promoted_path]
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            new_parts = []
        stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
        part_path = output_dir / f"part-{stamp}-{uuid.uuid4().hex}.parquet"
        tmp_path = part_path.with_suffix(".parquet.tmp")
        try:
            pq.write_table(tbl_out, tmp_path, compression=PARQUET_COMPRESSION)
            os.replace(tmp_path, part_path)
            new_parts.append(part_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        if verify_written:
            written = pq.read_table(part_path)
            if not written.schema.equals(tbl_out.schema, check_metadata=True) or not written.equals(tbl_out):
                raise SchemaError(f"Staged overlay verification failed for namespace '{namespace}'.")
        ledger_path = overlay_digest_ledger_path(output_dir)
        if ledger_path is not None and ledger_path.is_file():
            update_overlay_digest_ledger(output_dir, new_parts=new_parts)

        return rows_written, rows_incoming, rows_missing, part_path

    def _write_event_fields(
        result: tuple[int, int, int, Path],
    ) -> tuple[int, dict[str, object], dict[str, int], dict[str, dict[str, str]]]:
        rows_written, rows_incoming, rows_missing, _ = result
        args = _merge_write_overlay_part_event_args(
            namespace=namespace,
            key=key,
            columns=attach_cols,
            rows_incoming=rows_incoming,
            rows_written=rows_written,
            rows_missing=rows_missing,
            allow_missing=allow_missing,
            event_args=event_args,
        )
        metrics = {
            "rows_incoming": rows_incoming,
            "rows_written": rows_written,
            "rows_missing": rows_missing,
        }
        artifacts = {"overlay": {"namespace": namespace, "key": key}}
        return rows_written, args, metrics, artifacts

    def _record_write(
        result: tuple[int, int, int, Path],
        *,
        target_path: Path,
    ) -> int:
        rows_written, args, metrics, artifacts = _write_event_fields(result)
        dataset._record_event(
            "write_overlay_part",
            args=args,
            metrics=metrics,
            artifacts=artifacts,
            target_path=target_path,
            actor=actor,
        )
        return rows_written

    def _rollback_publication(
        publication: CreateOnlyDirectoryPublication,
    ) -> bool:
        try:
            return publication.rollback()
        except BaseException as rollback_error:
            raise PublicationError(
                f"Overlay namespace '{namespace}' rollback after a failed publication boundary also failed."
            ) from rollback_error

    def _rollback_when_event_uncommitted(
        publication: CreateOnlyDirectoryPublication,
        failure: BaseException,
        *,
        require_published: bool,
    ) -> None:
        rolled_back = _rollback_publication(publication)
        if not rolled_back and (require_published or _entry_exists(dir_path)):
            raise PublicationError(
                f"Overlay namespace '{namespace}' could not be rolled back safely after an uncommitted event."
            ) from failure

    with write_lock(dataset.dir):
        if create_only and (_entry_exists(file_path) or _entry_exists(dir_path)):
            raise FileExistsError(f"Overlay namespace '{namespace}' already exists for {dataset.name}.")
        dataset._auto_freeze_registry()
        if not create_only:
            result = _write_part()
            if result is None:
                return 0
            return _record_write(result, target_path=result[3])

        try:
            publication = CreateOnlyDirectoryPublication.prepare(dir_path)
        except PublicationError as exc:
            if _entry_exists(file_path) or _entry_exists(dir_path):
                raise FileExistsError(f"Overlay namespace '{namespace}' already exists for {dataset.name}.") from exc
            raise
        with publication:
            result = _write_part(
                output_dir=publication.stage,
                promote_existing_file=False,
                verify_written=True,
            )
            if result is None:
                return 0
            if _entry_exists(file_path):
                raise FileExistsError(f"Overlay namespace '{namespace}' already exists for {dataset.name}.")
            staged_part = result[3]
            rows_written, args, metrics, artifacts = _write_event_fields(result)
            event_append = EventAppendAttempt(
                prepare_event(
                    "write_overlay_part",
                    dataset=dataset.name,
                    args=args,
                    metrics=metrics,
                    artifacts=artifacts,
                    target_path=staged_part,
                    dataset_root=dataset.root,
                    actor=actor,
                )
            )
            publication_completed = False
            try:
                publication.publish(required_manifest=staged_part.name)
                publication_completed = True
                event_append.append_to(dataset.events_path)
                return rows_written
            except PublicationError as exc:
                if publication_completed:
                    raise
                if _rollback_publication(publication):
                    raise
                if _entry_exists(file_path) or _entry_exists(dir_path):
                    raise FileExistsError(
                        f"Overlay namespace '{namespace}' already exists for {dataset.name}."
                    ) from exc
                raise
            except EventAppendFailure as event_error:
                if event_error.state is not EventAppendState.RESTORED:
                    raise
                _rollback_when_event_uncommitted(
                    publication,
                    event_error,
                    require_published=True,
                )
                raise
            except BaseException as boundary_error:
                if event_append.started:
                    raise
                _rollback_when_event_uncommitted(
                    publication,
                    boundary_error,
                    require_published=False,
                )
                raise
