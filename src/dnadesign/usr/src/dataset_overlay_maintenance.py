"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_overlay_maintenance.py

Overlay inventory and maintenance operations for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq

from .errors import SchemaError
from .maintenance import require_maintenance
from .overlays import (
    OVERLAY_META_CREATED,
    OVERLAY_META_REGISTRY_HASH,
    list_overlays,
    overlay_dir_path,
    overlay_metadata,
    overlay_path,
    overlay_schema,
)
from .registry import validate_overlay_schema
from .storage.locking import dataset_write_lock
from .storage.parquet import now_utc, write_parquet_atomic_batches
from .types import Fingerprint, OverlayInfo


class DatasetOverlayMaintenanceHost(Protocol):
    dir: Path

    def _require_exists(self) -> None: ...

    def _require_registry_for_mutation(self, action: str) -> dict: ...

    def _auto_freeze_registry(self, *, record_auto_event: bool = True) -> tuple[Path, str, bool]: ...

    def _registry(self, *, required: bool) -> dict: ...

    def _registry_hash(self, *, required: bool) -> str | None: ...

    def _record_event(
        self,
        action: str,
        *,
        args: dict | None = None,
        metrics: dict | None = None,
        artifacts: dict | None = None,
        maintenance: dict | None = None,
        target_path: Path | None = None,
        registry_hash: str | None = None,
        actor: dict | None = None,
    ) -> None: ...


def list_overlay_infos(dataset: DatasetOverlayMaintenanceHost) -> list[OverlayInfo]:
    dataset._require_exists()
    overlays: list[OverlayInfo] = []
    for path in list_overlays(dataset.dir):
        meta = overlay_metadata(path)
        parts = sorted(path.glob("part-*.parquet")) if path.is_dir() else [path]
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


def remove_overlay_namespace(
    dataset: DatasetOverlayMaintenanceHost,
    namespace: str,
    *,
    mode: str = "error",
) -> dict:
    if mode not in {"error", "delete", "archive"}:
        raise SchemaError(f"Unsupported remove_overlay mode '{mode}'.")
    dataset._require_registry_for_mutation("remove_overlay")
    file_path = overlay_path(dataset.dir, namespace)
    dir_path = overlay_dir_path(dataset.dir, namespace)
    if file_path.exists() and dir_path.exists():
        raise SchemaError(f"Overlay '{namespace}' has both file and directory sources; resolve manually.")
    path = dir_path if dir_path.exists() else file_path
    if not path.exists():
        if mode == "error":
            raise SchemaError(f"Overlay '{namespace}' not found.")
        return {"removed": False, "namespace": namespace}

    with dataset_write_lock(dataset.dir):
        if mode == "archive":
            archive_dir = path.parent / "_archived"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = now_utc().replace(":", "").replace("-", "").replace(".", "")
            suffix = ".parquet" if path.is_file() else ""
            archived = archive_dir / f"{path.stem}-{stamp}{suffix}"
            path.replace(archived)
            dataset._record_event(
                "archive_overlay",
                args={"namespace": namespace, "archived": str(archived)},
            )
            return {"removed": True, "namespace": namespace, "archived_path": str(archived)}

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        dataset._record_event(
            "remove_overlay",
            args={"namespace": namespace},
        )
        return {"removed": True, "namespace": namespace}


def compact_overlay_namespace(
    dataset: DatasetOverlayMaintenanceHost,
    namespace: str,
    *,
    reserved_namespaces: set[str],
) -> Path:
    ctx = require_maintenance("compact_overlay")
    with dataset_write_lock(dataset.dir):
        dataset._auto_freeze_registry()
        file_path = overlay_path(dataset.dir, namespace)
        dir_path = overlay_dir_path(dataset.dir, namespace)
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
        if namespace not in reserved_namespaces:
            registry = dataset._registry(required=True)
            validate_overlay_schema(namespace, schema, registry=registry, key=key)

        try:
            import pyarrow.dataset as ds
        except Exception as e:
            raise SchemaError(f"Parquet dataset support is required for compact_overlay: {e}") from e

        compact_dataset = ds.dataset([str(p) for p in parts], format="parquet")
        batches = compact_dataset.to_batches(batch_size=65536)

        metadata = dict(schema.metadata or {})
        metadata[OVERLAY_META_CREATED.encode("utf-8")] = str(now_utc()).encode("utf-8")
        reg_hash = dataset._registry_hash(required=namespace not in reserved_namespaces)
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

        dataset._record_event(
            "compact_overlay",
            args={"namespace": namespace, "archived": str(archived), "maintenance_reason": ctx.reason},
            maintenance={"reason": ctx.reason},
            target_path=file_path,
            actor=ctx.actor,
        )
        return file_path
