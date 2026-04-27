"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/lifecycle/registry.py

Dataset registry/lifecycle helpers extracted from the Dataset coordinator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import pyarrow.parquet as pq

from ...contracts import META_REGISTRY_HASH, SchemaError, SequencesError, merge_base_metadata
from ...maintenance import require_maintenance
from ...overlays import overlay_path
from ...registry import load_registry, registry_bytes, registry_hash
from ...storage.locking import dataset_write_lock
from ...storage.parquet import iter_parquet_batches, write_parquet_atomic_batches


class DatasetRegistryLifecycleHost(Protocol):
    root: Path
    dir: Path
    records_path: Path
    snapshot_dir: Path

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


def require_dataset_exists(*, records_path: Path) -> None:
    if not records_path.exists():
        raise SequencesError(f"Dataset not initialized: {records_path}")


def require_registry_for_mutation(*, root: Path, action: str) -> dict:
    try:
        return load_registry(root, required=True)
    except SchemaError as exc:
        raise SchemaError(f"Registry required for {action}. Create registry.yaml before mutating datasets.") from exc


def base_metadata(*, records_path: Path, root: Path, created_at: Optional[str] = None) -> dict[bytes, bytes]:
    metadata = None
    if records_path.exists():
        metadata = pq.ParquetFile(str(records_path)).schema_arrow.metadata
    reg_hash = registry_hash(root, required=True)
    return merge_base_metadata(metadata, created_at, reg_hash)


def tombstone_path(*, dataset_dir: Path, namespace: str) -> Path:
    return overlay_path(dataset_dir, namespace)


def load_dataset_registry(*, root: Path, required: bool) -> dict:
    return load_registry(root, required=required)


def dataset_registry_hash(*, root: Path, required: bool) -> Optional[str]:
    return registry_hash(root, required=required)


def stored_dataset_registry_hash(*, records_path: Path) -> str:
    parquet_file = pq.ParquetFile(str(records_path))
    metadata = parquet_file.schema_arrow.metadata or {}
    raw_hash = metadata.get(META_REGISTRY_HASH.encode("utf-8"))
    if not raw_hash:
        raise SchemaError("Dataset does not have a registry_hash; run `usr maintenance registry-freeze`.")
    return raw_hash.decode("utf-8")


def frozen_registry_path(*, dataset_dir: Path, records_path: Path) -> Path:
    reg_hash = stored_dataset_registry_hash(records_path=records_path)
    return dataset_dir / "_registry" / f"registry.{reg_hash}.yaml"


def auto_freeze_registry(
    dataset: DatasetRegistryLifecycleHost,
    *,
    record_auto_event: bool = True,
) -> tuple[Path, str, bool]:
    reg_hash = registry_hash(dataset.root, required=True)
    reg_bytes = registry_bytes(dataset.root)
    snap_dir = dataset.dir / "_registry"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"registry.{reg_hash}.yaml"
    created = False
    if not snap_path.exists():
        snap_path.write_bytes(reg_bytes)
        created = True

    parquet_file = pq.ParquetFile(str(dataset.records_path))
    metadata = parquet_file.schema_arrow.metadata or {}
    if metadata.get(META_REGISTRY_HASH.encode("utf-8")) != reg_hash.encode("utf-8"):

        def _iter_batches():
            yield from iter_parquet_batches(dataset.records_path)

        updated_metadata = merge_base_metadata(metadata, registry_hash=reg_hash)
        write_parquet_atomic_batches(
            _iter_batches(),
            parquet_file.schema_arrow,
            dataset.records_path,
            dataset.snapshot_dir,
            metadata=updated_metadata,
        )
        created = True

    if created and record_auto_event:
        dataset._record_event(
            "registry_freeze",
            args={"registry_hash": reg_hash, "snapshot": str(snap_path), "auto": True},
        )
    return snap_path, reg_hash, created


def freeze_registry(dataset: DatasetRegistryLifecycleHost) -> Path:
    context = require_maintenance("freeze_registry")
    require_dataset_exists(records_path=dataset.records_path)
    with dataset_write_lock(dataset.dir):
        snap_path, reg_hash, updated = auto_freeze_registry(dataset, record_auto_event=False)
        dataset._record_event(
            "registry_freeze",
            args={"registry_hash": reg_hash, "snapshot": str(snap_path), "auto": False, "updated": updated},
            maintenance={"reason": context.reason},
            actor=context.actor,
        )
        return snap_path
