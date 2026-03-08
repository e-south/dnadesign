"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_overlay_catalog.py

Overlay catalog loading and overlay-aware dataset schema metadata helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import NamespaceError, SchemaError
from .overlays import list_overlays, overlay_metadata, overlay_schema
from .registry import validate_overlay_schema
from .types import DatasetInfo

_LOAD_OVERLAYS_CACHE: dict[
    tuple[str, bool, tuple[str, ...] | None],
    tuple[tuple[tuple[str, int, int], ...], tuple[int, int] | None, tuple[dict, ...]],
] = {}
_LOAD_OVERLAYS_CACHE_MAX = 4_096


class DatasetOverlayCatalogHost(Protocol):
    dir: Path
    name: str
    records_path: Path

    def _require_exists(self) -> None: ...

    def _registry(self, *, required: bool) -> dict: ...


def load_overlay_catalog(
    dataset: DatasetOverlayCatalogHost,
    *,
    include_tombstone: bool = True,
    namespaces: Sequence[str] | None = None,
    reserved_namespaces: set[str] | frozenset[str],
) -> list[dict]:
    dataset._require_exists()
    overlays = []
    paths = list_overlays(dataset.dir)
    path_entries = []
    path_sig_rows: list[tuple[str, int, int]] = []
    for path in paths:
        path_stat = path.stat()
        meta = overlay_metadata(path)
        namespace = meta.get("namespace") or path.stem
        path_entries.append((path, meta, namespace))
        path_sig_rows.append((str(path), int(path_stat.st_mtime_ns), int(path_stat.st_size)))
    namespace_filter = set(namespaces) if namespaces else None
    require_registry = any(namespace not in reserved_namespaces for _, _, namespace in path_entries)
    namespace_key = tuple(sorted(namespace_filter)) if namespace_filter else None
    cache_key = (str(dataset.dir), bool(include_tombstone), namespace_key)
    path_sig = tuple(path_sig_rows)
    registry_sig: tuple[int, int] | None = None
    if require_registry:
        reg_path = dataset.dir / "_registry" / "registry.yaml"
        if reg_path.exists():
            reg_stat = reg_path.stat()
            registry_sig = (int(reg_stat.st_mtime_ns), int(reg_stat.st_size))
    cached = _LOAD_OVERLAYS_CACHE.get(cache_key)
    if cached is not None and cached[0] == path_sig and cached[1] == registry_sig:
        return [dict(overlay) for overlay in cached[2]]

    registry = dataset._registry(required=require_registry) if require_registry else {}
    seen: dict[str, Path] = {}
    for path, meta, namespace in path_entries:
        key = meta.get("key")
        if not key:
            raise SchemaError(f"Overlay missing required metadata key: {path}")
        if namespace in seen:
            raise SchemaError(
                f"Overlay namespace '{namespace}' has multiple sources: {seen[namespace]} and {path}. "
                "Resolve by compacting or removing one source."
            )
        seen[namespace] = path
        if not include_tombstone and namespace in reserved_namespaces:
            continue
        if namespace_filter and namespace not in namespace_filter:
            continue
        schema = overlay_schema(path)
        if namespace not in reserved_namespaces:
            validate_overlay_schema(namespace, schema, registry=registry, key=key)
        if key not in schema.names:
            raise SchemaError(f"Overlay missing key column '{key}': {path}")
        overlay_cols = [column for column in schema.names if column != key]
        read_path = str(path / "part-*.parquet") if path.is_dir() else str(path)
        overlays.append(
            {
                "namespace": namespace,
                "key": key,
                "cols": overlay_cols,
                "schema": schema,
                "path": path,
                "read_path": read_path,
            }
        )
    _LOAD_OVERLAYS_CACHE[cache_key] = (path_sig, registry_sig, tuple(dict(overlay) for overlay in overlays))
    if len(_LOAD_OVERLAYS_CACHE) > _LOAD_OVERLAYS_CACHE_MAX:
        _LOAD_OVERLAYS_CACHE.clear()
    return overlays


def build_dataset_info(
    dataset: DatasetOverlayCatalogHost,
    *,
    required_columns: Sequence[tuple[str, object]],
    reserved_namespaces: set[str] | frozenset[str],
) -> DatasetInfo:
    dataset._require_exists()
    parquet_file = pq.ParquetFile(str(dataset.records_path))
    columns = list(parquet_file.schema_arrow.names)
    derived_columns = []
    overlay_data = load_overlay_catalog(dataset, reserved_namespaces=reserved_namespaces)
    for overlay in overlay_data:
        derived_columns.extend(overlay["cols"])
    all_columns = list(columns)
    for column in derived_columns:
        if column not in all_columns:
            all_columns.append(column)
    required_names = {name for name, _ in required_columns}
    namespaces = sorted({column.split("__", 1)[0] for column in all_columns if column not in required_names and "__" in column})
    return DatasetInfo(
        name=dataset.name,
        path=str(dataset.records_path),
        rows=int(parquet_file.metadata.num_rows),
        columns=all_columns,
        namespaces=namespaces,
    )


def merge_dataset_schema(
    dataset: DatasetOverlayCatalogHost,
    *,
    reserved_namespaces: set[str] | frozenset[str],
) -> pa.Schema:
    dataset._require_exists()
    base_schema = pq.ParquetFile(str(dataset.records_path)).schema_arrow
    for overlay in load_overlay_catalog(dataset, reserved_namespaces=reserved_namespaces):
        for field in overlay["schema"]:
            if field.name == overlay["key"]:
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
