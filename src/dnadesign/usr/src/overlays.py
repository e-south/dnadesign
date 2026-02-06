"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlays.py

Overlay file management and metadata helpers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

DERIVED_DIR_NAME = "_derived"
OVERLAY_META_NAMESPACE = "usr:overlay_namespace"
OVERLAY_META_KEY = "usr:overlay_key"
OVERLAY_META_CREATED = "usr:overlay_created_at"
OVERLAY_META_REGISTRY_HASH = "usr:registry_hash"
OVERLAY_PART_PREFIX = "part-"


def derived_dir(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / DERIVED_DIR_NAME


def overlay_path(dataset_dir: Path, namespace: str) -> Path:
    return derived_dir(dataset_dir) / f"{namespace}.parquet"


def overlay_dir_path(dataset_dir: Path, namespace: str) -> Path:
    return derived_dir(dataset_dir) / str(namespace)


def list_overlays(dataset_dir: Path) -> List[Path]:
    d = derived_dir(dataset_dir)
    if not d.exists():
        return []
    overlays: List[Path] = []
    for entry in d.iterdir():
        if entry.is_file() and entry.suffix == ".parquet":
            overlays.append(entry)
            continue
        if entry.is_dir():
            parts = sorted(entry.glob(f"{OVERLAY_PART_PREFIX}*.parquet"))
            if parts:
                overlays.append(entry)
    return sorted(overlays, key=lambda p: p.name)


def overlay_parts(path: Path) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob(f"{OVERLAY_PART_PREFIX}*.parquet"))
    if p.is_file():
        return [p]
    return []


def _meta_get(md: Optional[Dict[bytes, bytes]], key: str) -> Optional[str]:
    if not md:
        return None
    raw = md.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


def overlay_metadata(path: Path) -> Dict[str, Optional[str]]:
    parts = overlay_parts(path)
    if not parts:
        raise FileNotFoundError(f"Overlay has no parquet parts: {path}")
    pf = pq.ParquetFile(str(parts[0]))
    md = pf.schema_arrow.metadata
    return {
        "namespace": _meta_get(md, OVERLAY_META_NAMESPACE),
        "key": _meta_get(md, OVERLAY_META_KEY),
        "created_at": _meta_get(md, OVERLAY_META_CREATED),
        "registry_hash": _meta_get(md, OVERLAY_META_REGISTRY_HASH),
    }


def overlay_schema(path: Path) -> pa.Schema:
    parts = overlay_parts(path)
    if not parts:
        raise FileNotFoundError(f"Overlay has no parquet parts: {path}")
    pf = pq.ParquetFile(str(parts[0]))
    return pf.schema_arrow


def with_overlay_metadata(
    table: pa.Table,
    *,
    namespace: str,
    key: str,
    created_at: str,
    registry_hash: str | None = None,
) -> pa.Table:
    md = dict(table.schema.metadata or {})
    md[OVERLAY_META_NAMESPACE.encode("utf-8")] = str(namespace).encode("utf-8")
    md[OVERLAY_META_KEY.encode("utf-8")] = str(key).encode("utf-8")
    md[OVERLAY_META_CREATED.encode("utf-8")] = str(created_at).encode("utf-8")
    if registry_hash:
        md[OVERLAY_META_REGISTRY_HASH.encode("utf-8")] = str(registry_hash).encode("utf-8")
    return table.replace_schema_metadata(md)
