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
_OVERLAY_HEAD_CACHE: dict[str, tuple[int, int, dict[str, Optional[str]], pa.Schema]] = {}
_OVERLAY_HEAD_CACHE_MAX = 20_000
_OVERLAY_PARTS_CACHE: dict[str, tuple[int, int, tuple[str, ...]]] = {}
_OVERLAY_PARTS_CACHE_MAX = 20_000
_OVERLAY_LIST_CACHE: dict[str, tuple[tuple[tuple[str, bool, int, int], ...], tuple[str, ...]]] = {}
_OVERLAY_LIST_CACHE_MAX = 4_000


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
    if not d.is_dir():
        return []

    entry_signatures: list[tuple[str, bool, int, int]] = []
    for entry in d.iterdir():
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        entry_signatures.append((entry.name, entry.is_dir(), int(stat.st_mtime_ns), int(stat.st_size)))
    signature = tuple(sorted(entry_signatures, key=lambda item: item[0]))

    cache_key = str(d.absolute())
    cached = _OVERLAY_LIST_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return [Path(path) for path in cached[1]]

    overlays: List[Path] = []
    for name, is_dir, _mtime_ns, _size in signature:
        entry = d / name
        if not is_dir and entry.suffix == ".parquet":
            overlays.append(entry)
            continue
        if is_dir and overlay_parts(entry):
            overlays.append(entry)
    overlays_sorted = sorted(overlays, key=lambda p: p.name)

    _OVERLAY_LIST_CACHE[cache_key] = (signature, tuple(str(path) for path in overlays_sorted))
    if len(_OVERLAY_LIST_CACHE) > _OVERLAY_LIST_CACHE_MAX:
        _OVERLAY_LIST_CACHE.clear()
    return overlays_sorted


def overlay_parts(path: Path) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        try:
            stat = p.stat()
        except FileNotFoundError:
            return []
        cache_key = str(p.absolute())
        stat_key = (int(stat.st_mtime_ns), int(stat.st_size))
        cached = _OVERLAY_PARTS_CACHE.get(cache_key)
        if cached is not None and cached[0] == stat_key[0] and cached[1] == stat_key[1]:
            return [Path(part_path) for part_path in cached[2]]
        parts = tuple(str(part) for part in sorted(p.glob(f"{OVERLAY_PART_PREFIX}*.parquet")))
        _OVERLAY_PARTS_CACHE[cache_key] = (stat_key[0], stat_key[1], parts)
        if len(_OVERLAY_PARTS_CACHE) > _OVERLAY_PARTS_CACHE_MAX:
            _OVERLAY_PARTS_CACHE.clear()
        return [Path(part_path) for part_path in parts]
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


def _overlay_head(path: Path) -> tuple[dict[str, Optional[str]], pa.Schema]:
    parts = overlay_parts(path)
    if not parts:
        raise FileNotFoundError(f"Overlay has no parquet parts: {path}")
    part = Path(parts[0]).absolute()
    try:
        stat = part.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Overlay has no parquet parts: {path}") from exc

    cache_key = str(part)
    cached = _OVERLAY_HEAD_CACHE.get(cache_key)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_meta, cached_schema = cached
        if cached_mtime_ns == int(stat.st_mtime_ns) and cached_size == int(stat.st_size):
            return dict(cached_meta), cached_schema

    pf = pq.ParquetFile(str(part))
    schema = pf.schema_arrow
    md = schema.metadata
    meta = {
        "namespace": _meta_get(md, OVERLAY_META_NAMESPACE),
        "key": _meta_get(md, OVERLAY_META_KEY),
        "created_at": _meta_get(md, OVERLAY_META_CREATED),
        "registry_hash": _meta_get(md, OVERLAY_META_REGISTRY_HASH),
    }
    _OVERLAY_HEAD_CACHE[cache_key] = (int(stat.st_mtime_ns), int(stat.st_size), dict(meta), schema)
    if len(_OVERLAY_HEAD_CACHE) > _OVERLAY_HEAD_CACHE_MAX:
        _OVERLAY_HEAD_CACHE.clear()
    return meta, schema


def overlay_metadata(path: Path) -> Dict[str, Optional[str]]:
    meta, _ = _overlay_head(path)
    return meta


def overlay_schema(path: Path) -> pa.Schema:
    _, schema = _overlay_head(path)
    return schema


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
