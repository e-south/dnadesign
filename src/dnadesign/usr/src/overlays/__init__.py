"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlays/__init__.py

Overlay file management and metadata helpers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import metadata as _metadata
from . import paths as _paths
from .constants import (
    DERIVED_DIR_NAME,
    OVERLAY_META_CREATED,
    OVERLAY_META_KEY,
    OVERLAY_META_NAMESPACE,
    OVERLAY_META_NAMESPACE_CONTRACT_HASH,
    OVERLAY_META_REGISTRY_HASH,
    OVERLAY_PART_PREFIX,
)

_OVERLAY_HEAD_CACHE: dict[
    str,
    tuple[
        tuple[tuple[str, int, int], ...],
        dict[str, str | None],
        pa.Schema,
    ],
] = {}
_OVERLAY_HEAD_CACHE_MAX = 20_000
_OVERLAY_PARTS_CACHE: dict[str, tuple[int, int, tuple[str, ...]]] = {}
_OVERLAY_PARTS_CACHE_MAX = 20_000
_OVERLAY_LIST_CACHE: dict[str, tuple[tuple[tuple[str, bool, int, int], ...], tuple[str, ...]]] = {}
_OVERLAY_LIST_CACHE_MAX = 4_000


def _is_temporary_overlay_entry(entry: Path) -> bool:
    return _paths.is_temporary_overlay_entry(entry)


def derived_dir(dataset_dir: Path) -> Path:
    return _paths.derived_dir(dataset_dir, derived_dir_name=DERIVED_DIR_NAME)


def overlay_path(dataset_dir: Path, namespace: str) -> Path:
    return _paths.overlay_path(dataset_dir, namespace, derived_dir_name=DERIVED_DIR_NAME)


def overlay_dir_path(dataset_dir: Path, namespace: str) -> Path:
    return _paths.overlay_dir_path(dataset_dir, namespace, derived_dir_name=DERIVED_DIR_NAME)


def list_overlays(dataset_dir: Path) -> list[Path]:
    return _paths.list_overlays(
        dataset_dir,
        derived_dir_name=DERIVED_DIR_NAME,
        overlay_parts=overlay_parts,
        list_cache=_OVERLAY_LIST_CACHE,
        list_cache_max=_OVERLAY_LIST_CACHE_MAX,
    )


def overlay_parts(path: Path) -> list[Path]:
    return _paths.overlay_parts(
        path,
        part_prefix=OVERLAY_PART_PREFIX,
        parts_cache=_OVERLAY_PARTS_CACHE,
        parts_cache_max=_OVERLAY_PARTS_CACHE_MAX,
    )


def _meta_get(md: dict[bytes, bytes] | None, key: str) -> str | None:
    return _metadata.meta_get(md, key)


def _overlay_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    return _metadata.overlay_signature(path, overlay_parts=overlay_parts)


def _overlay_head(path: Path) -> tuple[dict[str, str | None], pa.Schema]:
    return _metadata.overlay_head(
        path,
        overlay_parts=overlay_parts,
        head_cache=_OVERLAY_HEAD_CACHE,
        head_cache_max=_OVERLAY_HEAD_CACHE_MAX,
        namespace_key=OVERLAY_META_NAMESPACE,
        key_key=OVERLAY_META_KEY,
        created_key=OVERLAY_META_CREATED,
        registry_hash_key=OVERLAY_META_REGISTRY_HASH,
        namespace_contract_hash_key=OVERLAY_META_NAMESPACE_CONTRACT_HASH,
    )


def overlay_metadata(path: Path) -> dict[str, str | None]:
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
    namespace_contract_hash: str | None = None,
) -> pa.Table:
    return _metadata.with_overlay_metadata(
        table,
        namespace=namespace,
        key=key,
        created_at=created_at,
        registry_hash=registry_hash,
        namespace_contract_hash=namespace_contract_hash,
        namespace_key=OVERLAY_META_NAMESPACE,
        key_key=OVERLAY_META_KEY,
        created_key=OVERLAY_META_CREATED,
        registry_hash_key=OVERLAY_META_REGISTRY_HASH,
        namespace_contract_hash_key=OVERLAY_META_NAMESPACE_CONTRACT_HASH,
    )


__all__ = [
    "DERIVED_DIR_NAME",
    "OVERLAY_META_CREATED",
    "OVERLAY_META_KEY",
    "OVERLAY_META_NAMESPACE",
    "OVERLAY_META_NAMESPACE_CONTRACT_HASH",
    "OVERLAY_META_REGISTRY_HASH",
    "OVERLAY_PART_PREFIX",
    "_OVERLAY_HEAD_CACHE",
    "_OVERLAY_LIST_CACHE",
    "_OVERLAY_PARTS_CACHE",
    "_is_temporary_overlay_entry",
    "_meta_get",
    "_overlay_head",
    "_overlay_signature",
    "derived_dir",
    "list_overlays",
    "overlay_dir_path",
    "overlay_metadata",
    "overlay_parts",
    "overlay_path",
    "overlay_schema",
    "pq",
    "with_overlay_metadata",
]
