"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/overlays/metadata.py

Overlay metadata and schema inspection helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def meta_get(md: dict[bytes, bytes] | None, key: str) -> str | None:
    if not md:
        return None
    raw = md.get(key.encode("utf-8"))
    if raw is None:
        return None
    return raw.decode("utf-8")


def overlay_signature(
    path: Path,
    *,
    overlay_parts,
) -> tuple[tuple[str, int, int], ...]:
    parts = overlay_parts(path)
    if not parts:
        raise FileNotFoundError(f"Overlay has no parquet parts: {path}")
    signature_rows: list[tuple[str, int, int]] = []
    for part in parts:
        resolved = Path(part).absolute()
        try:
            stat = resolved.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Overlay has no parquet parts: {path}") from exc
        signature_rows.append((str(resolved), int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature_rows)


def overlay_head(
    path: Path,
    *,
    overlay_parts,
    head_cache: dict[str, tuple[tuple[tuple[str, int, int], ...], dict[str, str | None], pa.Schema]],
    head_cache_max: int,
    namespace_key: str,
    key_key: str,
    created_key: str,
    registry_hash_key: str,
    namespace_contract_hash_key: str,
) -> tuple[dict[str, str | None], pa.Schema]:
    signature = overlay_signature(path, overlay_parts=overlay_parts)
    cache_key = str(Path(path).absolute())
    cached = head_cache.get(cache_key)
    if cached is not None:
        cached_signature, cached_meta, cached_schema = cached
        if cached_signature == signature:
            return dict(cached_meta), cached_schema

    schema_parts: list[pa.Schema] = []
    meta: dict[str, str | None] | None = None
    schema_metadata: dict[bytes, bytes] | None = None
    for part_path, _mtime_ns, _size in signature:
        parquet_file = pq.ParquetFile(part_path)
        part_schema = parquet_file.schema_arrow
        schema_parts.append(part_schema)
        if meta is None:
            md = part_schema.metadata
            meta = {
                "namespace": meta_get(md, namespace_key),
                "key": meta_get(md, key_key),
                "created_at": meta_get(md, created_key),
                "registry_hash": meta_get(md, registry_hash_key),
                "namespace_contract_hash": meta_get(md, namespace_contract_hash_key),
            }
            schema_metadata = dict(md or {})
    schema = pa.unify_schemas(schema_parts, promote_options="permissive") if len(schema_parts) > 1 else schema_parts[0]
    if schema_metadata:
        schema = schema.with_metadata(schema_metadata)
    resolved_meta = meta or {
        "namespace": None,
        "key": None,
        "created_at": None,
        "registry_hash": None,
        "namespace_contract_hash": None,
    }
    head_cache[cache_key] = (signature, dict(resolved_meta), schema)
    if len(head_cache) > head_cache_max:
        head_cache.clear()
    return dict(resolved_meta), schema


def with_overlay_metadata(
    table: pa.Table,
    *,
    namespace: str,
    key: str,
    created_at: str,
    registry_hash: str | None = None,
    namespace_contract_hash: str | None = None,
    namespace_key: str,
    key_key: str,
    created_key: str,
    registry_hash_key: str,
    namespace_contract_hash_key: str,
) -> pa.Table:
    md = dict(table.schema.metadata or {})
    md[namespace_key.encode("utf-8")] = str(namespace).encode("utf-8")
    md[key_key.encode("utf-8")] = str(key).encode("utf-8")
    md[created_key.encode("utf-8")] = str(created_at).encode("utf-8")
    if registry_hash:
        md[registry_hash_key.encode("utf-8")] = str(registry_hash).encode("utf-8")
    if namespace_contract_hash:
        md[namespace_contract_hash_key.encode("utf-8")] = str(namespace_contract_hash).encode("utf-8")
    return table.replace_schema_metadata(md)
