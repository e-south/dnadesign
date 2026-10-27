"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/query/__init__.py

Overlay query helpers used by Dataset DuckDB joins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ...contracts import SchemaError
from ...overlays import overlay_metadata, overlay_parts, overlay_schema
from .catalog import build_dataset_info, load_overlay_catalog, merge_dataset_schema
from .planner import build_overlay_query

__all__ = [
    "build_dataset_info",
    "build_overlay_query",
    "create_overlay_view",
    "load_overlay_catalog",
    "merge_dataset_schema",
    "sql_ident",
    "sql_str",
]


_OVERLAY_PART_CREATED_AT_CACHE: dict[str, tuple[int, int, str]] = {}
_OVERLAY_PART_CREATED_AT_CACHE_MAX = 20_000
_OVERLAY_KEY_UNIQUENESS_CACHE: dict[tuple[str, str], tuple[int, int]] = {}
_OVERLAY_KEY_UNIQUENESS_CACHE_MAX = 20_000
_OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE: dict[tuple[str, str], tuple[tuple[str, int, int], ...]] = {}
_OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE_MAX = 4_000


def sql_ident(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def sql_str(value: str) -> str:
    return str(value).replace("'", "''")


def _read_parquet_with_filename_sql(paths: tuple[Path, ...]) -> str:
    if len(paths) == 1:
        return f"read_parquet('{sql_str(str(paths[0]))}', filename=true)"
    path_list = ", ".join(f"'{sql_str(str(path))}'" for path in paths)
    return f"read_parquet([{path_list}], filename=true, union_by_name=true)"


def _sql_values_rows(rows: tuple[tuple[str, str, str], ...]) -> str:
    return ", ".join(
        f"('{sql_str(file_path)}', '{sql_str(created_at)}', '{sql_str(file_name)}')"
        for file_path, created_at, file_name in rows
    )


def _overlay_part_stat_key(part: Path) -> tuple[int, int]:
    resolved = Path(part)
    try:
        stat = resolved.stat()
    except FileNotFoundError as exc:
        raise SchemaError(f"Overlay has no parquet parts: {part}") from exc
    return int(stat.st_mtime_ns), int(stat.st_size)


def _overlay_part_created_at(
    part: Path,
    *,
    stat_key: tuple[int, int] | None = None,
) -> str:
    resolved = Path(part)
    if stat_key is None:
        stat_key = _overlay_part_stat_key(resolved)
    cache_key = str(resolved)
    cached = _OVERLAY_PART_CREATED_AT_CACHE.get(cache_key)
    if cached is not None:
        cached_mtime_ns, cached_size, cached_created_at = cached
        if cached_mtime_ns == stat_key[0] and cached_size == stat_key[1]:
            return cached_created_at
    meta = overlay_metadata(resolved)
    created = meta.get("created_at")
    if not created:
        raise SchemaError(f"Overlay part missing required created_at metadata: {resolved}")
    _OVERLAY_PART_CREATED_AT_CACHE[cache_key] = (stat_key[0], stat_key[1], created)
    if len(_OVERLAY_PART_CREATED_AT_CACHE) > _OVERLAY_PART_CREATED_AT_CACHE_MAX:
        _OVERLAY_PART_CREATED_AT_CACHE.clear()
    return created


def _assert_single_overlay_key_unique(
    con: Any,
    *,
    part: Path,
    key_q: str,
    key: str,
    stat_key: tuple[int, int] | None = None,
) -> None:
    resolved = Path(part)
    if stat_key is None:
        stat_key = _overlay_part_stat_key(resolved)
    cache_key = (str(resolved), key)
    cached = _OVERLAY_KEY_UNIQUENESS_CACHE.get(cache_key)
    if cached == stat_key:
        return
    dup_exists = con.execute(
        f"SELECT 1 FROM read_parquet('{sql_str(str(resolved))}') GROUP BY {key_q} HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if dup_exists is not None:
        raise SchemaError(f"Overlay part has duplicate keys for '{key}': {resolved}")
    _OVERLAY_KEY_UNIQUENESS_CACHE[cache_key] = stat_key
    if len(_OVERLAY_KEY_UNIQUENESS_CACHE) > _OVERLAY_KEY_UNIQUENESS_CACHE_MAX:
        _OVERLAY_KEY_UNIQUENESS_CACHE.clear()


def create_overlay_view(
    con: Any,
    *,
    view_name: str,
    path: Path,
    key: str,
    columns: Sequence[str] | None = None,
) -> str:
    key_q = sql_ident(key)
    raw_parts = overlay_parts(path)
    parts = tuple(Path(part).absolute() for part in raw_parts)
    if not parts:
        raise SchemaError(f"Overlay has no parquet parts: {path}")
    schema_names = list(overlay_schema(path).names)
    requested_columns = list(dict.fromkeys(columns or schema_names))
    if key not in requested_columns:
        requested_columns.insert(0, key)
    missing_columns = [name for name in requested_columns if name not in schema_names]
    if missing_columns:
        raise SchemaError(f"Overlay is missing requested columns: {', '.join(missing_columns)}")
    data_columns = [name for name in requested_columns if name != key]
    projection_sql = ", ".join(sql_ident(name) for name in requested_columns)

    if len(parts) == 1:
        part = parts[0]
        part_stat_key = _overlay_part_stat_key(part)
        _overlay_part_created_at(part, stat_key=part_stat_key)
        _assert_single_overlay_key_unique(con, part=part, key_q=key_q, key=key, stat_key=part_stat_key)
        return f"(SELECT {projection_sql} FROM read_parquet('{sql_str(str(part))}'))"

    part_rows: list[tuple[str, str, str]] = []
    part_signature_rows: list[tuple[str, int, int]] = []
    for part in parts:
        stat_key = _overlay_part_stat_key(part)
        created = _overlay_part_created_at(part, stat_key=stat_key)
        part_rows.append((str(part), created, part.name))
        part_signature_rows.append((str(part), stat_key[0], stat_key[1]))

    part_set_cache_key = (str(Path(path).absolute()), key)
    part_signature = tuple(part_signature_rows)
    cached_part_signature = _OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE.get(part_set_cache_key)

    source_sql = _read_parquet_with_filename_sql(parts)
    if cached_part_signature != part_signature:
        dup_row = con.execute(
            f"SELECT filename FROM {source_sql} GROUP BY filename, {key_q} HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if dup_row is not None:
            raise SchemaError(f"Overlay part has duplicate keys for '{key}': {Path(str(dup_row[0]))}")
        _OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE[part_set_cache_key] = part_signature
        if len(_OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE) > _OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE_MAX:
            _OVERLAY_PART_SET_KEY_UNIQUENESS_CACHE.clear()

    metadata_view = f"{view_name}_meta"
    staging_view = f"{view_name}_staging"
    con.execute(
        f"CREATE TEMP TABLE {metadata_view} ("
        "__usr_overlay_file_path VARCHAR, "
        "__usr_overlay_created_at VARCHAR, "
        "__usr_overlay_filename VARCHAR)"
    )
    for offset in range(0, len(part_rows), 256):
        batch = tuple(part_rows[offset : offset + 256])
        con.execute(f"INSERT INTO {metadata_view} VALUES {_sql_values_rows(batch)}")
    # Keep the wide overlay payload lazy here. Materializing a temporary staging
    # table doubles peak memory for large vector columns before the final
    # arg_max/group-by collapse chooses the newest row per key.
    con.execute(
        f"CREATE TEMP VIEW {staging_view} AS "
        f"SELECT {', '.join(f'p.{sql_ident(name)}' for name in requested_columns)}, "
        "m.__usr_overlay_created_at, "
        "m.__usr_overlay_filename "
        f"FROM {source_sql} p "
        f"JOIN {metadata_view} m ON p.filename = m.__usr_overlay_file_path"
    )

    con.execute(
        f"CREATE TEMP VIEW {view_name} AS "
        "WITH ranked AS ("
        "SELECT *, "
        "concat(__usr_overlay_created_at, '\x1f', __usr_overlay_filename) AS __usr_overlay_sort_key "
        f"FROM {staging_view}"
        ") "
        "SELECT "
        + ", ".join(
            [f"{key_q}"]
            + [
                f"arg_max({sql_ident(column_name)}, __usr_overlay_sort_key) AS {sql_ident(column_name)}"
                for column_name in data_columns
            ]
        )
        + " "
        "FROM ranked "
        f"GROUP BY {key_q}"
    )
    return view_name
