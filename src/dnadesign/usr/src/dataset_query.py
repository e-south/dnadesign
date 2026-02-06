"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_query.py

Overlay query helpers used by Dataset duckdb joins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import SchemaError
from .overlays import overlay_metadata


def sql_ident(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def sql_str(value: str) -> str:
    return str(value).replace("'", "''")


def create_overlay_view(
    con: Any,
    *,
    view_name: str,
    path: Path,
    key: str,
) -> None:
    key_q = sql_ident(key)
    if path.is_dir():
        parts = sorted(path.glob("part-*.parquet"))
    else:
        parts = [path]
    if not parts:
        raise SchemaError(f"Overlay has no parquet parts: {path}")

    part_selects: list[tuple[str, str, str]] = []
    for idx, part in enumerate(parts):
        part_view = f"{view_name}_part_{idx}"
        part_sql = sql_str(str(part))
        con.execute(f"CREATE TEMP VIEW {part_view} AS SELECT * FROM read_parquet('{part_sql}')")
        dup_part = int(
            con.execute(
                f"SELECT COUNT(*) FROM (SELECT {key_q} FROM {part_view} GROUP BY {key_q} HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )
        if dup_part:
            raise SchemaError(f"Overlay part has duplicate keys for '{key}': {part}")
        meta = overlay_metadata(part)
        created = meta.get("created_at")
        if not created:
            raise SchemaError(f"Overlay part missing required created_at metadata: {part}")
        part_selects.append((part_view, created, part.name))

    if len(part_selects) == 1:
        con.execute(f"CREATE TEMP VIEW {view_name} AS SELECT * FROM {part_selects[0][0]}")
        return

    ranked_selects = []
    for part_view, created, file_name in part_selects:
        created_sql = sql_str(created)
        file_sql = sql_str(file_name)
        ranked_selects.append(
            "SELECT p.*, "
            f"'{created_sql}' AS __usr_overlay_created_at, "
            f"'{file_sql}' AS __usr_overlay_filename "
            f"FROM {part_view} p"
        )

    raw_view = f"{view_name}_raw"
    con.execute(f"CREATE TEMP VIEW {raw_view} AS {' UNION ALL '.join(ranked_selects)}")
    con.execute(
        f"CREATE TEMP VIEW {view_name} AS "
        "SELECT * EXCLUDE (__usr_overlay_rownum, __usr_overlay_created_at, __usr_overlay_filename) "
        "FROM ("
        "SELECT r.*, "
        "ROW_NUMBER() OVER ("
        f"PARTITION BY {key_q} "
        "ORDER BY __usr_overlay_created_at DESC, __usr_overlay_filename DESC"
        ") "
        f"AS __usr_overlay_rownum FROM {raw_view} r"
        ") ranked WHERE __usr_overlay_rownum = 1"
    )
