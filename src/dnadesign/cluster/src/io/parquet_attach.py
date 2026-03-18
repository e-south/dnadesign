"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/io/parquet_attach.py

Parquet-native attachment materialization for large generic file writes.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("PyArrow is required for Parquet column inspection.") from exc
    return list(pq.read_schema(path).names)


def _duckdb_connection():
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("duckdb is required for optimized Parquet attachment writes.") from exc
    return duckdb.connect()


def _output_path(*, src_file: Path, inplace: bool, out: Path | None) -> Path:
    if inplace and out is not None:
        raise ValueError("Pass either --inplace or --out, not both.")
    if not inplace and out is None:
        raise ValueError("Provide --out when not using --inplace.")
    return src_file if inplace else Path(out)


def write_parquet_with_attached_columns(
    *,
    src_file: Path,
    cols_df: pd.DataFrame,
    key_col: str,
    allow_overwrite: bool,
    inplace: bool,
    out: Path | None,
    backup_fn,
    base_df: pd.DataFrame | None = None,
) -> Path:
    source_columns = list(base_df.columns) if base_df is not None else _read_parquet_columns(src_file)
    if key_col not in source_columns:
        raise KeyError(f"Parquet source is missing key column '{key_col}'.")
    if key_col not in cols_df.columns:
        raise KeyError(f"Attachment columns are missing key column '{key_col}'.")

    attach_columns = [column for column in cols_df.columns if column != key_col]
    if not attach_columns:
        return _output_path(src_file=src_file, inplace=inplace, out=out)

    existing = [column for column in attach_columns if column in source_columns]
    if existing and not allow_overwrite:
        raise RuntimeError(
            "Columns already exist: "
            + ", ".join(existing[:8])
            + (" ..." if len(existing) > 8 else "")
            + ". Re-run with `-y/--allow-overwrite` or use a new --name."
        )

    target = _output_path(src_file=src_file, inplace=inplace, out=out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if inplace:
        backup_fn(src_file)

    attach_columns_set = set(attach_columns)
    select_parts: list[str] = []
    for column in source_columns:
        quoted = _quote_ident(column)
        if column in attach_columns_set and column != key_col:
            select_parts.append(f"ov.{quoted} AS {quoted}")
        else:
            select_parts.append(f"src.{quoted}")
    for column in attach_columns:
        if column not in source_columns:
            quoted = _quote_ident(column)
            select_parts.append(f"ov.{quoted} AS {quoted}")

    key_sql = _quote_ident(key_col)
    con = _duckdb_connection()
    try:
        if base_df is not None:
            con.register("src_df", base_df)
            source_ref = "src_df AS src"
        else:
            source_ref = f"read_parquet('{_quote_path(src_file)}') AS src"
        con.register("ov_df", cols_df)
        output_tmp = target.with_suffix(target.suffix + ".tmp")
        query = (
            f"COPY (SELECT {', '.join(select_parts)} "
            f"FROM {source_ref} "
            f"LEFT JOIN ov_df AS ov USING ({key_sql})) "
            f"TO '{_quote_path(output_tmp)}' (FORMAT PARQUET)"
        )
        con.execute(query)
    finally:
        con.close()

    output_tmp.replace(target)
    return target


__all__ = ["write_parquet_with_attached_columns"]
