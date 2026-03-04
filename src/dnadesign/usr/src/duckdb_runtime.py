"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/duckdb_runtime.py

DuckDB session helpers with explicit UTC timezone enforcement.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .errors import SchemaError

_DUCKDB_ROOT_SESSION = None


def _duckdb_session_cursor(duckdb_module):
    global _DUCKDB_ROOT_SESSION
    if _DUCKDB_ROOT_SESSION is None:
        _DUCKDB_ROOT_SESSION = duckdb_module.connect()
    try:
        return _DUCKDB_ROOT_SESSION.cursor()
    except Exception:
        try:
            _DUCKDB_ROOT_SESSION.close()
        except Exception:
            pass
        _DUCKDB_ROOT_SESSION = duckdb_module.connect()
        return _DUCKDB_ROOT_SESSION.cursor()


def connect_duckdb_utc(
    *,
    missing_dependency_message: str = "duckdb is required (install duckdb).",
    error_context: str,
):
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SchemaError(str(missing_dependency_message)) from exc

    con = _duckdb_session_cursor(duckdb)
    try:
        con.execute("SET TimeZone='UTC'")
        row = con.execute("SELECT current_setting('TimeZone')").fetchone()
    except Exception as exc:
        con.close()
        raise SchemaError(f"{error_context}: failed to initialize DuckDB session timezone to UTC.") from exc

    value = str(row[0]).strip().upper() if row and row[0] is not None else ""
    if value != "UTC":
        con.close()
        raise SchemaError(f"{error_context}: expected DuckDB TimeZone=UTC, got '{value or 'unset'}'.")
    return con
