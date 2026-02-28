"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_duckdb_session_contract.py

DuckDB session hardening contracts for USR read/write paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.usr import Dataset
from dnadesign.usr.tests.registry_helpers import ensure_registry


def test_connect_duckdb_utc_sets_session_timezone() -> None:
    from dnadesign.usr.src.duckdb_runtime import connect_duckdb_utc

    con = connect_duckdb_utc(error_context="duckdb session contract test")
    try:
        current_tz = con.execute("SELECT current_setting('TimeZone')").fetchone()[0]
    finally:
        con.close()
    assert str(current_tz).strip().upper() == "UTC"


def test_dataset_head_created_at_is_utc_offset_zero(tmp_path: Path) -> None:
    ensure_registry(tmp_path)
    ds = Dataset(tmp_path, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {
                "sequence": "ACGT",
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": "unit-test",
            }
        ],
        source="unit-test",
    )

    out = ds.head(1)
    ts = out["created_at"].iloc[0]
    assert isinstance(ts, pd.Timestamp)
    assert ts.tzinfo is not None
    assert ts.utcoffset().total_seconds() == 0
