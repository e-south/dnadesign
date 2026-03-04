"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_duckdb_session_contract.py

DuckDB session hardening contracts for USR read/write paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_connect_duckdb_utc_reuses_root_connection(monkeypatch) -> None:
    from dnadesign.usr.src import duckdb_runtime as module

    class _FakeResult:
        def __init__(self, value: str) -> None:
            self._value = value

        def fetchone(self):
            return (self._value,)

    class _FakeSession:
        def __init__(self) -> None:
            self._timezone = "UTC"

        def execute(self, sql: str):
            if sql == "SET TimeZone='UTC'":
                self._timezone = "UTC"
                return self
            if sql == "SELECT current_setting('TimeZone')":
                return _FakeResult(self._timezone)
            raise AssertionError(f"Unexpected SQL in fake session: {sql}")

        def close(self) -> None:
            return None

    class _FakeRoot:
        def cursor(self):
            return _FakeSession()

        def close(self) -> None:
            return None

    calls = {"connect": 0}

    def _fake_connect():
        calls["connect"] += 1
        return _FakeRoot()

    monkeypatch.setitem(sys.modules, "duckdb", SimpleNamespace(connect=_fake_connect))
    module._DUCKDB_ROOT_SESSION = None
    try:
        con_a = module.connect_duckdb_utc(error_context="reuse contract test")
        con_a.close()
        con_b = module.connect_duckdb_utc(error_context="reuse contract test")
        con_b.close()
        assert calls["connect"] == 1
    finally:
        module._DUCKDB_ROOT_SESSION = None
