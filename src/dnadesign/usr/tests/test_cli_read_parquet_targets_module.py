"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_read_parquet_targets_module.py

Contracts for parquet target helper extraction from read view handlers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_parquet(path: Path, *, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"value": [payload]})
    pq.write_table(table, path)


def test_read_parquet_targets_module_is_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read_parquet_targets")
    assert hasattr(module, "_resolve_parquet_from_dir")
    assert hasattr(module, "_resolve_parquet_target")
    assert hasattr(module, "_list_parquet_candidates")


def test_resolve_parquet_from_dir_prefers_events_parquet_when_present(tmp_path: Path) -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read_parquet_targets")
    target_dir = tmp_path / "dataset"
    _write_parquet(target_dir / "records.parquet", payload="records")
    _write_parquet(target_dir / "events-20260227T010101.parquet", payload="events-timestamped")
    _write_parquet(target_dir / "events.parquet", payload="events")

    selected = module._resolve_parquet_from_dir(target_dir)

    assert selected == target_dir / "events.parquet"


def test_resolve_parquet_from_dir_selects_newest_match_for_glob(tmp_path: Path) -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.read_parquet_targets")
    target_dir = tmp_path / "logs"
    older = target_dir / "events-001.parquet"
    newer = target_dir / "events-002.parquet"
    _write_parquet(older, payload="older")
    _write_parquet(newer, payload="newer")

    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))

    selected = module._resolve_parquet_from_dir(target_dir, glob="events-*.parquet")

    assert selected == newer
