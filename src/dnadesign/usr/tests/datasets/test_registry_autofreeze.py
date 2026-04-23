"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_registry_autofreeze.py

Auto-freeze registry behavior tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.usr import Dataset
from dnadesign.usr.src.registry import registry_hash
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_auto_freeze_registry_on_init(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit")

    reg_hash = registry_hash(root, required=True)
    snap_dir = ds.dir / "_registry"
    assert (snap_dir / f"registry.{reg_hash}.yaml").exists()

    md = pq.ParquetFile(str(ds.records_path)).schema_arrow.metadata or {}
    assert md.get(b"usr:registry_hash") == reg_hash.encode("utf-8")
