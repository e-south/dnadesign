"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_mock_overlay.py

Tests that mock dataset creation uses overlay-first layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.usr.src.mock import MockSpec, create_mock_dataset
from dnadesign.usr.src.schema import REQUIRED_COLUMNS
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_make_mock_creates_overlay_and_base_only(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(
        root,
        namespace="demo",
        columns_spec="demo__x_representation:list<float32>,demo__label_vec8:list<float32>",
    )
    spec = MockSpec(n=3, length=4, x_dim=2, y_dim=2, namespace="demo")
    create_mock_dataset(root, "demo", spec, force=True)

    base_path = root / "demo" / "records.parquet"
    base_cols = pq.ParquetFile(str(base_path)).schema_arrow.names
    assert set(base_cols) == {name for name, _ in REQUIRED_COLUMNS}

    overlay_path = root / "demo" / "_derived" / "demo.parquet"
    assert overlay_path.exists()
    overlay_cols = pq.ParquetFile(str(overlay_path)).schema_arrow.names
    assert "demo__x_representation" in overlay_cols
    assert "demo__label_vec8" in overlay_cols


def test_make_mock_writes_structured_event(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(
        root,
        namespace="demo",
        columns_spec="demo__x_representation:list<float32>,demo__label_vec8:list<float32>",
    )
    spec = MockSpec(n=1, length=4, x_dim=2, y_dim=2, namespace="demo")
    create_mock_dataset(root, "demo", spec, force=True)

    events_path = root / "demo" / ".events.log"
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["action"] == "make_mock"
    assert "fingerprint" in payload
    assert payload["fingerprint"]["rows"] == 1
    assert payload["fingerprint"]["cols"] >= 1
