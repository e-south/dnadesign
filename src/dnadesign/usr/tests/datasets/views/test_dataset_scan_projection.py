"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/views/test_dataset_scan_projection.py

Tests for bounded overlay projection planning in USR dataset scans.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.datasets.demo.mock import MockSpec, create_mock_dataset


def test_duckdb_query_passes_requested_overlay_columns_to_overlay_view(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="mock",
        columns_spec="mock__x_representation:list<float32>,mock__label_vec8:list<float32>",
    )
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__x_representation:list<float32>,infer__aux_representation:list<float32>",
    )
    create_mock_dataset(
        root,
        "demo_overlay_scan",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )

    dataset = Dataset(root, "demo_overlay_scan")
    ids = dataset.head(n=3, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__x_representation": pa.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], type=pa.list_(pa.float32())),
                "infer__aux_representation": pa.array(
                    [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        key="id",
    )

    captured: dict[str, object] = {}

    def _capture_overlay_view(con, *, view_name: str, path: Path, key: str, columns=None) -> str:
        captured["view_name"] = view_name
        captured["path"] = path
        captured["key"] = key
        captured["columns"] = list(columns) if columns is not None else None
        from dnadesign.usr.src.datasets.query import create_overlay_view

        return create_overlay_view(con, view_name=view_name, path=path, key=key, columns=columns)

    monkeypatch.setattr(dataset, "_create_overlay_view", _capture_overlay_view)

    con, _, _ = dataset._duckdb_query(
        columns=["id", "infer__x_representation"],
        include_overlays=True,
        include_deleted=False,
    )
    try:
        assert captured["key"] == "id"
        assert captured["columns"] == ["id", "infer__x_representation"]
    finally:
        con.close()


def test_duckdb_query_applies_low_memory_settings_for_unbounded_overlay_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "usr_root"
    register_test_namespace(
        root,
        namespace="mock",
        columns_spec="mock__x_representation:list<float32>,mock__label_vec8:list<float32>",
    )
    register_test_namespace(
        root,
        namespace="infer",
        columns_spec="infer__x_representation:list<float32>",
    )
    create_mock_dataset(
        root,
        "demo_overlay_scan",
        MockSpec(n=3, length=12, x_dim=2, y_dim=2, namespace="mock"),
        force=True,
    )

    dataset = Dataset(root, "demo_overlay_scan")
    ids = dataset.head(n=3, columns=["id"], include_derived=False)["id"].tolist()
    dataset.write_overlay_part(
        "infer",
        pa.table(
            {
                "id": ids,
                "infer__x_representation": pa.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], type=pa.list_(pa.float32())),
            }
        ),
        key="id",
    )

    from dnadesign.usr.src.datasets.query import planner as dataset_overlay_query_module

    recorded_sql: list[str] = []
    connect_default = dataset_overlay_query_module.connect_duckdb_utc

    class _CountingConnection:
        def __init__(self, inner) -> None:
            self._inner = inner

        def execute(self, sql: str, *args, **kwargs):
            recorded_sql.append(sql)
            return self._inner.execute(sql, *args, **kwargs)

        def close(self) -> None:
            self._inner.close()

    def _connect_counting(*args, **kwargs):
        return _CountingConnection(connect_default(*args, **kwargs))

    monkeypatch.setattr(dataset_overlay_query_module, "connect_duckdb_utc", _connect_counting)

    con, _, _ = dataset._duckdb_query(
        columns=["id", "infer__x_representation"],
        include_overlays=True,
        include_deleted=False,
    )
    try:
        assert "SET threads TO 1" in recorded_sql
        assert "SET preserve_insertion_order TO false" in recorded_sql
    finally:
        con.close()
