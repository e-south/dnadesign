"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_registry.py

Namespace registry enforcement tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.errors import SchemaError
from dnadesign.usr.src.overlays import overlay_path, with_overlay_metadata


def _make_dataset(root: Path) -> Dataset:
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows(
        [
            {"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
            {"sequence": "TGCA", "bio_type": "dna", "alphabet": "dna_4", "source": "unit"},
        ],
        source="unit",
    )
    return ds


def _write_registry(root: Path, namespaces: dict) -> Path:
    path = root / "registry.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"namespaces": namespaces}, f, sort_keys=True)
    return path


def test_attach_requires_registry(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)

    with pytest.raises(SchemaError, match="Registry"):
        ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)

    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "float64"}],
            }
        },
    )

    rows = ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)
    assert rows == 2


def test_registry_type_mismatch_is_error(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)
    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "float64"}],
            }
        },
    )

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": ["x", "y"]})
    pq.write_table(tbl, attach_path)

    with pytest.raises(SchemaError, match="type"):
        ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)


def test_overlays_require_registry_for_reads(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    overlay_dir = ds.dir / "_derived"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlay_path(ds.dir, "mock")
    overlay_tbl = pa.table({"id": ds.head(2)["id"].tolist(), "mock__score": [1.0, 2.0]})
    overlay_tbl = with_overlay_metadata(overlay_tbl, namespace="mock", key="id", created_at="2026-02-05T00:00:00Z")
    pq.write_table(overlay_tbl, out_path)

    with pytest.raises(SchemaError, match="Registry"):
        ds.head(1)

    df = ds.head(1, include_derived=False)
    assert df.shape[0] == 1
