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
from dnadesign.usr.src.registry import (
    USR_STATE_COLUMNS,
    USR_STATE_NAMESPACE,
    RegistryColumn,
    arrow_type_str,
    load_registry,
    register_namespace,
)
from dnadesign.usr.src.schema import META_REGISTRY_HASH


def _make_dataset(root: Path) -> Dataset:
    if not (root / "registry.yaml").exists():
        _write_registry(root, {}, include_usr_state=True)
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


def _write_registry(root: Path, namespaces: dict, *, include_usr_state: bool = True) -> Path:
    path = root / "registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if include_usr_state and USR_STATE_NAMESPACE not in namespaces:
        namespaces = {
            **namespaces,
            USR_STATE_NAMESPACE: {
                "owner": "usr",
                "description": "reserved record-state overlay",
                "columns": [{"name": c.name, "type": c.type} for c in USR_STATE_COLUMNS],
            },
        }
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

    with pytest.raises(SchemaError, match="Namespace 'mock' is not registered"):
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


def test_overlay_registry_hash_mismatch_is_error(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
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
    ds = _make_dataset(root)

    out_path = overlay_path(ds.dir, "mock")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_tbl = pa.table({"id": ds.head(2)["id"].tolist(), "mock__score": [1.0, 2.0]})
    overlay_tbl = with_overlay_metadata(
        overlay_tbl,
        namespace="mock",
        key="id",
        created_at="2026-02-06T00:00:00Z",
        registry_hash="deadbeef",
    )
    pq.write_table(overlay_tbl, out_path)

    with pytest.raises(SchemaError, match="registry_hash"):
        ds.validate(registry_mode="current")


def test_registry_requires_usr_state(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "float64"}],
            }
        },
        include_usr_state=False,
    )
    with pytest.raises(SchemaError, match="usr_state"):
        load_registry(root, required=True)


def test_register_namespace_bootstraps_usr_state(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_namespace(
        root,
        namespace="mock",
        columns=[RegistryColumn("mock__score", "float64")],
    )
    entries = load_registry(root, required=True)
    assert "mock" in entries
    assert USR_STATE_NAMESPACE in entries


def test_overlays_require_registry_for_reads(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    ds = _make_dataset(root)

    overlay_dir = ds.dir / "_derived"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    out_path = overlay_path(ds.dir, "mock")
    overlay_tbl = pa.table({"id": ds.head(2)["id"].tolist(), "mock__score": [1.0, 2.0]})
    overlay_tbl = with_overlay_metadata(overlay_tbl, namespace="mock", key="id", created_at="2026-02-05T00:00:00Z")
    pq.write_table(overlay_tbl, out_path)

    with pytest.raises(SchemaError, match="Namespace 'mock' is not registered"):
        ds.head(1)

    df = ds.head(1, include_derived=False)
    assert df.shape[0] == 1


def test_registry_hash_written_when_registry_present(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
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

    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    pf = pq.ParquetFile(str(ds.records_path))
    md = pf.schema_arrow.metadata or {}
    assert META_REGISTRY_HASH.encode("utf-8") in md


def test_validate_registry_modes(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
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
    ds = _make_dataset(root)

    attach_path = tmp_path / "attach.parquet"
    ids = ds.head(2)["id"].tolist()
    tbl = pa.table({"id": ids, "score": [0.1, 0.2]})
    pq.write_table(tbl, attach_path)
    ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)

    with ds.maintenance(reason="registry_freeze"):
        ds.freeze_registry()

    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "string"}],
            }
        },
    )

    with pytest.raises(SchemaError, match="registry_hash"):
        ds.validate(registry_mode="current")

    ds.validate(registry_mode="frozen")
    ds.validate(registry_mode="either")


def test_registry_type_supports_struct_and_fixed_list() -> None:
    dtype = pa.list_(pa.float32(), list_size=8)
    assert arrow_type_str(dtype) == "fixed_size_list<float32>[8]"

    struct = pa.struct(
        [
            pa.field("name", pa.string()),
            pa.field("values", pa.list_(pa.int64())),
        ]
    )
    assert arrow_type_str(struct) == "struct<name:string,values:list<int64>>"
