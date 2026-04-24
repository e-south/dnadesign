"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/datasets/lifecycle/test_registry.py

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

from dnadesign.usr.src.contracts import META_REGISTRY_HASH, SchemaError
from dnadesign.usr.src.dataset import Dataset
from dnadesign.usr.src.overlays import overlay_path, with_overlay_metadata
from dnadesign.usr.src.registry import (
    USR_STATE_COLUMNS,
    USR_STATE_NAMESPACE,
    RegistryColumn,
    arrow_type_str,
    load_registry,
    load_registry_file,
    namespace_contract_hash_for_entries,
    register_namespace,
    registry_bytes,
    registry_hash,
)


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


def test_namespace_contract_hash_ignores_catalog_fields_but_tracks_column_order(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _write_registry(
        root,
        {
            "mock": {
                "owner": "tool-a",
                "description": "first description",
                "columns": [
                    {"name": "mock__score", "type": "float64"},
                    {"name": "mock__rank", "type": "int64"},
                ],
            }
        },
    )
    first = namespace_contract_hash_for_entries(load_registry(root, required=True), "mock")

    _write_registry(
        root,
        {
            "mock": {
                "owner": "tool-b",
                "description": "second description",
                "columns": [
                    {"name": "mock__score", "type": "float64"},
                    {"name": "mock__rank", "type": "int64"},
                ],
            }
        },
    )
    second = namespace_contract_hash_for_entries(load_registry(root, required=True), "mock")

    _write_registry(
        root,
        {
            "mock": {
                "owner": "tool-b",
                "description": "second description",
                "columns": [
                    {"name": "mock__rank", "type": "int64"},
                    {"name": "mock__score", "type": "float64"},
                ],
            }
        },
    )
    reordered = namespace_contract_hash_for_entries(load_registry(root, required=True), "mock")

    assert first == second
    assert reordered != first


def test_registry_hash_and_bytes_reuse_cached_canonical_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    import dnadesign.usr.src.registry as registry_mod

    safe_dump_calls = {"count": 0}
    safe_dump_original = registry_mod.yaml.safe_dump

    def _count_safe_dump(*args, **kwargs):
        safe_dump_calls["count"] += 1
        return safe_dump_original(*args, **kwargs)

    monkeypatch.setattr(registry_mod.yaml, "safe_dump", _count_safe_dump)

    first_hash = registry_hash(root, required=True)
    first_bytes = registry_bytes(root)
    second_hash = registry_hash(root, required=True)
    second_bytes = registry_bytes(root)

    assert first_hash == second_hash
    assert first_bytes == second_bytes
    assert safe_dump_calls["count"] == 1


def test_registry_hash_cache_invalidates_after_registry_update(tmp_path: Path) -> None:
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
    first_hash = registry_hash(root, required=True)
    first_bytes = registry_bytes(root)

    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [
                    {"name": "mock__score", "type": "float64"},
                    {"name": "mock__rank", "type": "int64"},
                ],
            }
        },
    )
    second_hash = registry_hash(root, required=True)
    second_bytes = registry_bytes(root)

    assert first_hash != second_hash
    assert first_bytes != second_bytes


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


def test_namespace_current_mode_ignores_unrelated_registry_changes(tmp_path: Path) -> None:
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
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
    ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)

    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "float64"}],
            },
            "other": {
                "owner": "unit-test",
                "description": "unrelated namespace added later",
                "columns": [{"name": "other__score", "type": "float64"}],
            },
        },
    )

    with pytest.raises(SchemaError, match="registry_hash"):
        ds.validate(registry_mode="current")

    ds.validate(registry_mode="namespace-current")


def test_namespace_registry_modes_require_namespace_contract_hash(tmp_path: Path) -> None:
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

    with pytest.raises(SchemaError, match="namespace_contract_hash"):
        ds.validate(registry_mode="namespace-current")


def test_namespace_frozen_and_either_modes_use_frozen_registry_snapshot(tmp_path: Path) -> None:
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
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
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

    with pytest.raises(SchemaError, match="namespace_contract_hash"):
        ds.validate(registry_mode="namespace-current")

    ds.validate(registry_mode="namespace-frozen")
    ds.validate(registry_mode="namespace-either")


def test_strict_validate_materialized_base_uses_namespace_registry_modes(tmp_path: Path) -> None:
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
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
    ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)

    with ds.maintenance(reason="materialize"):
        ds.materialize(keep_overlays=False)

    ds.import_rows(
        [{"sequence": "CCCC", "bio_type": "dna", "alphabet": "dna_4", "source": "after-materialize"}],
        source="after-materialize",
    )
    ds.validate(strict=True, registry_mode="namespace-current")

    _write_registry(
        root,
        {
            "mock": {
                "owner": "unit-test",
                "description": "test namespace",
                "columns": [{"name": "mock__score", "type": "float64"}],
            },
            "other": {
                "owner": "unit-test",
                "description": "unrelated namespace added later",
                "columns": [{"name": "other__score", "type": "float64"}],
            },
        },
    )

    with pytest.raises(SchemaError, match="records.parquet registry_hash mismatch"):
        ds.validate(strict=True, registry_mode="current")

    ds.validate(strict=True, registry_mode="namespace-current")


def test_strict_validate_materialized_base_rejects_namespace_type_drift(tmp_path: Path) -> None:
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
    pq.write_table(pa.table({"id": ids, "score": [0.1, 0.2]}), attach_path)
    ds.attach(attach_path, namespace="mock", key="id", columns=["score"], parse_json=False)

    with ds.maintenance(reason="materialize"):
        ds.materialize(keep_overlays=False)

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

    with pytest.raises(SchemaError, match="mock__score"):
        ds.validate(strict=True, registry_mode="namespace-current")


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


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"namespaces": 1}, "must contain a 'namespaces' mapping"),
        (
            {"namespaces": {"mock": "bad"}},
            "Registry entry for 'mock' must be a mapping",
        ),
        (
            {"namespaces": {"mock": {"owner": "unit", "description": "demo", "columns": "bad"}}},
            "must define 'columns' as a list",
        ),
        (
            {"namespaces": {"mock": {"owner": "unit", "description": "demo", "columns": ["bad"]}}},
            "Registry column for 'mock' must be a mapping",
        ),
        (
            {"namespaces": {"mock": {"owner": "unit", "description": "demo", "columns": [{"name": "mock__score"}]}}},
            "requires name and type",
        ),
        (
            {
                "namespaces": {
                    "mock": {
                        "owner": "unit",
                        "description": "demo",
                        "columns": [
                            {"name": "mock__score", "type": "float64"},
                            {"name": "mock__score", "type": "float64"},
                        ],
                    }
                }
            },
            "duplicate column names",
        ),
        (
            {
                "namespaces": {
                    "mock": {
                        "owner": "unit",
                        "description": "demo",
                        "columns": [{"name": "score", "type": "float64"}],
                    }
                }
            },
            "must be namespaced",
        ),
        (
            {
                "namespaces": {
                    "mock": {
                        "owner": "unit",
                        "description": "demo",
                        "columns": [{"name": "mock__score", "type": "mystery"}],
                    }
                }
            },
            "Unsupported registry type",
        ),
    ],
)
def test_load_registry_file_rejects_malformed_registry_payloads(tmp_path: Path, payload: dict, match: str) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(SchemaError, match=match):
        load_registry_file(path)
