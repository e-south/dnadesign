"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/runtime/test_feature_alias_contracts.py

Regression tests for feature alias infer runtime.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.infer.src.features import aliases as alias_module
from dnadesign.infer.src.features.aliases import (
    compact_feature_sidecars_to_current_aliases,
    feature_alias_path,
    feature_vector_path,
    persist_feature_alias_rows,
    persist_feature_vector_rows,
    prune_stale_feature_alias_entries,
)
from dnadesign.infer.src.features.cache_keys import DNA_SEQUENCE_CASE_POLICY
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces


def _dataset(tmp_path: Path) -> tuple[Path, Dataset]:
    root = tmp_path / "usr"
    ensure_sequence_contract_namespaces(root)
    ds = Dataset(root, "demo")
    ds.init(source="test")
    return root, ds


def _current_alias_row(root: Path, dataset_id: str, alias_id: str = "alias_current") -> dict[str, object]:
    return {
        "_dataset_root": root.as_posix(),
        "_dataset_id": dataset_id,
        "alias_id": alias_id,
        "view_id": "view_current",
        "view_name": "current",
        "sequence_id": "seq_current",
        "feature_vector_key": "fv_current",
        "forward_pass_key": "fp_current",
        "provider": "evo2",
        "model_name": "evo2_7b",
        "model_revision": None,
        "layer_name": "block26_mlp_out",
        "representation_kind": "intermediate_embedding",
        "pooling_operation": "seq_mean",
        "pooling_start_0": None,
        "pooling_end_0": None,
        "orientation": "forward",
        "source_dataset_id": dataset_id,
        "feature_request_digest": "digest_current",
        "runtime_fingerprint_key": "runtime_fingerprint_current",
        "sequence_case_policy": DNA_SEQUENCE_CASE_POLICY,
        "created_at": "2026-05-06T00:00:00+00:00",
    }


def test_feature_alias_writer_rejects_missing_runtime_contract(tmp_path: Path) -> None:
    root, ds = _dataset(tmp_path)
    row = _current_alias_row(root, ds.name)
    row.pop("runtime_fingerprint_key")

    with pytest.raises(ValueError, match="missing current runtime contract"):
        persist_feature_alias_rows([row])


def test_feature_alias_writer_prunes_fingerprintless_aliases_on_current_write(tmp_path: Path) -> None:
    root, ds = _dataset(tmp_path)
    path = feature_alias_path(dataset_root=root, dataset_id=ds.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "alias_id": "alias_stale",
                    "view_id": "view_stale",
                    "view_name": "stale",
                    "sequence_id": "seq_stale",
                    "feature_vector_key": "fv_stale",
                    "forward_pass_key": "fp_stale",
                    "provider": "evo2",
                    "model_name": "evo2_7b",
                    "model_revision": None,
                    "layer_name": "block26_mlp_out",
                    "representation_kind": "intermediate_embedding",
                    "pooling_operation": "seq_mean",
                    "pooling_start_0": None,
                    "pooling_end_0": None,
                    "orientation": "forward",
                    "source_dataset_id": ds.name,
                    "feature_request_digest": "digest_stale",
                    "created_at": "2026-04-01T00:00:00+00:00",
                }
            ]
        ),
        path,
    )

    persist_feature_alias_rows([_current_alias_row(root, ds.name)])

    rows = pq.read_table(path).to_pylist()
    assert [row["alias_id"] for row in rows] == ["alias_current"]
    assert rows[0]["runtime_fingerprint_key"] == "runtime_fingerprint_current"
    assert rows[0]["sequence_case_policy"] == DNA_SEQUENCE_CASE_POLICY


def test_feature_alias_writer_uses_atomic_temp_promote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, ds = _dataset(tmp_path)
    path = feature_alias_path(dataset_root=root, dataset_id=ds.name)
    original_write_table = alias_module.pq.write_table
    write_paths: list[Path] = []

    def _capture_write_table(table, where, *args, **kwargs):
        write_paths.append(Path(where))
        return original_write_table(table, where, *args, **kwargs)

    monkeypatch.setattr(alias_module.pq, "write_table", _capture_write_table)

    persist_feature_alias_rows([_current_alias_row(root, ds.name)])

    assert path.exists()
    assert path not in write_paths
    assert any(
        write_path.parent == path.parent and write_path.name.startswith(f".{path.name}.") for write_path in write_paths
    )


def test_feature_alias_writer_creates_dataset_local_lock(tmp_path: Path) -> None:
    root, ds = _dataset(tmp_path)

    persist_feature_alias_rows([_current_alias_row(root, ds.name)])

    assert (root / ds.name / "_derived/infer/.sidecar.lock").exists()


def test_sidecar_compaction_prunes_unreferenced_payload_rows(tmp_path: Path) -> None:
    root, ds = _dataset(tmp_path)
    persist_feature_alias_rows([_current_alias_row(root, ds.name)])
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": root.as_posix(),
                "_dataset_id": ds.name,
                "feature_vector_key": "fv_current",
                "value": [1.0, 2.0],
                "created_at": "2026-05-06T00:00:00+00:00",
            },
            {
                "_dataset_root": root.as_posix(),
                "_dataset_id": ds.name,
                "feature_vector_key": "fv_stale",
                "value": [3.0, 4.0],
                "created_at": "2026-04-01T00:00:00+00:00",
            },
        ]
    )

    result = compact_feature_sidecars_to_current_aliases(dataset_root=root, dataset_id=ds.name)

    rows = pq.read_table(feature_vector_path(dataset_root=root, dataset_id=ds.name)).to_pylist()
    assert result["removed_vector_rows"] == 1
    assert [row["feature_vector_key"] for row in rows] == ["fv_current"]


def test_alias_prune_removes_same_slot_superseded_runtime_before_payload_compaction(tmp_path: Path) -> None:
    root, ds = _dataset(tmp_path)
    stale = _current_alias_row(root, ds.name, alias_id="alias_old_runtime")
    stale["feature_vector_key"] = "fv_old_runtime"
    stale["runtime_fingerprint_key"] = "runtime_fingerprint_old"
    current = _current_alias_row(root, ds.name)

    persist_feature_alias_rows([stale])
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": root.as_posix(),
                "_dataset_id": ds.name,
                "feature_vector_key": "fv_old_runtime",
                "value": [0.0, 0.0],
                "created_at": "2026-04-01T00:00:00+00:00",
            }
        ]
    )
    persist_feature_alias_rows([current])
    persist_feature_vector_rows(
        [
            {
                "_dataset_root": root.as_posix(),
                "_dataset_id": ds.name,
                "feature_vector_key": "fv_current",
                "value": [1.0, 2.0],
                "created_at": "2026-05-06T00:00:00+00:00",
            }
        ]
    )

    assert current["_dataset_root"] == root.as_posix()
    prune_result = prune_stale_feature_alias_entries(
        current_vector_alias_rows=[current],
        current_scalar_alias_rows=[],
    )
    compact_result = compact_feature_sidecars_to_current_aliases(dataset_root=root, dataset_id=ds.name)

    alias_rows = pq.read_table(feature_alias_path(dataset_root=root, dataset_id=ds.name)).to_pylist()
    vector_rows = pq.read_table(feature_vector_path(dataset_root=root, dataset_id=ds.name)).to_pylist()
    assert prune_result["removed_vector_alias_rows"] == 1
    assert compact_result["removed_vector_rows"] == 1
    assert [row["alias_id"] for row in alias_rows] == ["alias_current"]
    assert [row["feature_vector_key"] for row in vector_rows] == ["fv_current"]
