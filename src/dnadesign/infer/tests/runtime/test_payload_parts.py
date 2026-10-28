"""Committed feature parts retain exact values across append and migration."""

import json
import math
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.contracts.feature_payloads import manifest_path, parts_path, payload_files
from dnadesign.infer.src.features import aliases, payload_store
from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces


def _dataset(tmp_path: Path) -> Dataset:
    ensure_sequence_contract_namespaces(tmp_path)
    ds = Dataset(tmp_path, "demo")
    ds.init(source="test")
    return ds


def _row(ds: Dataset, key: str, value: list[float]) -> dict:
    return {
        "_dataset_root": str(ds.root),
        "_dataset_id": ds.name,
        "feature_vector_key": key,
        "value": value,
        "created_at": "2026-10-28T00:00:00+00:00",
    }


def test_appending_payloads_does_not_rewrite_committed_values(tmp_path):
    ds = _dataset(tmp_path)
    aliases.persist_feature_vector_rows([_row(ds, "one", [1.0, 2.0])])
    before = {p: sha256(p.read_bytes()).hexdigest() for p in (ds.dir / "_derived/infer").rglob("*.parquet")}
    assert before
    aliases.persist_feature_vector_rows([_row(ds, "two", [3.0, 4.0])])
    assert all(p.exists() and sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one", "two"]) == {
        "one": [1.0, 2.0],
        "two": [3.0, 4.0],
    }


def test_single_file_payload_requires_explicit_migration_before_appending(tmp_path):
    ds = _dataset(tmp_path)
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _row(ds, "one", [1.0, 2.0])
    pq.write_table(pa.Table.from_pylist([row], schema=aliases._FEATURE_VECTOR_SCHEMA), path)
    before = path.read_bytes()
    assert aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one"]) == {
        "one": [1.0, 2.0]
    }
    with pytest.raises(ValueError, match="migrat"):
        aliases.persist_feature_vector_rows([_row(ds, "two", [3.0, 4.0])])
    assert path.read_bytes() == before

    result = aliases.migrate_feature_payloads(dataset_root=ds.root, dataset_id=ds.name)
    assert result["vectors"]["migrated_rows"] == 1
    assert not path.exists()
    assert payload_files(path)[0].read_bytes() == before
    aliases.persist_feature_vector_rows([_row(ds, "two", [3.0, 4.0])])
    assert aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one", "two"]) == {
        "one": [1.0, 2.0],
        "two": [3.0, 4.0],
    }
    assert aliases.migrate_feature_payloads(dataset_root=ds.root, dataset_id=ds.name)["vectors"]["migrated_rows"] == 0


def test_failed_commit_and_orphan_do_not_change_visible_payloads(tmp_path, monkeypatch):
    ds = _dataset(tmp_path)
    aliases.persist_feature_vector_rows([_row(ds, "one", [1.0, 2.0])])
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    before = manifest_path(path).read_bytes()
    original = payload_store._write_manifest

    def fail(*args, **kwargs):
        raise OSError("injected before manifest publication")

    monkeypatch.setattr(payload_store, "_write_manifest", fail)
    with pytest.raises(OSError, match="injected"):
        aliases.persist_feature_vector_rows([_row(ds, "two", [3.0, 4.0])])
    assert manifest_path(path).read_bytes() == before
    orphan = parts_path(path) / ("part-" + "a" * 32 + ".parquet")
    pq.write_table(pa.Table.from_pylist([_row(ds, "orphan", [99.0])], schema=aliases._FEATURE_VECTOR_SCHEMA), orphan)
    assert aliases.load_feature_vector_rows(
        dataset_root=ds.root, dataset_id=ds.name, keys=["one", "two", "orphan"]
    ) == {"one": [1.0, 2.0]}
    monkeypatch.setattr(payload_store, "_write_manifest", original)
    aliases.persist_feature_vector_rows([_row(ds, "two", [3.0, 4.0])])
    aliases.compact_feature_sidecars(dataset_root=ds.root, dataset_id=ds.name)
    assert len(payload_files(path)) == 1
    assert not orphan.exists()
    assert aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one", "two"]) == {
        "one": [1.0, 2.0],
        "two": [3.0, 4.0],
    }


def test_collision_does_not_publish_another_part(tmp_path):
    ds = _dataset(tmp_path)
    row = _row(ds, "one", [1.0, 2.0])
    assert aliases.persist_feature_vector_rows([row]) == 1
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    before = manifest_path(path).read_bytes()
    assert aliases.persist_feature_vector_rows([row]) == 0
    with pytest.raises(ValueError, match="collision"):
        aliases.persist_feature_vector_rows([{**row, "value": [9.0]}])
    assert manifest_path(path).read_bytes() == before


@pytest.mark.parametrize("bad_name", ["../outside.parquet", "/tmp/outside.parquet", "part-invalid.parquet"])
def test_manifest_cannot_reference_files_outside_its_parts_directory(tmp_path, bad_name):
    ds = _dataset(tmp_path)
    aliases.persist_feature_vector_rows([_row(ds, "one", [1.0])])
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    manifest = json.loads(manifest_path(path).read_text())
    manifest["parts"][0]["file"] = bad_name
    manifest_path(path).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one"])


def test_compaction_rejects_changed_part_bytes(tmp_path):
    ds = _dataset(tmp_path)
    aliases.persist_feature_vector_rows([_row(ds, "one", [1.0])])
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    part = payload_files(path)[0]
    data = bytearray(part.read_bytes())
    data[10] ^= 1
    part.write_bytes(data)
    before = manifest_path(path).read_bytes()
    with pytest.raises(ValueError, match="checksum mismatch"):
        payload_store.compact_payload(path, schema=aliases._FEATURE_VECTOR_SCHEMA, keep_keys={"one"}, batch_size=8)
    assert manifest_path(path).read_bytes() == before


def test_cli_migration_and_compaction_preserve_payload_keys(tmp_path):
    from typer.testing import CliRunner

    from dnadesign.infer.src.cli import app

    ds = _dataset(tmp_path)
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([_row(ds, "one", [1.0])], schema=aliases._FEATURE_VECTOR_SCHEMA), path)
    for command in ("migrate", "compact"):
        result = CliRunner().invoke(app, ["features", command, "--root", str(ds.root), "--dataset", ds.name])
        assert result.exit_code == 0, result.output or repr(result.exception)
    assert aliases.load_feature_vector_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["one"]) == {"one": [1.0]}


def test_scalar_nan_round_trips_without_changing_its_value(tmp_path):
    ds = _dataset(tmp_path)
    aliases.persist_feature_scalar_rows(
        [
            {
                "_dataset_root": str(ds.root),
                "_dataset_id": ds.name,
                "feature_scalar_key": "scalar",
                "value": float("nan"),
                "created_at": "2026-10-28T00:00:00+00:00",
            }
        ]
    )
    result = aliases.load_feature_scalar_rows(dataset_root=ds.root, dataset_id=ds.name, keys=["scalar"])
    assert math.isnan(result["scalar"])


def test_streaming_reader_holds_lock_until_snapshot_is_closed(tmp_path):
    import fcntl

    from dnadesign.contracts.feature_payloads import iter_payload_batches

    ds = _dataset(tmp_path)
    aliases.persist_feature_vector_rows([_row(ds, "one", [1.0]), _row(ds, "two", [2.0])])
    path = aliases.feature_vector_path(dataset_root=ds.root, dataset_id=ds.name)
    stream = iter_payload_batches(path, batch_size=1)
    next(stream)
    try:
        with (path.parent / ".sidecar.lock").open() as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stream.close()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()
