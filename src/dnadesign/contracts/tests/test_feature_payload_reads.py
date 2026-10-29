"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/test_feature_payload_reads.py

Feature consumers verify committed content and support explicit snapshots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.contracts import feature_payloads as payloads


def _table(value):
    return pa.Table.from_pylist(
        [{"feature_vector_key": "one", "value": [value, value + 1], "created_at": "2026-10-28"}],
        schema=payloads.VECTOR_SCHEMA,
    )


def _bundle(tmp_path):
    path = tmp_path / "feature_vectors.parquet"
    directory = payloads.parts_path(path)
    directory.mkdir()
    part = directory / ("part-" + "a" * 32 + ".parquet")
    pq.write_table(_table(0.25), part)
    document = {
        "format": payloads.FORMAT,
        "key_column": "feature_vector_key",
        "parts": [
            {
                "file": part.name,
                "rows": 1,
                "bytes": part.stat().st_size,
                "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            }
        ],
    }
    payloads.manifest_path(path).write_text(json.dumps(document))
    return path, part, document


@pytest.mark.parametrize("streaming", [False, True])
def test_same_size_content_change_invalidates_verified_reads(tmp_path, streaming):
    path, part, _ = _bundle(tmp_path)
    payloads.read_payload_table(path)
    original = part.stat()
    pq.write_table(_table(0.5), part)
    assert part.stat().st_size == original.st_size
    os.utime(part, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum mismatch"):
        if streaming:
            list(payloads.iter_payload_batches(path))
        else:
            payloads.read_payload_table(path)


@pytest.mark.parametrize("field,value,match", [("rows", 2, "row count"), ("sha256", "0" * 64, "checksum mismatch")])
def test_changed_manifest_invalidates_verification(tmp_path, field, value, match):
    path, _, document = _bundle(tmp_path)
    payloads.read_payload_table(path)
    document["parts"][0][field] = value
    payloads.manifest_path(path).write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        payloads.read_payload_table(path)


def test_verified_part_must_have_declared_payload_schema(tmp_path):
    path, part, document = _bundle(tmp_path)
    pq.write_table(pa.table({"wrong": [1]}), part)
    document["parts"][0].update(bytes=part.stat().st_size, sha256=hashlib.sha256(part.read_bytes()).hexdigest())
    payloads.manifest_path(path).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema"):
        payloads.read_payload_table(path)


def test_unchanged_parts_are_not_rehashed_for_each_read(tmp_path, monkeypatch):
    path, _, _ = _bundle(tmp_path)
    original = hashlib.file_digest
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(hashlib, "file_digest", count)
    payloads.read_payload_table(path)
    list(payloads.iter_payload_batches(path))
    assert len(calls) == 1


def test_explicit_snapshot_reads_without_creating_lock(tmp_path):
    path, _, _ = _bundle(tmp_path)
    tmp_path.chmod(0o555)
    try:
        assert payloads.read_payload_table(path, read_mode="snapshot").num_rows == 1
        assert sum(b.num_rows for b in payloads.iter_payload_batches(path, read_mode="snapshot")) == 1
        assert not (tmp_path / ".sidecar.lock").exists()
    finally:
        tmp_path.chmod(0o755)


def test_snapshot_rejects_inventory_change_during_stream(tmp_path):
    path, _, document = _bundle(tmp_path)
    stream = payloads.iter_payload_batches(path, read_mode="snapshot")
    next(stream)
    document["parts"] = []
    payloads.manifest_path(path).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="changed during"):
        next(stream)


def test_invalid_read_mode_does_not_create_lock(tmp_path):
    path, _, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match="read_mode"):
        payloads.read_payload_table(path, read_mode="typo")
    assert not (tmp_path / ".sidecar.lock").exists()


def test_snapshot_rejects_manifest_publication_during_single_file_read(tmp_path):
    path = tmp_path / "feature_vectors.parquet"
    pq.write_table(_table(0.25), path)
    stream = payloads.iter_payload_batches(path, read_mode="snapshot")
    next(stream)
    _bundle(tmp_path)
    with pytest.raises(ValueError, match="changed during"):
        next(stream)
