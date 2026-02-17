"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_usr_writer.py

USR adapter tests for DenseGen output fidelity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dnadesign.densegen.src.adapters.outputs.usr_writer as usr_writer_module
import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.adapters.outputs.record import OutputRecord
from dnadesign.densegen.src.adapters.outputs.usr_writer import USRWriter
from dnadesign.densegen.src.artifacts.npz_store import ArtifactInfo
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _events_for_action(events_path: Path, action: str) -> list[dict]:
    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payloads = [json.loads(line) for line in lines]
    return [item for item in payloads if item.get("action") == action]


def test_usr_writer_writes_overlay_parts_with_typed_columns(tmp_path: Path) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
    )

    meta = output_meta(library_hash="demo", library_index=1)
    record = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(record)

    part_dir = root / "demo" / "_derived" / "densegen"
    parts = list(part_dir.glob("part-*.parquet"))
    assert len(parts) == 1

    schema = pq.ParquetFile(str(parts[0])).schema_arrow
    assert pa.types.is_list(schema.field("densegen__tf_list").type)
    assert pa.types.is_struct(schema.field("densegen__fixed_elements").type)


def test_usr_writer_offloads_npz_fields(tmp_path: Path) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        npz_fields=["used_tfbs_detail"],
    )

    meta = output_meta(library_hash="demo", library_index=1)
    record = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(record)

    part_dir = root / "demo" / "_derived" / "densegen"
    parts = list(part_dir.glob("part-*.parquet"))
    assert len(parts) == 1

    tbl = pq.read_table(parts[0])
    ref = tbl.column("densegen__npz_ref")[0].as_py()
    sha = tbl.column("densegen__npz_sha256")[0].as_py()
    size_bytes = tbl.column("densegen__npz_bytes")[0].as_py()
    fields = tbl.column("densegen__npz_fields")[0].as_py()
    assert ref
    assert isinstance(sha, str) and len(sha) == 64
    assert int(size_bytes) > 0
    assert "used_tfbs_detail" in fields
    assert tbl.column("densegen__used_tfbs_detail")[0].as_py() is None

    npz_path = root / "demo" / ref
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as data:
        assert "used_tfbs_detail" in data.files


def test_usr_writer_uses_atomic_npz_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        npz_fields=["used_tfbs_detail"],
    )

    called = {"count": 0}

    def _fake_write_npz_atomic(arrays: dict[str, np.ndarray], out_path: Path) -> ArtifactInfo:
        called["count"] += 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        size_bytes = int(out_path.stat().st_size)
        return ArtifactInfo(ref=str(out_path), sha256="a" * 64, bytes=size_bytes)

    monkeypatch.setattr(usr_writer_module, "write_npz_atomic", _fake_write_npz_atomic)

    meta = output_meta(library_hash="demo", library_index=1)
    record = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(record)
    assert called["count"] == 1


def test_usr_writer_emits_densegen_health_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    now = {"value": 0.0}
    monkeypatch.setattr(usr_writer_module.time, "monotonic", lambda: now["value"])
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        health_event_interval_seconds=60.0,
        run_quota=10,
    )

    meta = output_meta(library_hash="demo", library_index=1)
    record = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=meta,
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(record)

    health_events = _events_for_action(root / "demo" / ".events.log", "densegen_health")
    statuses = [str((event.get("args") or {}).get("status")) for event in health_events]
    assert statuses[:2] == ["started", "running"]
    running_events = [event for event in health_events if str((event.get("args") or {}).get("status")) == "running"]
    assert len(running_events) == 1
    event = running_events[0]
    assert event["metrics"]["rows_written_session"] == 1
    assert event["metrics"]["rows_incoming_session"] == 1
    assert event["metrics"]["run_quota"] == 10
    assert event["metrics"]["quota_progress_pct"] == pytest.approx(10.0)
    assert event["args"]["status"] == "running"


def test_usr_writer_health_events_are_time_throttled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    now = {"value": 0.0}
    monkeypatch.setattr(usr_writer_module.time, "monotonic", lambda: now["value"])
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        health_event_interval_seconds=60.0,
    )

    first = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    second = OutputRecord.from_sequence(
        sequence="ACGTACGTAC",
        meta=output_meta(library_hash="demo", library_index=2),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(first)
    now["value"] = 30.0
    assert writer.add(second)

    health_events = _events_for_action(root / "demo" / ".events.log", "densegen_health")
    statuses = [str((event.get("args") or {}).get("status")) for event in health_events]
    assert "started" in statuses
    running_events = [event for event in health_events if str((event.get("args") or {}).get("status")) == "running"]
    assert len(running_events) == 1


def test_usr_writer_finalize_emits_completed_health_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "usr"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    now = {"value": 0.0}
    monkeypatch.setattr(usr_writer_module.time, "monotonic", lambda: now["value"])
    writer = USRWriter(
        dataset="demo",
        root=root,
        namespace="densegen",
        chunk_size=1,
        health_event_interval_seconds=60.0,
    )
    record = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(record)
    now["value"] = 10.0
    writer.finalize()

    health_events = _events_for_action(root / "demo" / ".events.log", "densegen_health")
    statuses = [event["args"]["status"] for event in health_events]
    assert statuses[-1] == "completed"
