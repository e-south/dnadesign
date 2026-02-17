"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_usr_flush.py

Transactional flush tests for DenseGen USR output writing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import dnadesign.densegen.src.adapters.outputs.usr_writer as usr_writer_module
import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.adapters.outputs.record import OutputRecord
from dnadesign.densegen.src.adapters.outputs.usr_flush import DensegenUsrFlushError
from dnadesign.densegen.src.adapters.outputs.usr_writer import USRWriter
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _make_writer(tmp_path: Path) -> tuple[Path, USRWriter]:
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
    return root, writer


def _make_record() -> OutputRecord:
    return OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )


def test_usr_flush_writes_orphan_manifest_and_failure_event_on_overlay_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, writer = _make_writer(tmp_path)

    def _raise_overlay_failure(*_args, **_kwargs):
        raise RuntimeError("overlay write failed")

    monkeypatch.setattr(writer.ds, "write_overlay_part", _raise_overlay_failure)

    with pytest.raises(RuntimeError, match="overlay write failed"):
        writer.add(_make_record())

    part_dir = root / "demo" / "_derived" / "densegen"
    assert not list(part_dir.glob("part-*.parquet"))

    orphan_manifest = root / "demo" / "_artifacts" / "orphans.jsonl"
    assert orphan_manifest.exists()
    lines = [line for line in orphan_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    entry = json.loads(lines[-1])
    assert entry["npz_ref"]
    assert entry["run_id"] == "demo"

    events_path = root / "demo" / ".events.log"
    actions = [json.loads(line)["action"] for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert "densegen_flush_failed" in actions


def test_usr_flush_records_failure_event_on_npz_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, writer = _make_writer(tmp_path)

    def _raise_npz_failure(*_args, **_kwargs):
        raise OSError("npz write failed")

    monkeypatch.setattr(usr_writer_module, "write_npz_atomic", _raise_npz_failure)

    with pytest.raises(DensegenUsrFlushError, match="Failed to write NPZ artifact"):
        writer.add(_make_record())

    part_dir = root / "demo" / "_derived" / "densegen"
    assert not list(part_dir.glob("part-*.parquet"))

    events_path = root / "demo" / ".events.log"
    actions = [json.loads(line)["action"] for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert "densegen_flush_failed" in actions


def test_usr_flush_retries_overlay_after_partial_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, writer = _make_writer(tmp_path)
    original_write_overlay_part = writer.ds.write_overlay_part
    calls = {"n": 0}

    def _flaky_overlay_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("overlay write failed")
        return original_write_overlay_part(*args, **kwargs)

    monkeypatch.setattr(writer.ds, "write_overlay_part", _flaky_overlay_write)

    with pytest.raises(RuntimeError, match="overlay write failed"):
        writer.add(_make_record())

    # Base row is already imported, but overlay metadata must be replayable.
    assert writer.ds.head(1).shape[0] == 1

    writer.flush()

    part_dir = root / "demo" / "_derived" / "densegen"
    assert list(part_dir.glob("part-*.parquet"))
