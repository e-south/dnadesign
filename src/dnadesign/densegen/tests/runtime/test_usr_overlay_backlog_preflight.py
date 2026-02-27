"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_usr_overlay_backlog_preflight.py

Resume preflight tests for replaying pending USR overlay backlog parts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

import dnadesign.usr as usr_pkg
from dnadesign.densegen.src.adapters.outputs.record import OutputRecord
from dnadesign.densegen.src.adapters.outputs.usr_writer import USRWriter
from dnadesign.densegen.src.core.pipeline.orchestrator import _replay_usr_overlay_backlog_for_resume
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _copy_registry(usr_root: Path) -> None:
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = usr_root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")


def _densegen_cfg_stub(*, run_id: str = "demo", dataset: str = "demo_workspace"):
    return SimpleNamespace(
        run=SimpleNamespace(id=run_id),
        output=SimpleNamespace(
            targets=["usr"],
            usr=SimpleNamespace(
                root="outputs/usr",
                dataset=dataset,
                chunk_size=8,
            ),
        ),
    )


def test_resume_preflight_replays_pending_usr_overlay_backlog(tmp_path: Path, monkeypatch) -> None:
    usr_root = tmp_path / "outputs" / "usr"
    _copy_registry(usr_root)
    writer = USRWriter(
        dataset="demo_workspace",
        root=usr_root,
        namespace="densegen",
        chunk_size=8,
        run_id="demo",
    )
    rec = OutputRecord.from_sequence(
        sequence="ACGTACGTAA",
        meta=output_meta(library_hash="demo_hash", library_index=1),
        source="unit",
        bio_type="dna",
        alphabet="dna_4",
    )
    assert writer.add(rec) is True

    def _fail_overlay(*_args, **_kwargs):
        raise RuntimeError("simulated overlay write failure")

    monkeypatch.setattr(writer.ds, "write_overlay_part", _fail_overlay)
    with pytest.raises(RuntimeError, match="simulated overlay write failure"):
        writer.flush()

    dataset_dir = usr_root / "demo_workspace"
    backlog_root = dataset_dir / "_artifacts" / "pending_overlay"
    overlay_root = dataset_dir / "_derived" / "densegen"
    assert pq.read_table(dataset_dir / "records.parquet").num_rows == 1
    assert sorted(backlog_root.glob("part-*.parquet"))
    assert not sorted(overlay_root.glob("part-*.parquet"))

    replayed_parts = _replay_usr_overlay_backlog_for_resume(
        cfg=_densegen_cfg_stub(),
        cfg_path=tmp_path / "config.yaml",
        run_root=tmp_path,
    )

    assert replayed_parts == 1
    assert not sorted(backlog_root.glob("part-*.parquet"))
    assert sorted(overlay_root.glob("part-*.parquet"))


def test_resume_preflight_noops_without_pending_usr_overlay_backlog(tmp_path: Path) -> None:
    replayed_parts = _replay_usr_overlay_backlog_for_resume(
        cfg=_densegen_cfg_stub(),
        cfg_path=tmp_path / "config.yaml",
        run_root=tmp_path,
    )

    assert replayed_parts == 0
    assert not (tmp_path / "outputs" / "usr" / "demo_workspace").exists()


def test_resume_preflight_fails_when_backlog_exists_without_records(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "outputs" / "usr" / "demo_workspace"
    backlog_root = dataset_dir / "_artifacts" / "pending_overlay"
    backlog_root.mkdir(parents=True, exist_ok=True)
    (backlog_root / "part-1.parquet").write_text("pending", encoding="utf-8")
    _copy_registry(tmp_path / "outputs" / "usr")

    with pytest.raises(RuntimeError, match="records.parquet is missing"):
        _replay_usr_overlay_backlog_for_resume(
            cfg=_densegen_cfg_stub(),
            cfg_path=tmp_path / "config.yaml",
            run_root=tmp_path,
        )
