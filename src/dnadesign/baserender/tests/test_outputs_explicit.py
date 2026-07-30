"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_outputs_explicit.py

Tests that only explicitly declared outputs are produced.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.animation as animation
import pytest

from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.public import run_cruncher_showcase_job

from .conftest import densegen_job_payload, write_job, write_parquet


def _make_input_parquet(tmp_path: Path) -> Path:
    return write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            }
        ],
    )


def test_images_output_does_not_produce_video(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"

    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "images_only.yaml", payload)

    report = run_cruncher_showcase_job(str(job_path))

    images_dir = Path(report.outputs["images_dir"])
    assert images_dir.exists()
    assert any(p.suffix == ".png" for p in images_dir.iterdir())

    video_path = results_root / job_path.stem / f"{job_path.stem}.mp4"
    assert not video_path.exists()


def test_video_output_does_not_produce_images(tmp_path: Path) -> None:
    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("FFmpeg not available")

    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"

    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 1}],
    )
    job_path = write_job(tmp_path / "video_only.yaml", payload)

    report = run_cruncher_showcase_job(str(job_path))

    video_path = Path(report.outputs["video_path"])
    assert video_path.exists()
    assert video_path.suffix == ".mp4"

    images_dir = results_root / job_path.stem / "images"
    assert not images_dir.exists()


def test_video_output_reports_planned_frame_count(monkeypatch, tmp_path: Path) -> None:
    from dnadesign.baserender.src.outputs import video as video_module

    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            },
            {
                "id": "r2",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row2",
            },
        ],
    )
    results_root = tmp_path / "results"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 3}],
    )
    job_path = write_job(tmp_path / "video_report.yaml", payload)

    def _fake_write_video(records, *, output, renderer_name, style, palette):
        del renderer_name, style, palette
        records = list(records)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(b"fake-mp4")
        assert video_module.planned_video_frame_count(records, output=output) == 6
        return output.path

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_video", _fake_write_video)

    report = run_cruncher_showcase_job(str(job_path))

    assert report.output_metrics["video"] == {
        "record_count": 2,
        "planned_frame_count": 6,
        "fps": 2,
        "frames_per_record": 3,
    }


def test_strict_skip_failure_happens_before_artifact_writes(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            },
            {
                "id": "missing_sequence",
                "sequence": None,
                "densegen__used_tfbs_detail": [],
                "details": "row2",
            },
        ],
    )
    results_root = tmp_path / "results"
    report_path = results_root / "strict_report.json"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
        extra={
            "run": {
                "strict": True,
                "fail_on_skips": True,
                "emit_report": True,
                "report_path": str(report_path),
            }
        },
    )
    job_path = write_job(tmp_path / "strict_skip.yaml", payload)

    with pytest.raises(SchemaError, match="strict mode is enabled"):
        run_cruncher_showcase_job(str(job_path))

    assert not (results_root / job_path.stem / "images").exists()
    assert not report_path.exists()


def test_existing_batch_output_fails_without_mutation_by_default(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "conflict.yaml", payload)
    images_dir = results_root / job_path.stem / "images"
    images_dir.mkdir(parents=True)
    sentinel = images_dir / "keep.txt"
    sentinel.write_text("original\n", encoding="utf-8")

    with pytest.raises(SchemaError, match="already exists.*conflict_policy.*replace"):
        run_cruncher_showcase_job(str(job_path))

    assert sentinel.read_text(encoding="utf-8") == "original\n"
    assert list(images_dir.iterdir()) == [sentinel]


def test_replace_conflict_policy_replaces_complete_batch_output(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
        extra={
            "run": {
                "strict": False,
                "fail_on_skips": False,
                "emit_report": False,
                "report_path": None,
                "conflict_policy": "replace",
            }
        },
    )
    job_path = write_job(tmp_path / "replace.yaml", payload)
    images_dir = results_root / job_path.stem / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "stale.txt").write_text("stale\n", encoding="utf-8")

    report = run_cruncher_showcase_job(str(job_path))

    assert Path(report.outputs["images_dir"]) == images_dir.resolve()
    assert not (images_dir / "stale.txt").exists()
    assert [path.name for path in images_dir.iterdir()] == ["r1.png"]


def test_output_failure_publishes_no_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    report_path = results_root / "partial_report.json"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[
            {"kind": "images", "fmt": "png"},
            {"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 1},
        ],
        extra={
            "run": {
                "strict": False,
                "fail_on_skips": False,
                "emit_report": True,
                "report_path": str(report_path),
            }
        },
    )
    job_path = write_job(tmp_path / "partial.yaml", payload)

    def _fail_video(*args, **kwargs):
        del args, kwargs
        raise SchemaError("simulated video failure")

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_video", _fail_video)

    with pytest.raises(SchemaError, match="simulated video failure"):
        run_cruncher_showcase_job(str(job_path))

    batch_root = results_root / job_path.stem
    assert not (batch_root / "images").exists()
    assert not (batch_root / f"{job_path.stem}.mp4").exists()
    assert not report_path.exists()
    assert not list(results_root.rglob("*.staging-*"))


def test_publication_failure_restores_replaced_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.baserender.src.execution import runner as runner_module

    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[
            {"kind": "images", "fmt": "png"},
            {"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 1},
        ],
        extra={
            "run": {
                "strict": False,
                "fail_on_skips": False,
                "emit_report": False,
                "report_path": None,
                "conflict_policy": "replace",
            }
        },
    )
    job_path = write_job(tmp_path / "rollback.yaml", payload)
    batch_root = results_root / job_path.stem
    images_dir = batch_root / "images"
    video_path = batch_root / f"{job_path.stem}.mp4"
    images_dir.mkdir(parents=True)
    (images_dir / "original.txt").write_text("original images\n", encoding="utf-8")
    video_path.write_bytes(b"original video")

    def _fake_write_video(records, *, output, renderer_name, style, palette):
        del records, renderer_name, style, palette
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(b"replacement video")
        return output.path

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_video", _fake_write_video)
    real_replace = runner_module.os.replace

    def _fail_second_publication(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if ".staging-" in source_path.name and destination_path == video_path.resolve():
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runner_module.os, "replace", _fail_second_publication)

    with pytest.raises(OSError, match="simulated publication failure"):
        run_cruncher_showcase_job(str(job_path))

    assert (images_dir / "original.txt").read_text(encoding="utf-8") == "original images\n"
    assert video_path.read_bytes() == b"original video"
    assert not list(batch_root.glob(".*.staging-*"))
    assert not list(batch_root.glob(".*.backup-*"))


def test_concurrent_error_policy_job_cannot_overwrite_reserved_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "concurrent.yaml", payload)
    first_writer_entered = threading.Event()
    release_first_writer = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def _controlled_write_images(records, *, output, renderer_name, style, palette):
        nonlocal call_count
        del records, renderer_name, style, palette
        with call_lock:
            call_count += 1
            call_index = call_count
        if call_index == 1:
            first_writer_entered.set()
            assert release_first_writer.wait(timeout=10)
        assert output.dir is not None
        output.dir.mkdir(parents=True, exist_ok=True)
        (output.dir / "rendered.png").write_bytes(f"batch-{call_index}".encode())
        return output.dir

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _controlled_write_images)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_run = executor.submit(run_cruncher_showcase_job, str(job_path))
        assert first_writer_entered.wait(timeout=10)
        try:
            with pytest.raises(SchemaError, match="publication is already in progress"):
                run_cruncher_showcase_job(str(job_path))
        finally:
            release_first_writer.set()
        first_run.result(timeout=10)

    images_dir = results_root / job_path.stem / "images"
    assert (images_dir / "rendered.png").read_bytes() == b"batch-1"
    assert not list(images_dir.parent.glob(".*.publication.lock"))
