"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_outputs_explicit.py

Tests that only explicitly declared outputs are produced.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import subprocess
import sys
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


def _start_reservation_process(
    final: Path,
    staging: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    child_code = """
import signal
import sys
from pathlib import Path
from dnadesign.baserender.src.execution.runner import _PublicationTarget, _reserve_publication_targets

_reserve_publication_targets((_PublicationTarget(final=Path(sys.argv[1]), staging=Path(sys.argv[2])),))
print("ready", flush=True)
signal.pause()
"""
    return subprocess.Popen(
        [sys.executable, "-c", child_code, str(final), str(staging)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
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

    def _fail_second_publication(source, destination, **kwargs):
        source_path = Path(source)
        destination_path = Path(destination)
        if ".staging-" in source_path.name and destination_path.name == video_path.name:
            raise OSError("simulated publication failure")
        return real_replace(source, destination, **kwargs)

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


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process semantics")
def test_publication_reservation_is_released_when_worker_dies(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    final = tmp_path / "results" / "images"
    targets = (_PublicationTarget(final=final, staging=tmp_path / "staging"),)
    process = _start_reservation_process(final, tmp_path / "staging")
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(SchemaError, match="publication is already in progress"):
            _reserve_publication_targets(targets)

        process.kill()
        process.wait(timeout=10)

        locks = _reserve_publication_targets(targets)
        _release_publication_locks(locks)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process semantics")
@pytest.mark.parametrize(
    ("held_relative", "attempted_relative"),
    [("images", "images/plot.png"), ("images/plot.png", "images")],
)
def test_publication_reservation_rejects_concurrent_nested_destination(
    tmp_path: Path,
    held_relative: str,
    attempted_relative: str,
) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    results_root = tmp_path / "results"
    held_final = results_root / held_relative
    attempted_final = results_root / attempted_relative
    process = _start_reservation_process(held_final, tmp_path / "held-staging")

    def _reserve_nested_destination() -> None:
        locks = _reserve_publication_targets(
            (
                _PublicationTarget(
                    final=attempted_final,
                    staging=tmp_path / "attempted-staging",
                ),
            )
        )
        _release_publication_locks(locks)

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(SchemaError, match="publication is already in progress"):
            _reserve_nested_destination()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_publication_reservation_allows_concurrent_sibling_destinations(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    results_root = tmp_path / "results"
    left_locks = _reserve_publication_targets(
        (_PublicationTarget(final=results_root / "images", staging=tmp_path / "images-staging"),)
    )
    try:
        right_locks = _reserve_publication_targets(
            (_PublicationTarget(final=results_root / "video.mp4", staging=tmp_path / "video-staging"),)
        )
        _release_publication_locks(right_locks)
    finally:
        _release_publication_locks(left_locks)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process semantics")
def test_publication_reservation_is_shared_across_process_temp_directories(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    final = tmp_path / "results" / "images"
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    child_env = dict(os.environ)
    child_env["TMPDIR"] = str(child_temp)
    process = _start_reservation_process(final, tmp_path / "child-staging", env=child_env)

    def _reserve_same_destination() -> None:
        locks = _reserve_publication_targets((_PublicationTarget(final=final, staging=tmp_path / "parent-staging"),))
        _release_publication_locks(locks)

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(SchemaError, match="publication is already in progress"):
            _reserve_same_destination()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process semantics")
def test_publication_reservation_rejects_case_alias_on_insensitive_filesystem(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    probe = tmp_path / "CaseSensitivityProbe"
    probe.write_text("probe\n", encoding="utf-8")
    if not (tmp_path / "casesensitivityprobe").exists():
        pytest.skip("requires a case-insensitive destination filesystem")

    held_final = tmp_path / "results" / "Result.txt"
    attempted_final = tmp_path / "results" / "result.txt"
    process = _start_reservation_process(held_final, tmp_path / "held-staging")

    def _reserve_case_alias() -> None:
        locks = _reserve_publication_targets(
            (_PublicationTarget(final=attempted_final, staging=tmp_path / "attempted-staging"),)
        )
        _release_publication_locks(locks)

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(SchemaError, match="publication is already in progress"):
            _reserve_case_alias()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process semantics")
def test_publication_reservation_allows_case_distinct_siblings_on_sensitive_filesystem(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    probe = tmp_path / "CaseSensitivityProbe"
    probe.write_text("probe\n", encoding="utf-8")
    if (tmp_path / "casesensitivityprobe").exists():
        pytest.skip("requires a case-sensitive destination filesystem")

    results_root = tmp_path / "results"
    upper_locks = _reserve_publication_targets(
        (_PublicationTarget(final=results_root / "Result.txt", staging=tmp_path / "upper-staging"),)
    )
    try:
        lower_locks = _reserve_publication_targets(
            (_PublicationTarget(final=results_root / "result.txt", staging=tmp_path / "lower-staging"),)
        )
        _release_publication_locks(lower_locks)
    finally:
        _release_publication_locks(upper_locks)


def test_publication_does_not_retraverse_swapped_output_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.baserender.src.execution.runner import (
        _PublicationTarget,
        _release_publication_locks,
        _reserve_publication_targets,
    )

    parquet = _make_input_parquet(tmp_path)
    results_root = tmp_path / "results"
    results_root.mkdir()
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=results_root,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "ancestor-swap.yaml", payload)
    moved_root = tmp_path / "moved-results"
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    attacker_final = attacker_root / job_path.stem / "images"
    attacker_locks = _reserve_publication_targets(
        (_PublicationTarget(final=attacker_final, staging=tmp_path / "attacker-staging"),)
    )

    def _swap_then_write_images(records, *, output, renderer_name, style, palette):
        del records, renderer_name, style, palette
        results_root.rename(moved_root)
        results_root.symlink_to(attacker_root, target_is_directory=True)
        assert output.dir is not None
        output.dir.mkdir(parents=True, exist_ok=True)
        (output.dir / "rendered.png").write_bytes(b"redirected")
        return output.dir

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _swap_then_write_images)
    try:
        with pytest.raises(SchemaError, match="output parent changed during publication"):
            run_cruncher_showcase_job(str(job_path))
    finally:
        _release_publication_locks(attacker_locks)

    assert not attacker_final.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory descriptors")
@pytest.mark.parametrize("unsafe_kind", ["symlink", "mode", "owner"])
def test_publication_lock_registry_rejects_untrusted_preexisting_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    from dnadesign.baserender.src.execution import runner as runner_module

    registry_root = tmp_path / "registry-root"
    registry_root.mkdir()
    namespace = registry_root / f"dnadesign-baserender-locks-{os.getuid()}"
    if unsafe_kind == "symlink":
        target = tmp_path / "attacker-registry"
        target.mkdir()
        namespace.symlink_to(target, target_is_directory=True)
    else:
        namespace.mkdir(mode=0o700)
        if unsafe_kind == "mode":
            namespace.chmod(0o755)
        else:
            real_fstat = runner_module.os.fstat

            def _foreign_owner(descriptor: int):
                result = real_fstat(descriptor)
                values = list(result)
                values[4] = os.getuid() + 1
                return os.stat_result(values)

            monkeypatch.setattr(runner_module.os, "fstat", _foreign_owner)

    with pytest.raises(SchemaError, match="lock registry"):
        descriptor = runner_module._open_publication_lock_registry(root=registry_root)
        os.close(descriptor)
