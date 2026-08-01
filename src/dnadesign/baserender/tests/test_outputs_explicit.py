"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_outputs_explicit.py

Tests for immutable, atomic render-bundle publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.animation as animation
import pytest

from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.execution.runner import _prepare_bundle_publication, _publish_bundle
from dnadesign.baserender.src.public import run_render_job

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


def _fake_image_writer(records, *, output, renderer_name, style, palette):
    del records, renderer_name, style, palette
    assert output.dir is not None
    output.dir.mkdir(parents=True, exist_ok=True)
    (output.dir / "rendered.png").write_bytes(b"rendered")
    return output.dir


def test_images_output_publishes_one_manifested_bundle(tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "images_only.yaml", payload)

    report = run_render_job(str(job_path))

    images_dir = Path(report.outputs["images_dir"])
    assert images_dir == bundle / "images"
    assert any(path.suffix == ".png" for path in images_dir.iterdir())
    assert Path(report.outputs["bundle_root"]) == bundle
    assert Path(report.outputs["manifest_path"]) == bundle / "manifest.json"
    assert (bundle / "manifest.json").is_file()
    assert not (bundle / f"{job_path.stem}.mp4").exists()
    manifest_text = (bundle / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert str(tmp_path) not in manifest_text
    assert manifest["schema"] == "dnadesign.baserender.render_bundle_manifest.v1"
    artifacts = manifest["artifact_inventory"]["artifacts"]
    assert {item["path"] for item in artifacts} == {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in artifacts)


def test_portable_manifest_fails_when_source_evidence_is_unavailable(tmp_path: Path) -> None:
    from dnadesign.baserender.src.reporting import RunReport

    bundle = tmp_path / "render-v1"
    staging = tmp_path / "stage"
    staging.mkdir()
    report = RunReport(job_name="missing-source", input_path=str(tmp_path / "missing.parquet"), selection_path=None)

    with pytest.raises(ValueError, match="Render source is unavailable or unsafe"):
        report.write_portable_manifest(staging / "manifest.json", bundle_root=bundle, staging_root=staging)

    assert not (staging / "manifest.json").exists()


def test_publication_stage_is_private_and_final_bundle_has_deliberate_public_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.artifacts import publication as publication_module

    bundle = tmp_path / "results" / "render-v1"
    publication = _prepare_bundle_publication(bundle)
    observed_modes: dict[str, int] = {}
    real_rename = publication_module._rename_create_only
    try:
        nested = publication.stage / "images"
        nested.mkdir(mode=0o755)
        artifact = nested / "rendered.png"
        artifact.write_bytes(b"rendered")
        artifact.chmod(0o644)
        manifest = publication.stage / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        manifest.chmod(0o640)

        def _inspect_then_rename(parent_descriptor: int, source: str, destination: str) -> None:
            adjacent = bundle.parent / source
            observed_modes["root"] = stat.S_IMODE(adjacent.stat().st_mode)
            observed_modes["directory"] = stat.S_IMODE((adjacent / "images").stat().st_mode)
            observed_modes["artifact"] = stat.S_IMODE((adjacent / "images" / "rendered.png").stat().st_mode)
            observed_modes["manifest"] = stat.S_IMODE((adjacent / "manifest.json").stat().st_mode)
            real_rename(parent_descriptor, source, destination)

        monkeypatch.setattr(publication_module, "_rename_create_only", _inspect_then_rename)
        _publish_bundle(publication)
    finally:
        publication.close()

    assert observed_modes == {"root": 0o700, "directory": 0o700, "artifact": 0o600, "manifest": 0o600}
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o755
    assert stat.S_IMODE((bundle / "images").stat().st_mode) == 0o755
    assert stat.S_IMODE((bundle / "images" / "rendered.png").stat().st_mode) == 0o644
    assert stat.S_IMODE((bundle / "manifest.json").stat().st_mode) == 0o640


@pytest.mark.parametrize("required_manifest", ["/tmp/manifest.json", "../manifest.json", "nested/../../manifest.json"])
def test_publication_rejects_required_manifest_outside_stage_before_copy(
    tmp_path: Path,
    required_manifest: str,
) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(PublicationError, match="required manifest must stay inside publication staging"):
            publication.publish(required_manifest=required_manifest)
    finally:
        publication.close()

    assert not bundle.exists()
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


def test_publication_rejects_required_manifest_through_stage_symlink(tmp_path: Path) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "manifest.json").write_text("{}\n", encoding="utf-8")
    publication = CreateOnlyDirectoryPublication.prepare(tmp_path / "results" / "render-v1")
    try:
        (publication.stage / "redirect").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PublicationError, match="required manifest must stay inside publication staging"):
            publication.publish(required_manifest="redirect/manifest.json")
    finally:
        publication.close()


@pytest.mark.parametrize(
    "alias",
    ["MANIFEST.JSON", ".DNADESIGN-PUBLICATION-OWNER.JSON"],
)
def test_publication_rejects_portable_metadata_aliases(tmp_path: Path, alias: str) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

    publication = CreateOnlyDirectoryPublication.prepare(tmp_path / "results" / "render-v1")
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        (publication.stage / alias).write_text("alias\n", encoding="utf-8")

        required_manifest = alias if alias == "MANIFEST.JSON" else "manifest.json"
        with pytest.raises(PublicationError, match="portable filesystem identity|owner sentinel"):
            publication.publish(required_manifest=required_manifest)
    finally:
        publication.close()


def test_post_rename_failure_never_removes_replacement_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication
    from dnadesign.artifacts import publication as publication_module

    bundle = tmp_path / "results" / "render-v1"
    displaced = tmp_path / "results" / "displaced-owned-publication"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")

    def _replace_then_fail(source: Path, destination_descriptor: int) -> None:
        del source, destination_descriptor
        bundle.rename(displaced)
        bundle.mkdir()
        (bundle / "keep.txt").write_text("replacement\n", encoding="utf-8")
        raise OSError("simulated post-rename failure")

    monkeypatch.setattr(publication_module, "_restore_published_modes", _replace_then_fail)
    try:
        with pytest.raises(OSError, match="simulated post-rename failure"):
            publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()

    assert (bundle / "keep.txt").read_text(encoding="utf-8") == "replacement\n"
    assert displaced.is_dir()


def test_post_rename_failure_removes_only_the_proven_owned_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication
    from dnadesign.artifacts import publication as publication_module

    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")

    def _fail_after_rename(source: Path, destination_descriptor: int) -> None:
        del source, destination_descriptor
        raise OSError("simulated owned post-rename failure")

    monkeypatch.setattr(publication_module, "_restore_published_modes", _fail_after_rename)
    try:
        with pytest.raises(OSError, match="simulated owned post-rename failure"):
            publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()

    assert not bundle.exists()
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


def test_publication_rejects_symlinked_parent_without_creating_target_tree(tmp_path: Path) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

    outside = tmp_path / "outside"
    outside.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(outside, target_is_directory=True)
    requested = redirected / "new-parent" / "render-v1"

    with pytest.raises(PublicationError, match="symlinked path component"):
        CreateOnlyDirectoryPublication.prepare(requested)

    assert not (outside / "new-parent").exists()


def test_publication_rejects_parent_traversal_before_creating_target_tree(tmp_path: Path) -> None:
    from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError

    requested = tmp_path / "uncreated" / ".." / "redirected" / "render-v1"

    with pytest.raises(PublicationError, match="must not contain parent traversal"):
        CreateOnlyDirectoryPublication.prepare(requested)

    assert not (tmp_path / "uncreated").exists()
    assert not (tmp_path / "redirected").exists()


@pytest.mark.parametrize(
    "outputs, message",
    [
        ([{"kind": "images", "dir": "manifest.json/images", "fmt": "png"}], "beneath the bundle manifest"),
        (
            [
                {"kind": "images", "dir": "artifacts", "fmt": "png"},
                {"kind": "video", "path": "artifacts/movie.mp4", "fmt": "mp4"},
            ],
            "prefix collision",
        ),
    ],
)
def test_impossible_output_topologies_fail_before_rendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outputs: list[dict[str, object]],
    message: str,
) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    job_path = write_job(
        tmp_path / "invalid-topology.yaml",
        densegen_job_payload(parquet_path=parquet, bundle_path=bundle, outputs=outputs),
    )
    writer_called = False

    def _unexpected_writer(*args, **kwargs):
        nonlocal writer_called
        writer_called = True
        del args, kwargs

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _unexpected_writer)
    with pytest.raises(SchemaError, match=message):
        run_render_job(str(job_path))

    assert not writer_called
    assert not bundle.parent.exists()


def test_video_output_does_not_produce_images(tmp_path: Path) -> None:
    if not animation.writers.is_available("ffmpeg"):
        pytest.skip("FFmpeg not available")
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 1}],
    )
    job_path = write_job(tmp_path / "video_only.yaml", payload)

    report = run_render_job(str(job_path))

    video_path = Path(report.outputs["video_path"])
    assert video_path == bundle / "video_only.mp4"
    assert video_path.is_file()
    assert not (bundle / "images").exists()
    assert (bundle / "manifest.json").is_file()


def test_video_output_reports_planned_frame_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dnadesign.baserender.src.outputs import video as video_module

    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": identifier,
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": identifier,
            }
            for identifier in ("r1", "r2")
        ],
    )
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 3}],
    )
    job_path = write_job(tmp_path / "video_report.yaml", payload)

    def _fake_write_video(records, *, output, renderer_name, style, palette):
        del renderer_name, style, palette
        materialized = list(records)
        output.path.parent.mkdir(parents=True, exist_ok=True)
        output.path.write_bytes(b"fake-mp4")
        assert video_module.planned_video_frame_count(materialized, output=output) == 6
        return output.path

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_video", _fake_write_video)
    report = run_render_job(str(job_path))

    assert report.output_metrics["video"] == {
        "record_count": 2,
        "planned_frame_count": 6,
        "fps": 2,
        "frames_per_record": 3,
    }


def test_strict_skip_failure_happens_before_bundle_creation(tmp_path: Path) -> None:
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
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "images", "fmt": "png"}],
        extra={"run": {"strict": True, "fail_on_skips": True}},
    )
    job_path = write_job(tmp_path / "strict_skip.yaml", payload)

    with pytest.raises(SchemaError, match="strict mode is enabled"):
        run_render_job(str(job_path))

    assert not bundle.exists()
    assert not bundle.parent.exists()


def test_non_strict_empty_input_fails_before_bundle_parent_creation(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "missing_sequence",
                "sequence": None,
                "densegen__used_tfbs_detail": [],
                "details": "row",
            }
        ],
    )
    bundle = tmp_path / "results" / "render-v1"
    job_path = write_job(
        tmp_path / "empty.yaml",
        densegen_job_payload(
            parquet_path=parquet,
            bundle_path=bundle,
            outputs=[{"kind": "images", "fmt": "png"}],
        ),
    )

    with pytest.raises(SchemaError, match="No records to render"):
        run_render_job(str(job_path))

    assert not bundle.parent.exists()


def test_source_mutation_during_render_fails_closed_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    job_path = write_job(
        tmp_path / "source-mutation.yaml",
        densegen_job_payload(
            parquet_path=parquet,
            bundle_path=bundle,
            outputs=[{"kind": "images", "fmt": "png"}],
        ),
    )

    def _mutating_writer(records, *, output, renderer_name, style, palette):
        _fake_image_writer(records, output=output, renderer_name=renderer_name, style=style, palette=palette)
        parquet.write_bytes(b"changed after records were read")

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _mutating_writer)

    with pytest.raises(ValueError, match="Render source changed during execution"):
        run_render_job(str(job_path))

    assert not bundle.exists()


@pytest.mark.parametrize("preexisting", ["empty", "nonempty"])
def test_existing_bundle_is_immutable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preexisting: str) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    bundle.mkdir(parents=True)
    sentinel = bundle / "keep.txt"
    if preexisting == "nonempty":
        sentinel.write_text("original\n", encoding="utf-8")
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "conflict.yaml", payload)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _fake_image_writer)

    with pytest.raises(SchemaError, match="already exists and is immutable"):
        run_render_job(str(job_path))

    assert bundle.is_dir()
    assert list(bundle.iterdir()) == ([sentinel] if preexisting == "nonempty" else [])


def test_output_failure_leaves_no_final_or_adjacent_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[
            {"kind": "images", "fmt": "png"},
            {"kind": "video", "fmt": "mp4", "fps": 2, "frames_per_record": 1},
        ],
    )
    job_path = write_job(tmp_path / "partial.yaml", payload)

    def _fail_video(*args, **kwargs):
        del args, kwargs
        raise SchemaError("simulated video failure")

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_video", _fail_video)
    with pytest.raises(SchemaError, match="simulated video failure"):
        run_render_job(str(job_path))

    assert not bundle.exists()
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


def test_copy_failure_removes_adjacent_stage_and_never_exposes_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dnadesign.artifacts import publication as publication_module

    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "copy-failure.yaml", payload)
    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _fake_image_writer)
    real_copy = publication_module._copy_directory

    def _copy_then_fail(source: Path, parent_descriptor: int, name: str) -> None:
        real_copy(source, parent_descriptor, name)
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(publication_module, "_copy_directory", _copy_then_fail)
    with pytest.raises(OSError, match="simulated copy interruption"):
        run_render_job(str(job_path))

    assert not bundle.exists()
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


def test_concurrent_runs_publish_exactly_one_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    payload = densegen_job_payload(
        parquet_path=parquet,
        bundle_path=bundle,
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "concurrent.yaml", payload)
    barrier = threading.Barrier(2)
    call_lock = threading.Lock()
    call_count = 0

    def _concurrent_writer(records, *, output, renderer_name, style, palette):
        nonlocal call_count
        del records, renderer_name, style, palette
        with call_lock:
            call_count += 1
            index = call_count
        assert output.dir is not None
        output.dir.mkdir(parents=True, exist_ok=True)
        (output.dir / "rendered.png").write_bytes(f"batch-{index}".encode())
        barrier.wait(timeout=10)
        return output.dir

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _concurrent_writer)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_render_job, str(job_path)) for _ in range(2)]
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=15))
            except Exception as exc:  # noqa: BLE001 - assertions inspect the concurrent outcome
                outcomes.append(exc)

    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(failures) == 1
    assert isinstance(failures[0], SchemaError)
    assert "already exists and is immutable" in str(failures[0])
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "images" / "rendered.png").read_bytes() in {b"batch-1", b"batch-2"}
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX atomic directory publication")
def test_create_only_publication_is_shared_across_process_temp_directories(tmp_path: Path) -> None:
    bundle = tmp_path / "shared" / "render-v1"
    release = tmp_path / "release"
    child_code = """
import os
import shutil
import sys
import time
from pathlib import Path
from dnadesign.baserender.src.execution.runner import _prepare_bundle_publication, _publish_bundle

publication = _prepare_bundle_publication(Path(sys.argv[1]))
try:
    (publication.stage / "manifest.json").write_text(sys.argv[2], encoding="utf-8")
    Path(sys.argv[3]).write_text("ready", encoding="utf-8")
    while not Path(sys.argv[4]).exists():
        time.sleep(0.01)
    try:
        _publish_bundle(publication)
    except Exception as exc:
        print(f"error:{type(exc).__name__}:{exc}")
    else:
        print("ok")
finally:
    shutil.rmtree(publication.stage, ignore_errors=True)
    os.close(publication.parent_descriptor)
"""
    processes: list[subprocess.Popen[str]] = []
    ready_paths: list[Path] = []
    for index in range(2):
        private_temp = tmp_path / f"temp-{index}"
        private_temp.mkdir()
        ready = tmp_path / f"ready-{index}"
        ready_paths.append(ready)
        env = dict(os.environ)
        env["TMPDIR"] = str(private_temp)
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", child_code, str(bundle), f"batch-{index}", str(ready), str(release)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        )
    try:
        for _ in range(500):
            if all(path.exists() for path in ready_paths):
                break
            time.sleep(0.01)
        assert all(path.exists() for path in ready_paths)
        release.write_text("publish", encoding="utf-8")
        outcomes = [process.communicate(timeout=15) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    stdout = [result[0].strip() for result in outcomes]
    assert stdout.count("ok") == 1
    assert sum(line.startswith("error:SchemaError:") for line in stdout) == 1
    assert (bundle / "manifest.json").read_text(encoding="utf-8") in {"batch-0", "batch-1"}
    assert not list(bundle.parent.glob(".render-v1.staging-*"))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process termination")
def test_killed_copy_never_exposes_final_and_stale_stage_cannot_block_retry(tmp_path: Path) -> None:
    from dnadesign.baserender.src.execution.runner import _prepare_bundle_publication, _publish_bundle

    bundle = tmp_path / "results" / "render-v1"
    child_code = """
import os
import signal
import sys
from pathlib import Path
from dnadesign.artifacts.publication import _copy_directory
from dnadesign.baserender.src.execution.runner import _prepare_bundle_publication

publication = _prepare_bundle_publication(Path(sys.argv[1]))
(publication.stage / "manifest.json").write_text("interrupted", encoding="utf-8")
_copy_directory(
    publication.stage,
    publication.parent_descriptor,
    publication.adjacent_stage_name,
)
os.unlink(
    f"{publication.adjacent_stage_name}/.dnadesign-publication-owner.json",
    dir_fd=publication.parent_descriptor,
)
print("copied", flush=True)
signal.pause()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(bundle)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "copied"
        process.kill()
        process.wait(timeout=10)
        stale = list(bundle.parent.glob(".render-v1.staging-*"))
        assert len(stale) == 1
        assert not bundle.exists()

        retry = _prepare_bundle_publication(bundle)
        try:
            (retry.stage / "manifest.json").write_text("retry", encoding="utf-8")
            _publish_bundle(retry)
        finally:
            shutil.rmtree(retry.stage, ignore_errors=True)
            os.close(retry.parent_descriptor)

        assert (bundle / "manifest.json").read_text(encoding="utf-8") == "retry"
        assert not stale[0].exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for path in bundle.parent.glob(".render-v1.staging-*"):
            shutil.rmtree(path)


def test_unrelated_stale_temp_stage_is_not_reused_or_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private_temp = tmp_path / "private-temp"
    private_temp.mkdir()
    stale = private_temp / "dnadesign-baserender-stage-stale"
    stale.mkdir()
    (stale / "attacker.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setenv("TMPDIR", str(private_temp))
    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _fake_image_writer)
    parquet = _make_input_parquet(tmp_path)
    bundle = tmp_path / "results" / "render-v1"
    job_path = write_job(
        tmp_path / "stale.yaml",
        densegen_job_payload(
            parquet_path=parquet,
            bundle_path=bundle,
            outputs=[{"kind": "images", "fmt": "png"}],
        ),
    )

    run_render_job(str(job_path))

    assert (stale / "attacker.txt").read_text(encoding="utf-8") == "stale"
    assert (bundle / "manifest.json").is_file()


def test_publication_rejects_output_ancestor_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet = _make_input_parquet(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    bundle = results / "render-v1"
    moved_results = tmp_path / "moved-results"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    job_path = write_job(
        tmp_path / "ancestor-swap.yaml",
        densegen_job_payload(
            parquet_path=parquet,
            bundle_path=bundle,
            outputs=[{"kind": "images", "fmt": "png"}],
        ),
    )

    def _swap_then_write(records, *, output, renderer_name, style, palette):
        del records, renderer_name, style, palette
        results.rename(moved_results)
        results.symlink_to(attacker, target_is_directory=True)
        assert output.dir is not None
        output.dir.mkdir(parents=True, exist_ok=True)
        (output.dir / "rendered.png").write_bytes(b"rendered")
        return output.dir

    monkeypatch.setattr("dnadesign.baserender.src.outputs.write_images", _swap_then_write)
    with pytest.raises(SchemaError, match="bundle parent changed during publication"):
        run_render_job(str(job_path))

    assert not (attacker / "render-v1").exists()
    assert not (moved_results / "render-v1").exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX case behavior")
def test_case_aliases_obey_destination_filesystem_semantics(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    upper = _prepare_bundle_publication(parent / "Render-v1")
    lower = _prepare_bundle_publication(parent / "render-v1")
    try:
        for publication, marker in ((upper, "upper"), (lower, "lower")):
            (publication.stage / "manifest.json").write_text(marker, encoding="utf-8")
        _publish_bundle(upper)
        if (parent / "render-v1").exists():
            with pytest.raises(SchemaError, match="already exists and is immutable"):
                _publish_bundle(lower)
        else:
            _publish_bundle(lower)
            assert (parent / "Render-v1" / "manifest.json").read_text(encoding="utf-8") == "upper"
            assert (parent / "render-v1" / "manifest.json").read_text(encoding="utf-8") == "lower"
    finally:
        for publication in (upper, lower):
            if publication.stage.exists():
                shutil.rmtree(publication.stage)
            os.close(publication.parent_descriptor)
