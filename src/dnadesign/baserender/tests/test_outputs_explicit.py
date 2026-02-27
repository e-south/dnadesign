"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_outputs_explicit.py

Tests that only explicitly declared outputs are produced.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import pytest

from dnadesign.baserender.src.api import run_cruncher_showcase_job

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
