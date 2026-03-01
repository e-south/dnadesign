"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_video_boundaries.py

Characterization tests for trajectory-video timeline and render contract seams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dnadesign.cruncher.analysis import trajectory_video
from dnadesign.cruncher.analysis.trajectory_video_contract import build_sequence_rows_video_job
from dnadesign.cruncher.analysis.trajectory_video_timeline import (
    sample_frame_indices,
    source_indices_for_best_so_far_timeline,
)
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig


def test_timeline_sampling_retains_endpoints_and_monotonic_source_progression() -> None:
    objective = np.asarray([0.1, 0.25, 0.2, 0.6, 0.55, 0.9], dtype=float)
    sampled_indices, effective_stride, _ = sample_frame_indices(
        objective_values=objective,
        sampling_stride=2,
        include_best_updates=True,
        snapshot_budget=4,
    )
    assert sampled_indices[0] == 0
    assert sampled_indices[-1] == len(objective) - 1
    assert effective_stride >= 2

    source_indices = source_indices_for_best_so_far_timeline(
        objective_values=objective,
        sampled_indices=sampled_indices,
    )
    assert len(source_indices) == len(sampled_indices)
    assert source_indices == sorted(source_indices)


def test_sequence_rows_video_job_contract_is_strict_and_explicit(tmp_path: Path) -> None:
    cfg = AnalysisTrajectoryVideoConfig(
        playback={"target_duration_sec": 6.0, "fps": 10},
        sweep_inset={"enabled": True, "corner": "bottom_left"},
    )
    job = build_sequence_rows_video_job(
        records_path=tmp_path / "records.parquet",
        out_path=tmp_path / "plots" / "chain_trajectory_video.mp4",
        config=cfg,
        pauses={"chain_1_frame_0001": 0.2},
        title_text="Best-so-far motif placement improves over sweeps",
    )

    assert int(job["version"]) == 3
    assert str(job["input"]["kind"]) == "parquet"
    assert str(job["input"]["adapter"]["kind"]) == "generic_features"
    assert str(job["render"]["renderer"]) == "sequence_rows"
    assert bool(job["run"]["strict"]) is True
    assert bool(job["run"]["fail_on_skips"]) is True
    output_cfg = dict(job["outputs"][0])
    assert str(output_cfg["fmt"]) == "mp4"
    assert int(output_cfg["fps"]) == 10
    assert float(output_cfg["total_duration"]) == 6.0
    assert int(output_cfg["height_px"]) == 820
    assert str(output_cfg["title_text"]) == "Best-so-far motif placement improves over sweeps"
    assert str(output_cfg["title_align"]) == "center"
    style_cfg = dict(job["render"]["style"]["overrides"])
    assert bool(style_cfg["show_reverse_complement"]) is True


def test_trajectory_video_public_exports_stay_minimal() -> None:
    exports = set(getattr(trajectory_video, "__all__", []))
    assert exports == {"render_chain_trajectory_video"}
