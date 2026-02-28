"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_video.py

Unit tests for chain trajectory video selection and timeline sampling helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.trajectory_video import render_chain_trajectory_video
from dnadesign.cruncher.analysis.trajectory_video_timeline import (
    allocate_taper_extra_frames,
    filter_rows_by_phase_scope,
    sample_frame_indices,
    select_chain_rows,
    source_indices_for_best_so_far_timeline,
)
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig, AnalysisTrajectoryVideoSelectionConfig
from dnadesign.cruncher.core.pwm import PWM


def test_best_so_far_source_indices_are_monotonic() -> None:
    objective = np.asarray([0.1, 0.05, 0.4, 0.2, 0.9], dtype=float)
    sampled = [0, 1, 2, 3, 4]
    source = source_indices_for_best_so_far_timeline(objective_values=objective, sampled_indices=sampled)
    assert source == [0, 0, 2, 2, 4]


def test_sample_indices_respect_budget_and_endpoints() -> None:
    objective = np.asarray([0.1, 0.2, 0.15, 0.5, 0.4, 0.6, 0.55, 0.9], dtype=float)
    selected, effective_stride, best_updates = sample_frame_indices(
        objective_values=objective,
        sampling_stride=2,
        include_best_updates=True,
        snapshot_budget=4,
    )
    assert len(selected) == 4
    assert selected[0] == 0
    assert selected[-1] == len(objective) - 1
    assert effective_stride >= 2
    assert best_updates[0] == 0
    assert best_updates[-1] == len(objective) - 1


def test_linear_taper_extra_frames_prefers_later_frames() -> None:
    extras = allocate_taper_extra_frames(point_count=6, total_extra_frames=9)
    assert len(extras) == 6
    assert sum(extras) == 9
    assert extras[-1] >= extras[0]
    assert extras == sorted(extras)


def test_phase_filter_prefers_tune_and_draw_when_tune_exists() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0, 0],
            "sweep": [0, 1, 2],
            "phase": ["tune", "draw", "cooldown"],
            "objective_scalar": [0.1, 0.2, 0.05],
        }
    )
    filtered = filter_rows_by_phase_scope(frame, phase_scope="tune_and_draw_if_present")
    assert filtered["phase"].tolist() == ["tune", "draw"]


def test_phase_filter_if_present_fails_without_tune_or_draw() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep": [0, 1],
            "phase": ["cooldown", "anneal"],
            "objective_scalar": [0.1, 0.2],
        }
    )
    with np.testing.assert_raises_regex(
        ValueError,
        "no tune/draw rows",
    ):
        filter_rows_by_phase_scope(frame, phase_scope="tune_and_draw_if_present")


def test_phase_filter_required_fails_without_tune() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep": [0, 1],
            "phase": ["draw", "draw"],
            "objective_scalar": [0.1, 0.2],
        }
    )
    with np.testing.assert_raises_regex(
        ValueError,
        "requires tune rows",
    ):
        filter_rows_by_phase_scope(frame, phase_scope="tune_and_draw_required")


def test_select_chain_rows_uses_best_chain_by_default() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "objective_scalar": [0.2, 0.3, 0.4, 1.1],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig()
    selected, chain_idx = select_chain_rows(frame, config=cfg)
    assert chain_idx == 1
    assert selected["chain"].tolist() == [1, 1]


def test_select_chain_rows_explicit_chain_policy() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep": [0, 1, 0, 1],
            "phase": ["draw", "draw", "draw", "draw"],
            "objective_scalar": [0.2, 0.3, 0.4, 1.1],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig(
        selection=AnalysisTrajectoryVideoSelectionConfig(chain_policy="explicit", explicit_chain_1based=1)
    )
    selected, chain_idx = select_chain_rows(frame, config=cfg)
    assert chain_idx == 0
    assert selected["chain"].tolist() == [0, 0]


def test_select_chain_rows_rejects_negative_chain_or_sweep() -> None:
    frame = pd.DataFrame(
        {
            "chain": [-1, 0],
            "sweep": [0, -2],
            "phase": ["draw", "draw"],
            "objective_scalar": [0.2, 0.3],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig()
    with np.testing.assert_raises_regex(
        ValueError,
        "negative chain/sweep",
    ):
        select_chain_rows(frame, config=cfg)


def test_select_chain_rows_explicit_chain_must_exist() -> None:
    frame = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep": [0, 1],
            "phase": ["draw", "draw"],
            "objective_scalar": [0.2, 0.3],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig(
        selection=AnalysisTrajectoryVideoSelectionConfig(chain_policy="explicit", explicit_chain_1based=2)
    )
    with np.testing.assert_raises_regex(
        ValueError,
        "explicit chain 2 was not found",
    ):
        select_chain_rows(frame, config=cfg)


def test_render_chain_trajectory_video_uses_strict_baserender_run_contract(tmp_path, monkeypatch) -> None:
    pwm = PWM(
        name="lexA",
        matrix=np.asarray(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ],
            dtype=float,
        ),
    )
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0],
            "sweep": [0, 1, 2],
            "phase": ["draw", "draw", "draw"],
            "objective_scalar": [0.2, 0.4, 0.6],
            "sequence": ["ACGTACGT", "ACGTTCGT", "ACGTTTGT"],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig(
        playback={"target_duration_sec": 4.0, "fps": 8},
        limits={"max_total_frames": 32, "max_snapshots": 16, "max_estimated_render_sec": 30.0},
    )
    out_path = tmp_path / "video.mp4"
    captured: dict[str, object] = {}

    def _fake_run_job(job_mapping: dict[str, object], *, kind: str, caller_root: Path) -> None:
        captured["job"] = job_mapping
        captured["kind"] = kind
        captured["caller_root"] = caller_root
        captured["records_df"] = pd.read_parquet(Path(str(job_mapping["input"]["path"])))
        out = Path(str(job_mapping["outputs"][0]["path"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp4")

    monkeypatch.setattr("dnadesign.cruncher.analysis.trajectory_video.run_job", _fake_run_job)
    result = render_chain_trajectory_video(
        trajectory_df=trajectory_df,
        tf_names=["lexA"],
        pwms={"lexA": pwm},
        out_path=out_path,
        config=cfg,
        bidirectional=True,
        pwm_pseudocounts=0.0,
        log_odds_clip=None,
        tmp_root=tmp_path / "_tmp",
    )

    assert out_path.exists()
    assert result["snapshot_count"] >= 2
    assert captured["kind"] == "sequence_rows_v3"
    run_cfg = dict(captured["job"]["run"])
    assert run_cfg["strict"] is True
    assert run_cfg["fail_on_skips"] is True
    render_style = dict(captured["job"]["render"]["style"]["overrides"])
    assert render_style["font_size_label"] == 15
    assert render_style["show_reverse_complement"] is True

    output_cfg = dict(captured["job"]["outputs"][0])
    assert "width_px" not in output_cfg
    assert "aspect" not in output_cfg
    assert int(output_cfg["height_px"]) == 820
    assert str(output_cfg["title_text"]) == "Best-so-far motif placement improves over sweeps"
    assert int(output_cfg["title_font_size"]) == 12
    assert str(output_cfg["title_align"]) == "center"
    pauses = dict(output_cfg["pauses"])
    assert pauses
    assert float(pauses.get("chain_1_frame_0003", 0.0)) > float(pauses.get("chain_1_frame_0001", 0.0))

    records_df = captured["records_df"]
    overlays = [json.loads(str(raw)).get("overlay_text") for raw in records_df["display"].tolist()]
    assert all(overlay is None for overlay in overlays)
    assert not (tmp_path / "_tmp").exists()
    assert int(result["pause_events"]) >= 1


def test_render_chain_trajectory_video_can_embed_you_are_here_inset_payload(tmp_path, monkeypatch) -> None:
    pwm = PWM(
        name="lexA",
        matrix=np.asarray(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7],
            ],
            dtype=float,
        ),
    )
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0],
            "sweep": [0, 1, 2, 3],
            "phase": ["draw", "draw", "draw", "draw"],
            "objective_scalar": [0.2, 0.4, 0.35, 0.6],
            "sequence": ["ACGTACGT", "ACGTTCGT", "ACGTTTGT", "ACGTTTAT"],
        }
    )
    cfg = AnalysisTrajectoryVideoConfig(
        sweep_inset={"enabled": True, "corner": "bottom_left"},
        playback={"target_duration_sec": 4.0, "fps": 8},
        limits={"max_total_frames": 32, "max_snapshots": 16, "max_estimated_render_sec": 30.0},
    )
    out_path = tmp_path / "video.mp4"
    captured: dict[str, object] = {}

    def _fake_run_job(job_mapping: dict[str, object], *, kind: str, caller_root: Path) -> None:
        del kind, caller_root
        captured["job"] = job_mapping
        captured["records_df"] = pd.read_parquet(Path(str(job_mapping["input"]["path"])))
        out = Path(str(job_mapping["outputs"][0]["path"]))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-mp4")

    monkeypatch.setattr("dnadesign.cruncher.analysis.trajectory_video.run_job", _fake_run_job)
    result = render_chain_trajectory_video(
        trajectory_df=trajectory_df,
        tf_names=["lexA"],
        pwms={"lexA": pwm},
        out_path=out_path,
        config=cfg,
        bidirectional=True,
        pwm_pseudocounts=0.0,
        log_odds_clip=None,
        tmp_root=tmp_path / "_tmp",
    )
    assert result["snapshot_count"] >= 2
    records_df = captured["records_df"]
    first_display = json.loads(str(records_df.iloc[0]["display"]))
    inset = dict(first_display["trajectory_inset"])
    assert inset["corner"] == "bottom_left"
    assert "label" not in inset
    assert len(inset["x"]) == len(inset["y"])
    assert int(inset["point_index"]) >= 0


def test_render_chain_trajectory_video_supports_pandas_str_dtype_mode(tmp_path, monkeypatch) -> None:
    import pandas as pd

    prior_storage = pd.options.mode.string_storage
    pd.options.mode.string_storage = "auto"
    try:
        pwm = PWM(
            name="lexA",
            matrix=np.asarray(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.1, 0.1, 0.7],
                ],
                dtype=float,
            ),
        )
        trajectory_df = pd.DataFrame(
            {
                "chain": [0, 0, 0],
                "sweep": [0, 1, 2],
                "phase": ["draw", "draw", "draw"],
                "objective_scalar": [0.2, 0.4, 0.6],
                "sequence": ["ACGTACGT", "ACGTTCGT", "ACGTTTGT"],
            }
        )
        cfg = AnalysisTrajectoryVideoConfig(
            playback={"target_duration_sec": 4.0, "fps": 8},
            limits={"max_total_frames": 32, "max_snapshots": 16, "max_estimated_render_sec": 30.0},
        )
        out_path = tmp_path / "video_str_mode.mp4"

        def _fake_run_job(job_mapping: dict[str, object], *, kind: str, caller_root: Path) -> None:
            out = Path(str(job_mapping["outputs"][0]["path"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake-mp4")

        monkeypatch.setattr("dnadesign.cruncher.analysis.trajectory_video.run_job", _fake_run_job)
        result = render_chain_trajectory_video(
            trajectory_df=trajectory_df,
            tf_names=["lexA"],
            pwms={"lexA": pwm},
            out_path=out_path,
            config=cfg,
            bidirectional=True,
            pwm_pseudocounts=0.0,
            log_odds_clip=None,
            tmp_root=tmp_path / "_tmp",
        )
    finally:
        pd.options.mode.string_storage = prior_storage

    assert out_path.exists()
    assert result["snapshot_count"] >= 2
