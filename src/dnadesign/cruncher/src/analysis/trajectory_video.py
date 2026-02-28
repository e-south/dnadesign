"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/trajectory_video.py

Build short chain-trajectory motif videos via the public baserender job API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from dnadesign.baserender import run_job
from dnadesign.cruncher.analysis.trajectory_video_contract import build_sequence_rows_video_job
from dnadesign.cruncher.analysis.trajectory_video_timeline import (
    allocate_taper_extra_frames,
    build_inset_line_indices,
    sample_frame_indices,
    select_chain_rows,
    source_indices_for_best_so_far_timeline,
)
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer

_DNA_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

__all__ = [
    "render_chain_trajectory_video",
]


def _revcomp(seq: str) -> str:
    return seq.translate(_DNA_COMP)[::-1]


def _encode_sequence(seq: str) -> np.ndarray:
    text = str(seq).strip().upper()
    if not text:
        raise ValueError("Trajectory video sequence is empty.")
    try:
        return np.asarray([_DNA_BASE_TO_INT[base] for base in text], dtype=np.int8)
    except KeyError as exc:
        raise ValueError(f"Trajectory video sequence contains invalid base(s): {seq!r}") from exc


def _matrix_from_pwm(pwm_obj: PWM) -> list[list[float]]:
    matrix = getattr(pwm_obj, "matrix", None)
    if matrix is None:
        raise ValueError("PWM object missing matrix")
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("PWM matrix must be 2D with at least 4 columns [A,C,G,T]")
    return [[float(v) for v in row[:4]] for row in arr.tolist()]


def render_chain_trajectory_video(
    *,
    trajectory_df: pd.DataFrame,
    tf_names: list[str],
    pwms: Mapping[str, PWM],
    out_path: Path,
    config: AnalysisTrajectoryVideoConfig,
    bidirectional: bool,
    pwm_pseudocounts: float,
    log_odds_clip: float | None,
    tmp_root: Path,
) -> dict[str, object]:
    if not tf_names:
        raise ValueError("Trajectory video requires at least one TF name.")
    if "sequence" not in trajectory_df.columns:
        raise ValueError("Trajectory video requires trajectory column 'sequence'.")
    missing_pwms = [tf for tf in tf_names if tf not in pwms]
    if missing_pwms:
        raise ValueError(f"Trajectory video missing PWMs for TFs: {missing_pwms}")

    chain_rows, selected_chain = select_chain_rows(trajectory_df, config=config)
    objective_column = str(config.selection.objective_column)
    objective_values = chain_rows[objective_column].astype(float).to_numpy()
    if not np.isfinite(objective_values).any():
        raise ValueError("Trajectory video objective values are non-finite for selected chain.")

    target_total_frames = int(round(float(config.playback.target_duration_sec) * float(config.playback.fps)))
    if target_total_frames < 2:
        raise ValueError("Trajectory video frame budget must be >= 2.")
    if target_total_frames > int(config.limits.max_total_frames):
        raise ValueError(
            "Trajectory video target frames exceed limits.max_total_frames; "
            "reduce playback duration/fps or raise limits.max_total_frames."
        )

    pause_frames_per_update = int(round(float(config.playback.pause_on_best_update_sec) * float(config.playback.fps)))
    taper_reserved_frames = int(round(float(config.playback.sweep_taper_fraction) * float(target_total_frames)))
    sampling_frame_budget = max(2, int(target_total_frames - taper_reserved_frames))
    snapshot_budget = min(
        int(config.limits.max_snapshots),
        int(sampling_frame_budget // max(1, 1 + pause_frames_per_update)),
    )
    if snapshot_budget < 2:
        raise ValueError(
            "Trajectory video budget is too small after pause settings; reduce pause_on_best_update_sec "
            "or increase target_duration_sec."
        )

    sampled_indices, effective_stride, best_updates = sample_frame_indices(
        objective_values=objective_values,
        sampling_stride=int(config.sampling.stride),
        include_best_updates=bool(config.sampling.include_best_updates),
        snapshot_budget=snapshot_budget,
    )

    if str(config.timeline_mode) == "best_so_far":
        source_indices = source_indices_for_best_so_far_timeline(
            objective_values=objective_values,
            sampled_indices=sampled_indices,
        )
    else:
        source_indices = list(sampled_indices)

    if len(source_indices) != len(sampled_indices):
        raise ValueError("Trajectory video source/frame index mismatch.")
    if len(source_indices) > int(config.limits.max_snapshots):
        raise ValueError("Trajectory video selected snapshots exceed limits.max_snapshots.")

    inset_x: list[float] | None = None
    inset_y: list[float] | None = None
    inset_point_index_by_source: dict[int, int] = {}
    if bool(config.sweep_inset.enabled):
        line_budget = min(240, max(2, int(config.limits.max_snapshots) * 2))
        line_indices = build_inset_line_indices(
            point_count=int(objective_values.size),
            budget=line_budget,
            required_indices=list(source_indices),
        )
        inset_x = [float(chain_rows.iloc[int(idx)]["sweep"]) for idx in line_indices]
        best_curve = np.maximum.accumulate(objective_values)
        inset_y = [float(best_curve[int(idx)]) for idx in line_indices]
        index_by_source = {int(idx): pos for pos, idx in enumerate(line_indices)}
        for source_idx in source_indices:
            key = int(source_idx)
            if key in index_by_source:
                inset_point_index_by_source[key] = int(index_by_source[key])
                continue
            fallback = int(np.searchsorted(np.asarray(line_indices, dtype=int), key, side="right") - 1)
            fallback = max(0, min(fallback, len(line_indices) - 1))
            inset_point_index_by_source[key] = fallback

    scorer = Scorer(
        dict(pwms),
        background=(0.25, 0.25, 0.25, 0.25),
        bidirectional=bool(bidirectional),
        scale="normalized-llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    pwm_matrices = {tf: _matrix_from_pwm(pwms[tf]) for tf in tf_names}

    snapshot_rows: list[dict[str, object]] = []
    pauses: dict[str, float] = {}
    previous_source_idx: int | None = None
    best_update_set = set(best_updates)
    if str(config.timeline_mode) == "best_so_far":
        overlay_title = "Best-so-far motif placement improves over sweeps"
    else:
        overlay_title = "Sampled motif placement across sweeps"
    for frame_no, (sampled_idx, source_idx) in enumerate(zip(sampled_indices, source_indices), start=1):
        source_row = chain_rows.iloc[int(source_idx)]
        sequence = str(source_row.get("sequence", "")).strip().upper()
        if not sequence:
            raise ValueError("Trajectory video source row is missing sequence.")
        seq_arr = _encode_sequence(sequence)
        _, hit_map = scorer.compute_all_per_pwm_and_hits(seq_arr, int(seq_arr.size))

        feature_rows: list[dict[str, object]] = []
        effect_rows: list[dict[str, object]] = []
        tag_labels: dict[str, str] = {}
        for tf_idx, tf_name in enumerate(tf_names):
            hit = hit_map.get(tf_name)
            if hit is None:
                raise ValueError(f"Trajectory video missing hit for TF '{tf_name}'.")
            start = int(hit["best_start"])
            window_seq = str(hit["best_window_seq"]).upper()
            width = len(window_seq)
            if width < 1:
                raise ValueError(f"Trajectory video hit width is invalid for TF '{tf_name}'.")
            strand_raw = str(hit["strand"]).strip()
            if strand_raw == "+":
                strand = "fwd"
            elif strand_raw == "-":
                strand = "rev"
            else:
                raise ValueError(f"Trajectory video hit has invalid strand for TF '{tf_name}': {strand_raw!r}")
            end = start + width
            if start < 0 or end > len(sequence):
                raise ValueError(
                    f"Trajectory video hit span is out of bounds for TF '{tf_name}': [{start}, {end}) "
                    f"for sequence length {len(sequence)}."
                )
            label = window_seq if strand == "fwd" else _revcomp(window_seq)
            feature_id = f"frame_{frame_no}:best_window:{tf_name}:{tf_idx}"
            tag = f"tf:{tf_name}"
            matrix = pwm_matrices[tf_name]
            if len(matrix) != width:
                raise ValueError(
                    f"Trajectory video PWM length mismatch for TF '{tf_name}': "
                    f"matrix rows={len(matrix)} hit width={width}"
                )
            feature_rows.append(
                {
                    "id": feature_id,
                    "kind": "kmer",
                    "span": {"start": start, "end": end, "strand": strand},
                    "label": label,
                    "tags": [tag],
                    "attrs": {"tf": tf_name},
                    "render": {"priority": 10},
                }
            )
            effect_rows.append(
                {
                    "kind": "motif_logo",
                    "target": {"feature_id": feature_id},
                    "params": {"matrix": matrix},
                    "render": {"priority": 20},
                }
            )
            tag_labels[tag] = tf_name

        frame_id = f"chain_{int(selected_chain) + 1}_frame_{frame_no:04d}"
        if str(config.timeline_mode) == "best_so_far":
            is_best_update_frame = previous_source_idx is None or int(source_idx) != int(previous_source_idx)
        else:
            is_best_update_frame = int(sampled_idx) in best_update_set
        previous_source_idx = int(source_idx)

        display_payload: dict[str, object] = {"overlay_text": None, "tag_labels": tag_labels}
        if inset_x is not None and inset_y is not None:
            display_payload["trajectory_inset"] = {
                "x": inset_x,
                "y": inset_y,
                "point_index": int(inset_point_index_by_source.get(int(source_idx), 0)),
                "corner": str(config.sweep_inset.corner),
            }

        snapshot_rows.append(
            {
                "id": frame_id,
                "sequence": sequence,
                "features": json.dumps(feature_rows, separators=(",", ":")),
                "effects": json.dumps(effect_rows, separators=(",", ":")),
                "display": json.dumps(display_payload, separators=(",", ":")),
            }
        )
        if is_best_update_frame and float(config.playback.pause_on_best_update_sec) > 0:
            pauses[frame_id] = float(config.playback.pause_on_best_update_sec)

    existing_pause_frames = int(sum(int(round(float(sec) * float(config.playback.fps))) for sec in pauses.values()))
    remaining_frames_for_taper = max(0, int(target_total_frames - len(snapshot_rows) - existing_pause_frames))
    taper_extra_frames = allocate_taper_extra_frames(
        point_count=len(snapshot_rows),
        total_extra_frames=remaining_frames_for_taper,
    )
    if len(taper_extra_frames) != len(snapshot_rows):
        raise ValueError("Trajectory video taper frame allocation mismatch.")
    for row, extra_frames in zip(snapshot_rows, taper_extra_frames):
        if extra_frames <= 0:
            continue
        frame_id = str(row["id"])
        pauses[frame_id] = float(pauses.get(frame_id, 0.0)) + (float(extra_frames) / float(config.playback.fps))

    pause_frames_total = int(sum(int(round(float(sec) * float(config.playback.fps))) for sec in pauses.values()))
    total_frames = int(len(snapshot_rows) + pause_frames_total)
    if total_frames > target_total_frames:
        raise ValueError(
            "Trajectory video pauses exceed playback frame budget; reduce pause_on_best_update_sec, "
            "reduce duration, or increase sampling stride."
        )
    if total_frames > int(config.limits.max_total_frames):
        raise ValueError("Trajectory video total frames exceed limits.max_total_frames.")
    estimated_render_sec = float(total_frames / float(config.playback.fps)) * 1.5
    if estimated_render_sec > float(config.limits.max_estimated_render_sec):
        raise ValueError(
            "Trajectory video estimated render time exceeds limits.max_estimated_render_sec; "
            "reduce total frames or raise the limit explicitly."
        )

    scratch_parent = tmp_root.parent if tmp_root.parent.exists() else tmp_root
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_trajectory_video_tmp_", dir=scratch_parent) as tmp_dir:
        work_root = Path(tmp_dir)
        records_path = work_root / "_trajectory_video_records.parquet"
        pd.DataFrame(snapshot_rows).to_parquet(records_path, engine="pyarrow")

        job_mapping = build_sequence_rows_video_job(
            records_path=records_path,
            out_path=out_path,
            config=config,
            pauses=pauses,
            title_text=overlay_title,
        )
        run_job(job_mapping, kind="sequence_rows_v3", caller_root=work_root)
    if not out_path.exists():
        raise ValueError(f"Trajectory video output was not created: {out_path}")

    return {
        "chain_1based": int(selected_chain) + 1,
        "timeline_mode": str(config.timeline_mode),
        "objective_column": objective_column,
        "snapshot_count": int(len(snapshot_rows)),
        "stride_effective": int(effective_stride),
        "target_total_frames": int(target_total_frames),
        "total_frames": int(total_frames),
        "pause_events": int(len(pauses)),
        "estimated_render_sec": float(estimated_render_sec),
    }
