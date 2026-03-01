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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from dnadesign.baserender import run_job
from dnadesign.cruncher.analysis.objective_labels import objective_scale_label
from dnadesign.cruncher.analysis.trajectory_video_contract import build_sequence_rows_video_job
from dnadesign.cruncher.analysis.trajectory_video_timeline import (
    allocate_taper_extra_frames,
    build_panel_line_indices,
    sample_frame_indices,
    select_chain_rows,
    source_indices_for_best_so_far_timeline,
)
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer

_DNA_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
_VIDEO_SUBTITLE_LINE_CHARS = 68

__all__ = [
    "render_chain_trajectory_video",
]


@dataclass(frozen=True)
class _FrameBudget:
    target_total_frames: int
    snapshot_budget: int


@dataclass(frozen=True)
class _FrameSelection:
    sampled_indices: list[int]
    source_indices: list[int]
    best_updates: list[int]
    effective_stride: int


@dataclass(frozen=True)
class _PanelPayload:
    x: list[float]
    y: list[float]
    y_label: str
    point_index_by_source: dict[int, int]


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


def _clamp_normalized_score(value: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("Trajectory video subtitle scores must be finite.")
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _wrap_subtitle_tokens(tokens: list[str], *, line_chars: int) -> str:
    lines: list[str] = []
    current = ""
    for token in tokens:
        text = str(token).strip()
        if not text:
            continue
        if not current:
            current = text
            continue
        candidate = f"{current} {text}"
        if len(candidate) <= int(line_chars):
            current = candidate
            continue
        lines.append(current)
        current = text
    if current:
        lines.append(current)
    return "\n".join(lines)


def _video_subtitle_text(*, tf_names: list[str], per_tf_map: Mapping[str, float]) -> str:
    tokens: list[str] = []
    for tf_name in tf_names:
        if tf_name not in per_tf_map:
            raise ValueError(f"Trajectory video subtitle missing normalized score for TF '{tf_name}'.")
        score_value = _clamp_normalized_score(float(per_tf_map[tf_name]))
        tokens.append(f"{tf_name}={score_value:.2f}")
    return _wrap_subtitle_tokens(tokens, line_chars=_VIDEO_SUBTITLE_LINE_CHARS)


def _panel_y_label(*, objective_column: str, objective_config: Mapping[str, object] | None) -> str:
    column = str(objective_column).strip()
    if column == "raw_llr_objective":
        return "Best objective (raw-LLR)"
    if column == "norm_llr_objective":
        return "Best objective (norm-LLR)"
    if column != "objective_scalar":
        return f"Best objective ({column})"
    cfg = objective_config if isinstance(objective_config, Mapping) else {}
    combine = str(cfg.get("combine") or "min").strip().lower()
    scale_label = objective_scale_label(cfg)
    softmin_cfg = cfg.get("softmin")
    softmin_enabled = isinstance(softmin_cfg, Mapping) and bool(softmin_cfg.get("enabled"))
    if combine == "sum":
        return f"Best sum-TF {scale_label}"
    if combine == "min" and softmin_enabled:
        return f"Best soft-min TF {scale_label}"
    return f"Best min-TF {scale_label}"


def _parse_hit_fields(*, hit: Mapping[str, object], tf_name: str, sequence_len: int) -> tuple[int, int, str, str]:
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
    if start < 0 or end > int(sequence_len):
        raise ValueError(
            f"Trajectory video hit span is out of bounds for TF '{tf_name}': [{start}, {end}) "
            f"for sequence length {int(sequence_len)}."
        )
    return start, end, strand, window_seq


def _compute_frame_budget(*, config: AnalysisTrajectoryVideoConfig) -> _FrameBudget:
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
    return _FrameBudget(
        target_total_frames=int(target_total_frames),
        snapshot_budget=int(snapshot_budget),
    )


def _select_frame_indices(
    *,
    objective_values: np.ndarray,
    config: AnalysisTrajectoryVideoConfig,
    snapshot_budget: int,
) -> _FrameSelection:
    sampled_indices, effective_stride, best_updates = sample_frame_indices(
        objective_values=objective_values,
        sampling_stride=int(config.sampling.stride),
        include_best_updates=bool(config.sampling.include_best_updates),
        snapshot_budget=int(snapshot_budget),
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
    return _FrameSelection(
        sampled_indices=[int(idx) for idx in sampled_indices],
        source_indices=[int(idx) for idx in source_indices],
        best_updates=[int(idx) for idx in best_updates],
        effective_stride=int(effective_stride),
    )


def _score_source_sequences(
    *,
    chain_rows: pd.DataFrame,
    source_indices: list[int],
    scorer: Scorer,
) -> tuple[list[str], list[Mapping[str, float]], list[Mapping[str, Mapping[str, object]]]]:
    frame_sequences: list[str] = []
    frame_per_tf_maps: list[Mapping[str, float]] = []
    frame_hit_maps: list[Mapping[str, Mapping[str, object]]] = []
    for source_idx in source_indices:
        source_row = chain_rows.iloc[int(source_idx)]
        sequence = str(source_row.get("sequence", "")).strip().upper()
        if not sequence:
            raise ValueError("Trajectory video source row is missing sequence.")
        seq_arr = _encode_sequence(sequence)
        per_tf_map, hit_map = scorer.compute_all_per_pwm_and_hits(seq_arr, int(seq_arr.size))
        frame_sequences.append(sequence)
        frame_per_tf_maps.append(per_tf_map)
        frame_hit_maps.append(hit_map)
    return frame_sequences, frame_per_tf_maps, frame_hit_maps


def _build_panel_payload(
    *,
    chain_rows: pd.DataFrame,
    objective_values: np.ndarray,
    source_indices: list[int],
    max_snapshots: int,
    objective_column: str,
    objective_config: Mapping[str, object] | None,
) -> _PanelPayload:
    line_budget = min(240, max(2, int(max_snapshots) * 2))
    line_indices = build_panel_line_indices(
        point_count=int(objective_values.size),
        budget=line_budget,
        required_indices=[int(idx) for idx in source_indices],
    )
    panel_x = [float(chain_rows.iloc[int(idx)]["sweep"]) for idx in line_indices]
    best_curve = np.maximum.accumulate(objective_values)
    panel_y = [float(best_curve[int(idx)]) for idx in line_indices]
    panel_y_label = _panel_y_label(
        objective_column=objective_column,
        objective_config=objective_config,
    )
    panel_point_index_by_source: dict[int, int] = {}
    index_by_source = {int(idx): pos for pos, idx in enumerate(line_indices)}
    line_index_arr = np.asarray(line_indices, dtype=int)
    for source_idx in source_indices:
        key = int(source_idx)
        if key in index_by_source:
            panel_point_index_by_source[key] = int(index_by_source[key])
            continue
        fallback = int(np.searchsorted(line_index_arr, key, side="right") - 1)
        fallback = max(0, min(fallback, len(line_indices) - 1))
        panel_point_index_by_source[key] = fallback
    return _PanelPayload(
        x=panel_x,
        y=panel_y,
        y_label=panel_y_label,
        point_index_by_source=panel_point_index_by_source,
    )


def _build_snapshot_row(
    *,
    frame_id: str,
    frame_no: int,
    source_idx: int,
    sequence: str,
    tf_names: list[str],
    per_tf_map: Mapping[str, float],
    hit_map: Mapping[str, Mapping[str, object]],
    pwm_matrices: Mapping[str, list[list[float]]],
    panel_payload: _PanelPayload,
) -> dict[str, object]:
    frame_subtitle = _video_subtitle_text(tf_names=tf_names, per_tf_map=per_tf_map)
    feature_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    tag_labels: dict[str, str] = {}
    for tf_idx, tf_name in enumerate(tf_names):
        hit = hit_map.get(tf_name)
        if hit is None:
            raise ValueError(f"Trajectory video missing hit for TF '{tf_name}'.")
        start, end, strand, window_seq = _parse_hit_fields(hit=hit, tf_name=tf_name, sequence_len=len(sequence))
        width = int(end - start)
        label = window_seq if strand == "fwd" else _revcomp(window_seq)
        feature_id = f"frame_{frame_no}:best_window:{tf_name}:{tf_idx}"
        tag = f"tf:{tf_name}"
        matrix = pwm_matrices[tf_name]
        if len(matrix) != width:
            raise ValueError(
                f"Trajectory video PWM length mismatch for TF '{tf_name}': matrix rows={len(matrix)} hit width={width}"
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

    display_payload: dict[str, object] = {
        "overlay_text": None,
        "video_subtitle": frame_subtitle,
        "tag_labels": tag_labels,
        "trajectory_panel": {
            "x": panel_payload.x,
            "y": panel_payload.y,
            "point_index": int(panel_payload.point_index_by_source.get(int(source_idx), 0)),
            "x_label": "Sweep",
            "y_label": panel_payload.y_label,
        },
    }
    return {
        "id": frame_id,
        "sequence": sequence,
        "features": json.dumps(feature_rows, separators=(",", ":")),
        "effects": json.dumps(effect_rows, separators=(",", ":")),
        "display": json.dumps(display_payload, separators=(",", ":")),
    }


def _build_snapshot_rows(
    *,
    sampled_indices: list[int],
    source_indices: list[int],
    frame_sequences: list[str],
    frame_per_tf_maps: list[Mapping[str, float]],
    frame_hit_maps: list[Mapping[str, Mapping[str, object]]],
    tf_names: list[str],
    selected_chain: int,
    timeline_mode: str,
    best_updates: list[int],
    pause_on_best_update_sec: float,
    pwm_matrices: Mapping[str, list[list[float]]],
    panel_payload: _PanelPayload,
) -> tuple[list[dict[str, object]], dict[str, float], str]:
    snapshot_rows: list[dict[str, object]] = []
    pauses: dict[str, float] = {}
    previous_source_idx: int | None = None
    best_update_set = set(int(idx) for idx in best_updates)
    overlay_title = (
        "Best-so-far motif placement improves over sweeps"
        if str(timeline_mode) == "best_so_far"
        else "Sampled motif placement across sweeps"
    )
    for frame_no, (sampled_idx, source_idx, sequence, per_tf_map, hit_map) in enumerate(
        zip(sampled_indices, source_indices, frame_sequences, frame_per_tf_maps, frame_hit_maps),
        start=1,
    ):
        frame_id = f"chain_{int(selected_chain) + 1}_frame_{frame_no:04d}"
        if str(timeline_mode) == "best_so_far":
            is_best_update_frame = previous_source_idx is None or int(source_idx) != int(previous_source_idx)
        else:
            is_best_update_frame = int(sampled_idx) in best_update_set
        previous_source_idx = int(source_idx)
        snapshot_rows.append(
            _build_snapshot_row(
                frame_id=frame_id,
                frame_no=int(frame_no),
                source_idx=int(source_idx),
                sequence=str(sequence),
                tf_names=tf_names,
                per_tf_map=per_tf_map,
                hit_map=hit_map,
                pwm_matrices=pwm_matrices,
                panel_payload=panel_payload,
            )
        )
        if is_best_update_frame and float(pause_on_best_update_sec) > 0:
            pauses[frame_id] = float(pause_on_best_update_sec)
    return snapshot_rows, pauses, overlay_title


def _replace_final_snapshot_with_polished_sequence(
    *,
    polished_final_sequence: str | None,
    snapshot_rows: list[dict[str, object]],
    scorer: Scorer,
    source_indices: list[int],
    tf_names: list[str],
    pwm_matrices: Mapping[str, list[list[float]]],
    panel_payload: _PanelPayload,
) -> None:
    polished_sequence = str(polished_final_sequence or "").strip().upper()
    if not polished_sequence or not snapshot_rows:
        return
    final_seq_now = str(snapshot_rows[-1]["sequence"]).strip().upper()
    if polished_sequence == final_seq_now:
        return
    seq_arr = _encode_sequence(polished_sequence)
    per_tf_map, hit_map = scorer.compute_all_per_pwm_and_hits(seq_arr, int(seq_arr.size))
    source_idx = int(source_indices[-1])
    frame_no = int(len(snapshot_rows))
    frame_id = str(snapshot_rows[-1]["id"])
    snapshot_rows[-1] = _build_snapshot_row(
        frame_id=frame_id,
        frame_no=frame_no,
        source_idx=source_idx,
        sequence=polished_sequence,
        tf_names=tf_names,
        per_tf_map=per_tf_map,
        hit_map=hit_map,
        pwm_matrices=pwm_matrices,
        panel_payload=panel_payload,
    )


def _pause_frame_count(*, pauses: Mapping[str, float], fps: float) -> int:
    return int(sum(int(round(float(sec) * float(fps))) for sec in pauses.values()))


def _apply_taper_pause_extension(
    *,
    snapshot_rows: list[dict[str, object]],
    pauses: dict[str, float],
    target_total_frames: int,
    fps: float,
) -> None:
    existing_pause_frames = _pause_frame_count(pauses=pauses, fps=fps)
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
        pauses[frame_id] = float(pauses.get(frame_id, 0.0)) + (float(extra_frames) / float(fps))


def _validate_frame_limits(
    *,
    snapshot_rows: list[dict[str, object]],
    pauses: Mapping[str, float],
    config: AnalysisTrajectoryVideoConfig,
    target_total_frames: int,
) -> tuple[int, float]:
    pause_frames_total = _pause_frame_count(pauses=pauses, fps=float(config.playback.fps))
    total_frames = int(len(snapshot_rows) + pause_frames_total)
    if total_frames > int(target_total_frames):
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
    return int(total_frames), float(estimated_render_sec)


def _render_video_job(
    *,
    snapshot_rows: list[dict[str, object]],
    out_path: Path,
    config: AnalysisTrajectoryVideoConfig,
    pauses: dict[str, float],
    title_text: str,
    tmp_root: Path,
) -> None:
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
            title_text=title_text,
        )
        run_job(job_mapping, kind="sequence_rows_v3", caller_root=work_root)
    if not out_path.exists():
        raise ValueError(f"Trajectory video output was not created: {out_path}")


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
    polished_final_sequence: str | None = None,
    objective_from_manifest: Mapping[str, object] | None = None,
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

    frame_budget = _compute_frame_budget(config=config)
    frame_selection = _select_frame_indices(
        objective_values=objective_values,
        config=config,
        snapshot_budget=int(frame_budget.snapshot_budget),
    )

    scorer = Scorer(
        dict(pwms),
        background=(0.25, 0.25, 0.25, 0.25),
        bidirectional=bool(bidirectional),
        scale="normalized-llr",
        pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
    )
    pwm_matrices = {tf: _matrix_from_pwm(pwms[tf]) for tf in tf_names}
    frame_sequences, frame_per_tf_maps, frame_hit_maps = _score_source_sequences(
        chain_rows=chain_rows,
        source_indices=frame_selection.source_indices,
        scorer=scorer,
    )
    panel_payload = _build_panel_payload(
        chain_rows=chain_rows,
        objective_values=objective_values,
        source_indices=frame_selection.source_indices,
        max_snapshots=int(config.limits.max_snapshots),
        objective_column=objective_column,
        objective_config=objective_from_manifest,
    )

    snapshot_rows, pauses, overlay_title = _build_snapshot_rows(
        sampled_indices=frame_selection.sampled_indices,
        source_indices=frame_selection.source_indices,
        frame_sequences=frame_sequences,
        frame_per_tf_maps=frame_per_tf_maps,
        frame_hit_maps=frame_hit_maps,
        tf_names=tf_names,
        selected_chain=int(selected_chain),
        timeline_mode=str(config.timeline_mode),
        best_updates=frame_selection.best_updates,
        pause_on_best_update_sec=float(config.playback.pause_on_best_update_sec),
        pwm_matrices=pwm_matrices,
        panel_payload=panel_payload,
    )

    _replace_final_snapshot_with_polished_sequence(
        polished_final_sequence=polished_final_sequence,
        snapshot_rows=snapshot_rows,
        scorer=scorer,
        source_indices=frame_selection.source_indices,
        tf_names=tf_names,
        pwm_matrices=pwm_matrices,
        panel_payload=panel_payload,
    )

    _apply_taper_pause_extension(
        snapshot_rows=snapshot_rows,
        pauses=pauses,
        target_total_frames=int(frame_budget.target_total_frames),
        fps=float(config.playback.fps),
    )

    total_frames, estimated_render_sec = _validate_frame_limits(
        snapshot_rows=snapshot_rows,
        pauses=pauses,
        config=config,
        target_total_frames=int(frame_budget.target_total_frames),
    )

    _render_video_job(
        snapshot_rows=snapshot_rows,
        out_path=out_path,
        config=config,
        pauses=pauses,
        title_text=overlay_title,
        tmp_root=tmp_root,
    )

    return {
        "chain_1based": int(selected_chain) + 1,
        "timeline_mode": str(config.timeline_mode),
        "objective_column": objective_column,
        "snapshot_count": int(len(snapshot_rows)),
        "stride_effective": int(frame_selection.effective_stride),
        "target_total_frames": int(frame_budget.target_total_frames),
        "total_frames": int(total_frames),
        "pause_events": int(len(pauses)),
        "estimated_render_sec": float(estimated_render_sec),
    }
