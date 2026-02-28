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
import math
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from dnadesign.baserender import cruncher_showcase_style_overrides, run_job
from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer

_DNA_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def _trajectory_video_style_overrides() -> dict[str, object]:
    overrides = dict(cruncher_showcase_style_overrides())
    overrides["figure_scale"] = 0.98
    overrides["padding_y"] = 3.0
    overrides["font_size_label"] = 15
    overrides["overlay_align"] = "center"
    overrides["show_reverse_complement"] = True
    overrides["baseline_spacing"] = 36.0
    overrides["track_spacing"] = 10.0

    layout = dict(overrides.get("layout", {}))
    layout["outer_pad_cells"] = 0.0
    overrides["layout"] = layout

    motif_logo = dict(overrides.get("motif_logo", {}))
    motif_logo["layout"] = "overlay"
    motif_logo["bits_to_cells"] = 0.70
    overrides["motif_logo"] = motif_logo
    return overrides


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


def _best_update_indices(values: np.ndarray) -> list[int]:
    updates: list[int] = []
    best = float("-inf")
    for idx, raw in enumerate(values.tolist()):
        value = float(raw)
        if not math.isfinite(value):
            continue
        if value > best + 1.0e-12:
            updates.append(int(idx))
            best = value
    if not updates:
        updates = [0]
    last = int(values.size - 1)
    if last not in updates:
        updates.append(last)
    return updates


def _uniform_pick(indices: list[int], k: int) -> list[int]:
    if k <= 0 or not indices:
        return []
    if len(indices) <= k:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, k).round().astype(int).tolist()
    picked: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        idx = int(indices[int(pos)])
        if idx in seen:
            continue
        picked.append(idx)
        seen.add(idx)
    if len(picked) >= k:
        return picked[:k]
    for idx in indices:
        if idx in seen:
            continue
        picked.append(int(idx))
        seen.add(int(idx))
        if len(picked) >= k:
            break
    return picked[:k]


def _phase_filter_rows(
    trajectory_df: pd.DataFrame,
    *,
    phase_scope: str,
) -> pd.DataFrame:
    if trajectory_df.empty:
        return trajectory_df
    if "phase" not in trajectory_df.columns:
        return trajectory_df
    phase = trajectory_df["phase"].astype(str).str.strip().str.lower()
    has_tune = bool((phase == "tune").any())
    if phase_scope == "draw_only":
        out = trajectory_df.loc[phase == "draw"].copy()
        if out.empty:
            raise ValueError("Trajectory video draw_only scope has no draw-phase rows.")
        return out
    if phase_scope == "tune_and_draw_required":
        if not has_tune:
            raise ValueError(
                "Trajectory video requires tune rows for phase_scope=tune_and_draw_required, but none were found."
            )
        keep = phase.isin({"tune", "draw"})
        out = trajectory_df.loc[keep].copy()
        if out.empty:
            raise ValueError("Trajectory video has no tune/draw rows after phase filtering.")
        return out
    if has_tune:
        keep = phase.isin({"tune", "draw"})
        out = trajectory_df.loc[keep].copy()
        if out.empty:
            raise ValueError("Trajectory video has no tune/draw rows after phase filtering.")
        return out
    out = trajectory_df.loc[phase == "draw"].copy()
    if out.empty:
        raise ValueError("Trajectory video has no tune/draw rows after phase filtering.")
    return out


def _select_chain_rows(
    trajectory_df: pd.DataFrame,
    *,
    config: AnalysisTrajectoryVideoConfig,
) -> tuple[pd.DataFrame, int]:
    if trajectory_df is None or trajectory_df.empty:
        raise ValueError("Trajectory video requires non-empty trajectory data.")
    if "chain" not in trajectory_df.columns:
        raise ValueError("Trajectory video requires trajectory column 'chain'.")
    if "sweep" not in trajectory_df.columns:
        raise ValueError("Trajectory video requires trajectory column 'sweep'.")
    objective_column = str(config.selection.objective_column)
    if objective_column not in trajectory_df.columns:
        raise ValueError(f"Trajectory video objective column '{objective_column}' is missing.")

    rows = trajectory_df.copy()
    rows["chain"] = pd.to_numeric(rows["chain"], errors="coerce")
    rows["sweep"] = pd.to_numeric(rows["sweep"], errors="coerce")
    rows[objective_column] = pd.to_numeric(rows[objective_column], errors="coerce")
    rows = rows.dropna(subset=["chain", "sweep", objective_column])
    if rows.empty:
        raise ValueError("Trajectory video has no rows with numeric chain/sweep/objective values.")
    rows["chain"] = rows["chain"].astype(int)
    rows["sweep"] = rows["sweep"].astype(int)
    if bool((rows["chain"] < 0).any()) or bool((rows["sweep"] < 0).any()):
        raise ValueError("Trajectory video rows contain negative chain/sweep values.")

    rows = _phase_filter_rows(rows, phase_scope=str(config.selection.phase_scope))
    if rows.empty:
        raise ValueError("Trajectory video has no rows after phase filtering.")
    rows = rows.sort_values(["chain", "sweep"]).reset_index(drop=True)

    if str(config.selection.chain_policy) == "explicit":
        explicit = int(config.selection.explicit_chain_1based or 0) - 1
        if explicit < 0:
            raise ValueError("Trajectory video explicit chain index must be >= 1.")
        chain_rows = rows.loc[rows["chain"] == explicit].copy()
        if chain_rows.empty:
            raise ValueError(f"Trajectory video explicit chain {explicit + 1} was not found in trajectory rows.")
        return chain_rows.reset_index(drop=True), explicit

    best_chain = None
    best_score = float("-inf")
    grouped = rows.groupby("chain", sort=True, dropna=False)
    for chain_id, chain_rows in grouped:
        chain_best = float(chain_rows[objective_column].max())
        if chain_best > best_score + 1.0e-12:
            best_score = chain_best
            best_chain = int(chain_id)
    if best_chain is None:
        raise ValueError("Trajectory video could not select a best chain.")
    selected = rows.loc[rows["chain"] == best_chain].copy()
    if selected.empty:
        raise ValueError("Trajectory video selected chain has no rows.")
    return selected.reset_index(drop=True), int(best_chain)


def _sample_indices(
    *,
    objective_values: np.ndarray,
    sampling_stride: int,
    include_best_updates: bool,
    snapshot_budget: int,
) -> tuple[list[int], int, list[int]]:
    n = int(objective_values.size)
    if n < 2:
        raise ValueError("Trajectory video requires at least two trajectory rows for rendering.")
    dynamic_stride = max(1, int(math.ceil(float(n) / float(max(1, snapshot_budget)))))
    effective_stride = max(int(sampling_stride), dynamic_stride)
    base = list(range(0, n, effective_stride))
    if (n - 1) not in base:
        base.append(n - 1)

    priority: list[int] = [0, n - 1]
    best_updates = _best_update_indices(objective_values) if include_best_updates else []
    priority.extend(best_updates)
    candidate = sorted({*base, *priority})
    if len(candidate) <= snapshot_budget:
        return candidate, effective_stride, best_updates

    required = [0, n - 1]
    selected_set = {0, n - 1}
    slots = snapshot_budget - len(required)
    if slots <= 0:
        return required[:snapshot_budget], effective_stride, best_updates

    preferred = [idx for idx in sorted(set(priority)) if idx not in selected_set]
    preferred_pick = _uniform_pick(preferred, min(slots, len(preferred)))
    for idx in preferred_pick:
        selected_set.add(int(idx))
    slots = snapshot_budget - len(selected_set)

    if slots > 0:
        remainder = [idx for idx in candidate if idx not in selected_set]
        for idx in _uniform_pick(remainder, slots):
            selected_set.add(int(idx))

    selected = sorted(selected_set)
    if len(selected) > snapshot_budget:
        core = [idx for idx in selected if idx not in {0, n - 1}]
        keep_core = _uniform_pick(core, max(0, snapshot_budget - 2))
        selected = sorted({0, n - 1, *keep_core})
    return selected, effective_stride, best_updates


def _best_so_far_source_indices(
    *,
    objective_values: np.ndarray,
    sampled_indices: list[int],
) -> list[int]:
    state_indices: list[int] = []
    current_best_idx = 0
    current_best_score = float("-inf")
    scan_start = 0
    for sampled in sampled_indices:
        stop = int(sampled) + 1
        for idx in range(scan_start, stop):
            score = float(objective_values[idx])
            if not math.isfinite(score):
                continue
            if score > current_best_score + 1.0e-12:
                current_best_score = score
                current_best_idx = int(idx)
        state_indices.append(int(current_best_idx))
        scan_start = stop
    return state_indices


def _inset_line_indices(*, point_count: int, budget: int, required_indices: list[int]) -> list[int]:
    if point_count < 2:
        return [0]
    if budget < 2:
        budget = 2
    full = list(range(point_count))
    if point_count <= budget:
        return full

    required = sorted({0, point_count - 1, *[int(v) for v in required_indices if 0 <= int(v) < point_count]})
    if len(required) >= budget:
        picked = _uniform_pick(required, budget)
        picked = sorted({0, point_count - 1, *picked})
        if len(picked) > budget:
            core = [idx for idx in picked if idx not in {0, point_count - 1}]
            picked = sorted({0, point_count - 1, *_uniform_pick(core, max(0, budget - 2))})
        return picked

    selected = set(required)
    slots = budget - len(selected)
    remainder = [idx for idx in full if idx not in selected]
    for idx in _uniform_pick(remainder, slots):
        selected.add(int(idx))
    return sorted(selected)


def _linear_taper_extra_frames(*, point_count: int, total_extra_frames: int) -> list[int]:
    if point_count < 1:
        return []
    if total_extra_frames <= 0:
        return [0 for _ in range(point_count)]
    if point_count == 1:
        return [int(total_extra_frames)]

    weights = np.linspace(0.0, 1.0, point_count, dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        out = [0 for _ in range(point_count)]
        out[-1] = int(total_extra_frames)
        return out

    raw = [(float(weights[idx]) / weight_sum) * float(total_extra_frames) for idx in range(point_count)]
    base = [int(math.floor(value)) for value in raw]
    remaining = int(total_extra_frames - sum(base))
    idx = point_count - 1
    while remaining > 0:
        base[idx] += 1
        idx -= 1
        if idx < 0:
            idx = point_count - 1
        remaining -= 1
    return base


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
    missing_pwms = [tf for tf in tf_names if tf not in pwms]
    if missing_pwms:
        raise ValueError(f"Trajectory video missing PWMs for TFs: {missing_pwms}")

    chain_rows, selected_chain = _select_chain_rows(trajectory_df, config=config)
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

    sampled_indices, effective_stride, best_updates = _sample_indices(
        objective_values=objective_values,
        sampling_stride=int(config.sampling.stride),
        include_best_updates=bool(config.sampling.include_best_updates),
        snapshot_budget=snapshot_budget,
    )

    if str(config.timeline_mode) == "best_so_far":
        source_indices = _best_so_far_source_indices(
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
        line_indices = _inset_line_indices(
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
    taper_extra_frames = _linear_taper_extra_frames(
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

        job_mapping: dict[str, object] = {
            "version": 3,
            "input": {
                "kind": "parquet",
                "path": str(records_path),
                "adapter": {
                    "kind": "generic_features",
                    "columns": {
                        "id": "id",
                        "sequence": "sequence",
                        "features": "features",
                        "effects": "effects",
                        "display": "display",
                    },
                    "policies": {},
                },
                "alphabet": "DNA",
            },
            "render": {
                "renderer": "sequence_rows",
                "style": {"overrides": dict(_trajectory_video_style_overrides())},
            },
            "outputs": [
                {
                    "kind": "video",
                    "path": str(out_path),
                    "fmt": "mp4",
                    "fps": int(config.playback.fps),
                    "frames_per_record": 1,
                    "pauses": pauses,
                    "total_duration": float(config.playback.target_duration_sec),
                    "height_px": 820,
                    "title_text": overlay_title,
                    "title_font_size": 12,
                    "title_align": "center",
                }
            ],
            "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
        }
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
