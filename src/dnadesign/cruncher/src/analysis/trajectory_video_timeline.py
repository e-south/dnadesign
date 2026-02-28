"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/trajectory_video_timeline.py

Timeline and sampling helpers for chain-trajectory video frame selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dnadesign.cruncher.config.schema_v3 import AnalysisTrajectoryVideoConfig


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


def filter_rows_by_phase_scope(
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


def select_chain_rows(
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

    rows = filter_rows_by_phase_scope(rows, phase_scope=str(config.selection.phase_scope))
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


def sample_frame_indices(
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


def source_indices_for_best_so_far_timeline(
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


def build_inset_line_indices(*, point_count: int, budget: int, required_indices: list[int]) -> list[int]:
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


def allocate_taper_extra_frames(*, point_count: int, total_extra_frames: int) -> list[int]:
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


__all__ = [
    "allocate_taper_extra_frames",
    "build_inset_line_indices",
    "filter_rows_by_phase_scope",
    "sample_frame_indices",
    "select_chain_rows",
    "source_indices_for_best_so_far_timeline",
]
