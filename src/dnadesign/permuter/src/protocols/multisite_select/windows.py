"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/protocols/multisite_select/windows.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    window_id: int
    start_aa: int  # inclusive, 1-indexed
    end_aa: int  # inclusive, 1-indexed
    covered_idx: np.ndarray  # sorted variant indices (global indexing)
    F_w: float  # sum score_plus of covered variants
    covered_count: int


def _make_windows(
    L_total: int,
    *,
    length_aa: int,
    stride_aa: int,
) -> List[Tuple[int, int]]:
    S: List[Tuple[int, int]] = []
    L = int(length_aa)
    step = int(max(1, stride_aa))
    # Slide 1..L_total (inclusive bounds)
    for s in range(1, L_total - L + 2, step):
        e = s + L - 1
        S.append((s, e))
    if not S:
        # fallback single window when L_total < L → clamp to sequence
        S.append((1, L_total))
    return S


def find_covered_variants(
    pos_min: np.ndarray, pos_max: np.ndarray, *, s: int, e: int
) -> np.ndarray:
    mask = (pos_min >= s) & (pos_max <= e)
    return np.nonzero(mask)[0]


def greedy_cover(
    windows: List[Window],
    score_plus: np.ndarray,
    *,
    W: int,
) -> Tuple[List[Window], List[float]]:
    """
    Greedy maximize F(S) with positive-part guard — standard monotone coverage.
    Returns (selected_windows, gains_per_step).
    """
    W = int(max(1, W))
    selected: List[Window] = []
    gains: List[float] = []
    covered = np.zeros_like(score_plus, dtype=bool)

    remain = {w.window_id: w for w in windows}
    for _ in range(W):
        best_id = None
        best_gain = 0.0
        for wid, w in remain.items():
            # marginal gain = sum score_plus of uncovered covered_idx
            idx = w.covered_idx[~covered[w.covered_idx]]
            gain = float(score_plus[idx].sum()) if idx.size else 0.0
            if gain > best_gain:
                best_gain = gain
                best_id = wid
        if best_id is None or best_gain <= 0.0:
            # stop when no positive marginal gains
            break
        w = remain.pop(best_id)
        selected.append(w)
        gains.append(best_gain)
        covered[w.covered_idx] |= True
    return selected, gains


def compute_windows_and_scores(
    *,
    aa_pos_lists: Sequence[Sequence[int]],
    score_plus: np.ndarray,
    L_total: int,
    length_aa: int,
    stride_aa: int,
) -> Tuple[List[Window], pd.DataFrame]:
    """
    Build windows and compute coverage sets & F_w for diagnostics.
    """
    pos_min = np.array(
        [min(x) if len(x) else np.inf for x in aa_pos_lists], dtype=float
    )
    pos_max = np.array(
        [max(x) if len(x) else -np.inf for x in aa_pos_lists], dtype=float
    )

    bounds = _make_windows(L_total, length_aa=length_aa, stride_aa=stride_aa)
    win_objs: List[Window] = []
    for wid, (s, e) in enumerate(bounds):
        idx = find_covered_variants(pos_min, pos_max, s=s, e=e)
        Fw = float(score_plus[idx].sum()) if idx.size else 0.0
        win_objs.append(
            Window(
                window_id=wid,
                start_aa=s,
                end_aa=e,
                covered_idx=np.array(sorted(idx.tolist()), dtype=int),
                F_w=Fw,
                covered_count=int(idx.size),
            )
        )

    # summary frame (all windows)
    wdf = pd.DataFrame(
        {
            "window_id": [w.window_id for w in win_objs],
            "start_aa": [w.start_aa for w in win_objs],
            "end_aa": [w.end_aa for w in win_objs],
            "F_w": [w.F_w for w in win_objs],
            "covered_count": [w.covered_count for w in win_objs],
        }
    ).sort_values(["F_w", "covered_count", "start_aa"], ascending=[False, False, True])
    return win_objs, wdf
