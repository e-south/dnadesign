"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/window_score_mass.py

Mass of positive-part scores along the AA axis, with selected windows shaded.
Split from protocols/multisite_select to central plots/ module.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_mass(
    *,
    L_total: int,
    aa_pos_lists: Sequence[Sequence[int]],
    score_plus: np.ndarray,
    normalize_by_k: bool = False,
) -> pd.DataFrame:
    """
    mass[j] = Σ score_plus(v)/k(v) if normalize_by_k, else Σ score_plus(v) for j ∈ mutated positions of v
    """
    mass = np.zeros(L_total + 1, dtype=float)  # 1-indexed
    mutated_any = np.zeros(L_total + 1, dtype=bool)
    for pos_list, s in zip(aa_pos_lists, score_plus):
        if s <= 0 or not pos_list:
            continue
        denom = max(1, len(pos_list)) if normalize_by_k else 1
        for j in pos_list:
            if 1 <= j <= L_total:
                mass[j] += s / denom
                mutated_any[j] = True
    idx = np.arange(1, L_total + 1)
    df = pd.DataFrame({"aa_index": idx, "mass": mass[1:], "mutated_any": mutated_any[1:]})
    return df


def render_mass(
    *,
    df_mass: pd.DataFrame,
    windows: pd.DataFrame,  # has 'start_aa','end_aa','rank'
    aa_letters: Optional[Sequence[str]],
    out_png: str | Path,
    title: str,
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
):
    fs = float(font_scale) if font_scale else 1.0
    x = df_mass["aa_index"].to_numpy()
    y = df_mass["mass"].to_numpy()

    if figsize is None and aa_letters is not None:
        L = len(aa_letters)
        # ~0.06" per residue gives room for per-residue labels; min 12", max 60"
        width = max(12.0, min(60.0, 0.06 * L))
        figsize = (width, 3.2)
    fig = plt.figure(figsize=figsize or (12, 3.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(x, y, lw=1.2)
    ax.fill_between(x, y, 0.0, alpha=0.25, step="pre")

    # shade selected windows (rank 1 darkest)
    if windows is not None and not windows.empty:
        for _, r in windows.sort_values("rank").iterrows():
            s, e, rk = int(r["start_aa"]), int(r["end_aa"]), int(r["rank"])
            alpha = 0.25 if rk > 1 else 0.40
            ax.axvspan(s, e, color="gray", alpha=alpha, lw=0)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel("Score mass (Σ score⁺)")
    ax.set_xlabel("AA position (1-indexed)")

    if aa_letters is not None and len(aa_letters) == len(x):
        L = len(aa_letters)
        # per-residue tick granularity
        xticks = np.arange(1, L + 1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([aa_letters[i - 1] for i in xticks], rotation=0, fontsize=int(round(8 * fs)))

    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
