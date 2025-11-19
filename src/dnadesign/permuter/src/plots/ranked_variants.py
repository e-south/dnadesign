"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/ranked_variants.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _require(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"ranked_variants: missing required column(s) for {ctx}: {missing}\n"
            "Expected canonical columns from combine/eval/epistasis stages."
        )


def _observed_col(metric_id: str) -> str:
    return f"permuter__observed__{metric_id}"


def _stable_jitter(keys: List[str], width: float) -> np.ndarray:
    """
    Deterministic jitter in [-width, +width] per key, using MD5 (stable across runs).
    """
    denom = float(2**128)
    vals = []
    for k in keys:
        h = hashlib.md5((k or "").encode("utf-8")).hexdigest()
        u = int(h, 16) / denom  # [0,1)
        vals.append((u * 2.0 - 1.0) * width)
    return np.asarray(vals, dtype="float64")


def plot(
    elite_df: pd.DataFrame,
    all_df: pd.DataFrame,
    output_path: Path,
    job_name: str,
    ref_sequence: Optional[str] = None,  # unused
    metric_id: Optional[str] = None,
    evaluators: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    font_scale: Optional[float] = None,
    ranked_jitter: float = 0.18,
    ranked_point_size: float = 18.0,
    ranked_alpha: float = 0.45,
    ranked_cmap: str = "coolwarm",
) -> None:
    if not metric_id:
        raise RuntimeError("ranked_variants: metric_id is required")
    df = all_df.copy()

    # Prefer round-2 combinations
    if "permuter__round" in df.columns and (df["permuter__round"] == 2).any():
        df = df[df["permuter__round"] == 2].copy()
    obs_col = _observed_col(metric_id)
    _require(
        df, ["mut_count", "aa_combo_str", obs_col, "epistasis"], "k‑categorical scatter"
    )
    df = df.dropna(subset=[obs_col, "mut_count", "epistasis"]).copy()
    if df.empty:
        raise RuntimeError("ranked_variants: no rows with observed/epistasis to plot")
    df["mut_count"] = df["mut_count"].astype(int)

    # Title scaffold
    fs = float(font_scale) if font_scale else 1.0
    if figsize:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    else:
        fig_w, fig_h = (10.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.90", linewidth=0.7)

    # Categorical x positions for each k and deterministic jitter per variant
    ks: List[int] = sorted(df["mut_count"].unique().tolist())
    x_index: Dict[int, float] = {k: float(i) for i, k in enumerate(ks)}
    x_vals = df["mut_count"].map(x_index).astype(float).to_numpy()
    if ranked_jitter > 0:
        x_vals = x_vals + _stable_jitter(
            df["aa_combo_str"].astype(str).tolist(), float(ranked_jitter)
        )

    # Scatter: hue = epistasis (continuous)
    sc = ax.scatter(
        x_vals,
        df[obs_col].astype(float).to_numpy(),
        c=df["epistasis"].astype(float).to_numpy(),
        s=float(ranked_point_size),
        alpha=float(ranked_alpha),
        edgecolors="none",
        cmap=str(ranked_cmap),
        zorder=1.0,
    )

    # X ticks as categories
    ax.set_xticks([x_index[k] for k in ks])
    ax.set_xticklabels([str(k) for k in ks], rotation=0)
    ax.set_xlabel("Mutation count k", fontsize=int(round(11 * fs)))
    ax.set_ylabel(f"Observed {metric_id}", fontsize=int(round(11 * fs)))
    ax.tick_params(axis="x", labelsize=int(round(10 * fs)))
    ax.tick_params(axis="y", labelsize=int(round(10 * fs)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title = f"{job_name}"
    fig.suptitle(title, fontsize=int(round(13 * fs)))
    if evaluators:
        fig.text(
            0.5,
            0.96,
            evaluators,
            ha="center",
            va="top",
            fontsize=int(round(9.5 * fs)),
            alpha=0.75,
        )

    # Colorbar for epistasis
    cbar = fig.colorbar(sc, ax=ax, shrink=0.88)
    try:
        cbar.outline.set_visible(False)
    except Exception:
        pass
    cbar.set_label(
        "Epistasis (observed - expected)", rotation=90, fontsize=int(round(10 * fs))
    )
    cbar.ax.tick_params(labelsize=int(round(9 * fs)))

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
