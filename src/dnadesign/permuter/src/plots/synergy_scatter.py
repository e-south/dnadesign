"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/synergy_scatter.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_LOG = logging.getLogger("permuter.plot.synergy_scatter")


def _require(df: pd.DataFrame, cols: List[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"synergy_scatter: missing required column(s) for {ctx}: {missing}\n"
            "Expected canonical columns from evaluation/epistasis normalization."
        )


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
) -> None:
    if not metric_id:
        raise RuntimeError("synergy_scatter: metric_id is required")
    obs_col = f"permuter__observed__{metric_id}"
    exp_col = f"permuter__expected__{metric_id}"
    _require(all_df, [obs_col, exp_col], "canonical observed/expected")

    # Prefer round-2 variants
    df = all_df.copy()
    if "permuter__round" in df.columns and (df["permuter__round"] == 2).any():
        df = df[df["permuter__round"] == 2]

    keep_cols: List[str] = [obs_col, exp_col]
    if "aa_combo_str" in df.columns:
        keep_cols.append("aa_combo_str")
    df = df[keep_cols].dropna().copy()
    if df.empty:
        raise RuntimeError(
            "synergy_scatter: no rows with both observed and expected values"
        )

    obs = df[obs_col].astype(float).to_numpy()
    exp = df[exp_col].astype(float).to_numpy()
    delta = obs - exp

    # Preview (top‑5 by |Δ|) with additive vs observed (optional, for logs).
    try:
        preview = (
            df.assign(_obs=obs, _exp=exp, _delta=delta)
            .assign(_abs=lambda x: x["_delta"].abs())
            .sort_values("_abs", ascending=False, kind="mergesort")
            .head(5)
        )
        blocks = []
        for _, r in preview.iterrows():
            combo = str(r["aa_combo_str"]) if "aa_combo_str" in df.columns else ""
            plus = combo.replace("|", " + ") if combo else "combo"
            semi = combo.replace("|", ";") if combo else "combo"
            line1 = f"{plus} = {r['_exp']:+.3f}  (expected)"
            line2 = f"{semi:<{len(plus)}} = {r['_obs']:+.3f}  (observed)  → Δ={r['_delta']:+.3f}"
            blocks.append("  " + line1 + "\n  " + line2)
        if blocks:
            _LOG.info("[synergy] top‑5 by |Δ|:\n%s", "\n".join(blocks))
    except Exception as _e:
        _LOG.debug("synergy preview skipped: %s", _e)

    fs = float(font_scale) if font_scale else 1.0
    if figsize:
        fig_w, fig_h = float(figsize[0]), float(figsize[1])
    else:
        fig_w, fig_h = (6.0, 6.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    # y=x diagonal
    lim_min = float(min(np.min(obs), np.min(exp)))
    lim_max = float(max(np.max(obs), np.max(exp)))
    if lim_min == lim_max:
        lim_min, lim_max = lim_min - 1e-6, lim_max + 1e-6
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max], color="0.80", linewidth=1.2, zorder=0.3
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    # Scatter colored by Δ
    sc = ax.scatter(
        exp,
        obs,
        c=delta,
        s=22,
        alpha=0.38,
        cmap="coolwarm",
        edgecolors="none",
        zorder=1.6,
    )

    rmse = float(np.sqrt(np.mean((obs - exp) ** 2)))
    text = f"RMSE={rmse:.3f}"

    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=int(round(9.5 * fs)),
        bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.8, pad=5.0),
    )

    # Labels
    ax.set_xlabel("Expected additive (from singles)", fontsize=int(round(11 * fs)))
    ax.set_ylabel(f"Observed {metric_id}", fontsize=int(round(11 * fs)))
    ax.tick_params(labelsize=int(round(10 * fs)))
    ax.set_axisbelow(True)
    ax.grid(True, color="0.92", linewidth=0.7, zorder=0.1)
    ax.margins(x=0.02, y=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title + subtitle
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

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.84)
    try:
        cbar.outline.set_visible(False)
    except Exception:
        pass
    cbar.set_label("Δ = observed − expected", rotation=90, fontsize=int(round(10 * fs)))
    cbar.ax.tick_params(labelsize=int(round(9 * fs)))

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
